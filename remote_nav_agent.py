"""GN0 agent wrapper for generic remote navigation policies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from GN_Bench.core.agent import Agent
from remote_nav_client import RemoteNavClient


ACTION_STOP = 0
ACTION_ALIASES = {
    "stop": 0,
    "<stop>": 0,
    "move_forward": 1,
    "forward": 1,
    "fwd": 1,
    "<fwd>": 1,
    "turn_left": 2,
    "left": 2,
    "<left>": 2,
    "turn_right": 3,
    "right": 3,
    "<right>": 3,
}


class RemoteNavAgent(Agent):
    """Calls a remote WebSocket policy and adapts actions to GN0.

    Supported response schemas:
      - {"action_type": "discrete", "actions": [0, 1, 2, 3]}
      - {"action_type": "nav_delta", "actions": [[dx, dy, dyaw], ...]}
      - Legacy nested nav_delta: {"action.nav_delta": [[dx, dy, dyaw], ...]}

    The agent executes at most ``action_num`` returned actions. If the server
    returns fewer actions, the next request sends exactly the RGB frames observed
    while executing those actions.
    """

    def __init__(
        self,
        model_path: str | None = None,
        result_path: str = "tmp/remote_nav",
        prompt_type: str = "V1",
        action_num: int = 1,
        server_host: str = "127.0.0.1",
        server_port: int = 8000,
        request_timeout: float = 120.0,
        connect_timeout: float = 600.0,
        metadata_timeout: float = 2.0,
        stop_distance: float | None = None,
        stop_eps: float = 1e-3,
        translation_frame: str = "chunk_start",
        max_translation: float = 0.0,
        max_yaw: float = 0.0,
        save_images: bool = False,
        action_format: str = "auto",
        send_state: bool = False,
        **_: Any,
    ) -> None:
        del model_path, prompt_type, stop_distance
        self.result_path = result_path
        self.action_num = max(1, int(action_num))
        self.stop_eps = max(0.0, float(stop_eps))
        self.translation_frame = str(translation_frame)
        if self.translation_frame not in {"chunk_start", "current"}:
            raise ValueError(
                "translation_frame must be 'chunk_start' or 'current', "
                f"got {self.translation_frame!r}"
            )
        self.max_translation = float(max_translation)
        self.max_yaw = float(max_yaw)
        self.save_images = bool(save_images)
        self.action_format = self._normalize_action_type(action_format)
        if self.action_format not in {"auto", "discrete", "nav_delta"}:
            raise ValueError(
                "action_format must be auto, discrete, or nav_delta, "
                f"got {action_format!r}"
            )
        self.send_state = bool(send_state)

        print(
            "Initialize remote nav client: "
            f"ws://{server_host}:{server_port}, action_num={self.action_num}, "
            f"action_format={self.action_format}, "
            f"translation_frame={self.translation_frame}"
        )
        self.client = RemoteNavClient(
            host=server_host,
            port=server_port,
            request_timeout=request_timeout,
            connect_timeout=connect_timeout,
            metadata_timeout=metadata_timeout,
        )
        print("Remote nav server metadata:", self.client.metadata)

        self.episode_id: str | None = None
        self.image_path: Path | None = None
        self.pending_actions: list[Any] = []
        self.pending_action_type = "auto"
        self.chunk_base_rotation: R | None = None
        self.frames_since_query: list[np.ndarray] = []
        self.query_count = 0
        self.step_idx = 0
        self.history_actions: list[Any] = []

    def reset(self, episode_ref, sim=None) -> None:
        del sim
        episode_ref = Path(episode_ref)
        self.episode_id = f"{episode_ref.parent.name}_{episode_ref.stem}"
        self.pending_actions = []
        self.pending_action_type = "auto"
        self.chunk_base_rotation = None
        self.frames_since_query = []
        self.query_count = 0
        self.step_idx = 0
        self.history_actions = []
        self.image_path = Path(self.result_path) / self.episode_id / "image"
        if self.save_images:
            self.image_path.mkdir(parents=True, exist_ok=True)
        try:
            self.client.reset()
        except Exception as exc:
            print(
                "[RemoteNav] reset request failed, continuing with session_id reset: "
                f"{exc}"
            )
        print("RemoteNav Reset Complete for Episode:", self.episode_id)

    def act(self, observations, info):
        sim = observations.get("sim")
        if sim is None:
            raise ValueError(
                "RemoteNavAgent requires observations['sim'] from run_remote_eval.py"
            )

        rgb = observations.get("rgb")
        if rgb is not None:
            self._record_rgb(rgb)
            if self.save_images:
                self._save_observation_images(sim, rgb)

        _ = info

        if not self.pending_actions:
            self._query_new_actions(observations, sim)

        if not self.pending_actions:
            return {"action": ACTION_STOP}

        action = self.pending_actions.pop(0)
        action_type = self.pending_action_type
        self.history_actions.append(self._history_value(action, action_type))
        self.step_idx += 1

        if action_type == "discrete":
            action_id = int(action)
            if action_id == ACTION_STOP:
                self.pending_actions = []
            return {"action": action_id}

        nav_delta = np.asarray(action, dtype=np.float32)
        if self._is_stop_nav_delta(nav_delta):
            self.pending_actions = []
            return {"action": ACTION_STOP}
        return self._nav_delta_to_env_action(sim, nav_delta)

    def _query_new_actions(self, observations, sim) -> None:
        rgb = observations.get("rgb")
        instruction_payload = observations.get("instruction", {})
        if isinstance(instruction_payload, dict):
            instruction = instruction_payload.get("text", "")
        else:
            instruction = str(instruction_payload or "")

        if rgb is None:
            raise ValueError("RemoteNavAgent requires observations['rgb']")
        if self.episode_id is None:
            raise RuntimeError("RemoteNavAgent.act called before reset")

        agent_state = sim.get_agent_state()
        self.chunk_base_rotation = R.from_quat(agent_state.rotation)

        current_rgb, image_chunk = self._build_request_images(rgb)
        state = self._nav_pose(sim) if self.send_state else None
        response = self.client.infer(
            rgb=current_rgb,
            images=image_chunk,
            instruction=instruction,
            session_id=self.episode_id,
            state=state,
        )
        self.query_count += 1
        self.frames_since_query = []

        action_type, actions = self._extract_actions(response)
        keep = min(self.action_num, len(actions))
        self.pending_actions = list(actions[:keep])
        self.pending_action_type = action_type

        request_frames = 1 if image_chunk is None else int(image_chunk.shape[0])
        print(
            f"[RemoteNav] step={self.step_idx} request_frames={request_frames} "
            f"action_type={action_type} returned={len(actions)} keep={keep}"
        )

    def _record_rgb(self, rgb: np.ndarray) -> None:
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            return
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.frames_since_query.append(np.ascontiguousarray(frame.copy()))
        max_history = max(64, self.action_num * 4)
        if len(self.frames_since_query) > max_history:
            self.frames_since_query = self.frames_since_query[-max_history:]

    def _build_request_images(
        self, current_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        current = np.asarray(current_rgb)
        if current.dtype != np.uint8:
            current = np.clip(current, 0, 255).astype(np.uint8)
        if self.query_count == 0 or not self.frames_since_query:
            return current, None
        return current, np.stack(self.frames_since_query, axis=0)

    def _extract_actions(self, response: dict[str, Any]) -> tuple[str, list[Any]]:
        action_type, raw_actions = self._select_action_payload(response)
        action_type = self._resolve_action_type(action_type, raw_actions)
        if action_type == "discrete":
            return action_type, self._normalize_discrete_actions(raw_actions)
        return action_type, self._normalize_nav_delta_actions(raw_actions)

    def _select_action_payload(self, response: dict[str, Any]) -> tuple[str, Any]:
        explicit_type = (
            response.get("action_type")
            or response.get("action.type")
            or response.get("action_format")
        )
        action_type = (
            self._normalize_action_type(explicit_type) if explicit_type else "auto"
        )

        if "actions" in response:
            return action_type, response["actions"]
        if "action.nav_delta" in response:
            return "nav_delta", response["action.nav_delta"]
        if "action.discrete" in response:
            return "discrete", response["action.discrete"]
        if "action.vlnce" in response:
            return "discrete", response["action.vlnce"]
        if "action" in response:
            return action_type, response["action"]
        raise KeyError(f"Remote response has no action field: {response.keys()}")

    def _resolve_action_type(self, action_type: str, raw_actions: Any) -> str:
        if self.action_format != "auto":
            return self.action_format
        if action_type != "auto":
            return action_type

        if isinstance(raw_actions, (str, bytes)):
            return "discrete"
        arr = np.asarray(raw_actions)
        if arr.shape == ():
            return "discrete"
        if arr.dtype.kind in {"i", "u", "b"}:
            return "discrete"
        if arr.ndim >= 2 and arr.shape[-1] >= 3:
            return "nav_delta"
        if arr.ndim == 1 and arr.shape[0] == 3 and arr.dtype.kind == "f":
            return "nav_delta"
        return "discrete"

    @staticmethod
    def _normalize_action_type(value: Any) -> str:
        if value is None:
            return "auto"
        text = str(value).strip().lower().replace("-", "_")
        aliases = {
            "": "auto",
            "auto": "auto",
            "vlnce": "discrete",
            "vln_ce": "discrete",
            "discrete": "discrete",
            "int": "discrete",
            "nav": "nav_delta",
            "delta": "nav_delta",
            "nav_delta": "nav_delta",
            "continuous": "nav_delta",
            "relative_pose": "nav_delta",
            "relative": "nav_delta",
        }
        return aliases.get(text, text)

    @staticmethod
    def _normalize_discrete_actions(raw_actions: Any) -> list[int]:
        if isinstance(raw_actions, (str, bytes)):
            raw_actions = [raw_actions]
        arr = np.asarray(raw_actions, dtype=object)
        if arr.shape == ():
            values = [arr.item()]
        else:
            values = arr.reshape(-1).tolist()

        actions: list[int] = []
        for value in values:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str):
                key = value.strip().lower()
                if key not in ACTION_ALIASES:
                    raise ValueError(f"Unknown discrete action token: {value!r}")
                action = ACTION_ALIASES[key]
            else:
                action = int(value)
            actions.append(action)
        return actions

    @staticmethod
    def _normalize_nav_delta_actions(raw_actions: Any) -> list[np.ndarray]:
        arr = np.asarray(raw_actions, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] < 3:
                raise ValueError(f"Expected nav_delta shape (N, >=3), got {arr.shape}")
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[-1] < 3:
            raise ValueError(f"Expected nav_delta shape (N, >=3), got {arr.shape}")
        arr = arr[:, :3]
        finite_mask = np.isfinite(arr).all(axis=1)
        return [row.astype(np.float32) for row in arr[finite_mask]]

    def _is_stop_nav_delta(self, nav_delta: np.ndarray) -> bool:
        action = np.asarray(nav_delta, dtype=np.float32)[:3]
        return bool(np.all(np.abs(action) <= self.stop_eps))

    def _nav_delta_to_env_action(self, sim, nav_delta: np.ndarray) -> dict[str, Any]:
        dx, dy, dyaw = [float(v) for v in nav_delta[:3]]
        if self.max_translation > 0:
            dist = math.hypot(dx, dy)
            if dist > self.max_translation:
                scale = self.max_translation / max(dist, 1e-8)
                dx *= scale
                dy *= scale
        if self.max_yaw > 0:
            dyaw = max(-self.max_yaw, min(self.max_yaw, dyaw))

        state = sim.get_agent_state()
        curr_pos = np.array(state.position, dtype=np.float32)
        curr_rot = R.from_quat(state.rotation)

        if (
            self.translation_frame == "chunk_start"
            and self.chunk_base_rotation is not None
        ):
            trans_rot = self.chunk_base_rotation
        else:
            trans_rot = curr_rot

        world_delta = trans_rot.apply(np.array([dx, dy, 0.0], dtype=np.float32))
        new_pos = curr_pos + world_delta.astype(np.float32)
        new_pos[2] = curr_pos[2]

        new_rot = curr_rot * R.from_euler("z", dyaw, degrees=False)
        return {
            "action": {
                "position": new_pos.astype(float).tolist(),
                "rotation": new_rot.as_quat().astype(float).tolist(),
            }
        }

    @staticmethod
    def _nav_pose(sim) -> np.ndarray:
        state = sim.get_agent_state()
        pos = np.asarray(state.position, dtype=np.float32)
        yaw = R.from_quat(state.rotation).as_euler("zyx", degrees=False)[0]
        return np.asarray([pos[0], pos[1], yaw], dtype=np.float32)

    @staticmethod
    def _history_value(action: Any, action_type: str) -> Any:
        if action_type == "discrete":
            return int(action)
        return np.asarray(action, dtype=np.float32).astype(float).tolist()

    def _save_observation_images(self, sim, rgb: np.ndarray) -> None:
        self._save_image(rgb, "rgb")
        bev_traj = sim.get_bev_map_with_trajectory()
        self._save_image(bev_traj, "bev_traj")

    def _save_image(self, img: np.ndarray, prefix: str) -> None:
        if self.image_path is None or img is None:
            return
        save_dir = self.image_path / prefix
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"{self.step_idx}.png"
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(file_path), arr)
