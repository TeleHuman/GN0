import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R

from GN_Bench.core.agent import Agent
from bae import BAEInference
from bae.constants import (
    ACTION_STOP,
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
)


ACTION_TOKENS = {
    ACTION_STOP: "<STOP>",
    ACTION_MOVE_FORWARD: "<FWD>",
    ACTION_TURN_LEFT: "<LEFT>",
    ACTION_TURN_RIGHT: "<RIGHT>",
}
HISTORY_ACTION_LIMIT = 5
IMAGE_DIRS = {
    "rgb": "model_input/rgb",
    "hist": "model_input/rgb_history",
    "occ": "model_input/occ",
    "occ_traj": "trajectory_vis/occ_executed",
    "bev_traj": "trajectory_vis/bev_executed",
}


class BAEAgentBase(Agent):
    """Shared BAE agent utilities for evaluation and DAgger collection."""

    def __init__(
        self,
        model_path,
        result_path,
        prompt_type,
        action_num=1,
        dtype="bf16",
    ):
        print("Initialize BAE")

        self.result_path = result_path
        self.prompt_type = prompt_type
        self.action_num = max(1, int(action_num))

        self.inference = BAEInference(
            model_path=model_path,
            prompt_type=prompt_type,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            max_new_tokens=512,
        )

        self.rgb_list = []
        self.episode_id = None
        self.image_path = None
        self.history_action_list = []

        print("BAE Initialization Complete")

    def reset_episode_state(self, episode_ref) -> None:
        self.rgb_list = []
        self.history_action_list = []
        self.episode_id = f"{episode_ref.parent.name}_{episode_ref.stem}"

        self.image_path = Path(self.result_path) / self.episode_id / "image"
        self.image_path.mkdir(parents=True, exist_ok=True)

    def save_observation_images(self, sim, rgb):
        occ_map = sim.get_occ_map()
        bev_map = sim.get_bev_map() if self.prompt_type in {"V1", "V2"} else None
        occ_traj = sim.get_occ_map_with_trajectory()
        bev_traj = sim.get_bev_map_with_trajectory()

        occ_h, occ_w = occ_map.shape[:2]
        self.rgb_list.append(rgb)

        paths = {
            "rgb": self.save_image(rgb, IMAGE_DIRS["rgb"]),
            "occ": self.save_image(occ_map, IMAGE_DIRS["occ"]),
            "bev": self.to_pil_image(bev_map) if bev_map is not None else None,
            "occ_traj": self.save_image(occ_traj, IMAGE_DIRS["occ_traj"]),
            "bev_traj": self.save_image(bev_traj, IMAGE_DIRS["bev_traj"]),
        }
        return paths, occ_h, occ_w

    def build_prompt_image_paths(self, paths):
        if self.prompt_type in {"V1", "V3"}:
            hist = self.build_history_mosaic(
                self.rgb_list[:-1],
                self.history_action_list,
            )
            paths["hist"] = self.save_image(hist, IMAGE_DIRS["hist"])

        if self.prompt_type == "V1":
            self.require_bev_input(paths)
            return [paths["rgb"], paths["hist"], paths["bev"], paths["occ"]]
        if self.prompt_type == "V2":
            self.require_bev_input(paths)
            return [paths["bev"], paths["occ"]]
        if self.prompt_type == "V3":
            return [paths["rgb"], paths["hist"]]

        raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    @staticmethod
    def to_pil_image(img: np.ndarray) -> Image.Image:
        """Convert an RGB numpy image to an in-memory PIL image for model input."""
        if img is None:
            raise ValueError("Cannot convert empty image to PIL.")

        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img, mode="RGB")

    @staticmethod
    def require_bev_input(paths):
        if paths.get("bev") is None:
            raise ValueError(
                "Prompt V1/V2 requires a BEV image, but simulator returned none."
            )

    @staticmethod
    def get_current_pixel(sim):
        pixel_pos = sim.get_current_pixel_position()
        if pixel_pos is None or len(pixel_pos) < 2:
            return -1, -1
        return tuple(map(int, pixel_pos[:2]))

    @staticmethod
    def get_agent_pose(sim):
        agent_state = sim.get_agent_state()
        curr_pos = np.array(agent_state.position, dtype=np.float32)
        curr_rot = R.from_quat(agent_state.rotation)
        world_fwd = curr_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        curr_yaw_deg = float(np.degrees(np.arctan2(world_fwd[1], world_fwd[0])))
        return [float(curr_pos[0]), float(curr_pos[1]), curr_yaw_deg]

    def build_history_mosaic(
        self,
        rgb_list,
        history_action_list,
        grid_size=4,
        target_size=(640, 480),
        only_moved=False,
    ):
        tw, th = target_size
        num_tiles = grid_size**2
        tile_w, tile_h = tw // grid_size, th // grid_size

        if not only_moved:
            selected_frames = rgb_list.copy()
        else:
            selected_frames = []
            for idx, frame in enumerate(rgb_list):
                if idx == 0:
                    selected_frames.append(frame)
                    continue
                prev_action_idx = idx - 1
                if prev_action_idx >= len(history_action_list):
                    continue
                if int(history_action_list[prev_action_idx]) == ACTION_MOVE_FORWARD:
                    selected_frames.append(frame)

        recent_frames = selected_frames[-num_tiles:][::-1]
        padding = [np.zeros((tile_h, tile_w, 3), dtype=np.uint8)] * (
            num_tiles - len(recent_frames)
        )

        tiles = [cv2.resize(f, (tile_w, tile_h)) for f in recent_frames] + padding

        grid = np.array(tiles).reshape(grid_size, grid_size, tile_h, tile_w, 3)
        return grid.swapaxes(1, 2).reshape(th, tw, 3)

    def save_image(self, img: np.ndarray, prefix: str) -> str:
        """Save a numpy RGB image for the current step and return its path."""
        if img is None:
            raise ValueError(f"Cannot save empty image for prefix '{prefix}'.")

        save_dir = Path(self.image_path) / prefix
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{len(self.rgb_list) - 1}.png"
        filepath = save_dir / file_name

        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return str(filepath)

    def build_prev_actions_xml(self) -> str:
        recent = list(reversed(self.history_action_list))[:HISTORY_ACTION_LIMIT]
        tokens = [ACTION_TOKENS.get(int(a), "<None>") for a in recent]
        while len(tokens) < HISTORY_ACTION_LIMIT:
            tokens.append("<None>")
        return "<action>" + ",".join(tokens) + "</action>"
