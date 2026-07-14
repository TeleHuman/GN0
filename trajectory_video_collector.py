from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from GN_Bench.tasks.nav.mpc_tools import smooth_polyline


DEFAULT_VIDEO_FPS = 10.0
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_SMOOTH_WINDOW = 9
DEFAULT_RESAMPLE_STEP_M = 0.05
DEFAULT_HEADING_LOOKAHEAD_M = 0.5
DEFAULT_YAW_SMOOTH_WINDOW = 9


class TrajectoryVideoCollector:
    """Render smoothed ground-truth trajectory paths as per-episode mp4 files."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        fps: float = DEFAULT_VIDEO_FPS,
        codec: str = DEFAULT_VIDEO_CODEC,
        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
        resample_step_m: float = DEFAULT_RESAMPLE_STEP_M,
        heading_lookahead_m: float = DEFAULT_HEADING_LOOKAHEAD_M,
        yaw_smooth_window: int = DEFAULT_YAW_SMOOTH_WINDOW,
    ) -> None:
        self.output_root = Path(output_root)
        self.fps = float(fps)
        self.codec = str(codec or DEFAULT_VIDEO_CODEC)
        self.smooth_window = int(smooth_window)
        self.resample_step_m = max(0.0, float(resample_step_m))
        self.heading_lookahead_m = max(0.0, float(heading_lookahead_m))
        self.yaw_smooth_window = int(yaw_smooth_window)

    def collect(self, *, episode, sim) -> tuple[Path, int]:
        positions = self._build_smoothed_positions(episode=episode, sim=sim)
        start_rotation = np.asarray(sim.get_agent_state().rotation, dtype=np.float32)
        rotations = self._rotations_from_path_positions(positions, start_rotation)
        frames = self._render_poses(sim=sim, positions=positions, rotations=rotations)
        if len(frames) != len(positions):
            raise RuntimeError(
                f"Rendered {len(frames)} frames for {len(positions)} trajectory poses."
            )

        output_path = self._output_path(episode)
        self._write_video(output_path, frames)
        return output_path, len(frames)

    def _build_smoothed_positions(self, *, episode, sim) -> list[np.ndarray]:
        path_xy, path_px = self._episode_path(episode)
        z = float(np.asarray(sim.get_agent_state().position, dtype=np.float32)[2])

        los_positions = self._line_of_sight_positions(
            sim=sim,
            path_px=path_px,
            z=z,
        )
        if los_positions:
            return los_positions

        if path_xy.shape[0] == 0 and getattr(sim, "raster_world", None) is not None:
            path_xy = np.asarray(sim.raster_world, dtype=np.float64).reshape(-1, 2)
        if path_xy.shape[0] == 0:
            current = np.asarray(sim.get_agent_state().position, dtype=np.float32)
            return [current.copy()]

        smoothed_xy = smooth_polyline(
            path_xy,
            mode="vel",
            win=self.smooth_window,
            kind="tri",
        )
        return [
            np.array([float(x), float(y), z], dtype=np.float32)
            for x, y in np.asarray(smoothed_xy, dtype=np.float32)
        ]

    @staticmethod
    def _episode_path(episode) -> tuple[np.ndarray, list[tuple[int, int]]]:
        path_info: dict[str, Any] | None = getattr(episode, "path_info", None)
        if path_info is None:
            trajectory_data = getattr(episode, "trajectory_data", {}) or {}
            path_info = trajectory_data.get("path", {})

        raster_world = (path_info or {}).get("raster_world", [])
        world_points: list[tuple[float, float]] = []
        for point in raster_world:
            if isinstance(point, dict):
                world_points.append((float(point["x"]), float(point["y"])))
            else:
                world_points.append((float(point[0]), float(point[1])))

        raster_pixel = (path_info or {}).get("raster_pixel", [])
        pixel_points: list[tuple[int, int]] = []
        for point in raster_pixel:
            pixel_points.append((int(point[0]), int(point[1])))

        return np.asarray(world_points, dtype=np.float64).reshape(-1, 2), pixel_points

    def _line_of_sight_positions(
        self,
        *,
        sim,
        path_px: list[tuple[int, int]],
        z: float,
    ) -> list[np.ndarray]:
        if not path_px or not hasattr(sim, "_smooth_path"):
            return []
        try:
            smooth_px = sim._smooth_path(path_px)
        except Exception as exc:
            print(f"[Collecter] line-of-sight path smoothing failed: {exc}")
            return []
        if not smooth_px:
            return []

        world_points = []
        for px in smooth_px:
            wx, wy = sim.transform_from_pixel_to_world(px)
            world_points.append(np.array([wx, wy, z], dtype=np.float32))

        if self.resample_step_m <= 1e-6:
            return world_points
        return self._resample_world_path(world_points, step_m=self.resample_step_m)

    @staticmethod
    def _resample_world_path(points, *, step_m: float) -> list[np.ndarray]:
        cleaned = []
        for point in points:
            arr = np.asarray(point, dtype=np.float32).reshape(-1)
            if arr.size < 3:
                arr = np.pad(arr, (0, 3 - arr.size), mode="constant")
            arr = arr[:3].astype(np.float32, copy=True)
            if cleaned and float(np.linalg.norm(arr[:2] - cleaned[-1][:2])) < 1e-6:
                continue
            cleaned.append(arr)

        if len(cleaned) <= 1:
            return cleaned

        points_np = np.stack(cleaned, axis=0)
        segment_lengths = np.linalg.norm(points_np[1:, :2] - points_np[:-1, :2], axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = float(cumulative[-1])
        if total_length <= 1e-6:
            return [points_np[0].copy()]

        num_points = max(2, int(math.ceil(total_length / float(step_m))) + 1)
        sampled = []
        segment_index = 0
        for target in np.linspace(0.0, total_length, num_points):
            while (
                segment_index + 1 < len(cumulative) - 1
                and cumulative[segment_index + 1] < target
            ):
                segment_index += 1
            seg_start = cumulative[segment_index]
            seg_end = cumulative[segment_index + 1]
            alpha = (
                0.0
                if seg_end <= seg_start
                else (target - seg_start) / (seg_end - seg_start)
            )
            point = (
                points_np[segment_index] * (1.0 - alpha)
                + points_np[segment_index + 1] * alpha
            )
            sampled.append(point.astype(np.float32))
        return sampled

    def _rotations_from_path_positions(
        self,
        positions,
        start_rotation,
    ) -> list[np.ndarray]:
        if not positions:
            return []

        start_forward = R.from_quat(start_rotation).apply(
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        start_yaw = float(math.atan2(start_forward[1], start_forward[0]))
        points = [np.asarray(position, dtype=np.float32) for position in positions]
        yaws = self._path_yaws(points, start_yaw)
        yaws = self._smooth_yaws(yaws)

        rotations = []
        for yaw in yaws:
            rotations.append(
                R.from_euler("z", float(yaw), degrees=False)
                .as_quat()
                .astype(np.float32)
            )
        return rotations

    def _path_yaws(self, points: list[np.ndarray], start_yaw: float) -> np.ndarray:
        if len(points) == 1:
            return np.asarray([start_yaw], dtype=np.float64)

        points_xy = np.asarray([point[:2] for point in points], dtype=np.float64)
        segment_lengths = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        last_yaw = float(start_yaw)
        yaws = [last_yaw]

        for index in range(1, len(points)):
            if self.heading_lookahead_m > 1e-6:
                target_s = min(
                    float(cumulative[-1]),
                    float(cumulative[index]) + self.heading_lookahead_m,
                )
                if target_s > float(cumulative[index]) + 1e-8:
                    target_xy = self._point_at_s(points_xy, cumulative, target_s)
                    delta = target_xy - points_xy[index]
                else:
                    delta = points_xy[index] - points_xy[index - 1]
            else:
                if index + 1 < len(points):
                    delta = points_xy[index + 1] - points_xy[index]
                else:
                    delta = points_xy[index] - points_xy[index - 1]
            if float(np.linalg.norm(delta)) > 1e-6:
                last_yaw = float(math.atan2(float(delta[1]), float(delta[0])))
            yaws.append(last_yaw)
        return np.asarray(yaws, dtype=np.float64)

    @staticmethod
    def _point_at_s(
        points_xy: np.ndarray, cumulative: np.ndarray, s: float
    ) -> np.ndarray:
        s = float(np.clip(s, 0.0, float(cumulative[-1])))
        index = int(np.searchsorted(cumulative, s, side="right") - 1)
        index = max(0, min(index, len(cumulative) - 2))
        denom = max(float(cumulative[index + 1] - cumulative[index]), 1e-12)
        alpha = (s - float(cumulative[index])) / denom
        return points_xy[index] * (1.0 - alpha) + points_xy[index + 1] * alpha

    def _smooth_yaws(self, yaws: np.ndarray) -> np.ndarray:
        win = int(self.yaw_smooth_window)
        if win <= 1 or len(yaws) < 3:
            return yaws
        if win % 2 == 0:
            win += 1
        if len(yaws) < win:
            return yaws

        unwrapped = np.unwrap(np.asarray(yaws, dtype=np.float64))
        pad = win // 2
        kernel = np.concatenate([np.arange(1, pad + 2), np.arange(pad, 0, -1)]).astype(
            np.float64
        )
        kernel /= np.sum(kernel)
        smoothed = np.convolve(
            np.pad(unwrapped, (pad, pad), mode="edge"),
            kernel,
            mode="valid",
        )
        smoothed[0] = unwrapped[0]
        return smoothed

    def _render_poses(self, *, sim, positions, rotations) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        saved_state = sim.get_agent_state()
        saved_position = np.asarray(saved_state.position, dtype=np.float32).copy()
        saved_rotation = np.asarray(saved_state.rotation, dtype=np.float32).copy()
        saved_actor_cursor = getattr(sim, "actor_render_cursor", None)
        try:
            for position, rotation in zip(positions, rotations):
                obs = sim.get_observations_at(
                    position=np.asarray(position, dtype=np.float32).tolist(),
                    rotation=np.asarray(rotation, dtype=np.float32).tolist(),
                    keep_agent_at_new_pose=False,
                )
                if obs is None or "rgb" not in obs:
                    break
                frames.append(self._ensure_rgb(obs["rgb"]))
        finally:
            sim.set_agent_state(saved_position, saved_rotation, reset_sensors=False)
            if saved_actor_cursor is not None:
                sim.actor_render_cursor = saved_actor_cursor
        return frames

    def _output_path(self, episode) -> Path:
        ref_path = Path(getattr(episode, "ref_json"))
        return self.output_root / ref_path.parent.name / f"{ref_path.stem}.mp4"

    @staticmethod
    def _ensure_rgb(frame) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame HWC, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _write_video(self, path: Path, frames: list[np.ndarray]) -> None:
        if not frames:
            raise ValueError(f"No frames to write for {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.stem}.tmp.{os.getpid()}.mp4")

        first = self._ensure_rgb(frames[0])
        height, width = first.shape[:2]
        if self._is_h264_codec():
            self._write_h264_video(tmp_path, frames, width=width, height=height)
            os.replace(tmp_path, path)
            return

        writer = cv2.VideoWriter(
            str(tmp_path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {tmp_path}")
        try:
            for frame in frames:
                rgb = self._ensure_rgb(frame)
                if rgb.shape[:2] != (height, width):
                    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

        os.replace(tmp_path, path)

    def _is_h264_codec(self) -> bool:
        return self.codec.lower() in {"h264", "x264", "avc1", "libx264"}

    def _write_h264_video(
        self,
        path: Path,
        frames: list[np.ndarray],
        *,
        width: int,
        height: int,
    ) -> None:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(path),
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        try:
            for frame in frames:
                rgb = self._ensure_rgb(frame)
                if rgb.shape[:2] != (height, width):
                    rgb = cv2.resize(
                        rgb,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )
                writer.append_data(rgb)
        finally:
            writer.close()
