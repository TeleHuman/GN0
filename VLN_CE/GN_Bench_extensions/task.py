import attr
from pathlib import Path

from typing import Dict, List, Optional, Any
from GN_Bench.config import Config
from GN_Bench.core.dataset import Dataset, Episode
from GN_Bench.tasks.nav.nav import NavigationGoal
from GN_Bench.core.registry import registry


@attr.s(auto_attribs=True, kw_only=True)
class GNBenchEpisode(Episode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    instruction: str = attr.ib(default=None)
    grounded_instruction: str = attr.ib(default=None)
    ref_json: str = attr.ib(default=None)
    trajectory_data: Dict[str, Any] = attr.ib(default=None)

    label_info: Dict[str, Any] = attr.ib(default=None)
    start_info: Dict[str, Any] = attr.ib(default=None)
    goal_info: Dict[str, Any] = attr.ib(default=None)
    path_info: Dict[str, Any] = attr.ib(default=None)
    start_facing_info: Dict[str, Any] = attr.ib(default=None)


@registry.register_dataset(name="GN_Matrix")
class GN_MatrixV0(Dataset):
    """Loads the GN_Bench Dataset."""

    episodes: List[GNBenchEpisode]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        parquet_path = self._get_parquet_path(config)
        self._load_from_parquet(
            parquet_path=parquet_path,
            scenes_dir_root=config.SCENES_DIR,
        )

    @staticmethod
    def _get_parquet_path(config: Config) -> Path:
        explicit_path = getattr(config, "PARQUET_PATH", None)
        if explicit_path:
            parquet_path = Path(str(explicit_path))
        else:
            parquet_path = Path(str(config.DATA_PATH))

        if parquet_path.suffix.lower() != ".parquet":
            raise ValueError(
                "GN_Matrix now expects a Parquet trajectory file. "
                f"Got DATA_PATH={config.DATA_PATH!r}"
            )

        return parquet_path

    @staticmethod
    def _read_parquet_rows(parquet_path: Path) -> List[Dict[str, Any]]:
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required when DATASET.DATA_PATH points to a Parquet "
                "file. Install it with: pip install pyarrow"
            ) from exc

        return pq.read_table(parquet_path).to_pylist()

    def _load_from_parquet(self, parquet_path: Path, scenes_dir_root: str) -> None:
        rows = self._read_parquet_rows(parquet_path)

        for episode_id, row in enumerate(rows, start=1):
            self.from_parquet_row(
                row=row,
                scenes_dir_root=scenes_dir_root,
                episode_id=episode_id,
            )

    @staticmethod
    def _require_row_value(row: Dict[str, Any], key: str) -> Any:
        if key not in row or row[key] is None:
            raise ValueError(f"Parquet row is missing required column '{key}'")
        return row[key]

    @classmethod
    def _build_path_info(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        world_x = cls._require_row_value(row, "path_raster_world_x")
        world_y = cls._require_row_value(row, "path_raster_world_y")
        pixel_x = cls._require_row_value(row, "path_raster_pixel_x")
        pixel_y = cls._require_row_value(row, "path_raster_pixel_y")

        lengths = {len(world_x), len(world_y), len(pixel_x), len(pixel_y)}
        if len(lengths) != 1:
            raise ValueError(
                "Parquet row has inconsistent path list lengths: "
                f"world_x={len(world_x)}, world_y={len(world_y)}, "
                f"pixel_x={len(pixel_x)}, pixel_y={len(pixel_y)}"
            )

        return {
            "raster_world": [
                {"x": float(x), "y": float(y)} for x, y in zip(world_x, world_y)
            ],
            "raster_pixel": [[int(x), int(y)] for x, y in zip(pixel_x, pixel_y)],
        }

    @classmethod
    def _build_trajectory_data(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        path_info = cls._build_path_info(row)
        label_info = {
            "ins_id": cls._require_row_value(row, "label_ins_id"),
            "label": cls._require_row_value(row, "label_label"),
            "goal_pixel": [
                int(cls._require_row_value(row, "label_goal_pixel_x")),
                int(cls._require_row_value(row, "label_goal_pixel_y")),
            ],
        }
        start_info = {
            "world": {
                "x": float(cls._require_row_value(row, "start_world_x")),
                "y": float(cls._require_row_value(row, "start_world_y")),
            }
        }
        start_facing_info = {
            "world": {
                "unit_dx": float(cls._require_row_value(row, "start_facing_unit_dx")),
                "unit_dy": float(cls._require_row_value(row, "start_facing_unit_dy")),
                "heading_degrees": float(
                    cls._require_row_value(row, "start_facing_heading_degrees")
                ),
            }
        }
        goal_info = {
            "world": {
                "x": float(cls._require_row_value(row, "goal_world_x")),
                "y": float(cls._require_row_value(row, "goal_world_y")),
            }
        }

        return {
            "scene": cls._require_row_value(row, "scene"),
            "label": label_info,
            "start": start_info,
            "start_facing": start_facing_info,
            "goal": goal_info,
            "grounded_instruction": cls._require_row_value(row, "grounded_instruction"),
            "instruction": cls._require_row_value(row, "instruction"),
            "path": path_info,
        }

    def from_parquet_row(
        self,
        row: Dict[str, Any],
        scenes_dir_root: str,
        episode_id: int,
    ) -> bool:
        data = self._build_trajectory_data(row)
        scene_name = str(data["scene"])
        trajectory_id = str(self._require_row_value(row, "trajectory_id"))
        ref_json = f"{scene_name}/{trajectory_id}.json"
        scene_id = str(Path(scenes_dir_root) / scene_name)

        start_position = [
            float(data["start"]["world"]["x"]),
            float(data["start"]["world"]["y"]),
        ]
        start_rotation = [float(data["start_facing"]["world"]["heading_degrees"])]

        goal_info = data["goal"]
        episode = GNBenchEpisode(
            episode_id=str(episode_id),
            scene_id=scene_id,
            ref_json=str(ref_json),
            start_position=start_position,
            start_rotation=start_rotation,
            instruction=data.get("instruction"),
            grounded_instruction=data.get("grounded_instruction"),
            trajectory_data=data,
            label_info=data.get("label"),
            start_info=data.get("start"),
            goal_info=goal_info,
            path_info=data.get("path"),
            start_facing_info=data.get("start_facing"),
        )
        episode.goals = [
            NavigationGoal(
                position=[
                    goal_info["world"]["x"],
                    goal_info["world"]["y"],
                    1.3,
                ]
            )
        ]

        self.episodes.append(episode)
        return True
