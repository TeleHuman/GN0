import json
import attr
from pathlib import Path

from typing import Dict, List, Optional, Any
from GN_Bench.config import Config
from GN_Bench.core.dataset import Dataset, Episode
from GN_Bench.tasks.nav.nav import NavigationGoal
from GN_Bench.core.registry import registry


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"
ALL_EPISODES_MASK = "*"


@attr.s(auto_attribs=True, kw_only=True)
class GNBenchEpisode(Episode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    instruction: str = attr.ib(default=None)
    grounded_instruction: str = attr.ib(default=None)
    ref_json: str = attr.ib(default=None)

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

        scene_ids = []
        scenes_dir = Path(config.DATA_PATH)
        scene_to_traj_ids = {}

        if not scenes_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {scenes_dir}")

        dataset_config_path = getattr(config, "DATASET_CONFIG", None)
        if dataset_config_path:
            config_path = Path(str(dataset_config_path))
            if not config_path.exists():
                fallback_path = scenes_dir / config_path
                if fallback_path.exists():
                    config_path = fallback_path

            if not config_path.exists():
                raise FileNotFoundError(
                    f"DATASET_CONFIG file does not exist: {dataset_config_path}"
                )

            with config_path.open("r", encoding="utf-8") as f:
                dataset_config_data = json.load(f)

            target_splits = ["easy", "hard", "medium"]

            for split in target_splits:
                scenes = dataset_config_data.get(split)

                if not isinstance(scenes, dict):
                    raise ValueError(
                        f"DATASET_CONFIG must contain a dict under '{split}'"
                    )

                for scene_name, traj_info in scenes.items():
                    scene_name = str(scene_name).strip()
                    if not scene_name:
                        continue
                    if not isinstance(traj_info, dict):
                        continue

                    traj_ids = {
                        str(traj_id).strip()
                        for traj_id in traj_info.keys()
                        if str(traj_id).strip() and str(traj_id).strip().isdigit()
                    }

                    if traj_ids:
                        if scene_name not in scene_to_traj_ids:
                            scene_to_traj_ids[scene_name] = set()
                        scene_to_traj_ids[scene_name].update(traj_ids)

            scene_ids = list(scene_to_traj_ids.keys())

        elif "*" in config.CONTENT_SCENES:
            scene_ids = [p.name for p in scenes_dir.iterdir() if p.is_dir()]
        else:
            scene_ids = config.CONTENT_SCENES

        global_episode_id = 1

        for scene_name in sorted(scene_ids):
            scene_path = scenes_dir / scene_name

            if not scene_path.is_dir():
                continue

            if scene_name in scene_to_traj_ids:
                json_files = [
                    scene_path / f"{traj_id}.json"
                    for traj_id in sorted(scene_to_traj_ids[scene_name], key=int)
                ]
                json_files = [p for p in json_files if p.is_file()]
            else:
                json_files = list(scene_path.glob("*.json"))

            for json_file in sorted(json_files):
                name_without_ext = json_file.stem

                if not name_without_ext.isdigit():
                    continue

                with json_file.open("r", encoding="utf-8") as f:
                    if self.from_json(
                        f.read(),
                        scene_name=scene_name,
                        scenes_dir_root=config.SCENES_DIR,
                        episode_id=global_episode_id,
                        ref_json=str(json_file),
                    ):
                        global_episode_id += 1

    def from_json(
        self,
        json_str: str,
        scene_name: str,
        scenes_dir_root: str,
        episode_id: int,
        ref_json: str,
    ) -> None:
        data = json.loads(json_str)
        scene_id = str(Path(scenes_dir_root) / scene_name)

        raw_start_x = data["start"]["world"]["x"]
        raw_start_y = data["start"]["world"]["y"]
        start_position = [float(raw_start_x), float(raw_start_y)]

        try:
            raw_heading = data["start_facing"]["world"]["heading_degrees"]
        except:  # noqa: E722
            return False
        start_rotation = [float(raw_heading)]

        episode = GNBenchEpisode(
            episode_id=str(episode_id),
            scene_id=scene_id,
            ref_json=ref_json,
            start_position=start_position,
            start_rotation=start_rotation,
            instruction=data.get("instruction"),
            grounded_instruction=data.get("grounded_instruction"),
            label_info=data.get("label"),
            start_info=data.get("start"),
            path_info=data.get("path"),
            start_facing_info=data.get("start_facing"),
        )

        if data.get("goal") is not None:
            raw_goals = [
                {
                    "position": [
                        data["goal"]["world"]["x"],
                        data["goal"]["world"]["y"],
                        1.3,
                    ]
                }
            ]
            episode.goals = [NavigationGoal(**g) for g in raw_goals]

        self.episodes.append(episode)
        return True
