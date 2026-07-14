#!/usr/bin/env python3

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange

from GN_Bench import Env
from GN_Bench.datasets import make_dataset
from VLN_CE.GN_Bench_extensions.config.default import get_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

TARGET_METRICS = (
    "distance_to_goal",
    "success",
    "spl",
    "path_length",
    "oracle_success",
)
DATASET_SEED = 42
STOP_ACTION_ID = 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GN_Bench Navigation Evaluation")

    io_group = parser.add_argument_group("Paths & IO")
    io_group.add_argument(
        "--exp-config", type=str, required=True, help="Config YAML path"
    )
    io_group.add_argument(
        "--model-path", type=str, required=True, help="Model weights location"
    )
    io_group.add_argument(
        "--result-path", type=str, required=True, help="Save directory"
    )

    model_group = parser.add_argument_group("Model & Task")
    model_group.add_argument("--model-name", type=str, required=True)
    model_group.add_argument(
        "--prompt-type", type=str, default="V3", choices=["V1", "V2", "V3"]
    )
    model_group.add_argument(
        "--action-num",
        type=int,
        default=1,
        help="Number of predicted actions to cache in pending_action_list",
    )
    model_group.add_argument("--split-num", type=int, required=True)
    model_group.add_argument("--split-id", type=int, required=True)

    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument("--debug", action="store_true")
    debug_group.add_argument("--dagger", action="store_true")
    debug_group.add_argument("--start-idx", type=int, default=0, help="StartIndex")
    debug_group.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Exclusive end index. Omit to run through the dataset end.",
    )

    collecter_group = parser.add_argument_group("Trajectory Video Collecter")
    collecter_group.add_argument(
        "--collecter",
        action="store_true",
        help="Render each smoothed dataset trajectory to <result-path>/<scene>/<trajectory>.mp4.",
    )

    return parser.parse_args()


def setup_debugger() -> None:
    import debugpy

    debugpy.listen(3356)
    LOGGER.info("Waiting for debugger attach on port 3356...")
    debugpy.wait_for_client()
    LOGGER.info("Debugger attached.")


def get_episode_id(ref_json: str) -> str:
    ref_path = Path(ref_json)
    return f"{ref_path.parent.name}_{ref_path.stem}"


def filter_pending_episodes(episodes: list, finished_ids: set[str]) -> list:
    return [ep for ep in episodes if get_episode_id(ep.ref_json) not in finished_ids]


def validate_split_args(split_num: int, split_id: int) -> None:
    if split_num <= 0:
        raise ValueError(f"split_num must be > 0, got {split_num}")
    if split_id < 0 or split_id >= split_num:
        raise ValueError(
            f"split_id must be in [0, split_num), got split_id={split_id}, split_num={split_num}"
        )


def shard_episodes(episodes: list, split_num: int, split_id: int) -> list:
    # Strided slicing guarantees disjoint shards across split_id values.
    return episodes[split_id::split_num]


def make_log_dir(result_path: str) -> Path:
    log_dir = Path(result_path) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def load_dataset_split(config):
    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(DATASET_SEED)
    return dataset.get_splits(1)[0]


def resolve_end_idx(episodes: list, end_idx: int | None) -> int:
    if end_idx is None or end_idx == -1:
        return len(episodes)
    if end_idx < -1:
        raise ValueError(f"end_idx must be >= 0, -1, or omitted; got {end_idx}")
    return int(end_idx)


def select_eval_episodes(
    dataset_split, log_dir: Path, start_idx: int, end_idx: int | None
):
    end_idx = resolve_end_idx(dataset_split.episodes, end_idx)
    finished_ids = {p.stem for p in log_dir.glob("*.json")}
    episodes = dataset_split.episodes[start_idx:end_idx]
    return filter_pending_episodes(episodes, finished_ids), end_idx


def select_collecter_episodes(dataset_split, start_idx: int, end_idx: int | None):
    end_idx = resolve_end_idx(dataset_split.episodes, end_idx)
    return dataset_split.episodes[start_idx:end_idx], end_idx


def resolve_collecter_settings(config):
    collecter_cfg = getattr(config.EVAL, "COLLECTER", None)

    def pick(name: str, default):
        if collecter_cfg is not None and name in collecter_cfg:
            return collecter_cfg[name]
        return default

    return {
        "video_fps": float(pick("VIDEO_FPS", 10.0)),
        "video_codec": str(pick("VIDEO_CODEC", "libx264")),
        "smooth_window": int(pick("SMOOTH_WINDOW", 9)),
        "resample_step_m": float(pick("RESAMPLE_STEP_M", 0.05)),
        "heading_lookahead_m": float(pick("HEADING_LOOKAHEAD_M", 0.5)),
        "yaw_smooth_window": int(pick("YAW_SMOOTH_WINDOW", 9)),
    }


def build_agent(dagger: bool, model_path, result_path, prompt_type, action_num):
    if dagger:
        from bae_agent_dagger import BAEAgent
    else:
        from bae_agent import BAEAgent

    return BAEAgent(model_path, result_path, prompt_type, action_num)


def run_exp(
    exp_config,
    split_num,
    split_id,
    debug=False,
    start_idx=0,
    end_idx=None,
    collecter=False,
    **kwargs,
):
    validate_split_args(split_num, split_id)

    if debug:
        setup_debugger()

    config = get_config(exp_config)
    result_path = kwargs["result_path"]
    Path(result_path).mkdir(parents=True, exist_ok=True)
    log_dir = None if collecter else make_log_dir(result_path)
    dataset_split = load_dataset_split(config)

    if collecter:
        global_episodes, resolved_end_idx = select_collecter_episodes(
            dataset_split, start_idx, end_idx
        )
    else:
        global_episodes, resolved_end_idx = select_eval_episodes(
            dataset_split, log_dir, start_idx, end_idx
        )

    if split_id == 0:
        mode = "Collecter" if collecter else "Eval"
        print(
            f"Global {mode} Range: [{start_idx}, {resolved_end_idx}] "
            f"(Total target: {len(global_episodes)})"
        )

    dataset_split.episodes = (
        shard_episodes(global_episodes, split_num, split_id) if global_episodes else []
    )

    print(
        f"[Split {split_id}/{split_num}] GPU assigned {len(dataset_split.episodes)} episodes."
    )

    if not dataset_split.episodes:
        print(f"[Split {split_id}/{split_num}] No episodes to process.")
        return

    if collecter:
        collecter_settings = resolve_collecter_settings(config)
        collect_trajectory_videos(
            config=config,
            split_id=split_id,
            dataset=dataset_split,
            result_path=result_path,
            **collecter_settings,
        )
        return

    evaluate_agent(config, split_id, dataset_split, log_dir, **kwargs)


def collect_trajectory_videos(
    config,
    split_id,
    dataset,
    result_path,
    video_fps,
    video_codec,
    smooth_window,
    resample_step_m,
    heading_lookahead_m,
    yaw_smooth_window,
) -> None:
    from trajectory_video_collector import TrajectoryVideoCollector

    env = Env(config.TASK_CONFIG, dataset)
    collector = TrajectoryVideoCollector(
        result_path,
        fps=video_fps,
        codec=video_codec,
        smooth_window=smooth_window,
        resample_step_m=resample_step_m,
        heading_lookahead_m=heading_lookahead_m,
        yaw_smooth_window=yaw_smooth_window,
    )
    print(
        f"[Collecter] settings: fps={video_fps}, "
        f"codec={video_codec}, smooth_window={smooth_window}, "
        f"resample_step_m={resample_step_m}, "
        f"heading_lookahead_m={heading_lookahead_m}, "
        f"yaw_smooth_window={yaw_smooth_window}"
    )

    desc = f"{config.EVAL.IDENTIFICATION}-collecter-{split_id}"
    for _ in trange(len(env.episodes), desc=desc):
        env.reset()
        video_path, frame_count = collector.collect(
            episode=env.current_episode,
            sim=env.sim,
        )
        print(f"[Collecter] {video_path} ({frame_count} frames)")


def evaluate_agent(
    config,
    split_id,
    dataset,
    log_dir,
    model_path,
    result_path,
    prompt_type,
    action_num,
    dagger,
    **kwargs,
) -> None:
    env = Env(config.TASK_CONFIG, dataset)
    agent = build_agent(dagger, model_path, result_path, prompt_type, action_num)

    desc = f"{config.EVAL.IDENTIFICATION}-{split_id}"
    for _ in trange(len(env.episodes), desc=desc):
        run_episode(env, agent, config)
        save_episode_metrics(env, log_dir)


def run_episode(env: Env, agent, config) -> None:
    obs = env.reset()
    agent.reset(Path(env.current_episode.ref_json), sim=env.sim)

    last_dtg = float("inf")
    continuous_rotation_count = 0

    for step in range(config.EVAL.EARLY_STOP_STEPS + 1):
        if env.episode_over:
            break

        info = env.get_metrics()
        curr_dtg = info.get("distance_to_goal", 0)

        continuous_rotation_count = (
            0 if curr_dtg != last_dtg else continuous_rotation_count + 1
        )
        last_dtg = curr_dtg

        obs.update(
            {"sim": env.sim, "goal_position": env.current_episode.goals[0].position}
        )

        if continuous_rotation_count > config.EVAL.EARLY_STOP_ROTATION:
            LOGGER.warning("[Early Stop] Rotation limit at step %s", step)
            action = {"action": STOP_ACTION_ID}
        else:
            action = agent.act(obs, info)

        obs = env.step(action)


def save_episode_metrics(env: Env, log_dir: Path) -> None:
    final_info = env.get_metrics()
    results = {k: final_info[k] for k in TARGET_METRICS if k in final_info}
    results["id"] = get_episode_id(env.current_episode.ref_json)

    save_path = log_dir / f"{results['id']}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=_json_default)


if __name__ == "__main__":
    args = parse_args()
    run_exp(**vars(args))
