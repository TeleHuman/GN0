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
    parser = argparse.ArgumentParser(
        description="Run GN_Bench evaluation against a remote navigation server"
    )

    io_group = parser.add_argument_group("Paths & IO")
    io_group.add_argument(
        "--exp-config", type=str, required=True, help="Config YAML path"
    )
    io_group.add_argument(
        "--result-path", type=str, required=True, help="Save directory"
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--action-num",
        type=int,
        default=8,
        help="Maximum number of returned remote actions to execute before the next query",
    )
    eval_group.add_argument("--split-num", type=int, required=True)
    eval_group.add_argument("--split-id", type=int, required=True)
    eval_group.add_argument("--start-idx", type=int, default=0, help="StartIndex")
    eval_group.add_argument("--end-idx", type=int, default=-1, help="EndIndex")

    remote_group = parser.add_argument_group("Remote Navigation Client")
    remote_group.add_argument("--server-host", type=str, default="127.0.0.1")
    remote_group.add_argument("--server-port", type=int, default=8000)
    remote_group.add_argument("--remote-timeout", type=float, default=120.0)
    remote_group.add_argument("--remote-connect-timeout", type=float, default=600.0)
    remote_group.add_argument(
        "--remote-stop-distance",
        type=float,
        default=None,
        help="Deprecated/ignored: Remote policies should return STOP explicitly or a zero nav_delta.",
    )
    remote_group.add_argument(
        "--remote-stop-eps",
        type=float,
        default=1e-3,
        help="Treat a model nav_delta with all |dx,dy,dyaw| <= eps as STOP.",
    )
    remote_group.add_argument(
        "--remote-translation-frame",
        type=str,
        default="chunk_start",
        choices=["chunk_start", "current"],
    )
    remote_group.add_argument("--remote-max-translation", type=float, default=0.0)
    remote_group.add_argument("--remote-max-yaw", type=float, default=0.0)
    remote_group.add_argument("--remote-save-images", action="store_true")
    remote_group.add_argument(
        "--action-format",
        type=str,
        default="auto",
        choices=["auto", "discrete", "nav_delta"],
        help="How to interpret remote actions. auto supports action_type, action.nav_delta, and integer action lists.",
    )
    remote_group.add_argument(
        "--remote-metadata-timeout",
        type=float,
        default=2.0,
        help="Seconds to wait for optional server metadata after WebSocket connect.",
    )
    remote_group.add_argument("--remote-send-state", action="store_true")

    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument("--debug", action="store_true")

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


def select_eval_episodes(dataset_split, log_dir: Path, start_idx: int, end_idx: int):
    if end_idx == -1:
        end_idx = len(dataset_split.episodes)

    finished_ids = {p.stem for p in log_dir.glob("*.json") if p.is_file()}
    episodes = dataset_split.episodes[start_idx:end_idx]
    pending = filter_pending_episodes(episodes, finished_ids)
    skipped = len(episodes) - len(pending)
    return pending, end_idx, skipped, len(finished_ids)


def build_agent(
    result_path,
    action_num,
    server_host,
    server_port,
    remote_timeout,
    remote_connect_timeout,
    remote_stop_distance,
    remote_stop_eps,
    remote_translation_frame,
    remote_max_translation,
    remote_max_yaw,
    remote_save_images,
    action_format,
    remote_metadata_timeout,
    remote_send_state,
):
    from remote_nav_agent import RemoteNavAgent

    return RemoteNavAgent(
        model_path=None,
        result_path=result_path,
        action_num=action_num,
        server_host=server_host,
        server_port=server_port,
        request_timeout=remote_timeout,
        connect_timeout=remote_connect_timeout,
        metadata_timeout=remote_metadata_timeout,
        stop_distance=remote_stop_distance,
        stop_eps=remote_stop_eps,
        translation_frame=remote_translation_frame,
        max_translation=remote_max_translation,
        max_yaw=remote_max_yaw,
        save_images=remote_save_images,
        action_format=action_format,
        send_state=remote_send_state,
    )


def run_exp(
    exp_config,
    split_num,
    split_id,
    debug=False,
    start_idx=0,
    end_idx=-1,
    **kwargs,
):
    validate_split_args(split_num, split_id)

    if debug:
        setup_debugger()

    config = get_config(exp_config)
    log_dir = make_log_dir(kwargs["result_path"])
    dataset_split = load_dataset_split(config)

    global_episodes, resolved_end_idx, skipped, finished_count = select_eval_episodes(
        dataset_split, log_dir, start_idx, end_idx
    )

    if split_id == 0:
        print(
            f"Global Eval Range: [{start_idx}, {resolved_end_idx}] "
            f"(pending: {len(global_episodes)}, skipped in range: {skipped}, "
            f"existing logs: {finished_count})"
        )

    dataset_split.episodes = (
        shard_episodes(global_episodes, split_num, split_id) if global_episodes else []
    )

    print(
        f"[Split {split_id}/{split_num}] Assigned {len(dataset_split.episodes)} episodes."
    )

    evaluate_agent(config, split_id, dataset_split, log_dir, **kwargs)


def evaluate_agent(
    config,
    split_id,
    dataset,
    log_dir,
    result_path,
    action_num,
    server_host,
    server_port,
    remote_timeout,
    remote_connect_timeout,
    remote_stop_distance,
    remote_stop_eps,
    remote_translation_frame,
    remote_max_translation,
    remote_max_yaw,
    remote_save_images,
    action_format,
    remote_metadata_timeout,
    remote_send_state,
    **kwargs,
) -> None:
    del kwargs
    env = Env(config.TASK_CONFIG, dataset)
    agent = build_agent(
        result_path=result_path,
        action_num=action_num,
        server_host=server_host,
        server_port=server_port,
        remote_timeout=remote_timeout,
        remote_connect_timeout=remote_connect_timeout,
        remote_stop_distance=remote_stop_distance,
        remote_stop_eps=remote_stop_eps,
        remote_translation_frame=remote_translation_frame,
        remote_max_translation=remote_max_translation,
        remote_max_yaw=remote_max_yaw,
        remote_save_images=remote_save_images,
        action_format=action_format,
        remote_metadata_timeout=remote_metadata_timeout,
        remote_send_state=remote_send_state,
    )

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
