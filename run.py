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
    debug_group.add_argument("--end-idx", type=int, default=-1, help="EndIndex")

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


def select_eval_episodes(dataset_split, log_dir: Path, start_idx: int, end_idx: int):
    if end_idx == -1:
        end_idx = len(dataset_split.episodes)

    finished_ids = {p.stem for p in log_dir.glob("*.json")}
    episodes = dataset_split.episodes[start_idx:end_idx]
    return filter_pending_episodes(episodes, finished_ids), end_idx


def build_agent(dagger: bool, model_path, result_path, prompt_type, action_num):
    if dagger:
        from bae_agent_dagger import BAEAgent
    else:
        from bae_agent import BAEAgent

    return BAEAgent(model_path, result_path, prompt_type, action_num)


def run_exp(
    exp_config, split_num, split_id, debug=False, start_idx=0, end_idx=-1, **kwargs
):
    validate_split_args(split_num, split_id)

    if debug:
        setup_debugger()

    config = get_config(exp_config)
    log_dir = make_log_dir(kwargs["result_path"])
    dataset_split = load_dataset_split(config)

    global_episodes, resolved_end_idx = select_eval_episodes(
        dataset_split, log_dir, start_idx, end_idx
    )

    if split_id == 0:
        print(
            f"Global Eval Range: [{start_idx}, {resolved_end_idx}] "
            f"(Total target: {len(global_episodes)})"
        )

    dataset_split.episodes = (
        shard_episodes(global_episodes, split_num, split_id) if global_episodes else []
    )

    print(
        f"[Split {split_id}/{split_num}] GPU assigned {len(dataset_split.episodes)} episodes."
    )

    evaluate_agent(config, split_id, dataset_split, log_dir, **kwargs)


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
