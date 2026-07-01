from typing import Any, Sequence

import numpy as np
from GN_Bench.config import Config
from GN_Bench.core.embodied_task import EmbodiedTask, Measure
from GN_Bench.core.registry import registry
from GN_Bench.core.simulator import Simulator
from GN_Bench.tasks.nav.nav import DistanceToGoal


def euclidean_distance(pos_a: Sequence[float], pos_b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2))


@registry.register_measure
class PathLength(Measure):
    """Path Length (PL)."""

    cls_uuid: str = "path_length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(current_position, self._previous_position)
        self._previous_position = current_position


@registry.register_measure
class OracleSuccess(Measure):
    """Oracle Success Rate (OSR)."""

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Config, **kwargs: Any):
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_goal = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = float(
            bool(self._metric) or distance_to_goal < self._config.SUCCESS_DISTANCE
        )


@registry.register_measure
class StepsTaken(Measure):
    """Counts executed actions. STOP counts as an action."""

    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        self._metric += 1.0
