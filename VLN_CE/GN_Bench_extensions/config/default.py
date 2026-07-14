from typing import List, Optional, Union

from GN_Bench.config.default import CONFIG_FILE_SEPARATOR
from GN_Bench.config.default import Config as CN
from GN_Bench.config.default import get_config as get_base_config

_C = get_base_config()
_C.defrost()

# ----------------------------------------------------------------------------
# PANORAMA SETTINGS
# ----------------------------------------------------------------------------
_C.TASK.PANO_ROTATIONS = 12
# ----------------------------------------------------------------------------
# GPS SENSOR
# ----------------------------------------------------------------------------
_C.TASK.GLOBAL_GPS_SENSOR = CN()
_C.TASK.GLOBAL_GPS_SENSOR.TYPE = "GlobalGPSSensor"
_C.TASK.GLOBAL_GPS_SENSOR.DIMENSIONALITY = 2
# ----------------------------------------------------------------------------
# ORACLE ACTION SENSOR
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_ACTION_SENSOR = CN()
_C.TASK.ORACLE_ACTION_SENSOR.TYPE = "OracleActionSensor"
_C.TASK.ORACLE_ACTION_SENSOR.GOAL_RADIUS = 0.5
# ----------------------------------------------------------------------------
# # RXR INSTRUCTION SENSOR
# ----------------------------------------------------------------------------
_C.TASK.RXR_INSTRUCTION_SENSOR = CN()
_C.TASK.RXR_INSTRUCTION_SENSOR.TYPE = "RxRInstructionSensor"
_C.TASK.RXR_INSTRUCTION_SENSOR.features_path = "data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{id:06}_{lang}_text_features.npz"
_C.TASK.INSTRUCTION_SENSOR_UUID = "rxr_instruction"
# ----------------------------------------------------------------------------
# SHORTEST PATH SENSOR
# ----------------------------------------------------------------------------
_C.TASK.SHORTEST_PATH_SENSOR = CN()
_C.TASK.SHORTEST_PATH_SENSOR.TYPE = "ShortestPathSensor"
# all goals can be navigated to within 0.5m.
_C.TASK.SHORTEST_PATH_SENSOR.GOAL_RADIUS = 0.5
# compatibility with the oracle used during dataset generation.
# if False, use the current version of the GN_Bench-Lab ShortestPathFollower
_C.TASK.SHORTEST_PATH_SENSOR.USE_ORIGINAL_FOLLOWER = False
# -----------------------------------------------------------------------------
# VLN ORACLE PROGRESS SENSOR
# ----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR = CN()
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR.TYPE = "VLNOracleProgressSensor"
# ----------------------------------------------------------------------------
# PANO ANGLE FEATURE SENSOR
# ----------------------------------------------------------------------------
_C.TASK.PANO_ANGLE_FEATURE_SENSOR = CN()
_C.TASK.PANO_ANGLE_FEATURE_SENSOR.TYPE = "AngleFeaturesSensor"
_C.TASK.PANO_ANGLE_FEATURE_SENSOR.CAMERA_NUM = 12
# ----------------------------------------------------------------------------
# GO_TOWARD_POINT ACTION
# ----------------------------------------------------------------------------
_C.TASK.ACTIONS.GO_TOWARD_POINT = CN()
_C.TASK.ACTIONS.GO_TOWARD_POINT.TYPE = "GoTowardPoint"
# if True, update the heading to face away from where the agent came from
_C.TASK.ACTIONS.GO_TOWARD_POINT.rotate_agent = True
# PATH_LENGTH MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.PATH_LENGTH = CN()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"
# ----------------------------------------------------------------------------
# ORACLE_SUCCESS MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.ORACLE_SUCCESS = CN()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 3.0
# ----------------------------------------------------------------------------
# STEPS_TAKEN MEASUREMENT
# ----------------------------------------------------------------------------
_C.TASK.STEPS_TAKEN = CN()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"
# ----------------------------------------------------------------------------
# DATASET EXTENSIONS
# ----------------------------------------------------------------------------
_C.DATASET.ROLES = ["guide"]  # options: "guide", "follower"
# language options: "te-IN", "hi-IN", "en-US", "en-IN"
_C.DATASET.LANGUAGES = ["*"]
# a list of episode IDs to allow in dataset creation.
_C.DATASET.EPISODES_ALLOWED = ["*"]


# ----------------------------------------------------------------------------
# GN0 EVALUATION CONFIG
# ----------------------------------------------------------------------------
_EVAL_C = CN()
_EVAL_C.BASE_TASK_CONFIG_PATH = ""
_EVAL_C.TASK_CONFIG = CN()
_EVAL_C.CMD_TRAILING_OPTS = []
_EVAL_C.EVAL = CN()
_EVAL_C.EVAL.IDENTIFICATION = "gs"
_EVAL_C.EVAL.EARLY_STOP_ROTATION = 25
_EVAL_C.EVAL.EARLY_STOP_STEPS = 500
_EVAL_C.EVAL.COLLECTER = CN()
_EVAL_C.EVAL.COLLECTER.VIDEO_FPS = 10.0
_EVAL_C.EVAL.COLLECTER.VIDEO_CODEC = "libx264"
_EVAL_C.EVAL.COLLECTER.SMOOTH_WINDOW = 9
_EVAL_C.EVAL.COLLECTER.RESAMPLE_STEP_M = 0.05
_EVAL_C.EVAL.COLLECTER.HEADING_LOOKAHEAD_M = 0.5
_EVAL_C.EVAL.COLLECTER.YAW_SMOOTH_WINDOW = 9


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    inline_config: Optional[CN] = None,
) -> CN:
    """Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if inline_config is not None and len(inline_config) > 0:
        config.merge_from_other_cfg(inline_config)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create the GN0 evaluation config without VLN-CE baseline defaults."""
    config = _EVAL_C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.TASK_CONFIG = get_extended_config(
        config.BASE_TASK_CONFIG_PATH or None,
        inline_config=config.TASK_CONFIG,
    )
    config.freeze()
    return config
