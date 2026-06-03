#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prompt templates for BAE navigation agent (norm1000 + token actions)."""

import math

NORM_N = 1000  # normalized token range: <0>..<999>


def _round_half_up_float(x: float) -> int:
    """Round-half-up for any real x."""
    if x >= 0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(-x + 0.5))


def _norm_xy_from_pixel(x: int, y: int, occ_w: int, occ_h: int) -> tuple[int, int]:
    """Convert original OCC pixel coords to normalized token indices (0..999)."""
    if occ_w <= 0 or occ_h <= 0:
        raise ValueError(f"bad OCC size W={occ_w} H={occ_h}")
    xx = max(0, min(int(x), occ_w - 1))
    yy = max(0, min(int(y), occ_h - 1))
    x_hat = _round_half_up_float((xx / float(occ_w)) * float(NORM_N))
    y_hat = _round_half_up_float((yy / float(occ_h)) * float(NORM_N))
    x_hat = max(0, min(x_hat, NORM_N - 1))
    y_hat = max(0, min(y_hat, NORM_N - 1))
    return x_hat, y_hat


def _tok(n: int) -> str:
    return f"<{int(n)}>"


PROMPT_V1 = r"""<image><image><image><image>
You are given FOUR images from the same navigation episode.

Image 1: CURRENT first-person RGB.
Image 2: HISTORY mosaic (4x4), NEW->OLD sampled by discrete action steps (current excluded; missing => black).
Image 3: Scene-level BEV map.
Image 4: OCC / traversability map (ORIGINAL resolution). Green=start, Red=current, white=traversable.

TASK_TYPE: PLANNING
Instruction: "{INSTRUCTION_TEXT}"

Your task:
- Actions: exactly 6 tokens from {{<FWD>,<LEFT>,<RIGHT>,<STOP>}}.
    - Action semantics: <FWD> moves forward 0.25m; <LEFT>/ <RIGHT> rotate 15 degrees in place (no translation); <STOP> means stop.

Output (STRICT): JSON with key "vlnce".
- "vlnce": "<action><FWD>,...,<STOP></action>"
No extra text.
"""


PROMPT_V2 = r"""<image><image>
You are given TWO images from the same navigation episode.

Image 1: Scene-level BEV map.
Image 2: OCC / traversability map (ORIGINAL resolution). Green=start, Red=current, white=traversable.

TASK_TYPE: PLANNING
Instruction: "{INSTRUCTION_TEXT}"

Your task:
- Actions: exactly 6 tokens from {{<FWD>,<LEFT>,<RIGHT>,<STOP>}}.
    - Action semantics: <FWD> moves forward 0.25m; <LEFT>/ <RIGHT> rotate 15 degrees in place (no translation); <STOP> means stop.

Output (STRICT): JSON with key "vlnce".
- "vlnce": "<action><FWD>,...,<STOP></action>"
No extra text.
"""


PROMPT_V3 = r"""<image><image>
You are given TWO images from the same navigation episode.

Image 1: CURRENT first-person RGB.
Image 2: HISTORY mosaic (4x4), NEW->OLD sampled by discrete action steps (current excluded; missing => black).

TASK_TYPE: PLANNING
Instruction: "{INSTRUCTION_TEXT}"

Your task:
- Actions: exactly 6 tokens from {{<FWD>,<LEFT>,<RIGHT>,<STOP>}}.
    - Action semantics: <FWD> moves forward 0.25m; <LEFT>/ <RIGHT> rotate 15 degrees in place (no translation); <STOP> means stop.

Output (STRICT): JSON with key "vlnce".
- "vlnce": "<action><FWD>,...,<STOP></action>"
No extra text.
"""


def _render_prompt(prompt: str, **kwargs) -> str:
    return prompt.format(**kwargs)


def build_prompt(
    prompt_type: str,
    instruction: str,
    cur_x: int | None = None,
    cur_y: int | None = None,
    occ_w: int | None = None,
    occ_h: int | None = None,
    occ_meter_per_px: float = 0.05,
    occ_rot_deg: int = 0,
    prev_actions: str | None = None,
) -> str:
    """
    Build a prompt string based on the prompt type.
    """
    pt = str(prompt_type).upper().strip()
    prev_actions = prev_actions or "<action><None>,<None>,<None>,<None>,<None></action>"

    if pt == "V3":
        return _render_prompt(
            PROMPT_V3,
            INSTRUCTION_TEXT=instruction,
            PREV_ACTIONS=prev_actions,
        )

    if occ_w is None or occ_h is None or cur_x is None or cur_y is None:
        raise ValueError("occ_w, occ_h, cur_x, cur_y are required for V1/V2")

    cur_xn, cur_yn = _norm_xy_from_pixel(cur_x, cur_y, occ_w, occ_h)
    cur_xn_tok = _tok(cur_xn)
    cur_yn_tok = _tok(cur_yn)
    x_unit_m = (float(occ_meter_per_px) * float(occ_w)) / 1000.0
    y_unit_m = (float(occ_meter_per_px) * float(occ_h)) / 1000.0

    if pt == "V2":
        return _render_prompt(
            PROMPT_V2,
            INSTRUCTION_TEXT=instruction,
            OCC_W=occ_w,
            OCC_H=occ_h,
            OCC_ROT_DEG=int(occ_rot_deg),
            OCC_M_PER_PX=float(occ_meter_per_px),
            X_NORM_UNIT_M=x_unit_m,
            Y_NORM_UNIT_M=y_unit_m,
            CUR_XN=cur_xn_tok,
            CUR_YN=cur_yn_tok,
            PREV_ACTIONS=prev_actions,
        )

    return _render_prompt(
        PROMPT_V1,
        INSTRUCTION_TEXT=instruction,
        OCC_W=occ_w,
        OCC_H=occ_h,
        OCC_ROT_DEG=int(occ_rot_deg),
        OCC_M_PER_PX=float(occ_meter_per_px),
        X_NORM_UNIT_M=x_unit_m,
        Y_NORM_UNIT_M=y_unit_m,
        CUR_XN=cur_xn_tok,
        CUR_YN=cur_yn_tok,
        PREV_ACTIONS=prev_actions,
    )
