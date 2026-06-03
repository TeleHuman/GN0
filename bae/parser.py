#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parser for BAE model outputs."""

import json
import re
from typing import List, Optional

_RE_JSON_OBJ = re.compile(r"\{.*?\}", re.DOTALL)
_RE_ACTION_XML = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_RE_PIXEL_TOKEN_PAIR = re.compile(
    r"\[\s*<\s*(\d{1,4})\s*>\s*,\s*<\s*(\d{1,4})\s*>\s*\]"
)

ACTION_TOKEN_TO_ID = {
    "STOP": 0,
    "FWD": 1,
    "LEFT": 2,
    "RIGHT": 3,
}


def _parse_action_token(tok: str) -> Optional[int]:
    t = tok.strip()
    if not t:
        return None

    # Old numeric format: 0,1,2,3
    try:
        x = int(t)
        if 0 <= x <= 3:
            return x
    except Exception:
        pass

    # New token format: <FWD>/<LEFT>/<RIGHT>/<STOP> (or without <>)
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1]
    t = t.strip().upper()
    return ACTION_TOKEN_TO_ID.get(t)


def parse_actions6(text: str) -> Optional[List[int]]:
    """
    Parse exactly 6 VLN-CE actions from model output.

    Expected format examples:
    - JSON: {"vlnce":"<action><FWD>,<LEFT>,<FWD>,<RIGHT>,<FWD>,<STOP></action>"}
    - XML: <action><FWD>,<LEFT>,<FWD>,<RIGHT>,<FWD>,<STOP></action>
    - Backward-compatible numeric form is also supported.

    Returns:
        List of 6 integers [0-3], or None if parsing fails
    """
    s = text.strip()

    # Try JSON format first
    m = _RE_JSON_OBJ.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("vlnce"), str):
                mm = _RE_ACTION_XML.search(obj["vlnce"])
                if mm:
                    arr = []
                    for x in mm.group(1).split(","):
                        xi = _parse_action_token(x)
                        if xi is not None:
                            arr.append(xi)
                    if len(arr) >= 6:
                        return arr[:6]
        except Exception:
            pass

    # Try direct XML format
    mm = _RE_ACTION_XML.search(s)
    if mm:
        try:
            arr = []
            for x in mm.group(1).split(","):
                xi = _parse_action_token(x)
                if xi is not None:
                    arr.append(xi)
            if len(arr) >= 6:
                return arr[:6]
        except Exception:
            pass

    return None


def validate_actions(actions: List[int]) -> bool:
    """Validate that all actions are in valid range [0-3]."""
    return all(0 <= a <= 3 for a in actions)


def parse_pixels(text: str) -> Optional[List[List[int]]]:
    """
    Parse pixel waypoints from model output.

    Expected format:
    - New: {"pixel":"[[<x0>,<y0>],[<x1>,<y1>],...]"}
    - Old: {"pixel":[[x0,y0],[x1,y1],...]}

    Only reads the FIRST JSON object in the text.

    Returns:
        List of [x, y] coordinate pairs, or None if parsing fails
    """
    s = text.strip()

    # Find the first JSON object
    m = _RE_JSON_OBJ.search(s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if not isinstance(obj, dict) or "pixel" not in obj:
                return None

            pixels = obj["pixel"]

            # New token-string format.
            if isinstance(pixels, str):
                result = []
                for x_str, y_str in _RE_PIXEL_TOKEN_PAIR.findall(pixels):
                    xi = int(x_str)
                    yi = int(y_str)
                    if not (0 <= xi <= 999 and 0 <= yi <= 999):
                        return None
                    result.append([xi, yi])
                return result if result else None

            # Old numeric list format (backward compatible).
            if isinstance(pixels, list):
                result = []
                for point in pixels:
                    if not (isinstance(point, list) and len(point) == 2):
                        return None
                    try:
                        result.append([int(point[0]), int(point[1])])
                    except Exception:
                        return None
                return result if result else None
        except Exception:
            pass

    return None
