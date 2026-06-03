# BAE Constants

# Action mapping
ACTION_STOP = 0
ACTION_MOVE_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3

# Prompt types
PROMPT_V1 = "V1"  # Full: rgb, hist, bev, occ + coordinates
PROMPT_V2 = "V2"  # Map-only: bev, occ + coordinates
PROMPT_V3 = "V3"  # Vision-only: rgb, hist

# Scale: 1 pixel = 0.05 meters
PIXEL_TO_METER = 0.05
METER_TO_PIXEL = 20.0

# Action sequence length
NUM_ACTIONS = 6
