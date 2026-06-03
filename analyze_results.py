import argparse
import json
import math
from pathlib import Path


def sanitize_metric(value):
    """Converts the value to float and turns NaN/Inf into 0.0."""
    try:
        val = float(value)
        return val if math.isfinite(val) else 0.0
    except (ValueError, TypeError):
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics from JSON logs in a directory."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Root directory containing the 'log' folder.",
    )
    args = parser.parse_args()

    log_dir = Path(args.path) / "log"
    if not log_dir.is_dir():
        print(f"Error: Log directory '{log_dir}' not found.")
        return

    log_files = list(log_dir.glob("*.json"))

    total_files = len(log_files)
    if total_files == 0:
        print("Warning: No log files found in the directory.")
        return

    succ = spl = distance_to_goal = oracle_succ = path_length = 0.0

    for file_path in log_files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            succ += sanitize_metric(data.get("success", 0))
            spl += sanitize_metric(data.get("spl", 0))
            distance_to_goal += sanitize_metric(data.get("distance_to_goal", 0))
            oracle_succ += sanitize_metric(data.get("oracle_success", 0))
            path_length += float(data.get("path_length", 0))

        except json.JSONDecodeError:
            print(f"Parsing Error: {file_path.name} is not a valid JSON file.")
        except Exception as e:
            print(f"Unknown Error: Occurred while processing {file_path.name} -> {e}")

    print(f"TL: {path_length / total_files:.3f}")
    print(f"NE: {distance_to_goal / total_files:.3f}")
    print(f"OS: {int(oracle_succ)}/{total_files} ({oracle_succ / total_files:.3f})")
    print(f"SR: {int(succ)}/{total_files} ({succ / total_files:.3f})")
    print(f"SPL:{spl:.3f}/{total_files} ({spl / total_files:.3f})")


if __name__ == "__main__":
    main()
