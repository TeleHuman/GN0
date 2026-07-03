#!/bin/bash

set -euo pipefail

INCLUDE_SERVER="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-server)
            INCLUDE_SERVER="true"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

PATTERNS=(
    "python .*run_remote_eval.py"
    "(bash|zsh) .*eval_remote.sh"
)

if [[ "$INCLUDE_SERVER" == "true" ]]; then
    PATTERNS+=("interiorgs_nav_server.py")
fi

kill_pattern() {
    local pattern="$1"
    local signal="$2"
    local pids

    pids="$(pgrep -f "$pattern" || true)"
    if [[ -z "$pids" ]]; then
        return
    fi

    echo "Sending ${signal:-TERM} to: $pattern"
    echo "$pids" | xargs -r kill $signal 2>/dev/null || true
}

for pattern in "${PATTERNS[@]}"; do
    kill_pattern "$pattern" ""
done

sleep 2

for pattern in "${PATTERNS[@]}"; do
    kill_pattern "$pattern" "-9"
done

echo "Remote GN-Bench eval processes cleaned."
