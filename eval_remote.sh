#!/bin/bash

# Evaluate GN-Bench navigation through generic remote WebSocket policies.
# The remote policy may return either discrete VLN-CE actions or nav_delta chunks.

MODEL_NAME="remote_nav"
CONFIG_PATH="VLN_CE/GN_Bench_extensions/config/bae_InteriorGS.yaml"

TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
SAVE_PATH="tmp/${MODEL_NAME}_${TIMESTAMP}"

DEBUG_FLAG=""
ACTION_NUM=8
ACTION_FORMAT="auto"
REMOTE_SEND_STATE_FLAG=""

START_IDX="0"
END_IDX="-1"
CHUNKS=1
PROCS_PER_GPU=1

SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
SERVER_PORTS=""
REMOTE_TIMEOUT="120"
REMOTE_CONNECT_TIMEOUT="600"
REMOTE_METADATA_TIMEOUT="2"
REMOTE_STOP_EPS="1e-3"
REMOTE_TRANSLATION_FRAME="chunk_start"
REMOTE_MAX_TRANSLATION="0.0"
REMOTE_MAX_YAW="0.0"
REMOTE_SAVE_IMAGES_FLAG="--remote-save-images"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG_FLAG="--debug"; shift ;;
        --action-num) ACTION_NUM="$2"; shift 2 ;;
        --action-format) ACTION_FORMAT="$2"; shift 2 ;;
        --remote-send-state) REMOTE_SEND_STATE_FLAG="--remote-send-state"; shift ;;
        --start-idx) START_IDX="$2"; shift 2 ;;
        --end-idx) END_IDX="$2"; shift 2 ;;
        --chunks) CHUNKS="$2"; shift 2 ;;
        --procs-per-gpu) PROCS_PER_GPU="$2"; shift 2 ;;
        --save-path) SAVE_PATH="$2"; shift 2 ;;
        --server-host) SERVER_HOST="$2"; shift 2 ;;
        --server-port) SERVER_PORT="$2"; shift 2 ;;
        --server-ports) SERVER_PORTS="$2"; shift 2 ;;
        --remote-timeout) REMOTE_TIMEOUT="$2"; shift 2 ;;
        --remote-connect-timeout) REMOTE_CONNECT_TIMEOUT="$2"; shift 2 ;;
        --remote-metadata-timeout) REMOTE_METADATA_TIMEOUT="$2"; shift 2 ;;
        --remote-stop-eps) REMOTE_STOP_EPS="$2"; shift 2 ;;
        --remote-translation-frame) REMOTE_TRANSLATION_FRAME="$2"; shift 2 ;;
        --remote-max-translation) REMOTE_MAX_TRANSLATION="$2"; shift 2 ;;
        --remote-max-yaw) REMOTE_MAX_YAW="$2"; shift 2 ;;
        --remote-save-images) REMOTE_SAVE_IMAGES_FLAG="--remote-save-images"; shift ;;
        --no-remote-save-images) REMOTE_SAVE_IMAGES_FLAG=""; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$SAVE_PATH"

echo "Starting evaluation: $MODEL_NAME"
echo "Results will be saved to: $SAVE_PATH"
echo "Action format: $ACTION_FORMAT"
echo "Max actions per request: $ACTION_NUM"
echo "Remote stop epsilon: $REMOTE_STOP_EPS"
if [[ -n "$REMOTE_SAVE_IMAGES_FLAG" ]]; then
    echo "Save images: true (rgb, bev_traj)"
else
    echo "Save images: false"
fi

if ! [[ "$CHUNKS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$PROCS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --chunks and --procs-per-gpu must be positive integers."
    exit 1
fi
if ! [[ "$SERVER_PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: --server-port must be an integer base port."
    exit 1
fi

TOTAL_WORKERS=$(( CHUNKS * PROCS_PER_GPU ))
echo "Evaluating episodes from $START_IDX to $END_IDX"
echo "GN-Bench GPU slots: $CHUNKS"
echo "Processes per GPU: $PROCS_PER_GPU"
echo "Total GN-Bench workers: $TOTAL_WORKERS"

SERVER_PORT_ARRAY=()
if [[ -n "$SERVER_PORTS" ]]; then
    IFS=',' read -r -a SERVER_PORT_ARRAY <<< "$SERVER_PORTS"
    if [[ "${#SERVER_PORT_ARRAY[@]}" -ne "$TOTAL_WORKERS" ]]; then
        echo "Error: --server-ports must contain exactly $TOTAL_WORKERS comma-separated ports." >&2
        exit 1
    fi
else
    for ((idx=0; idx<TOTAL_WORKERS; idx++)); do
        SERVER_PORT_ARRAY+=( "$(( SERVER_PORT + idx ))" )
    done
fi

echo "Remote server host: $SERVER_HOST"
echo "Remote server ports: ${SERVER_PORT_ARRAY[*]}"
echo "Note: this script starts GN-Bench workers only. Start one remote policy server per port first."

cleanup() {
    trap - SIGINT SIGTERM
    echo "Terminating..."
    local pids
    pids="$(jobs -pr)"
    if [[ -n "$pids" ]]; then
        kill $pids 2>/dev/null || true
        wait $pids 2>/dev/null || true
    fi
    exit 130
}

trap cleanup SIGINT SIGTERM

for ((gpu_id=0; gpu_id<CHUNKS; gpu_id++)); do
    for ((slot=0; slot<PROCS_PER_GPU; slot++)); do
        split_id=$(( gpu_id * PROCS_PER_GPU + slot ))
        worker_server_port="${SERVER_PORT_ARRAY[$split_id]}"

        echo "[Worker $split_id/$TOTAL_WORKERS] Launching on GN-Bench GPU $gpu_id (slot $slot), server port $worker_server_port..."

        CUDA_VISIBLE_DEVICES=$gpu_id python run_remote_eval.py \
            --exp-config "$CONFIG_PATH" \
            --split-num "$TOTAL_WORKERS" \
            --split-id "$split_id" \
            --result-path "$SAVE_PATH" \
            --action-num "$ACTION_NUM" \
            --action-format "$ACTION_FORMAT" \
            --start-idx "$START_IDX" \
            --end-idx "$END_IDX" \
            --server-host "$SERVER_HOST" \
            --server-port "$worker_server_port" \
            --remote-timeout "$REMOTE_TIMEOUT" \
            --remote-connect-timeout "$REMOTE_CONNECT_TIMEOUT" \
            --remote-metadata-timeout "$REMOTE_METADATA_TIMEOUT" \
            --remote-stop-eps "$REMOTE_STOP_EPS" \
            --remote-translation-frame "$REMOTE_TRANSLATION_FRAME" \
            --remote-max-translation "$REMOTE_MAX_TRANSLATION" \
            --remote-max-yaw "$REMOTE_MAX_YAW" \
            $REMOTE_SAVE_IMAGES_FLAG \
            $REMOTE_SEND_STATE_FLAG \
            $DEBUG_FLAG &
    done
done

wait
echo "All chunks finished."
