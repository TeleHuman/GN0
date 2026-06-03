#!/bin/bash

# ==========================================
# Configuration Area
# ==========================================
MODEL_PATH="model_zoo/bae"
MODEL_NAME="bae"
CONFIG_PATH="VLN_CE/vlnce_baselines/config/baselines/bae_InteriorGS.yaml"

# Format timestamp for unique result directory
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
SAVE_PATH="tmp/${MODEL_NAME}_${TIMESTAMP}"

# ==========================================
# Argument Parsing
# ==========================================
DEBUG_FLAG=""
DAGGER_FLAG=""
PROMPT_TYPE="V1"
ACTION_NUM=1

START_IDX="0"
END_IDX="-1"
CHUNKS=1
PROCS_PER_GPU=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)     DEBUG_FLAG="--debug"; shift ;;
        --dagger)   DAGGER_FLAG="--dagger"; shift ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --prompt-type) PROMPT_TYPE="$2"; shift 2 ;;
        --action-num)  ACTION_NUM="$2"; shift 2 ;;
        --start-idx)   START_IDX="$2"; shift 2 ;;
        --end-idx)     END_IDX="$2"; shift 2 ;;
        --chunks)      CHUNKS="$2"; shift 2 ;;
        --procs-per-gpu) PROCS_PER_GPU="$2"; shift 2 ;;
        --save-path)   SAVE_PATH="$2"; shift 2 ;;
        
        *) echo "Unknown option: $1"; shift ;;
    esac
done

mkdir -p "$SAVE_PATH"

# ==========================================
# Execution Logic
# ==========================================
echo "Starting evaluation: $MODEL_NAME with prompt $PROMPT_TYPE"
echo "Results will be saved to: $SAVE_PATH"
echo "Evaluating episodes from $START_IDX to $END_IDX across $CHUNKS GPUs"
echo "Processes per GPU: $PROCS_PER_GPU"

if ! [[ "$CHUNKS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$PROCS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --chunks and --procs-per-gpu must be positive integers."
    exit 1
fi

TOTAL_WORKERS=$(( CHUNKS * PROCS_PER_GPU ))
echo "Total workers: $TOTAL_WORKERS"

# Ensure all background processes are killed on exit (Ctrl+C)
trap 'echo "Terminating..."; kill 0' SIGINT SIGTERM

for ((gpu_id=0; gpu_id<CHUNKS; gpu_id++)); do
    for ((slot=0; slot<PROCS_PER_GPU; slot++)); do
        split_id=$(( gpu_id * PROCS_PER_GPU + slot ))

        echo "[Worker $split_id/$TOTAL_WORKERS] Launching on GPU $gpu_id (slot $slot)..."

        CUDA_VISIBLE_DEVICES=$gpu_id python run.py \
            --exp-config "$CONFIG_PATH" \
            --split-num "$TOTAL_WORKERS" \
            --split-id "$split_id" \
            --model-path "$MODEL_PATH" \
            --result-path "$SAVE_PATH" \
            --model-name "$MODEL_NAME" \
            --prompt-type "$PROMPT_TYPE" \
            --action-num "$ACTION_NUM" \
            --start-idx "$START_IDX" \
            --end-idx "$END_IDX" \
            $DAGGER_FLAG \
            $DEBUG_FLAG &
    done
done

# Wait for all background chunks to complete
wait
echo "All chunks finished."
