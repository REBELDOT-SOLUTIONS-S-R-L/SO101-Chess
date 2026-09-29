#!/usr/bin/env bash
set -Eeuo pipefail

# Queue the two balanced_lowpoly_2 XVLA runs behind the active LeRobot
# conversion. Each run goes through wait_convert_train_xvla.sh so source and
# converted-dataset validation remain identical to the standard pipeline.

WORKSPACE_ROOT="/home/roboticslab/IsaacTools/IsaacTasks/so101_chess"
SOURCE_HDF5="${WORKSPACE_ROOT}/datasets/chess_generated_env_rand_balanced_lowpoly_2.hdf5"
LEROBOT_OUTPUT="/home/roboticslab/datasets/so101_chess_balanced_lowpoly_2"
DATASET_REPO_ID="local/so101_chess_balanced_lowpoly_2"
CONVERTER_PID="${CONVERTER_PID:-}"
QUEUE_LOG="${WORKSPACE_ROOT}/logs/balanced_lowpoly_2_xvla_queue.log"

mkdir -p "${WORKSPACE_ROOT}/logs"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

if [[ -n "${CONVERTER_PID}" ]] && [[ -d "/proc/${CONVERTER_PID}" ]]; then
    log "Waiting for LeRobot converter PID ${CONVERTER_PID}."
    while [[ -d "/proc/${CONVERTER_PID}" ]]; do
        sleep 30
    done
fi

run_training() {
    local learning_rate="$1"
    local output_suffix="$2"
    local job_suffix="$3"

    log "Starting pipeline for learning rate ${learning_rate}."
    SOURCE_HDF5="${SOURCE_HDF5}" \
    EXPECTED_EPISODES=1000 \
    GENERATOR_PID=0 \
    LEROBOT_OUTPUT="${LEROBOT_OUTPUT}" \
    DATASET_REPO_ID="${DATASET_REPO_ID}" \
    MODEL_OUTPUT="/home/roboticslab/finetuned-models/xvla-chess-balanced-lowpoly-2-${output_suffix}" \
    TRAIN_STEPS=120000 \
    SAVE_FREQ=15000 \
    BATCH_SIZE=8 \
    LEARNING_RATE="${learning_rate}" \
    WARMUP_STEPS=4000 \
    DECAY_STEPS=120000 \
    DECAY_LR=2.5e-6 \
    JOB_NAME="xvla_chess_balanced_lowpoly_2_${job_suffix}" \
    PIPELINE_LOG="${WORKSPACE_ROOT}/logs/balanced_lowpoly_2_${output_suffix}_pipeline.log" \
    PIPELINE_PID_FILE="${WORKSPACE_ROOT}/logs/balanced_lowpoly_2_${output_suffix}_pipeline.pid" \
    LOCK_FILE="${WORKSPACE_ROOT}/logs/balanced_lowpoly_2_convert_train.lock" \
        "${WORKSPACE_ROOT}/scripts/wait_convert_train_xvla.sh"
}

run_training "1e-4" "lr1e-4" "lr1e-4"
run_training "5e-5" "lr5e-5" "lr5e-5"

log "Both XVLA fine-tuning runs completed successfully."
