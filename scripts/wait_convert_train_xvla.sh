#!/usr/bin/env bash
set -Eeuo pipefail

# Unattended balanced_3 handoff:
#   running Isaac Mimic generator -> validated LeRobot v3 dataset -> XVLA fine-tuning.
# Every path and training hyperparameter can be overridden through the environment.

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/roboticslab/IsaacTools/IsaacTasks/so101_chess}"
SOURCE_HDF5="${SOURCE_HDF5:-${WORKSPACE_ROOT}/datasets/chess_generated_env_rand_balanced_3.hdf5}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-500}"
GENERATOR_PID="${GENERATOR_PID:-3881119}"

REBELHDF5_ROOT="${REBELHDF5_ROOT:-/home/roboticslab/rebelHDF5}"
REBELHDF5_PYTHON="${REBELHDF5_PYTHON:-${REBELHDF5_ROOT}/.venv/bin/python}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/roboticslab/lerobot}"
UV_BIN="${UV_BIN:-/home/alexluci/.local/bin/uv}"
LEROBOT_OUTPUT="${LEROBOT_OUTPUT:-/home/roboticslab/datasets/so101_chess_balanced_3}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/so101_chess_balanced_3}"
MODEL_OUTPUT="${MODEL_OUTPUT:-/home/roboticslab/finetuned-models/xvla-chess-balanced-3-lr5e-5}"

TRAIN_STEPS="${TRAIN_STEPS:-120000}"
SAVE_FREQ="${SAVE_FREQ:-15000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-4000}"
DECAY_STEPS="${DECAY_STEPS:-120000}"
DECAY_LR="${DECAY_LR:-2.5e-6}"
JOB_NAME="${JOB_NAME:-xvla_chess_balanced_3_lr5e-5}"

LOG_DIR="${LOG_DIR:-${WORKSPACE_ROOT}/logs}"
PIPELINE_LOG="${PIPELINE_LOG:-${LOG_DIR}/balanced_3_convert_train.log}"
PIPELINE_PID_FILE="${PIPELINE_PID_FILE:-${LOG_DIR}/balanced_3_convert_train.pid}"
LOCK_FILE="${LOCK_FILE:-${LOG_DIR}/balanced_3_convert_train.lock}"

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another balanced_3 conversion/training handoff is already running." >&2
    exit 1
fi

exec > >(tee -a "${PIPELINE_LOG}") 2>&1
printf '%s\n' "$$" > "${PIPELINE_PID_FILE}"
trap 'status=$?; rm -f "${PIPELINE_PID_FILE}"; echo "[$(date --iso-8601=seconds)] Pipeline exited with status ${status}."; exit "${status}"' EXIT

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

require_executable() {
    if [[ ! -x "$1" ]]; then
        log "ERROR: required executable is missing: $1"
        exit 1
    fi
}

require_executable "${REBELHDF5_PYTHON}"
require_executable "${LEROBOT_ROOT}/.venv/bin/python"
require_executable "${UV_BIN}"
if [[ ! -x "${LEROBOT_ROOT}/.venv/bin/lerobot-train" ]]; then
    log "ERROR: lerobot-train is missing from ${LEROBOT_ROOT}/.venv/bin"
    exit 1
fi

log "Unattended pipeline started (PID $$)."
log "Waiting for generator PID ${GENERATOR_PID}; source=${SOURCE_HDF5}"

if [[ -d "/proc/${GENERATOR_PID}" ]]; then
    generator_command="$(tr '\0' ' ' < "/proc/${GENERATOR_PID}/cmdline" 2>/dev/null || true)"
    if [[ "${generator_command}" != *"${SOURCE_HDF5}"* ]]; then
        log "ERROR: PID ${GENERATOR_PID} is not writing the expected source dataset."
        log "Observed command: ${generator_command}"
        exit 1
    fi

    wait_minutes=0
    while [[ -d "/proc/${GENERATOR_PID}" ]]; do
        sleep 60
        wait_minutes=$((wait_minutes + 1))
        if (( wait_minutes % 10 == 0 )); then
            source_size="$(stat -c '%s' "${SOURCE_HDF5}" 2>/dev/null || echo 0)"
            log "Generator is still running after ${wait_minutes} min; source bytes=${source_size}."
        fi
    done
    log "Generator PID ${GENERATOR_PID} exited."
else
    log "Generator PID ${GENERATOR_PID} is already absent; validating the source immediately."
fi

# A completed writer must leave exactly EXPECTED_EPISODES successful demos.
# The converter also validates every episode's success attribute before writing.
log "Validating and converting the completed HDF5 dataset."
"${REBELHDF5_PYTHON}" "${WORKSPACE_ROOT}/scripts/convert_hdf5_to_lerobot_v3.py" \
    --input "${SOURCE_HDF5}" \
    --output "${LEROBOT_OUTPUT}" \
    --modality-json "${WORKSPACE_ROOT}/configs/lerobot/so101_chess_modality.json" \
    --conversion-config "${WORKSPACE_ROOT}/configs/lerobot/so101_chess_conversion.json" \
    --expected-episodes "${EXPECTED_EPISODES}" \
    --rebelhdf5-root "${REBELHDF5_ROOT}"

if [[ -e "${MODEL_OUTPUT}" ]]; then
    log "ERROR: refusing to overwrite existing model output: ${MODEL_OUTPUT}"
    exit 1
fi

log "LeRobot dataset is ready at ${LEROBOT_OUTPUT}."
log "Starting XVLA fine-tuning: lr=${LEARNING_RATE}, batch=${BATCH_SIZE}, steps=${TRAIN_STEPS}."

cd "${LEROBOT_ROOT}"
"${UV_BIN}" run --no-sync lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${LEROBOT_OUTPUT}" \
    --policy.path=lerobot/xvla-base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.action_mode=auto \
    --policy.push_to_hub=false \
    --policy.optimizer_lr="${LEARNING_RATE}" \
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
    --policy.scheduler_decay_steps="${DECAY_STEPS}" \
    --policy.scheduler_decay_lr="${DECAY_LR}" \
    --batch_size="${BATCH_SIZE}" \
    --accelerator.gradient_accumulation.steps=1 \
    --steps="${TRAIN_STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --output_dir="${MODEL_OUTPUT}" \
    --job_name="${JOB_NAME}" \
    --rename_map='{"observation.images.top_camera":"observation.images.image","observation.images.right_wrist_camera":"observation.images.image2"}'

log "XVLA fine-tuning completed successfully."
