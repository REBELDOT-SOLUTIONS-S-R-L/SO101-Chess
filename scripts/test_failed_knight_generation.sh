#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/roboticslab/IsaacTools/IsaacTasks/so101_chess"
PYTHON_BIN="${PYTHON_BIN:-/home/roboticslab/IsaacTools/.venv/bin/python}"
SOURCE_DATASET="${SOURCE_DATASET:-${ROOT}/datasets/annotated_dataset_balanced_lowpoly.hdf5}"
FAILED_CONFIGS="${FAILED_CONFIGS:-${ROOT}/datasets/chess_generated_env_rand_balanced_lowpoly_2_failed.hdf5}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT}/datasets/chess_failed_knight_config_test.hdf5}"
ATTEMPTS="${ATTEMPTS:-100}"
CONFIG_LIMIT="${CONFIG_LIMIT:-${ATTEMPTS}}"
CONFIG_SEED="${CONFIG_SEED:-0}"

if [[ ! -f "${SOURCE_DATASET}" ]]; then
    echo "Missing annotated source dataset: ${SOURCE_DATASET}" >&2
    exit 1
fi
if [[ ! -f "${FAILED_CONFIGS}" ]]; then
    echo "Missing failed-configuration dataset: ${FAILED_CONFIGS}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_FILE}" || -e "${OUTPUT_FILE%.hdf5}_failed.hdf5" ]]; then
    echo "Refusing to overwrite existing test output: ${OUTPUT_FILE}" >&2
    exit 1
fi

exec "${PYTHON_BIN}" "${ROOT}/scripts/generate_dataset.py" \
    --task LeIsaac-SO101-Chess-v0-Mimic \
    --num_envs 1 \
    --generation_num_trials "${ATTEMPTS}" \
    --input_file "${SOURCE_DATASET}" \
    --output_file "${OUTPUT_FILE}" \
    --dataset_schema standard \
    --failed_config_file "${FAILED_CONFIGS}" \
    --failed_piece_type knight \
    --failed_config_limit "${CONFIG_LIMIT}" \
    --failed_config_seed "${CONFIG_SEED}" \
    --enable_pinocchio \
    --enable_cameras \
    --headless \
    "$@"
