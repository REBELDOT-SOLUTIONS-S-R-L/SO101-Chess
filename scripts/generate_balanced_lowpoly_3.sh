#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/roboticslab/IsaacTools/IsaacTasks/so101_chess"
PYTHON_BIN="${PYTHON_BIN:-/home/roboticslab/IsaacTools/.venv/bin/python}"
SOURCE_DATASET="${SOURCE_DATASET:-${ROOT}/datasets/annotated_dataset_balanced_lowpoly_20_each.hdf5}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT}/datasets/chess_generated_env_rand_balanced_lowpoly_3.hdf5}"
NUM_EPISODES="${NUM_EPISODES:-1000}"

if [[ ! -f "${SOURCE_DATASET}" ]]; then
    echo "Missing merged annotated source dataset: ${SOURCE_DATASET}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_FILE}" || -e "${OUTPUT_FILE%.hdf5}_failed.hdf5" ]]; then
    echo "Refusing to overwrite generated output: ${OUTPUT_FILE}" >&2
    exit 1
fi
if (( NUM_EPISODES < 1 )); then
    echo "NUM_EPISODES must be positive." >&2
    exit 1
fi

# Normal generation retains the task's type-balanced random reset strategy.
# generation_guarantee remains enabled, so NUM_EPISODES means successful
# episodes; failed attempts are written to the companion *_failed.hdf5 file.
exec "${PYTHON_BIN}" "${ROOT}/scripts/generate_dataset.py" \
    --task LeIsaac-SO101-Chess-v0-Mimic \
    --num_envs 1 \
    --generation_num_trials "${NUM_EPISODES}" \
    --input_file "${SOURCE_DATASET}" \
    --output_file "${OUTPUT_FILE}" \
    --dataset_schema standard \
    --enable_pinocchio \
    --enable_cameras \
    --headless \
    "$@"
