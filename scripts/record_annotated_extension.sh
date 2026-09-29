#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/roboticslab/IsaacTools/IsaacTasks/so101_chess"
PYTHON_BIN="${PYTHON_BIN:-/home/roboticslab/IsaacTools/.venv/bin/python}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT}/datasets/annotated_dataset_balanced_lowpoly_extension_10_each.hdf5}"
SOBOL_START_INDEX="${SOBOL_START_INDEX:-10}"
NUM_DEMOS="${NUM_DEMOS:-60}"

if [[ -e "${OUTPUT_FILE}" ]]; then
    echo "Refusing to overwrite existing extension dataset: ${OUTPUT_FILE}" >&2
    exit 1
fi
if (( NUM_DEMOS % 6 != 0 )); then
    echo "NUM_DEMOS must be divisible by 6 for equal per-piece coverage." >&2
    exit 1
fi
for argument in "$@"; do
    case "${argument}" in
        --dataset_file|--dataset_file=*|--num_demos|--num_demos=*|--piece_type|--piece_type=*|\
        --sobol_start_index|--sobol_start_index=*|--sobol_plan|--sobol_plan=*|--enable_cameras)
            echo "This workflow manages or forbids recorder argument: ${argument}" >&2
            exit 1
            ;;
    esac
done

echo "Recording $((NUM_DEMOS / 6)) new demonstrations per piece."
echo "Continuing each per-piece Sobol stream at draw ${SOBOL_START_INDEX}."
echo "Output: ${OUTPUT_FILE}"

# Additional arguments are forwarded to the recorder. For example:
#   --teleop_device so101leader --port /dev/ttyACM0 --enable_pinocchio
# Do not pass --enable_cameras: the existing annotated source dataset has no
# camera arrays, and the extension must retain the same schema.
exec "${PYTHON_BIN}" "${ROOT}/scripts/record_annotated_demos_balanced.py" \
    --task LeIsaac-SO101-Chess-v0-Mimic \
    --dataset_file "${OUTPUT_FILE}" \
    --num_demos "${NUM_DEMOS}" \
    --sobol_start_index "${SOBOL_START_INDEX}" \
    "$@"
