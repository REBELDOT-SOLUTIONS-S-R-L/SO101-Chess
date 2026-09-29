#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/roboticslab/IsaacTools/IsaacTasks/so101_chess"
SOURCE="${ROOT}/datasets/annotated_dataset_balanced_lowpoly_20_each.hdf5"
GENERATED="${ROOT}/datasets/chess_generated_env_rand_balanced_lowpoly_3.hdf5"
FAILED="${ROOT}/datasets/chess_generated_env_rand_balanced_lowpoly_3_failed.hdf5"
LEROBOT="/home/roboticslab/datasets/so101_chess_balanced_lowpoly_3"
REPO_ID="alex-luci/so101_chess_balanced_lowpoly_3"
TASK="Move the chess piece from the red square to the green square"
EXPECTED_EPISODES=1000

ISAAC_PYTHON="/home/roboticslab/IsaacTools/.venv/bin/python"
REBEL_PYTHON="/home/roboticslab/rebelHDF5/.venv/bin/python"
HF_PYTHON="/home/alexluci/.local/share/uv/tools/huggingface-hub/bin/python"
HF_CLI="/home/alexluci/.local/bin/hf"
FFPROBE="/usr/bin/ffprobe"

LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/balanced_lowpoly_3_pipeline.log"
STATUS_FILE="${LOG_DIR}/balanced_lowpoly_3_pipeline.status"
LOCK_FILE="${LOG_DIR}/balanced_lowpoly_3_pipeline.lock"
MIN_FREE_BYTES=$((300 * 1024 * 1024 * 1024))
PREFLIGHT_ONLY=false
CURRENT_PHASE="startup"

if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--preflight-only]" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another balanced_lowpoly_3 pipeline already holds ${LOCK_FILE}." >&2
    exit 1
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

write_status() {
    local state="$1"
    local message="$2"
    local temporary="${STATUS_FILE}.tmp"
    {
        printf 'state=%s\n' "${state}"
        printf 'phase=%s\n' "${CURRENT_PHASE}"
        printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'pid=%s\n' "$$"
        printf 'message=%s\n' "${message}"
        printf 'log=%s\n' "${LOG_FILE}"
        printf 'repo=https://huggingface.co/datasets/%s\n' "${REPO_ID}"
    } > "${temporary}"
    mv "${temporary}" "${STATUS_FILE}"
}

set_phase() {
    CURRENT_PHASE="$1"
    write_status "RUNNING" "$2"
    printf '\n[%s] phase=%s %s\n' "$(date --iso-8601=seconds)" "${CURRENT_PHASE}" "$2"
}

on_exit() {
    local code=$?
    if (( code == 0 )); then
        if [[ "${PREFLIGHT_ONLY}" == true ]]; then
            write_status "PREFLIGHT_OK" "All preflight checks passed."
        else
            write_status "COMPLETE" "Generation, conversion, upload, and remote validation completed."
        fi
    else
        write_status "FAILED" "Pipeline exited with code ${code}; inspect the log."
    fi
}
trap on_exit EXIT

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        return 1
    fi
}

set_phase "preflight" "Checking source data, runtimes, storage, and remote repository."
for path in \
    "${SOURCE}" \
    "${ROOT}/scripts/generate_balanced_lowpoly_3.sh" \
    "${ROOT}/scripts/convert_hdf5_to_lerobot_v3.py" \
    "${ROOT}/scripts/validate_chess_hdf5.py" \
    "${ROOT}/scripts/validate_lerobot_v3_dataset.py" \
    "${ROOT}/scripts/validate_hf_lerobot_upload.py" \
    "${ROOT}/configs/lerobot/so101_chess_modality.json" \
    "${ROOT}/configs/lerobot/so101_chess_conversion.json" \
    "${ROOT}/configs/lerobot/README_lowpoly_3.md" \
    "${ISAAC_PYTHON}" \
    "${REBEL_PYTHON}" \
    "${HF_PYTHON}" \
    "${HF_CLI}" \
    "${FFPROBE}"; do
    require_file "${path}"
done
for command in flock tee; do
    command -v "${command}" >/dev/null || { echo "Missing command: ${command}" >&2; exit 1; }
done

if [[ -e "${GENERATED}" || -e "${FAILED}" || -e "${LEROBOT}" ]]; then
    echo "Refusing to start because an intended output already exists:" >&2
    printf '  %s\n' "${GENERATED}" "${FAILED}" "${LEROBOT}" >&2
    exit 1
fi
free_bytes="$(df -PB1 "${ROOT}" | awk 'NR == 2 {print $4}')"
if (( free_bytes < MIN_FREE_BYTES )); then
    echo "Need at least ${MIN_FREE_BYTES} free bytes; only ${free_bytes} are available." >&2
    exit 1
fi

"${ISAAC_PYTHON}" "${ROOT}/scripts/validate_chess_hdf5.py" \
    --dataset "${SOURCE}" \
    --expected-episodes 120 \
    --expected-per-piece 20 \
    --expected-success true

PYTHONPATH="/home/roboticslab/rebelHDF5/scripts" "${REBEL_PYTHON}" -c \
    'from backend.lerobot import _require_pyarrow, select_video_encoder; _require_pyarrow(); print("video_encoder=", select_video_encoder("h264"), flush=True)'
"${HF_CLI}" auth whoami
"${HF_PYTHON}" -c \
    'from huggingface_hub import HfApi; import sys; r=HfApi().repo_info(sys.argv[1], repo_type="dataset"); assert not r.private; print(f"repo={r.id} public={not r.private} revision={r.sha}")' \
    "${REPO_ID}"
printf 'free_space_gib=%s\n' "$((free_bytes / 1024 / 1024 / 1024))"

if [[ "${PREFLIGHT_ONLY}" == true ]]; then
    echo "Preflight completed successfully."
    exit 0
fi

set_phase "generation" "Generating ${EXPECTED_EPISODES} successful episodes."
"${ROOT}/scripts/generate_balanced_lowpoly_3.sh"

set_phase "hdf5_validation" "Validating successful and failed HDF5 outputs."
"${ISAAC_PYTHON}" "${ROOT}/scripts/validate_chess_hdf5.py" \
    --dataset "${GENERATED}" \
    --expected-episodes "${EXPECTED_EPISODES}" \
    --expected-success true \
    --require-balanced-types \
    --require-cameras
"${ISAAC_PYTHON}" "${ROOT}/scripts/validate_chess_hdf5.py" \
    --dataset "${FAILED}" \
    --expected-success false \
    --require-cameras

set_phase "conversion" "Converting the successful HDF5 episodes to LeRobot v3."
"${REBEL_PYTHON}" "${ROOT}/scripts/convert_hdf5_to_lerobot_v3.py" \
    --input "${GENERATED}" \
    --output "${LEROBOT}" \
    --modality-json "${ROOT}/configs/lerobot/so101_chess_modality.json" \
    --conversion-config "${ROOT}/configs/lerobot/so101_chess_conversion.json" \
    --expected-episodes "${EXPECTED_EPISODES}"
cp "${ROOT}/configs/lerobot/README_lowpoly_3.md" "${LEROBOT}/README.md"

set_phase "lerobot_validation" "Validating metadata, Parquet rows, task text, and every video."
"${REBEL_PYTHON}" "${ROOT}/scripts/validate_lerobot_v3_dataset.py" \
    --dataset "${LEROBOT}" \
    --expected-episodes "${EXPECTED_EPISODES}" \
    --expected-task "${TASK}" \
    --ffprobe "${FFPROBE}"

set_phase "upload" "Uploading the validated LeRobot dataset to ${REPO_ID}."
"${HF_CLI}" upload-large-folder "${REPO_ID}" "${LEROBOT}" \
    --repo-type dataset \
    --num-workers 4 \
    --no-bars

set_phase "remote_validation" "Comparing every local file with the Hugging Face repository."
"${HF_PYTHON}" "${ROOT}/scripts/validate_hf_lerobot_upload.py" \
    --dataset "${LEROBOT}" \
    --repo-id "${REPO_ID}" \
    --expected-episodes "${EXPECTED_EPISODES}"

CURRENT_PHASE="complete"
echo "Pipeline complete: https://huggingface.co/datasets/${REPO_ID}"
