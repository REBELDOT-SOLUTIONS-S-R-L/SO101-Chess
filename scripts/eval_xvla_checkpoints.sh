#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate every numbered XVLA checkpoint in the supplied checkpoint roots.
# With no positional arguments, both balanced-lowpoly-2 runs are swept.
#
# Configuration is available through environment variables:
#   EPISODES=25 EPISODE_LENGTH_S=12 ACTION_HORIZON=30 SEED=42
#   OUTPUT_DIR=/path/to/results PORT=8765 HEADLESS=1
#
# Examples:
#   scripts/eval_xvla_checkpoints.sh
#   EPISODES=5 scripts/eval_xvla_checkpoints.sh \
#     /path/to/model/checkpoints/060000/pretrained_model
#   scripts/eval_xvla_checkpoints.sh /path/to/model/checkpoints

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/roboticslab/IsaacTools/.venv/bin/python}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/roboticslab/lerobot/.venv/bin/python}"
EPISODES="${EPISODES:-25}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-12}"
ACTION_HORIZON="${ACTION_HORIZON:-30}"
SEED="${SEED:-42}"
PORT="${PORT:-8765}"
HEADLESS="${HEADLESS:-1}"
RESET_CAMERA_RENDERS="${RESET_CAMERA_RENDERS:-60}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-180}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/logs/xvla_checkpoint_eval_$(date +%Y%m%d_%H%M%S)}"

if (( $# > 0 )); then
    checkpoint_sources=("$@")
else
    checkpoint_sources=(
        "/home/roboticslab/finetuned-models/xvla-chess-balanced-lowpoly-2-lr1e-4/checkpoints"
        "/home/roboticslab/finetuned-models/xvla-chess-balanced-lowpoly-2-lr5e-5/checkpoints"
    )
fi

for executable in "${ISAAC_PYTHON}" "${LEROBOT_PYTHON}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Required Python executable is missing: ${executable}" >&2
        exit 2
    fi
done
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for inference-server health checks." >&2
    exit 2
fi
if ! [[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODES must be a positive integer, got ${EPISODES}." >&2
    exit 2
fi
if ! [[ "${ACTION_HORIZON}" =~ ^([1-9]|[12][0-9]|30)$ ]]; then
    echo "ACTION_HORIZON must be an integer from 1 through 30, got ${ACTION_HORIZON}." >&2
    exit 2
fi

checkpoints=()
shopt -s nullglob
for source in "${checkpoint_sources[@]}"; do
    source="${source%/}"
    if [[ -f "${source}/model.safetensors" ]]; then
        checkpoints+=("$(readlink -f "${source}")")
    elif [[ -f "${source}/pretrained_model/model.safetensors" ]]; then
        checkpoints+=("$(readlink -f "${source}/pretrained_model")")
    elif [[ -d "${source}" ]]; then
        for candidate in \
            "${source}"/[0-9]*/pretrained_model \
            "${source}"/checkpoints/[0-9]*/pretrained_model; do
            if [[ -f "${candidate}/model.safetensors" ]]; then
                checkpoints+=("$(readlink -f "${candidate}")")
            fi
        done
    else
        echo "Checkpoint source does not exist: ${source}" >&2
        exit 2
    fi
done
shopt -u nullglob

if (( ${#checkpoints[@]} == 0 )); then
    echo "No numbered pretrained_model checkpoints were found." >&2
    exit 2
fi
mapfile -t checkpoints < <(printf '%s\n' "${checkpoints[@]}" | sort -Vu)

mkdir -p "${OUTPUT_DIR}"
summary_file="${OUTPUT_DIR}/summary.tsv"
printf 'checkpoint\tstatus\tepisode_log\tconsole_log\tserver_log\n' > "${summary_file}"

server_pid=""
stop_server() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -INT "${server_pid}" 2>/dev/null || true
        for _ in {1..20}; do
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "${server_pid}" 2>/dev/null; then
            kill -TERM "${server_pid}" 2>/dev/null || true
            for _ in {1..10}; do
                if ! kill -0 "${server_pid}" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
        fi
        if kill -0 "${server_pid}" 2>/dev/null; then
            kill -KILL "${server_pid}" 2>/dev/null || true
        fi
        wait "${server_pid}" 2>/dev/null || true
    fi
    server_pid=""
}
trap stop_server EXIT
trap 'stop_server; exit 130' INT TERM

start_server() {
    local checkpoint="$1"
    local server_log="$2"
    if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Port ${PORT} already has a healthy server; refusing to replace an unowned process." >&2
        return 1
    fi

    "${LEROBOT_PYTHON}" "${WORKSPACE_ROOT}/scripts/xvla_inference_server.py" \
        --checkpoint "${checkpoint}" \
        --port "${PORT}" \
        --seed "${SEED}" \
        > "${server_log}" 2>&1 &
    server_pid=$!

    for ((second = 1; second <= SERVER_READY_TIMEOUT_S; second++)); do
        if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            wait "${server_pid}" 2>/dev/null || true
            echo "Inference server exited before becoming ready; see ${server_log}." >&2
            server_pid=""
            return 1
        fi
        sleep 1
    done

    echo "Inference server was not ready after ${SERVER_READY_TIMEOUT_S}s; see ${server_log}." >&2
    return 1
}

echo "Found ${#checkpoints[@]} checkpoints. Results: ${OUTPUT_DIR}"
failures=0
for checkpoint in "${checkpoints[@]}"; do
    step_name="$(basename "$(dirname "${checkpoint}")")"
    checkpoints_dir="$(dirname "$(dirname "${checkpoint}")")"
    run_name="$(basename "$(dirname "${checkpoints_dir}")")"
    label="${run_name}_${step_name}"
    result_dir="${OUTPUT_DIR}/${label}"
    episode_log="${result_dir}/episodes.jsonl"
    console_log="${result_dir}/eval_console.log"
    server_log="${result_dir}/server.log"

    if [[ -e "${result_dir}" ]]; then
        echo "Result directory already exists; refusing to mix runs: ${result_dir}" >&2
        printf '%s\t%s\t%s\t%s\t%s\n' "${checkpoint}" "result_dir_exists" "${episode_log}" "${console_log}" "${server_log}" >> "${summary_file}"
        failures=$((failures + 1))
        continue
    fi
    mkdir -p "${result_dir}"
    echo
    echo "=== Evaluating ${label} (${checkpoint}) ==="

    if ! start_server "${checkpoint}" "${server_log}"; then
        stop_server
        printf '%s\t%s\t%s\t%s\t%s\n' "${checkpoint}" "server_failed" "${episode_log}" "${console_log}" "${server_log}" >> "${summary_file}"
        failures=$((failures + 1))
        continue
    fi

    eval_command=(
        "${ISAAC_PYTHON}" "${WORKSPACE_ROOT}/scripts/eval_xvla.py"
        --enable_cameras
        --episodes "${EPISODES}"
        --episode_length_s "${EPISODE_LENGTH_S}"
        --action_horizon "${ACTION_HORIZON}"
        --seed "${SEED}"
        --reset_camera_renders "${RESET_CAMERA_RENDERS}"
        --policy_url "http://127.0.0.1:${PORT}/infer"
        --checkpoint_label "${checkpoint}"
        --log_file "${episode_log}"
    )
    if [[ "${HEADLESS}" == "1" ]]; then
        eval_command+=(--headless)
    fi

    set +e
    "${eval_command[@]}" 2>&1 | tee "${console_log}"
    eval_status=${PIPESTATUS[0]}
    set -e
    stop_server

    if (( eval_status == 0 )); then
        status="complete"
    else
        status="eval_failed_${eval_status}"
        failures=$((failures + 1))
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "${checkpoint}" "${status}" "${episode_log}" "${console_log}" "${server_log}" >> "${summary_file}"
done

echo
echo "Checkpoint sweep finished with ${failures} failure(s). Summary: ${summary_file}"
if (( failures > 0 )); then
    exit 1
fi
