#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-sweep}"
LEROBOT_ROOT="/home/shadeform/lerobot/lerobot"
TRAIN_BIN="/home/shadeform/lerobot/.venv/bin/lerobot-train"
PYTHON_BIN="/home/shadeform/lerobot/.venv/bin/python"
DATASET_ROOT="/home/shadeform/datasets"
DATASET_REPO="alex-luci/so101_chess_balanced_lowpoly_3"
OUTPUT_ROOT="/ephemeral/xvla-hpo"
LOG_ROOT="${OUTPUT_ROOT}/logs"
SUMMARY="${OUTPUT_ROOT}/summary.tsv"
LOCK="${OUTPUT_ROOT}/sweep.lock"
RENAME_MAP='{"observation.images.top_camera":"observation.images.image","observation.images.right_wrist_camera":"observation.images.image2"}'

mkdir -p "${LOG_ROOT}"
exec 9>"${LOCK}"
if ! flock -n 9; then
    echo "Another xVLA HPO process already holds ${LOCK}." >&2
    exit 1
fi

if [[ ! -x "${TRAIN_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "LeRobot virtual environment is incomplete." >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Dataset metadata is missing from ${DATASET_ROOT}." >&2
    exit 1
fi

# Passing episodes 0..799 and eval_split=0.125 yields exactly 700 training
# episodes and 100 validation episodes. Episodes 800..999 remain untouched.
TRAIN_VAL_EPISODES="$(${PYTHON_BIN} -c 'import json; print(json.dumps(list(range(800))))')"

declare -A LR WD FREEZE_LANG AUGMENT

LR[baseline]="1e-4";      WD[baseline]="0";    FREEZE_LANG[baseline]="false"; AUGMENT[baseline]="true"
LR[lr_7p5e5]="7.5e-5";    WD[lr_7p5e5]="0";    FREEZE_LANG[lr_7p5e5]="false"; AUGMENT[lr_7p5e5]="true"
LR[lr_1p5e4]="1.5e-4";    WD[lr_1p5e4]="0";    FREEZE_LANG[lr_1p5e4]="false"; AUGMENT[lr_1p5e4]="true"
LR[weight_decay]="1e-4";  WD[weight_decay]="1e-4"; FREEZE_LANG[weight_decay]="false"; AUGMENT[weight_decay]="true"
LR[freeze_language]="1e-4"; WD[freeze_language]="0"; FREEZE_LANG[freeze_language]="true"; AUGMENT[freeze_language]="true"
LR[no_augmentation]="1e-4"; WD[no_augmentation]="0"; FREEZE_LANG[no_augmentation]="false"; AUGMENT[no_augmentation]="false"
LR[low_combo]="7.5e-5";   WD[low_combo]="1e-4"; FREEZE_LANG[low_combo]="true"; AUGMENT[low_combo]="true"
LR[high_combo]="1.5e-4";  WD[high_combo]="1e-4"; FREEZE_LANG[high_combo]="true"; AUGMENT[high_combo]="true"

write_header() {
    if [[ ! -f "${SUMMARY}" ]]; then
        printf 'stage\tname\tsteps\tlearning_rate\tweight_decay\tfreeze_language\taugmentation\teval_loss\telapsed_seconds\tstatus\n' > "${SUMMARY}"
    fi
}

existing_result() {
    local stage="$1"
    local name="$2"
    awk -F '\t' -v stage="${stage}" -v name="${name}" \
        '$1 == stage && $2 == name && $10 == "complete" {print $8; exit}' "${SUMMARY}" 2>/dev/null
}

run_trial() {
    local stage="$1"
    local name="$2"
    local steps="$3"
    local warmup="$4"
    local eval_samples="$5"
    local output_dir="${OUTPUT_ROOT}/${stage}_${name}"
    local log_file="${LOG_ROOT}/${stage}_${name}.log"
    local previous
    previous="$(existing_result "${stage}" "${name}")"
    if [[ -n "${previous}" ]]; then
        echo "Reusing completed ${stage}/${name}: eval_loss=${previous}"
        return 0
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "Refusing to overwrite incomplete trial output: ${output_dir}" >&2
        return 1
    fi

    local start end elapsed eval_loss
    start="$(date +%s)"
    echo "[$(date --iso-8601=seconds)] Starting ${stage}/${name}: steps=${steps} lr=${LR[${name}]} wd=${WD[${name}]} freeze_language=${FREEZE_LANG[${name}]} augmentation=${AUGMENT[${name}]}"

    cd "${LEROBOT_ROOT}"
    "${TRAIN_BIN}" \
        --dataset.repo_id="${DATASET_REPO}" \
        --dataset.root="${DATASET_ROOT}" \
        --dataset.episodes="${TRAIN_VAL_EPISODES}" \
        --dataset.eval_split=0.125 \
        --dataset.image_transforms.enable="${AUGMENT[${name}]}" \
        --policy.path=lerobot/xvla-base \
        --policy.device=cuda \
        --policy.dtype=bfloat16 \
        --policy.action_mode=auto \
        --policy.push_to_hub=false \
        --policy.optimizer_lr="${LR[${name}]}" \
        --policy.optimizer_weight_decay="${WD[${name}]}" \
        --policy.freeze_language_encoder="${FREEZE_LANG[${name}]}" \
        --policy.freeze_vision_encoder=false \
        --policy.scheduler_warmup_steps="${warmup}" \
        --policy.scheduler_decay_steps="${steps}" \
        --policy.scheduler_decay_lr=2.5e-6 \
        --batch_size=8 \
        --accelerator.gradient_accumulation.steps=1 \
        --steps="${steps}" \
        --env_eval_freq=0 \
        --eval_steps="${steps}" \
        --max_eval_samples="${eval_samples}" \
        --log_freq=200 \
        --save_checkpoint=false \
        --output_dir="${output_dir}" \
        --job_name="xvla_chess_hpo_${stage}_${name}" \
        --wandb.enable=false \
        --rename_map="${RENAME_MAP}" \
        2>&1 | tee "${log_file}"

    eval_loss="$(grep -Eo 'eval_loss=[0-9]+([.][0-9]+)?' "${log_file}" | tail -1 | cut -d= -f2)"
    if [[ -z "${eval_loss}" ]]; then
        echo "No eval_loss was emitted for ${stage}/${name}." >&2
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tNA\t0\tfailed\n' \
            "${stage}" "${name}" "${steps}" "${LR[${name}]}" "${WD[${name}]}" \
            "${FREEZE_LANG[${name}]}" "${AUGMENT[${name}]}" >> "${SUMMARY}"
        return 1
    fi
    end="$(date +%s)"
    elapsed=$((end - start))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tcomplete\n' \
        "${stage}" "${name}" "${steps}" "${LR[${name}]}" "${WD[${name}]}" \
        "${FREEZE_LANG[${name}]}" "${AUGMENT[${name}]}" "${eval_loss}" "${elapsed}" >> "${SUMMARY}"
    echo "[$(date --iso-8601=seconds)] Completed ${stage}/${name}: eval_loss=${eval_loss}, elapsed=${elapsed}s"
}

write_header

case "${MODE}" in
    pilot)
        run_trial "pilot" "baseline" 100 2 128
        ;;
    sweep)
        stage1=(baseline lr_7p5e5 lr_1p5e4 weight_decay freeze_language no_augmentation low_combo high_combo)
        for name in "${stage1[@]}"; do
            run_trial "screen" "${name}" 5000 100 512
        done

        mapfile -t finalists < <(
            awk -F '\t' '$1 == "screen" && $10 == "complete" {print $8 "\t" $2}' "${SUMMARY}" \
                | sort -g -k1,1 \
                | awk -F '\t' '!seen[$2]++ {print $2}' \
                | head -2
        )
        if (( ${#finalists[@]} != 2 )); then
            echo "Expected two screening finalists, found ${#finalists[@]}." >&2
            exit 1
        fi
        printf 'Screening finalists: %s, %s\n' "${finalists[0]}" "${finalists[1]}"
        for name in "${finalists[@]}"; do
            run_trial "finalist" "${name}" 20000 400 2048
        done
        ;;
    *)
        echo "Usage: $0 [pilot|sweep]" >&2
        exit 2
        ;;
esac

echo "Results: ${SUMMARY}"
