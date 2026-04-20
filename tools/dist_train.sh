#!/usr/bin/env bash

DEFAULT_CONFIG="projects/configs/maptrv2/maptrv2_nusc_r50_110ep.py"
DEFAULT_GPUS=8
DEFAULT_WORK_DIR="work_dirs/maptrv2_r50_110ep"
DEFAULT_RESUME_PTH="$DEFAULT_WORK_DIR/latest.pth"
DEFAULT_RESUME_PT="$DEFAULT_WORK_DIR/latest.pt"

CONFIG="${1:-$DEFAULT_CONFIG}"
GPUS="${2:-$DEFAULT_GPUS}"
PORT=${PORT:-28509}

# Keep backward compatibility with positional args while allowing defaults.
[[ $# -gt 0 ]] && shift
[[ $# -gt 0 ]] && shift

# Default W&B metadata for this training job.
# You can still override these via environment variables at runtime.
: "${WANDB_PROJECT:=Baseline_New_110ep-hyun}"
: "${WANDB_ENTITY:=IRCV_Mapping}"
: "${WANDB_NAME:=Newsplit-Maptrv2-CAM-110ep}"

EXTRA_ARGS=("$@")

has_flag() {
    local flag="$1"
    local arg
    for arg in "${EXTRA_ARGS[@]}"; do
        if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
            return 0
        fi
    done
    return 1
}

if ! has_flag --work-dir; then
    EXTRA_ARGS+=(--work-dir "$DEFAULT_WORK_DIR")
fi

if ! has_flag --resume-from; then
    if [[ -f "$DEFAULT_RESUME_PTH" ]]; then
        EXTRA_ARGS+=(--resume-from "$DEFAULT_RESUME_PTH")
    elif [[ -f "$DEFAULT_RESUME_PT" ]]; then
        EXTRA_ARGS+=(--resume-from "$DEFAULT_RESUME_PT")
    fi
fi

if [[ -n "${WANDB_PROJECT:-}" ]]; then
    EXTRA_ARGS+=(--wandb-project "$WANDB_PROJECT")
    [[ -n "${WANDB_ENTITY:-}" ]] && EXTRA_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    [[ -n "${WANDB_NAME:-}" ]] && EXTRA_ARGS+=(--wandb-name "$WANDB_NAME")
    [[ -n "${WANDB_GROUP:-}" ]] && EXTRA_ARGS+=(--wandb-group "$WANDB_GROUP")
    [[ -n "${WANDB_JOB_TYPE:-}" ]] && EXTRA_ARGS+=(--wandb-job-type "$WANDB_JOB_TYPE")
    [[ -n "${WANDB_NOTES:-}" ]] && EXTRA_ARGS+=(--wandb-notes "$WANDB_NOTES")
    [[ -n "${WANDB_TAGS:-}" ]] && EXTRA_ARGS+=(--wandb-tags "$WANDB_TAGS")
    [[ -n "${WANDB_DIR:-}" ]] && EXTRA_ARGS+=(--wandb-dir "$WANDB_DIR")
    [[ "${WANDB_OFFLINE:-0}" == "1" ]] && EXTRA_ARGS+=(--wandb-offline)
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py "$CONFIG" --launcher pytorch "${EXTRA_ARGS[@]}" --deterministic
