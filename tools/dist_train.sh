#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}
shift 2

# Default W&B metadata for this training job.
# You can still override these via environment variables at runtime.
: "${WANDB_PROJECT:=Baseline_New_110ep-hyun}"
: "${WANDB_ENTITY:=IRCV_Mapping}"
: "${WANDB_NAME:=Newsplit-Maptrv2-CAM-110ep}"

EXTRA_ARGS=("$@")
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
