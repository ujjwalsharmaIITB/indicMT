#!/bin/bash
#SBATCH --job-name=job13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --open-mode=append
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job13: fine-tuning on the human corpus, starting from a pre-trained checkpoint.
# 1 node, 4x A100 80 GB, DDP. One config per experiment, in training/configs/:
#   finetune_mined_human            mined-only checkpoint   -> human only
#   finetune_mined_human_replay     mined-only checkpoint   -> human + 20% mined replay
#   finetune_combined_human         combined 8:2 checkpoint -> human only   (the current best model)
#   finetune_combined_human_replay  combined 8:2 checkpoint -> human + 20% mined replay
#   finetune_combined_12e18d_human, finetune_combined_18e18d_human   architecture study, stage 2
#
# Usage (the config name and the starting checkpoint are required):
#   sbatch -p <partition> \
#          --export=ALL,CFG=finetune_combined_human,SRC=128k-spm/opennmt/bpcc-combined/checkpoints/model_step_13600.pt \
#          scripts/slurm/training_finetune.sh
#
# --reset_optim all: load the model weights, build a NEW optimizer and restart the step counter at
# 0, so the fine-tuning LR schedule in the config applies as written. Output checkpoints are named
# by fine-tuning step. Resume after a failure is opt-in as in the other training scripts:
# RESUME=1 continues from the newest complete checkpoint of the fine-tuning run itself (weights,
# optimizer state and step), which then takes precedence over SRC.
# See training/README.md.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

if [ -z "${CFG:-}" ] || [ -z "${SRC:-}" ]; then
    echo "CFG (config name without .4gpu.yaml) and SRC (starting checkpoint) must be exported"; exit 1
fi
CONFIG=training/configs/${CFG}.4gpu.yaml
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }
[ -f "$SRC" ] || { echo "no such checkpoint: $SRC"; exit 1; }
NAME=$(grep '^save_model:' "$CONFIG" | sed 's#.*/opennmt/\([^/]*\)/checkpoints.*#\1#')
CKPT_DIR=128k-spm/opennmt/$NAME/checkpoints
mkdir -p "$CKPT_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

FROM_ARG="--train_from $SRC --reset_optim all"
mapfile -t CK < <(ls -1 "$CKPT_DIR"/model_step_*.pt 2>/dev/null | sort -V)
NCK=${#CK[@]}
if [ "${RESUME:-0}" = "1" ] && [ "$NCK" -ge 2 ]; then
    LAST="${CK[$((NCK-1))]}"; PREV="${CK[$((NCK-2))]}"
    if [ "$(stat -c %s "$LAST")" = "$(stat -c %s "$PREV")" ]; then PICK="$LAST"; else PICK="$PREV"; fi
    FROM_ARG="--train_from $PICK"
    echo "Resuming the fine-tuning run from $PICK"
else
    echo "Fine-tuning $NAME from $SRC (size $(stat -c %s "$SRC"))"
fi

START_MARK=$(mktemp "$CKPT_DIR/.start.XXXXXX")
echo "Starting $NAME ($(date))"
CUDA_VISIBLE_DEVICES=0,1,2,3 onmt_train -config "$CONFIG" $FROM_ARG
rc=$?
# onmt_train can exit 0 when its DDP worker processes die at spawn (a shared-memory semaphore
# vanished on one node), so also require a checkpoint written by this run. (The config's log.txt
# only records process start-up; step lines go to the job's stderr.)
if [ "$rc" -ne 0 ] || [ -z "$(find "$CKPT_DIR" -maxdepth 1 -name 'model_step_*.pt' -newer "$START_MARK" 2>/dev/null | head -1)" ]; then
    echo "FAILED: onmt_train exit $rc, no checkpoint written by this run ($(date))"; exit 1
fi
rm -f "$START_MARK"
echo "Done ($(date))"
