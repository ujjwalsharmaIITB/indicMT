#!/bin/bash
#SBATCH --job-name=job09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job9: OpenNMT-py pre-training on bpcc-human-train only (the human-only arm of the data study).
# 2 GPUs x accum_count 8 = the same effective 262,144 tokens per step as the 4-GPU arms.
# Config: training/configs/pretrain_human.2gpu.yaml (see training/README.md).
# Run from the repository root: sbatch -p <partition> scripts/slurm/training_pretrain_human.sh
# If your cluster has a node with a stalled filesystem or heavy contention, exclude it at
# submission time (sbatch --exclude=<node>).

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

mkdir -p 128k-spm/opennmt/bpcc-human/checkpoints

echo "Starting bpcc-human-train run (2 GPUs x accum 8) ($(date))"
# CPU: every rank process needs a physical core of its own (data pipeline and kernel launches run
# in the main process, num_workers 0). A job without --cpus-per-task got 2 CPUs that were the two
# hyperthreads of ONE core, shared by all ranks, and the GPUs idled waiting on the CPU. Threads per
# process are pinned so the ranks do not oversubscribe each other.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Resume is OPT-IN: a run starts from step 0 unless RESUME=1 is exported (sbatch --export=ALL,RESUME=1 ...).
# When it is set, training continues from the newest complete checkpoint (weights, optimizer state and
# step). All checkpoints have the same size, so a newest file whose size differs from the previous one is
# a partial write and is skipped. A resume restarts the data iterator at the beginning of the training
# file (batches repeat) and resets the early-stopping counter.
CKPT_DIR=128k-spm/opennmt/bpcc-human/checkpoints
RESUME_ARG=""
mapfile -t CK < <(ls -1 "$CKPT_DIR"/model_step_*.pt 2>/dev/null | sort -V)
NCK=${#CK[@]}
if [ "${RESUME:-0}" = "1" ] && [ "$NCK" -ge 2 ]; then
    LAST="${CK[$((NCK-1))]}"; PREV="${CK[$((NCK-2))]}"
    if [ "$(stat -c %s "$LAST")" = "$(stat -c %s "$PREV")" ]; then PICK="$LAST"; else PICK="$PREV"; fi
    RESUME_ARG="--train_from $PICK"
    echo "Resuming from $PICK"
fi

CUDA_VISIBLE_DEVICES=0,1 onmt_train -config training/configs/pretrain_human.2gpu.yaml $RESUME_ARG
echo "Done ($(date))"
