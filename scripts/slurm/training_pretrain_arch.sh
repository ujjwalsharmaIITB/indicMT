#!/bin/bash
#SBATCH --job-name=job12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --open-mode=append
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job12: architecture study on the current best recipe. Combined (8:2 mined:human) pre-training with a
# different encoder/decoder depth. Same as scripts/slurm/training_pretrain_combined.sh, except that
# the config and the run directory are parameters. Usage:
#   sbatch -p <partition> --export=ALL,CFG=pretrain_combined_12e18d,RUN=bpcc-combined-12e18d \
#          scripts/slurm/training_pretrain_arch.sh
#   sbatch -p <partition> --export=ALL,CFG=pretrain_combined_18e18d,RUN=bpcc-combined-18e18d \
#          scripts/slurm/training_pretrain_arch.sh
# Resume is opt-in with RESUME=1, as in the baseline script. See training/README.md.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

: "${CFG:?export CFG (config name without .4gpu.yaml)}"
: "${RUN:?export RUN (run directory under 128k-spm/opennmt/)}"
CKPT_DIR=128k-spm/opennmt/$RUN/checkpoints
mkdir -p "$CKPT_DIR"

echo "Starting $RUN with training/configs/$CFG.4gpu.yaml ($(date))"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

RESUME_ARG=""
mapfile -t CK < <(ls -1 "$CKPT_DIR"/model_step_*.pt 2>/dev/null | sort -V)
NCK=${#CK[@]}
if [ "${RESUME:-0}" = "1" ] && [ "$NCK" -ge 2 ]; then
    LAST="${CK[$((NCK-1))]}"; PREV="${CK[$((NCK-2))]}"
    if [ "$(stat -c %s "$LAST")" = "$(stat -c %s "$PREV")" ]; then PICK="$LAST"; else PICK="$PREV"; fi
    RESUME_ARG="--train_from $PICK"
    echo "Resuming from $PICK"
elif [ "$NCK" -gt 0 ]; then
    echo "FAILED: $CKPT_DIR already has checkpoints and RESUME is not set; refusing to start over them" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 onmt_train -config training/configs/$CFG.4gpu.yaml $RESUME_ARG
echo "Done ($(date))"
