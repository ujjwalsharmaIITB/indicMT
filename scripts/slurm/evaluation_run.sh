#!/bin/bash
#SBATCH --job-name=job14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job14: evaluate one checkpoint on IN22-Gen and IN22-Conv (44 directions, beam 5,
# chrF++ and BLEU with sacrebleu). See evaluation/README.md and evaluation/run_eval.py.
# Usage: sbatch -p <partition> scripts/slurm/evaluation_run.sh <checkpoint.pt> <run-name> [extra run_eval.py args...]
#
# evaluation/build_eval_pairs.py must have been run once beforehand. It builds the 44
# tag-prepended SPM source files, shared by every checkpoint, so it is not re-run here.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

mkdir -p logs
MODEL="$1"
RUN_NAME="$2"
shift 2 2>/dev/null
if [ -z "$MODEL" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: sbatch scripts/slurm/evaluation_run.sh <checkpoint.pt> <run-name> [extra args...]"
    exit 1
fi

echo "Starting evaluation: model=$MODEL run_name=$RUN_NAME ($(date))"
python evaluation/run_eval.py --model "$MODEL" --run-name "$RUN_NAME" --beam-size 5 --gpu 0 "$@"
echo "Done ($(date))"
