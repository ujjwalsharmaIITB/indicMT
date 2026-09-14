#!/bin/bash
#SBATCH --job-name=job04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Job4: temperature-sampled mixtures. CPU only. Array task 0 = human (train-gold),
# task 1 = mined (mined-mixture). The two tasks are independent, so they run in parallel.
# build_mixture.py's --workers flag caps concurrency within a task. See mixing/README.md.
#
# Note: every array task counts as one submission against a per-user submit limit, if your
# cluster has one. Keep the array small.

SOURCES=(human mined)
SOURCE=${SOURCES[$SLURM_ARRAY_TASK_ID]}

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

echo "Building mixture for source=$SOURCE ($(date))"
python mixing/build_mixture.py --source "$SOURCE" --workers 16
echo "Done ($(date))"
