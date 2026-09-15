#!/bin/bash
#SBATCH --job-name=job08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Job8: build the training, dev and test data for the 128k-spm experiment. First the raw
# line-aligned src/tgt files (training/build_parallel_corpus.py), then the 128k-SPM-encoded
# files (training/encode_spm.py), for all 5 datasets. CPU only.
# One array task per dataset. bpcc-mined-train (task 1) is about 72 GB and 213.8M lines, which
# is why this is parallelised at all. Both scripts skip work whose output already exists
# (idempotent re-run); pass FRESH=1 to force a rebuild.
# See training/README.md.

DATASETS=(bpcc-human-train bpcc-mined-train bpcc-human-dev in22gen-test in22conv-test)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

# Force a rebuild even if output already exists:
#   sbatch --export=ALL,FRESH=1 scripts/slurm/training_build_dataset.sh
FRESH_FLAG=""
if [ "${FRESH:-0}" = "1" ]; then
  FRESH_FLAG="--fresh"
fi

echo "Building 128k-spm dataset=$DATASET fresh=${FRESH:-0} ($(date))"
python training/build_parallel_corpus.py --dataset "$DATASET" --workers 32 $FRESH_FLAG
echo "Raw done, encoding with 128k SPM ($(date))"
python training/encode_spm.py --dataset "$DATASET" --workers 32 $FRESH_FLAG
echo "Done ($(date))"
