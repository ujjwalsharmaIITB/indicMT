#!/bin/bash
#SBATCH --job-name=job02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job2: build the multilingual bpcc-mined dataset (script filter, global dedup, tag + bidirectional
# expansion). CPU only (no GPU requested); the raw input is over 100M rows.
# See data/README.md.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

python data/bpcc-mined/create_multilingual_dataset.py --workers 16
