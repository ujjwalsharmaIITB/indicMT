#!/bin/bash
#SBATCH --job-name=job01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job1: build the multilingual bpcc-human dataset (tag + bidirectional expansion).
# CPU only. Run from the repository root: sbatch -p <partition> scripts/slurm/data_human_multilingual_dataset.sh
# See data/README.md.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

python data/bpcc-human/create_multilingual_dataset.py
