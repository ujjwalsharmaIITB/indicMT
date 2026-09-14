#!/bin/bash
#SBATCH --job-name=job07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job7: corpus laws. Zipf's law and Heaps' law over about 7M deduplicated sentences across
# 23 languages, with subsampling. CPU only. See tokenizer/analysis/README.md.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

python tokenizer/analysis/scripts/corpus_laws.py
