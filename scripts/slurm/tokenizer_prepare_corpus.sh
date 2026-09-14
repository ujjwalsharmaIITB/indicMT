#!/bin/bash
#SBATCH --job-name=job05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job5: build the SentencePiece training corpus (human data plus a mined top-up per language).
# CPU only. See tokenizer/README.md.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

python tokenizer/prepare_corpus.py
