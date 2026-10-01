#!/bin/bash
#SBATCH --job-name=job17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Job17: CPU-only download of a Hugging Face model for the LLM judge into ${JUDGE_MODEL_DIR}.
# Usage: sbatch -p <partition> --export=ALL,REPO=google/gemma-4-31B-it,JUDGE_MODEL_DIR=/path/to/gemma-4-31B-it \
#                scripts/slurm/llm_judge_download_model.sh
# Only weights and tokenizer files are fetched (safetensors, no duplicate formats).
# If your compute nodes have no internet, run the download on a login node instead.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment
: "${REPO:?set REPO}"
: "${JUDGE_MODEL_DIR:?set JUDGE_MODEL_DIR}"
DEST="$JUDGE_MODEL_DIR"
echo "Downloading $REPO to $DEST ($(date))"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$REPO", local_dir="$DEST", allow_patterns=["*.json","*.safetensors","*.jinja","*.model","*.txt","*.py","tokenizer*"], max_workers=8)
print("done", p)
PY
du -sh "$DEST"
echo "Done ($(date))"
