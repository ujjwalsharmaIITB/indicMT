#!/bin/bash
#SBATCH --job-name=job06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job6: train the 7 SentencePiece models (vocabulary sizes 16k to 256k). CPU only: unigram
# training is memory-bound, not GPU work. Use a high-memory node.
#
# All 7 sizes run one after the other in a single job instead of a Slurm array. An array of 7
# can exceed a per-user submit limit, and one job avoids that. Smallest sizes first, so partial
# progress is visible early. train_spm.py overwrites its own output; if the job dies partway,
# re-run the remaining sizes by hand.
# See tokenizer/README.md.

VOCAB_SIZES=(16000 32000 64000 96000 128000 192000 256000)

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

for VOCAB_SIZE in "${VOCAB_SIZES[@]}"; do
  echo "=== Training SPM vocab_size=$VOCAB_SIZE ($(date)) ==="
  python tokenizer/train_spm.py --vocab-size "$VOCAB_SIZE"
done

echo "=== SPM sweep complete ($(date)) ==="
