#!/bin/bash
#SBATCH --job-name=job18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Job18: checksum the downloaded judge-model shards against the Hub's sha256, so a truncated or
# corrupted download is caught before a GPU job depends on it. CPU only, reads about 62 GB.
# The expected checksums are fetched beforehand (on a machine with internet) into
# logs/judge-model-expected-sha256.txt, one line per file: "<sha256> <size> <file>".
# Usage: sbatch -p <partition> --export=ALL,JUDGE_MODEL_DIR=/path/to/gemma-4-31B-it scripts/slurm/llm_judge_verify_model.sh

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
: "${JUDGE_MODEL_DIR:?set JUDGE_MODEL_DIR}"
M="$JUDGE_MODEL_DIR"
while read want size name; do
    f="$M/$name"
    echo "$(date) hashing $name (size $(stat -c %s "$f"), expected $size)"
    got=$(sha256sum "$f" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then echo "OK       $name"; else echo "MISMATCH $name want=$want got=$got"; fi
done < logs/judge-model-expected-sha256.txt
echo "Done ($(date))"
