#!/bin/bash
#SBATCH --job-name=job23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job23: builds the API container image, runs it and checks /api/health and /api/translate.
# Needs podman (rootless podman needs a subuid/subgid range for the user) and the staged model
# in deployment/model/ (see deployment/README.md, "Container build").

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

IMAGE="${IMAGE:-indicmt-deploy:latest}"
NAME="indicmt-check-${SLURM_JOB_ID:-$$}"

echo "Building $IMAGE ($(date))"
python deployment/stage_common.py
podman build -f deployment/Dockerfile -t "$IMAGE" deployment/ || exit 1
echo "Image size (bytes): $(podman inspect --format='{{.Size}}' "$IMAGE")"

echo "Starting the container ($(date))"
podman run -d --name "$NAME" -p 8000:8000 "$IMAGE" || exit 1
trap 'podman stop "$NAME" >/dev/null; podman rm "$NAME" >/dev/null' EXIT

for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/api/health > /dev/null 2>&1 && break
  sleep 1
done

curl -sf http://127.0.0.1:8000/api/health || { echo "health check failed"; exit 1; }
echo
curl -sf -X POST http://127.0.0.1:8000/api/translate -H "Content-Type: application/json" \
  -d '{"text":"Hello, how are you?","source_lang":"en","target_lang":"hi"}' || { echo "translate check failed"; exit 1; }
echo
echo "Container check passed ($(date))"
