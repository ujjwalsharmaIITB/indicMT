#!/bin/bash
#SBATCH --job-name=job23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job23: containerised build-and-run check of the API image on a compute node, once rootless
# podman works on your cluster (it needs a subuid/subgid range for the user; see
# deployment/README.md, "Container build"). Not runnable on the development cluster so far.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

echo "Checking podman rootless build capability ($(date))"
TMP_BUILD="${TMPDIR:-/tmp}/podman-smoke-$SLURM_JOB_ID"
mkdir -p "$TMP_BUILD"
cat > "$TMP_BUILD/Dockerfile" <<'DOCKER'
FROM python:3.11-slim
CMD ["python3", "-c", "print('ok')"]
DOCKER
if ! podman build -t smoke-test:latest "$TMP_BUILD"; then
  echo "podman build failing - the subuid/subgid blocker is likely not fixed. See deployment/README.md."
  exit 1
fi
podman run --rm smoke-test:latest

echo "Trivial build OK - building the real image ($(date))"
python deployment/stage_common.py
podman build -f deployment/Dockerfile -t indicmt-deploy:latest deployment/
podman inspect --format='{{.Size}}' indicmt-deploy:latest

echo "Smoke-running the real image ($(date))"
podman run -d --name indicmt-smoke -p 8000:8000 indicmt-deploy:latest
sleep 5
curl -sf http://127.0.0.1:8000/api/health
curl -sf -X POST http://127.0.0.1:8000/api/translate -H "Content-Type: application/json" \
  -d '{"text":"Hello, how are you?","source_lang":"en","target_lang":"hi"}'
podman stop indicmt-smoke && podman rm indicmt-smoke

echo "Done ($(date))"
