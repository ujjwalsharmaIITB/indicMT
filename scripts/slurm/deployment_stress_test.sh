#!/bin/bash
#SBATCH --job-name=job22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job22: CPU stress test of the live API. Starts uvicorn on the allocated node, waits for
# /api/health, runs deployment/stress_test.py (asyncio + httpx; p50/p95/p99 and requests per
# second across concurrency levels, payload sizes and direction classes), then stops the server.
# CPU only. The staged model must exist in deployment/model/ (see deployment/README.md).
# Status: written, not yet run (see docs/deployment.md).

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

mkdir -p logs

export MODEL_DIR="${MODEL_DIR:-deployment/model/ct2_model}"
export SPM_PATH="${SPM_PATH:-deployment/model/spm_128000.model}"
export CHECKPOINT_NAME="${CHECKPOINT_NAME:-indicmt-final-int8}"
export QUANTIZATION="int8"
export INTER_THREADS=4
export INTRA_THREADS=1
export BEAM_SIZE=2

echo "Starting API server ($(date))"
uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 > logs/api-under-test.out 2>&1 &
API_PID=$!

echo "Waiting for /api/health ..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
    echo "API ready after ${i}s"
    break
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
  echo "API never became healthy - aborting stress test"
  kill "$API_PID" 2>/dev/null
  exit 1
fi

echo "Running stress test ($(date))"
STRESS_EXIT=0
python deployment/stress_test.py --base-url http://127.0.0.1:8000 \
  --output logs/stress_test_results.json || STRESS_EXIT=$?

echo "Stopping API server"
kill "$API_PID" 2>/dev/null
wait "$API_PID" 2>/dev/null

# Explicit success check: a crashing stress_test.py must not look like a completed run.
if [ "$STRESS_EXIT" -ne 0 ] || [ ! -s logs/stress_test_results.json ]; then
  echo "STRESS TEST FAILED (exit=$STRESS_EXIT, results file missing/empty)"
  exit 1
fi

echo "Done ($(date))"
