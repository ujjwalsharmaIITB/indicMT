#!/bin/bash
#SBATCH --job-name=job15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job15: evaluate MANY checkpoints on ONE 2-GPU allocation (see evaluation/README.md).
#
# Packing: PER_GPU (default 3) concurrent evaluations per GPU, so 6 "lanes" per job. Each lane
# works through its share of the checkpoint list in turn, so any number of checkpoints can be
# passed without exceeding 3 per GPU.
#
# CPU: packed jobs given the scheduler default of 2 CPUs for 6 processes were CPU-bound (about
# 1.9 CPUs busy for the whole job). 24 CPUs = 4 per process; OMP/MKL threads are pinned to 4 so
# processes do not oversubscribe each other.
#
# Batch size: EVAL_BATCH_SIZE (default 2048 tokens). Peak memory is set by sentence count x beam,
# so it spikes on short-sentence directions: 4096 tokens x 6 processes per GPU ran out of memory
# (single processes seen at up to 19.77 GiB). 2048 x 3 per GPU keeps the worst case near
# 30 GiB of a 48 GB card (inferred, not measured), so a checkpoint that fails is retried once at
# 1024, and per-process GPU memory is logged to logs/gpumem-<jobid>.csv.
#
# Usage: sbatch -p <partition> scripts/slurm/evaluation_batch_run.sh <ckpt1.pt> <run-name1> [<ckpt2.pt> <run-name2> ...]
# Run names must be unique within a job (they name the per-checkpoint logs).
# Exits non-zero if any checkpoint failed at every batch size tried.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p logs

if [ "$#" -eq 0 ] || [ $(( $# % 2 )) -ne 0 ]; then
    echo "Usage: sbatch scripts/slurm/evaluation_batch_run.sh <ckpt1.pt> <run-name1> [<ckpt2.pt> <run-name2> ...]"
    exit 1
fi

BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
PER_GPU="${EVAL_PER_GPU:-3}"
N_GPUS=2
LANES=$(( N_GPUS * PER_GPU ))

SIZES=("$BATCH_SIZE")
[ "$BATCH_SIZE" -gt 1024 ] && SIZES+=(1024)

CKPTS=(); NAMES=()
while [ "$#" -gt 0 ]; do
    CKPTS+=("$1"); NAMES+=("$2"); shift 2
done
N=${#CKPTS[@]}
echo "$N checkpoints, $LANES lanes ($PER_GPU per GPU x $N_GPUS GPUs), batch sizes tried: ${SIZES[*]} ($(date))"

(
    while true; do
        echo "t=$(date +%s)"
        nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader
        sleep 20
    done
) > "logs/gpumem-${SLURM_JOB_ID}.csv" 2>&1 &
SAMPLER=$!

run_lane() {
    local lane="$1" gpu="$2" rc=0 i bs ok
    for (( i = lane; i < N; i += LANES )); do
        ok=0
        for bs in "${SIZES[@]}"; do
            echo "[lane $lane gpu $gpu] ${NAMES[$i]} batch_size=$bs start ($(date))"
            if python evaluation/run_eval.py --model "${CKPTS[$i]}" --run-name "${NAMES[$i]}" \
                    --beam-size 5 --batch-size "$bs" --gpu "$gpu" \
                    > "logs/batch-${SLURM_JOB_ID}-${NAMES[$i]}-bs${bs}.out" \
                    2> "logs/batch-${SLURM_JOB_ID}-${NAMES[$i]}-bs${bs}.err"; then
                ok=1; break
            fi
            echo "[lane $lane gpu $gpu] ${NAMES[$i]} FAILED at batch_size=$bs ($(date))"
        done
        [ "$ok" -eq 1 ] && echo "[lane $lane gpu $gpu] ${NAMES[$i]} done ($(date))" || rc=1
    done
    return $rc
}

pids=()
for (( l = 0; l < LANES && l < N; l++ )); do
    run_lane "$l" $(( l % N_GPUS )) &
    pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
kill "$SAMPLER" 2>/dev/null

if [ "$fail" -eq 1 ]; then
    echo "One or more checkpoints FAILED at every batch size - see logs/batch-${SLURM_JOB_ID}-*.err"
    exit 1
fi
echo "All $N evaluations in this job done ($(date))"
