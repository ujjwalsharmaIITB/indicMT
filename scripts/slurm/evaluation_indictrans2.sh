#!/bin/bash
#SBATCH --job-name=job24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job24: IndicTrans2 1B baseline on IN22-Gen and IN22-Conv, scored exactly like our own
# checkpoints (evaluation/run_eval_indictrans2.py). GPU 0 runs en->xx
# (indictrans2-en-indic-1B), GPU 1 runs xx->en (indictrans2-indic-en-1B), then both are merged
# into scores.json and report.md in evaluation/results/indictrans2-1B/.
# Usage: sbatch -p <partition> scripts/slurm/evaluation_indictrans2.sh [extra run_eval_indictrans2.py args]
#   e.g. two directions only: sbatch scripts/slurm/evaluation_indictrans2.sh --test-sets in22-gen \
#          --directions en-hi hi-en --output-dir evaluation/results/indictrans2-1B-en-hi
#
# Needs its own environment (see the docstring of evaluation/run_eval_indictrans2.py): the
# project's python and torch plus transformers 4.46.3 and IndicTransToolkit 1.1.1, kept apart from
# the OpenNMT-py environment. The raw IN22 files (Job8 output) must exist. Both models are gated,
# so download them to the Hugging Face cache beforehand (with HF_TOKEN set); compute nodes often
# have no internet, hence HF_HUB_OFFLINE=1 below.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${INDICTRANS2_ENV:-indictrans2}"  # adapt to your environment

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

mkdir -p logs
echo "Start ($(date)) on $(hostname); args: $*"

python evaluation/run_eval_indictrans2.py --bucket "en->xx" --gpu 0 --batch-size 64 "$@" \
    > "logs/indictrans2-${SLURM_JOB_ID}-en-xx.out" 2>&1 &
p1=$!
python evaluation/run_eval_indictrans2.py --bucket "xx->en" --gpu 1 --batch-size 64 "$@" \
    > "logs/indictrans2-${SLURM_JOB_ID}-xx-en.out" 2>&1 &
p2=$!

fail=0
wait "$p1" || { echo "en->xx FAILED - see logs/indictrans2-${SLURM_JOB_ID}-en-xx.out"; fail=1; }
wait "$p2" || { echo "xx->en FAILED - see logs/indictrans2-${SLURM_JOB_ID}-xx-en.out"; fail=1; }
[ "$fail" -eq 1 ] && exit 1

# --merge only needs the output directory, so pass that through if it was given.
merge_args=()
args=("$@")
for (( i = 0; i < ${#args[@]}; i++ )); do
    [ "${args[$i]}" = "--output-dir" ] && merge_args=(--output-dir "${args[$((i + 1))]}")
done
python evaluation/run_eval_indictrans2.py --merge "${merge_args[@]}"
echo "Done ($(date))"
