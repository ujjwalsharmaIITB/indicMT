#!/bin/bash
#SBATCH --job-name=job21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job21: is the int8/int16 quality difference from float32 real or noise? CPU only.
# 1) Re-translate IN22-Gen (44 directions) with each CTranslate2 variant of one checkpoint, saving
#    the detokenized hypotheses (deployment/ct2_eval.py --hyp-dir). The three variants run in
#    parallel, 8 intra-threads each (the same thread setting as the latency run); timing is skipped.
# 2) sacrebleu paired bootstrap on chrF++, float32 as the baseline (evaluation/quant_significance.py).
# Defaults to the current best model. Override with CT2_ROOT. See deployment/README.md.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment
set -euo pipefail

CT2_ROOT="${CT2_ROOT:-128k-spm/opennmt/bpcc-ft-combined-human/ct2/model_step_1800}"
mkdir -p logs "$CT2_ROOT/hyps"
echo "host $(hostname), CT2_ROOT $CT2_ROOT ($(date))"
pids=()
for v in float32 int16 int8; do
  python deployment/ct2_eval.py --model-dir "$CT2_ROOT/$v" --variant-name "$v" \
    --output "$CT2_ROOT/hyps/eval_${v}_rerun.json" --hyp-dir "$CT2_ROOT/hyps/$v" \
    --beam-size 5 --inter-threads 1 --intra-threads 8 --skip-speed \
    > "logs/quant-significance-${SLURM_JOB_ID}-$v.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
echo "translations done ($(date))"
python evaluation/quant_significance.py --hyp-root "$CT2_ROOT/hyps" --out "$CT2_ROOT/significance" --n-samples 1000
echo "Done ($(date))"
