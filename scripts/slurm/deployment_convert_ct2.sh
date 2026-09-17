#!/bin/bash
#SBATCH --job-name=job20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job20: convert a checkpoint to CTranslate2 in the three CPU-usable variants (float32, int16,
# int8; see deployment/convert_to_ct2.py for why float16 and int8_float16 are excluded), then run
# the full 44-direction IN22-Gen evaluation against EACH variant directly through CTranslate2
# (deployment/ct2_eval.py, the actual served inference path, not onmt_translate). Produces macro
# chrF++/BLEU plus CPU latency and throughput per variant. CPU only (no GPU requested).
# Defaults to the current best model. Override with CHECKPOINT, CT2_ROOT and LABEL, e.g.
#   sbatch -p <partition> --export=ALL,CHECKPOINT=path/to/model_step_N.pt,CT2_ROOT=path/to/ct2/model_step_N \
#          scripts/slurm/deployment_convert_ct2.sh
# See deployment/README.md.

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${CONDA_ENV:-indicmt}"  # adapt to your environment

# set -e only from here, not above: environment-activation commands can return non-zero on
# benign conditions. Without set -e a crashing converter or ct2_eval.py call inside the loop
# below would not stop the script, and it would write an empty report and exit 0.
set -euo pipefail

mkdir -p logs

CHECKPOINT="${CHECKPOINT:-128k-spm/opennmt/bpcc-ft-combined-human/checkpoints/model_step_1800.pt}"
CT2_ROOT="${CT2_ROOT:-128k-spm/opennmt/bpcc-ft-combined-human/ct2/model_step_1800}"
LABEL="${LABEL:-current best model, bpcc-ft-combined-human step 1800}"
# Record the CPU the latency numbers were measured on.
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)" || true
echo "host: $(hostname), threads: inter 1 / intra 8, beam 5"

echo "Starting CT2 conversion ($(date))"
python deployment/convert_to_ct2.py --checkpoint "$CHECKPOINT" --output-root "$CT2_ROOT"
echo "Conversion done ($(date))"

for variant in float32 int16 int8; do
  echo "Evaluating $variant on IN22-Gen ($(date))"
  python deployment/ct2_eval.py \
    --model-dir "$CT2_ROOT/$variant" \
    --variant-name "$variant" \
    --output "$CT2_ROOT/eval_${variant}.json" \
    --beam-size 5 --inter-threads 1 --intra-threads 8
done

echo "Building comparison report ($(date))"
python - "$CT2_ROOT" "$LABEL" <<'PYEOF'
import json, sys
from pathlib import Path

ct2_root = Path(sys.argv[1])
rows = []
missing = []
for variant in ("float32", "int16", "int8"):
    p = ct2_root / f"eval_{variant}.json"
    if not p.exists():
        missing.append(variant)
        continue
    d = json.loads(p.read_text())
    en_xx = d["in22-gen"].get("MACRO_en->xx", {})
    xx_en = d["in22-gen"].get("MACRO_xx->en", {})
    rows.append((variant, en_xx.get("chrf++"), en_xx.get("bleu"),
                 xx_en.get("chrf++"), xx_en.get("bleu"),
                 d["speed"]["latency_p50_ms"], d["speed"]["latency_p95_ms"],
                 d["speed"]["batch32_throughput_sentences_per_sec"]))

if missing:
    print(f"ERROR: missing eval output for variant(s): {missing} - not writing a "
          f"partial/misleading report", file=sys.stderr)
    sys.exit(1)

lines = [f"# CTranslate2 quantization comparison ({sys.argv[2]}, IN22-Gen)", "",
         "| Variant | en->xx chrF++ | en->xx BLEU | xx->en chrF++ | xx->en BLEU | "
         "p50 latency (ms) | p95 latency (ms) | batch-32 throughput (sent/s) |",
         "|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | "
                  f"{r[5]:.1f} | {r[6]:.1f} | {r[7]:.1f} |")

out = ct2_root / "quantization_report.md"
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
print("\n".join(lines))
PYEOF

echo "Done ($(date))"
