"""Paired significance test between two systems' IN22 outputs (chrF++, paired bootstrap).

Same method as evaluation/quant_significance.py - sacrebleu's PairedTest (test_type="bs") on chrF++
(word_order=2), per direction and pooled per bucket (all 22 directions of a bucket concatenated;
corpus-level chrF++ over the concatenation) - but for any two systems whose detokenized hypotheses
follow run_eval.py's naming, <dir>/<test-set>.<src>-<tgt>.hyp.txt.

    python evaluation/paired_significance.py \
        --baseline other=evaluation/results/<other-system> \
        --system indicmt=evaluation/results/bpcc-ft-combined-human/model_step_1800 \
        --test-set in22-gen --buckets en->xx xx->en --out evaluation/results/significance/indicmt-vs-other
"""
import argparse
import json
import sys
from pathlib import Path

from sacrebleu.metrics import CHRF
from sacrebleu.significance import PairedTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
from run_eval import EN_SHORT, INDIC_SHORT_TAGS, load_reference  # noqa: E402


def read(p):
    with open(p, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def paired(base_name, base, sys_name, hyps, refs, n):
    test = PairedTest([(base_name, base), (sys_name, hyps)], {"chrF++": CHRF(word_order=2)},
                      references=[refs], test_type="bs", n_samples=n)
    _, scores = test()
    mkey = [k for k in scores if k != "System"][0]
    out = {name: {"score": r.score, "p_value": r.p_value} for name, r in zip(scores["System"], scores[mkey])}
    out["n_sentences_differing"] = sum(1 for x, y in zip(base, hyps) if x != y)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="name=dir")
    ap.add_argument("--system", required=True, help="name=dir")
    ap.add_argument("--test-set", default="in22-gen")
    ap.add_argument("--buckets", nargs="+", default=["en->xx", "xx->en"])
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bname, bdir = a.baseline.split("=", 1)
    sname, sdir = a.system.split("=", 1)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {"baseline": a.baseline, "system": a.system, "test_set": a.test_set, "n_samples": a.n_samples,
              "alpha": a.alpha, "method": "sacrebleu PairedTest, test_type=bs, chrF++ (word_order=2)", "buckets": {}}
    for bucket in a.buckets:
        pairs = [(EN_SHORT, x) if bucket == "en->xx" else (x, EN_SHORT) for x in INDIC_SHORT_TAGS]
        per_dir, pool = {}, {"b": [], "s": [], "r": []}
        for src, tgt in pairs:
            f = f"{a.test_set}.{src}-{tgt}.hyp.txt"
            b, s, r = read(Path(bdir) / f), read(Path(sdir) / f), load_reference(a.test_set, tgt)
            if not (len(b) == len(s) == len(r)):
                raise SystemExit(f"{f}: line counts differ ({len(b)}, {len(s)}, {len(r)})")
            per_dir[f"{src}-{tgt}"] = paired(bname, b, sname, s, r, a.n_samples)
            pool["b"] += b
            pool["s"] += s
            pool["r"] += r
        pooled = paired(bname, pool["b"], sname, pool["s"], pool["r"], a.n_samples)
        deltas = [d[sname]["score"] - d[bname]["score"] for d in per_dir.values()]
        result["buckets"][bucket] = {
            "pooled": pooled,
            "macro_baseline": sum(d[bname]["score"] for d in per_dir.values()) / len(per_dir),
            "macro_system": sum(d[sname]["score"] for d in per_dir.values()) / len(per_dir),
            "directions_system_better_p_lt_alpha": sum(1 for d in per_dir.values() if d[sname]["p_value"] < a.alpha and d[sname]["score"] > d[bname]["score"]),
            "directions_baseline_better_p_lt_alpha": sum(1 for d in per_dir.values() if d[sname]["p_value"] < a.alpha and d[sname]["score"] < d[bname]["score"]),
            "directions_p_lt_bonferroni": sum(1 for d in per_dir.values() if d[sname]["p_value"] < a.alpha / len(per_dir)),
            "delta_range": [min(deltas), max(deltas)],
            "per_direction": per_dir,
        }
    (out_dir / "significance.json").write_text(json.dumps(result, indent=1))
    L = [f"# Paired significance: {sname} vs {bname} ({a.test_set}, chrF++)", "",
         f"{result['method']}, {a.n_samples} resamples, baseline {bname}, alpha {a.alpha}. Pooled = all 22 directions of a bucket "
         "concatenated (corpus-level chrF++ over the concatenation).", "",
         f"| Bucket | Macro {bname} | Macro {sname} | Pooled {bname} -> {sname} | Pooled p | Directions {sname} better (p<a) | "
         f"Directions {bname} better (p<a) | p<a/22 | Delta range |", "|---|---|---|---|---|---|---|---|---|"]
    for bucket, r in result["buckets"].items():
        p = r["pooled"]
        L.append(f"| {bucket} | {r['macro_baseline']:.2f} | {r['macro_system']:.2f} | {p[bname]['score']:.2f} -> {p[sname]['score']:.2f} | "
                 f"{p[sname]['p_value']} | {r['directions_system_better_p_lt_alpha']}/22 | {r['directions_baseline_better_p_lt_alpha']}/22 | "
                 f"{r['directions_p_lt_bonferroni']}/22 | {r['delta_range'][0]:+.2f} to {r['delta_range'][1]:+.2f} |")
    (out_dir / "significance.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
