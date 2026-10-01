"""Paired significance test: do the quantized CTranslate2 variants (int16, int8) translate
differently from float32?

Uses sacrebleu's paired bootstrap resampling (PairedTest, test_type="bs") on chrF++
(word_order=2, the project's primary metric), with float32 as the baseline system:

- per direction: each of the 44 IN22-Gen directions separately (1,024 sentences);
- pooled per bucket: all 22 en->xx directions concatenated into one test set, and the same for
  xx->en. The pooled score is corpus-level chrF++ over the concatenation, not the macro average
  of the per-direction scores, but it is a valid system-level paired test of the same outputs.

Reads the detokenized hypotheses written by `deployment/ct2_eval.py --hyp-dir`, and the
references through evaluation/run_eval.py's load_reference. Writes a JSON and a markdown report.

    python evaluation/quant_significance.py --hyp-root <ct2_root>/hyps --out <ct2_root>/significance
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

TEST_SET = "in22-gen"
VARIANTS = ["float32", "int16", "int8"]


def read(p):
    with open(p, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def paired(base, systems, refs, n):
    named = [("float32", base)] + list(systems.items())
    test = PairedTest(named, {"chrF++": CHRF(word_order=2)}, references=[refs],
                      test_type="bs", n_samples=n)
    _, scores = test()
    names = scores["System"]
    mkey = [k for k in scores if k != "System"][0]   # sacrebleu names the metric, e.g. "chrF2++"
    out = {}
    for name, res in zip(names, scores[mkey]):
        out[name] = {"score": res.score, "p_value": res.p_value,
                     "mean": getattr(res, "mean", None), "ci": getattr(res, "ci", None)}
    # sacrebleu's bootstrap is degenerate when a system's outputs equal the baseline's (every
    # resampled delta is 0 and it reports the minimum p, 1/(n+1)); report those as p = 1.
    for name, hyps in systems.items():
        n_diff = sum(1 for x, y in zip(base, hyps) if x != y)
        out[name]["n_sentences_differing"] = n_diff
        if n_diff == 0:
            out[name]["p_value"] = 1.0
            out[name]["identical_to_baseline"] = True
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hyp-root", required=True, help="dir with one subdir per variant")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    root, out = Path(a.hyp_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    per_dir, pooled = {}, {}
    pool = {b: {"refs": [], **{v: [] for v in VARIANTS}} for b in ("en->xx", "xx->en")}
    for short in INDIC_SHORT_TAGS:
        for src, tgt, bucket in ((EN_SHORT, short, "en->xx"), (short, EN_SHORT, "xx->en")):
            refs = load_reference(TEST_SET, tgt)
            hyps = {v: read(root / v / f"{TEST_SET}.{src}-{tgt}.hyp.txt") for v in VARIANTS}
            for v, h in hyps.items():
                if len(h) != len(refs):
                    raise SystemExit(f"{v} {src}-{tgt}: {len(h)} hyps vs {len(refs)} refs")
            key = f"{src}-{tgt}"
            per_dir[key] = {"bucket": bucket,
                            **paired(hyps["float32"], {v: hyps[v] for v in VARIANTS[1:]}, refs, a.n_samples)}
            pool[bucket]["refs"] += refs
            for v in VARIANTS:
                pool[bucket][v] += hyps[v]
            print(key, {v: round(per_dir[key][v]["score"], 2) for v in VARIANTS},
                  {v: per_dir[key][v]["p_value"] for v in VARIANTS[1:]}, flush=True)
    for bucket, d in pool.items():
        pooled[bucket] = paired(d["float32"], {v: d[v] for v in VARIANTS[1:]}, d["refs"], a.n_samples)

    summary = {}
    for bucket in ("en->xx", "xx->en"):
        rows = [v for v in per_dir.values() if v["bucket"] == bucket]
        summary[bucket] = {}
        for v in VARIANTS[1:]:
            deltas = [r[v]["score"] - r["float32"]["score"] for r in rows]
            summary[bucket][v] = {
                "n_directions": len(rows),
                "n_significant_p_lt_alpha": sum(1 for r in rows if r[v]["p_value"] is not None and r[v]["p_value"] < a.alpha),
                "n_significant_bonferroni": sum(1 for r in rows if r[v]["p_value"] is not None and r[v]["p_value"] < a.alpha / len(rows)),
                "macro_delta_vs_float32": sum(deltas) / len(deltas),
                "min_delta": min(deltas), "max_delta": max(deltas),
                "pooled_score": pooled[bucket][v]["score"],
                "pooled_float32_score": pooled[bucket]["float32"]["score"],
                "pooled_p_value": pooled[bucket][v]["p_value"],
                "pooled_sentences_differing": pooled[bucket][v]["n_sentences_differing"],
                "pooled_sentences": len(pool[bucket]["refs"]),
            }

    json.dump({"test": "sacrebleu paired bootstrap resampling, chrF++ (word_order=2)",
               "n_samples": a.n_samples, "alpha": a.alpha, "baseline": "float32",
               "summary": summary, "pooled": pooled, "per_direction": per_dir},
              open(out / "significance.json", "w"), indent=2, default=str)

    L = ["# Quantization significance test (IN22-Gen, chrF++, paired bootstrap)", "",
         f"sacrebleu PairedTest, test_type=bs, {a.n_samples} resamples, baseline float32, alpha {a.alpha}. "
         "Per-direction tests use 1,024 sentences; pooled tests concatenate the 22 directions of a bucket "
         "(corpus-level chrF++ over the concatenation, not the macro average).", "",
         "| Bucket | Variant | Pooled chrF++ (float32 -> variant) | Pooled p | Sentences differing from float32 | Macro delta | Delta range | Directions p<alpha | p<alpha/22 (Bonferroni) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for bucket, vs in summary.items():
        for v, s in vs.items():
            L.append(f"| {bucket} | {v} | {s['pooled_float32_score']:.2f} -> {s['pooled_score']:.2f} | "
                     f"{s['pooled_p_value']} | {s['pooled_sentences_differing']}/{s['pooled_sentences']} | {s['macro_delta_vs_float32']:+.3f} | "
                     f"{s['min_delta']:+.2f} to {s['max_delta']:+.2f} | "
                     f"{s['n_significant_p_lt_alpha']}/{s['n_directions']} | {s['n_significant_bonferroni']}/{s['n_directions']} |")
    (out / "significance.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
