"""CTranslate2-native counterpart to evaluation/run_eval.py. Runs
the full 44-direction IN22-Gen evaluation directly against a CTranslate2
model (ctranslate2.Translator.translate_batch), not onmt_translate - this is
literally the code path the deployed API will use, not a proxy for it.
Reuses evaluation/run_eval.py's score()/detokenize()/load_reference() and
evaluation/build_eval_pairs.py's already-built tagged SPM source files
directly (both only depend on sentencepiece+sacrebleu, not OpenNMT-py).

Also measures CPU latency (single-sentence) and throughput (batch of 32)
for the loaded variant, over a real sentence sample.

Run once per quantization variant (float32/int16/int8) via
scripts/slurm/deployment_convert_ct2.sh; the job script aggregates the three
variants' JSON outputs into one quantization_report.md comparison table.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import ctranslate2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
from run_eval import (  # noqa: E402
    EN_SHORT, INDIC_SHORT_TAGS, EVAL_SRC_DIR, RAW_TEST_DIR,
    detokenize, load_reference, score,
)

import sentencepiece as spm  # noqa: E402

SPM_MODEL_PATH = ROOT / "tokenizer" / "sweep" / "mixed" / "spm_128000.model"
TEST_SET = "in22-gen"


def directions_44():
    for short in INDIC_SHORT_TAGS:
        yield EN_SHORT, short, "en->xx"
        yield short, EN_SHORT, "xx->en"


def read_tagged_source(path: Path) -> list:
    """Each line is already SPM-tokenized with the target tag prepended as
    the first space-separated piece - exactly what translate_batch wants
    (List[List[str]]), no re-encoding needed."""
    with open(path, encoding="utf-8") as f:
        return [line.strip().split(" ") for line in f]


def evaluate_variant(translator: ctranslate2.Translator, sp, beam_size: int, batch_size: int,
                     hyp_dir: Path = None) -> dict:
    direction_results = {}
    for src_short, tgt_short, bucket in directions_44():
        src_file = EVAL_SRC_DIR / TEST_SET / f"{TEST_SET}.128k.spm.{src_short}-{tgt_short}.src"
        if not src_file.exists():
            raise SystemExit(f"{src_file} missing - run evaluation/build_eval_pairs.py first")
        sources = read_tagged_source(src_file)
        print(f"[ct2_eval] {src_short}->{tgt_short} ({len(sources)} lines) ...", flush=True)
        results = translator.translate_batch(
            sources, beam_size=beam_size, max_batch_size=batch_size, batch_type="tokens",
        )
        hyp_tokens_list = [r.hypotheses[0] for r in results]
        hyps = [sp.decode(toks) for toks in hyp_tokens_list]
        refs = load_reference(TEST_SET, tgt_short)
        if len(hyps) != len(refs):
            raise SystemExit(f"{src_short}->{tgt_short}: hyp/ref mismatch ({len(hyps)} vs {len(refs)})")
        if hyp_dir is not None:
            # detokenized hypotheses, one per line, for paired significance tests between variants
            with open(hyp_dir / f"{TEST_SET}.{src_short}-{tgt_short}.hyp.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(h.replace("\n", " ") for h in hyps) + "\n")
        s = score(hyps, refs)
        s["bucket"] = bucket
        direction_results[f"{src_short}-{tgt_short}"] = s
        print(f"[ct2_eval]   chrF++={s['chrf++']:.2f}  BLEU={s['bleu']:.2f}", flush=True)

    per_direction = [(k, v) for k, v in direction_results.items() if not k.startswith("MACRO")]
    for bucket in ("en->xx", "xx->en"):
        bucket_scores = [v for _, v in per_direction if v["bucket"] == bucket]
        if not bucket_scores:
            continue
        macro_chrf = sum(v["chrf++"] for v in bucket_scores) / len(bucket_scores)
        macro_bleu = sum(v["bleu"] for v in bucket_scores) / len(bucket_scores)
        direction_results[f"MACRO_{bucket}"] = {
            "chrf++": macro_chrf, "bleu": macro_bleu, "n": len(bucket_scores),
        }
    return direction_results


def measure_speed(translator: ctranslate2.Translator, sp, sample_texts: list, beam_size: int) -> dict:
    tagged = [["<en>"] + sp.encode(t, out_type=str) for t in sample_texts]

    # single-sentence latency
    latencies = []
    for src in tagged:
        t0 = time.time()
        translator.translate_batch([src], beam_size=beam_size, max_batch_size=1)
        latencies.append(time.time() - t0)
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies)) - 1] if len(latencies) > 1 else latencies[0]

    # batch-32 throughput
    batch = (tagged * ((32 // len(tagged)) + 1))[:32]
    t0 = time.time()
    translator.translate_batch(batch, beam_size=beam_size, max_batch_size=32, batch_type="examples")
    elapsed = time.time() - t0
    throughput = len(batch) / elapsed

    return {
        "latency_p50_ms": p50 * 1000, "latency_p95_ms": p95 * 1000,
        "batch32_throughput_sentences_per_sec": throughput,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--variant-name", required=True, help="label, e.g. float32/int16/int8")
    ap.add_argument("--output", required=True, help="path to write this variant's JSON report")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4096, help="tokens per batch")
    ap.add_argument("--inter-threads", type=int, default=4)
    ap.add_argument("--intra-threads", type=int, default=1)
    ap.add_argument("--hyp-dir", default=None,
                    help="also write each direction's detokenized hypotheses here (for significance tests)")
    ap.add_argument("--skip-speed", action="store_true", help="skip the latency/throughput measurement")
    args = ap.parse_args()
    hyp_dir = None
    if args.hyp_dir:
        hyp_dir = Path(args.hyp_dir)
        hyp_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL_PATH))
    translator = ctranslate2.Translator(
        args.model_dir, device="cpu",
        inter_threads=args.inter_threads, intra_threads=args.intra_threads,
    )

    t_load0 = time.time()
    eval_results = evaluate_variant(translator, sp, args.beam_size, args.batch_size, hyp_dir)
    eval_elapsed = time.time() - t_load0

    if args.skip_speed:
        speed = {"latency_p50_ms": float("nan"), "latency_p95_ms": float("nan"),
                 "batch32_throughput_sentences_per_sec": float("nan"), "skipped": True}
    else:
        sample = load_reference(TEST_SET, "en")[:50]
        speed = measure_speed(translator, sp, sample, args.beam_size)

    report = {
        "variant": args.variant_name,
        "in22-gen": eval_results,
        "speed": speed,
        "full_eval_wall_seconds": eval_elapsed,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[ct2_eval] wrote {out_path}")
    m_en = eval_results.get("MACRO_en->xx", {})
    m_xx = eval_results.get("MACRO_xx->en", {})
    print(f"[ct2_eval] {args.variant_name}: macro en->xx chrF++={m_en.get('chrf++', 0):.2f} "
          f"BLEU={m_en.get('bleu', 0):.2f} | macro xx->en chrF++={m_xx.get('chrf++', 0):.2f} "
          f"BLEU={m_xx.get('bleu', 0):.2f} | p50={speed['latency_p50_ms']:.0f}ms "
          f"throughput={speed['batch32_throughput_sentences_per_sec']:.1f} sent/s")


if __name__ == "__main__":
    main()
