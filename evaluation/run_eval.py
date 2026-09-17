"""Evaluate a trained OpenNMT-py checkpoint on IN22-Gen/IN22-Conv,
all 44 directions (22 Indic languages x {en->xx, xx->en}), beam size 5 (default), chrF++ and BLEU via sacrebleu.

Requires evaluation/build_eval_pairs.py to have been run first (builds the
tag-prepended SPM source files this script feeds to onmt_translate).

Per direction: onmt_translate -> SPM-tokenized hypothesis -> detokenized with
the frozen 128k SPM model (sp.decode on the piece list, not a naive
whitespace join - unigram SentencePiece pieces don't always reassemble
correctly with a plain space join, e.g. across punctuation) -> scored with
sacrebleu against the untokenized IN22 reference
(128k-spm/data/raw/test/<test-set>/<test-set>.<short-tag>, never
SPM-tokenized or otherwise touched by our pipeline - exactly what
AI4Bharat published). Macro score = unweighted mean across the 22 directions
in that macro-direction bucket (en->xx or xx->en) - matches how the
bodhan-ai/indic-translate model card reports its own "English -> Indic" /
"Indic -> English" chrF++ numbers (see evaluation/baselines.py), which this
script's report is meant to sit next to.

chrF++ = sacrebleu's corpus_chrf with word_order=2 (the standard recipe that
turns plain chrF into chrF++ - char_order/beta stay at sacrebleu's chrF
defaults).

Output lands in a unified, arm-organized location, decoupled from the training
pipeline's own 128k-spm/opennmt/ tree - 128k-spm/opennmt/<arm>/checkpoints/model_step_N.pt
evaluates into evaluation/results/<arm>/model_step_N/ (override with
--output-dir; a checkpoint path that isn't under a checkpoints/ folder falls
back to evaluation/results/<run-name>/, no arm subfolder - nothing to derive
one from). Deliberately NOT beside the checkpoint itself (an earlier version
of this script did that) - a single evaluation/results/ home is more
discoverable for anyone who wants to see how models perform without needing
to know the training-pipeline layout, and it never duplicates or symlinks
the (large) checkpoint files themselves, only the evaluation artifacts they
produced. Not tracked by git (regenerable output):
  - <test-set>.<src>-<tgt>.hyp.spm   raw onmt_translate output (predictions), per direction
  - <test-set>.<src>-<tgt>.hyp.txt   detokenized plain-text predictions, per direction -
                                      saved before scoring, for later qualitative analysis
  - scores.json                      every direction's chrF++/BLEU + macros
  - report.md                        human-readable table + baseline comparison

Run via scripts/slurm/evaluation_run.sh (SLURM, 1 GPU) rather than on a login
node: translating 44 directions x however many test sets is real GPU-bound work.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import sacrebleu
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from common.languages import LANGUAGE_CODES, LANGUAGES  # noqa: E402
from baselines import INDIC_TRANSLATE_IN22_GEN_CHRF  # noqa: E402

SPM_MODEL_PATH = ROOT / "tokenizer" / "sweep" / "mixed" / "spm_128000.model"
EVAL_SRC_DIR = ROOT / "128k-spm" / "data" / "eval"
RAW_TEST_DIR = ROOT / "128k-spm" / "data" / "raw" / "test"

EN_SHORT = "en"
INDIC_SHORT_TAGS = [LANGUAGE_CODES[lang[:3]].strip("<>") for lang in LANGUAGES]
DEFAULT_TEST_SETS = ["in22-gen", "in22-conv"]


def directions_for_test_set(only=None):
    """Yield (src_short, tgt_short, macro_bucket) for all 44 directions, or a
    restricted subset when `only` is a set of "src-tgt" strings (e.g. for a
    quick spot-check instead of the full 44-direction run)."""
    for short in INDIC_SHORT_TAGS:
        for src, tgt, bucket in ((EN_SHORT, short, "en->xx"), (short, EN_SHORT, "xx->en")):
            if only is None or f"{src}-{tgt}" in only:
                yield src, tgt, bucket


def run_onmt_translate(model: Path, src_file: Path, out_file: Path, gpu: int, beam_size: int, batch_size: int):
    cmd = [
        "onmt_translate",
        "-gpu", str(gpu),
        "-batch_size", str(batch_size),
        "-batch_type", "tokens",
        "-beam_size", str(beam_size),
        "-model", str(model),
        "-src", str(src_file),
        "-output", str(out_file),
    ]
    subprocess.run(cmd, check=True)


def detokenize(sp: spm.SentencePieceProcessor, spm_file: Path) -> list:
    lines = []
    with open(spm_file, encoding="utf-8") as f:
        for line in f:
            pieces = [p for p in line.strip().split(" ") if p]
            lines.append(sp.decode(pieces) if pieces else "")
    return lines


def load_reference(test_set: str, short_tag: str) -> list:
    path = RAW_TEST_DIR / test_set / f"{test_set}.{short_tag}"
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def score(hyps: list, refs: list) -> dict:
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return {"chrf++": chrf.score, "bleu": bleu.score}


def evaluate_test_set(test_set: str, model_path: Path, sp, out_dir: Path, gpu: int, beam_size: int, batch_size: int, only=None) -> dict:
    direction_results = {}
    for src_short, tgt_short, bucket in directions_for_test_set(only):
        src_file = EVAL_SRC_DIR / test_set / f"{test_set}.128k.spm.{src_short}-{tgt_short}.src"
        if not src_file.exists():
            raise SystemExit(f"{src_file} missing - run evaluation/build_eval_pairs.py first")

        hyp_spm_file = out_dir / f"{test_set}.{src_short}-{tgt_short}.hyp.spm"
        print(f"[run_eval] {test_set} {src_short}->{tgt_short} ...", flush=True)
        run_onmt_translate(model_path, src_file, hyp_spm_file, gpu, beam_size, batch_size)

        hyps = detokenize(sp, hyp_spm_file)
        # Save the detokenized (plain-text) predictions alongside the raw SPM
        # output, before scoring - for later qualitative analysis (reading
        # actual translations) without having to re-run SPM decode on the
        # .hyp.spm files by hand.
        hyp_txt_file = out_dir / f"{test_set}.{src_short}-{tgt_short}.hyp.txt"
        with open(hyp_txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(hyps) + "\n")
        refs = load_reference(test_set, tgt_short)
        if len(hyps) != len(refs):
            raise SystemExit(
                f"{test_set} {src_short}->{tgt_short}: hyp/ref line count mismatch "
                f"({len(hyps)} vs {len(refs)})"
            )
        s = score(hyps, refs)
        s["bucket"] = bucket
        direction_results[f"{src_short}-{tgt_short}"] = s
        print(f"[run_eval]   chrF++={s['chrf++']:.2f}  BLEU={s['bleu']:.2f}", flush=True)

    # Snapshot the per-direction entries before inserting any MACRO_ key below -
    # direction_results.values() would otherwise include a bucket's own
    # just-inserted MACRO_ entry (which has no "bucket" key) once a second
    # bucket is processed, crashing with a KeyError.
    per_direction = [(k, v) for k, v in direction_results.items() if not k.startswith("MACRO")]
    for bucket in ("en->xx", "xx->en"):
        bucket_scores = [v for _, v in per_direction if v["bucket"] == bucket]
        if not bucket_scores:
            continue  # e.g. a --directions spot-check that only touched one bucket
        macro_chrf = sum(v["chrf++"] for v in bucket_scores) / len(bucket_scores)
        macro_bleu = sum(v["bleu"] for v in bucket_scores) / len(bucket_scores)
        direction_results[f"MACRO_{bucket}"] = {
            "chrf++": macro_chrf, "bleu": macro_bleu, "n": len(bucket_scores),
        }
    return direction_results


def write_markdown_report(run_name: str, all_results: dict, out_dir: Path):
    lines = [f"# Evaluation: {run_name}", ""]
    for test_set, direction_results in all_results.items():
        lines.append(f"## {test_set}")
        lines.append("")
        lines.append("| Direction | chrF++ | BLEU |")
        lines.append("|---|---|---|")
        for name, s in sorted(direction_results.items()):
            if name.startswith("MACRO"):
                continue
            lines.append(f"| {name} | {s['chrf++']:.2f} | {s['bleu']:.2f} |")
        lines.append("")
        for bucket in ("en->xx", "xx->en"):
            m = direction_results.get(f"MACRO_{bucket}")
            if m:
                lines.append(f"**Macro {bucket}**: chrF++ = {m['chrf++']:.2f}, "
                              f"BLEU = {m['bleu']:.2f} (n={m['n']})")
        if test_set == "in22-gen":
            lines.append("")
            lines.append("**Baseline comparison** (bodhan-ai/indic-translate model "
                          "card, IN22-Gen, chrF++ only - see evaluation/baselines.py):")
            lines.append("")
            lines.append("| Model | en->xx chrF++ | xx->en chrF++ |")
            lines.append("|---|---|---|")
            for model_name, (en_xx, xx_en) in INDIC_TRANSLATE_IN22_GEN_CHRF.items():
                lines.append(f"| {model_name} | {en_xx if en_xx is not None else '-'} | "
                              f"{xx_en if xx_en is not None else '-'} |")
            m_en = all_results[test_set].get("MACRO_en->xx", {}).get("chrf++")
            m_xx = all_results[test_set].get("MACRO_xx->en", {}).get("chrf++")
            en_str = f"{m_en:.2f}" if m_en is not None else "-"
            xx_str = f"{m_xx:.2f}" if m_xx is not None else "-"
            lines.append(f"| **{run_name} (this run)** | {en_str} | {xx_str} |")
        lines.append("")

    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[run_eval] wrote {report_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to a .pt OpenNMT-py checkpoint")
    ap.add_argument("--run-name", default=None,
                     help="Label for this run, e.g. bpcc-human-step2728 - default: the "
                          "checkpoint's own filename stem (e.g. model_step_1800)")
    ap.add_argument("--test-sets", nargs="+", default=DEFAULT_TEST_SETS, choices=DEFAULT_TEST_SETS)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1024,
                     help="tokens per batch (onmt_translate -batch_size). "
                          "**Real incident, not a theoretical caveat**: 4096 was tried first "
                          "(measured ~4.5GB on a single en->hi test run) and used to pack 6 "
                          "concurrent evaluations per GPU - two separate batch jobs then hit "
                          "real CUDA OOM crashes, with individual processes observed spiking to "
                          "15.96/17.89/19.77 GiB (nvidia-smi, from the actual crash messages), "
                          "far beyond the single-direction test run. Root cause: batch_size is in "
                          "TOKENS, but onmt/translate/decode_strategy.py's tile() (replicating "
                          "encoder state across the beam dimension at the start of decoding) "
                          "scales with SENTENCE COUNT in the batch x beam_size - a fixed token "
                          "budget packs far more sentences for short-sentence directions, so "
                          "peak memory varies hugely by direction, not just by batch_size. "
                          "Reverted to a more conservative 1024 with reduced concurrent packing "
                          "(see scripts/slurm/evaluation_batch_run.sh) until a real, multi-direction "
                          "(not single-direction) memory profile justifies going higher again.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output-dir", default=None,
                     help="default: evaluation/results/<arm>/<checkpoint-stem>/, e.g. a "
                          "checkpoint at 128k-spm/opennmt/bpcc-mined/checkpoints/"
                          "model_step_1800.pt writes to "
                          "evaluation/results/bpcc-ft-combined-human/model_step_1800/")
    ap.add_argument("--directions", nargs="+", default=None,
                     help="restrict to specific 'src-tgt' pairs (e.g. en-hi hi-en) "
                          "instead of all 44 - for a quick spot-check")
    args = ap.parse_args()
    only = set(args.directions) if args.directions else None

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise SystemExit(f"checkpoint not found: {model_path}")
    if not SPM_MODEL_PATH.exists():
        raise SystemExit(f"SPM model not found: {SPM_MODEL_PATH}")

    run_name = args.run_name or model_path.stem

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif model_path.parent.name == "checkpoints":
        # .../opennmt/<arm>/checkpoints/model_step_N.pt -> evaluation/results/<arm>/model_step_N/
        arm = model_path.parent.parent.name
        out_dir = ROOT / "evaluation" / "results" / arm / model_path.stem
    else:
        out_dir = ROOT / "evaluation" / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL_PATH))

    all_results = {}
    for test_set in args.test_sets:
        all_results[test_set] = evaluate_test_set(
            test_set, model_path, sp, out_dir, args.gpu, args.beam_size, args.batch_size, only
        )

    results_path = out_dir / "scores.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[run_eval] wrote {results_path}")

    write_markdown_report(run_name, all_results, out_dir)


if __name__ == "__main__":
    main()
