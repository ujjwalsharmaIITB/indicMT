"""Evaluate AI4Bharat's IndicTrans2 1B models on IN22-Gen/IN22-Conv with this
project's own scoring, so the numbers sit next to evaluation/run_eval.py's.
This is the script behind the "IndicTrans2 under the same protocol" results in
docs/evaluation.md.

Models (Hugging Face, gated - token from data/common/auth.py):
  ai4bharat/indictrans2-en-indic-1B  for the 22 en->xx directions
  ai4bharat/indictrans2-indic-en-1B  for the 22 xx->en directions

Decoding follows the model cards: IndicProcessor (IndicTransToolkit) pre- and
post-processing, beam 5, max_length 256, float16. Input is the raw IN22 text
(128k-spm/data/raw/test/<set>/<set>.<tag>), not this project's SPM-encoded
sources. Scoring is identical to run_eval.py: sacrebleu corpus chrF++
(word_order=2) and corpus BLEU against the same untokenized references, and
the same per-bucket macro averages. These numbers are therefore comparable to
our own models', unlike the model-card baselines in baselines.py (sentence-level,
greedy).

Runs one bucket per process (each bucket uses a different model):
  python evaluation/run_eval_indictrans2.py --bucket en->xx --gpu 0
  python evaluation/run_eval_indictrans2.py --bucket xx->en --gpu 1
  python evaluation/run_eval_indictrans2.py --merge
Each bucket writes scores.<bucket>.json; --merge writes scores.json and
report.md in the same layout as run_eval.py. Results go to
evaluation/results/indictrans2-1B/ by default (--output-dir to change).
Directions that already have a hyp.txt are reused, so an interrupted run
resumes (--overwrite to redo them).

Extra requirements. Run this in its own environment, separate from the
OpenNMT-py one, because it needs older versions than the rest of the project
may use: transformers 4.46.3 and IndicTransToolkit 1.1.1, on top of torch,
sacrebleu and sentencepiece. A GPU is needed (float16). The models are gated:
accept their terms on Hugging Face and log in or set HF_TOKEN. If your compute
nodes have no internet, download both models to the Hugging Face cache first.
scripts/slurm/evaluation_indictrans2.sh runs both buckets on two GPUs and then
merges them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "evaluation"))
from common.languages import LANGUAGE_CODES, LANGUAGES  # noqa: E402
from run_eval import load_reference, score, write_markdown_report  # noqa: E402

RAW_TEST_DIR = ROOT / "128k-spm" / "data" / "raw" / "test"
DEFAULT_OUT_DIR = ROOT / "evaluation" / "results" / "indictrans2-1B"
TEST_SETS = ["in22-gen", "in22-conv"]
MODELS = {
    "en->xx": "ai4bharat/indictrans2-en-indic-1B",
    "xx->en": "ai4bharat/indictrans2-indic-en-1B",
}
EN_SHORT, EN_FLORES = "en", "eng_Latn"
# short tag (file suffix, e.g. "hi") -> FLORES code IndicTrans2 expects (e.g. "hin_Deva").
# IN22's columns use the same lang_Script codes as BPCC (data/common/languages.py).
SHORT_TO_FLORES = {LANGUAGE_CODES[code[:3]].strip("<>"): code for code in LANGUAGES}


def directions(bucket, only=None):
    for short in SHORT_TO_FLORES:
        src, tgt = (EN_SHORT, short) if bucket == "en->xx" else (short, EN_SHORT)
        if only is None or f"{src}-{tgt}" in only:
            yield src, tgt


def flores(short):
    return EN_FLORES if short == EN_SHORT else SHORT_TO_FLORES[short]


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class Translator:
    def __init__(self, repo, gpu, batch_size, beam_size, max_length):
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        sys.path.insert(0, str(ROOT / "data"))
        from common.auth import get_hf_token

        token = get_hf_token()
        self.torch = torch
        self.device = f"cuda:{gpu}"
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, token=token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=torch.float16, token=token
        ).to(self.device).eval()
        self.ip = IndicProcessor(inference=True)
        self.batch_size, self.beam_size, self.max_length = batch_size, beam_size, max_length

    def translate(self, sentences, src_lang, tgt_lang):
        # Longest first, so the first batch shows any out-of-memory at once and
        # padding within a batch is small; outputs are put back in input order.
        order = sorted(range(len(sentences)), key=lambda i: -len(sentences[i]))
        out = [None] * len(sentences)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            batch = self.ip.preprocess_batch([sentences[i] for i in idx], src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                                    return_attention_mask=True).to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, use_cache=True, min_length=0,
                                                max_length=self.max_length, num_beams=self.beam_size,
                                                num_return_sequences=1)
            # The tokenizer holds separate source and target vocabularies; its _decode switches to
            # the target one by itself (tokenization_indictrans.py).
            decoded = self.tokenizer.batch_decode(generated.detach().cpu().tolist(), skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
            for i, hyp in zip(idx, self.ip.postprocess_batch(decoded, lang=tgt_lang)):
                out[i] = hyp
        return out


def run_bucket(args, out_dir):
    only = set(args.directions) if args.directions else None
    tr = Translator(MODELS[args.bucket], args.gpu, args.batch_size, args.beam_size, args.max_length)
    results = {}
    for test_set in args.test_sets:
        res = {}
        for src, tgt in directions(args.bucket, only):
            hyp_file = out_dir / f"{test_set}.{src}-{tgt}.hyp.txt"
            refs = load_reference(test_set, tgt)
            if hyp_file.exists() and not args.overwrite:
                hyps = read_lines(hyp_file)  # resume: reuse finished directions
            else:
                t0 = time.time()
                print(f"[it2] {test_set} {src}->{tgt} ...", flush=True)
                srcs = read_lines(RAW_TEST_DIR / test_set / f"{test_set}.{src}")
                hyps = tr.translate(srcs, flores(src), flores(tgt))
                with open(hyp_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(hyps) + "\n")
                print(f"[it2]   {len(hyps)} sentences in {time.time() - t0:.0f}s", flush=True)
            if len(hyps) != len(refs):
                raise SystemExit(f"{test_set} {src}->{tgt}: hyp/ref line count mismatch ({len(hyps)} vs {len(refs)})")
            s = score(hyps, refs)
            s["bucket"] = args.bucket
            res[f"{src}-{tgt}"] = s
            print(f"[it2]   chrF++={s['chrf++']:.2f}  BLEU={s['bleu']:.2f}", flush=True)
        results[test_set] = res
    path = out_dir / f"scores.{args.bucket.replace('->', '-')}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[it2] wrote {path}")


def merge(out_dir):
    merged = {}
    for bucket in MODELS:
        part = json.loads((out_dir / f"scores.{bucket.replace('->', '-')}.json").read_text(encoding="utf-8"))
        for test_set, res in part.items():
            merged.setdefault(test_set, {}).update(res)
    for test_set, res in merged.items():
        per_direction = [v for k, v in res.items() if not k.startswith("MACRO")]
        for bucket in MODELS:
            b = [v for v in per_direction if v["bucket"] == bucket]
            if b:
                res[f"MACRO_{bucket}"] = {"chrf++": sum(v["chrf++"] for v in b) / len(b),
                                         "bleu": sum(v["bleu"] for v in b) / len(b), "n": len(b)}
    (out_dir / "scores.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report("indictrans2-1B", merged, out_dir)
    for test_set, res in merged.items():
        for bucket in MODELS:
            m = res.get(f"MACRO_{bucket}")
            if m:
                print(f"[it2] {test_set} {bucket}: chrF++ {m['chrf++']:.2f}  BLEU {m['bleu']:.2f}  (n={m['n']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", choices=list(MODELS))
    ap.add_argument("--merge", action="store_true", help="combine both buckets into scores.json + report.md")
    ap.add_argument("--test-sets", nargs="+", default=TEST_SETS, choices=TEST_SETS)
    ap.add_argument("--directions", nargs="+", default=None, help="restrict to 'src-tgt' pairs, e.g. en-hi")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32, help="sentences per batch")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--overwrite", action="store_true", help="re-translate directions whose hyp.txt exists")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(out_dir)
    elif args.bucket:
        run_bucket(args, out_dir)
    else:
        ap.error("pass --bucket or --merge")


if __name__ == "__main__":
    main()
