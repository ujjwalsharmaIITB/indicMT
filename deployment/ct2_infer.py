"""Shared CTranslate2 inference module - the one code path for tag + SPM-encode ->
translate_batch -> SPM-decode, used three ways so it never
drifts into three near-duplicate implementations: as a CLI smoke-test tool
(this file, run directly), imported by deployment/ct2_eval.py (the
quantization/quality sweep), and imported by deployment/api/translator.py
(the live API).

Tag scheme matches training exactly (see docs/data.md, "Language tags"): the
target-language tag and the raw SOURCE sentence are encoded together, as one
string ("<hi> some english text" -> ['▁', '<hi>', ...] -> Hindi output), by
data/common/tokenization.py's encode_sources, the same function the
evaluation uses. xx->xx pivots through English at
the caller level (translate_pivot here, or the API's own two-call routing) -
this module only ever does one direction per call.
"""

import argparse
import sys
from pathlib import Path

import ctranslate2
import sentencepiece as spm

# data/common/tokenization.py, the source encoding shared with training. In the Docker image (and
# after deployment/stage_common.py in the repo) it is the generated copy in api/common/; when run
# from the repo without staging, fall back to the original.
try:
    from api.common.tokenization import encode_sources
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
    from common.tokenization import encode_sources


class Ct2Translator:
    def __init__(self, model_dir: str, spm_path: str, device: str = "cpu",
                 inter_threads: int = 1, intra_threads: int = 0):
        self.translator = ctranslate2.Translator(
            model_dir, device=device, inter_threads=inter_threads, intra_threads=intra_threads,
        )
        self.sp = spm.SentencePieceProcessor(model_file=str(spm_path))

    def translate_one(self, text: str, target_tag: str, beam_size: int = 5,
                       max_batch_size: int = 0) -> str:
        return self.translate_batch([text], target_tag, beam_size=beam_size,
                                     max_batch_size=max_batch_size)[0]

    def translate_batch(self, texts: list, target_tag: str, beam_size: int = 5,
                         max_batch_size: int = 0) -> list:
        """texts: list of raw (untokenized) strings, all sharing the same
        target_tag (e.g. "<hi>"). Returns a list of detokenized translations,
        same order/length as texts."""
        sources = encode_sources(self.sp, target_tag, texts)  # same encoding as the training sources
        results = self.translator.translate_batch(
            sources, beam_size=beam_size, max_batch_size=max_batch_size,
            batch_type="tokens" if max_batch_size else "examples",
        )
        out = []
        for r in results:
            hyp_tokens = r.hypotheses[0]
            out.append(self.sp.decode(hyp_tokens))
        return out

    def translate_pivot(self, text: str, target_tag: str, beam_size: int = 5) -> tuple:
        """xx -> en -> xx, for when neither source nor target is English.
        Returns (final_translation, english_intermediate)."""
        english = self.translate_one(text, "<en>", beam_size=beam_size)
        final = self.translate_one(english, target_tag, beam_size=beam_size)
        return final, english


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="Path to a CTranslate2 model directory")
    ap.add_argument("--spm", required=True, help="Path to the SentencePiece .model file")
    ap.add_argument("--text", required=True)
    ap.add_argument("--target-lang", required=True, help="short tag without brackets, e.g. hi")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--inter-threads", type=int, default=1)
    ap.add_argument("--intra-threads", type=int, default=0)
    args = ap.parse_args()

    if not Path(args.model_dir).exists():
        raise SystemExit(f"model dir not found: {args.model_dir}")

    tr = Ct2Translator(args.model_dir, args.spm, inter_threads=args.inter_threads,
                        intra_threads=args.intra_threads)
    target_tag = f"<{args.target_lang}>"
    output = tr.translate_one(args.text, target_tag, beam_size=args.beam_size)
    print(f"[ct2_infer] input:  {args.text}")
    print(f"[ct2_infer] target: {target_tag}")
    print(f"[ct2_infer] output: {output}")


if __name__ == "__main__":
    main()
