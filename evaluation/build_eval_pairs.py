"""Eval prep: build the 44 direction-specific, tagged SentencePiece source
files needed to run onmt_translate (and deployment/ct2_eval.py) on
IN22-Gen/IN22-Conv, from the untokenized multi-parallel per-language test
files training/build_parallel_corpus.py already produced
(128k-spm/data/raw/test/{test-set}/{test-set}.<short-tag>, deliberately NOT
tag-prepended - "which direction's <xx> tag to prepend is an eval-time choice",
per that script's own docstring).

Each of the 22 Indic languages pairs with English in both directions (44
directions total per test set, matching this project's model shape:
44 directions, not full 22x22 Indic-Indic; see evaluation/README.md):
  - en -> xx: source = the English file, tagged "<xx>"
  - xx -> en: source = language xx's file, tagged "<en>"

Each source line is built exactly as the training and dev sources were: the raw
sentence with the tag in front, "<xx> raw sentence", encoded by SentencePiece in
one call (data/common/tokenization.py). With add_dummy_prefix (the SentencePiece
default) and "<xx>" a user-defined piece, this yields ['▁', '<xx>', '▁word', ...]:
a lone "▁" first, then the tag, as in every training and dev source line. Prepending "<xx>" to an already-encoded
sentence would drop that leading "▁" and no longer match training.

Output: 128k-spm/data/eval/{test-set}/{test-set}.128k.spm.{src}-{tgt}.src
(gitignored - already covered by .gitignore's `[0-9]*[a-z]*-spm/data/`
pattern - regenerable, matches 128k-spm/'s existing data/ layout).

Small and fast (1,024-1,503 lines x 44 pairs x 2 test sets). Cheap enough to
run directly, no job needed.
"""

import sys
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from common.languages import LANGUAGE_CODES, LANGUAGES  # noqa: E402
from common.tokenization import encode_sources  # noqa: E402

RAW_DIR = ROOT / "128k-spm" / "data" / "raw" / "test"
SPM_MODEL_PATH = ROOT / "tokenizer" / "sweep" / "mixed" / "spm_128000.model"
EVAL_DIR = ROOT / "128k-spm" / "data" / "eval"
TEST_SETS = ["in22-gen", "in22-conv"]

EN_SHORT = "en"
INDIC_SHORT_TAGS = [LANGUAGE_CODES[lang[:3]].strip("<>") for lang in LANGUAGES]


def build_pair(sp, test_set: str, src_short: str, tgt_short: str) -> int:
    raw_path = RAW_DIR / test_set / f"{test_set}.{src_short}"
    out_dir = EVAL_DIR / test_set
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{test_set}.128k.spm.{src_short}-{tgt_short}.src"
    tag = f"<{tgt_short}>"
    with open(raw_path, encoding="utf-8") as f_in:
        texts = [line.rstrip("\n") for line in f_in]
    pieces = encode_sources(sp, tag, texts)  # same encoding as the training sources
    with open(out_path, "w", encoding="utf-8") as f_out:
        for p in pieces:
            f_out.write(" ".join(p) + "\n")
    return len(pieces)


def main():
    if not RAW_DIR.exists():
        raise SystemExit(
            f"{RAW_DIR} does not exist - run training/build_parallel_corpus.py first."
        )
    if not SPM_MODEL_PATH.exists():
        raise SystemExit(f"{SPM_MODEL_PATH} not found - train the tokenizer first (tokenizer/README.md).")
    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL_PATH))
    total_pairs = 0
    for test_set in TEST_SETS:
        if not (RAW_DIR / test_set).exists():
            print(f"[build_eval_pairs] skipping {test_set}: not found under {RAW_DIR}")
            continue
        for short in INDIC_SHORT_TAGS:
            n1 = build_pair(sp, test_set, EN_SHORT, short)
            n2 = build_pair(sp, test_set, short, EN_SHORT)
            print(f"[build_eval_pairs] {test_set}: en->{short} ({n1} lines), "
                  f"{short}->en ({n2} lines)")
            total_pairs += 2
    print(f"[build_eval_pairs] done - {total_pairs} direction files built across "
          f"{len(TEST_SETS)} test set(s)")


if __name__ == "__main__":
    main()
