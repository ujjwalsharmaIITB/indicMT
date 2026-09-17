"""Eval prep: build the 44 direction-specific, tag-prepended SPM
source files needed to run onmt_translate against IN22-Gen/IN22-Conv, from
the per-language multi-parallel test files training/build_parallel_corpus.py
already produced (128k-spm/data/test/{test-set}/{test-set}.128k.spm.<short-tag>,
already 128k-SPM-tokenized, deliberately NOT tag-prepended - "which direction's
<xx> tag to prepend is an eval-time choice", per that script's own docstring).

Each of the 22 Indic languages pairs with English in both directions (44
directions total per test set, matching this project's model shape:
44 directions, not full 22x22 Indic-Indic; see evaluation/README.md):
  - en -> xx: source = the English file, "<xx> " prepended to every line
  - xx -> en: source = language xx's file, "<en> " prepended to every line

Prepending directly onto the already-SPM-tokenized (space-separated pieces)
line, not re-encoding raw text with the tag then re-tokenizing, is safe and
exactly equivalent here: every "<xx>" tag is a registered USER_DEFINED
SentencePiece symbol (see training/export_onmt_vocab.py), which SentencePiece
always segments as one atomic piece wherever it occurs in raw text,
regardless of context - so "<xx> " + already-tokenized-pieces reproduces
what encoding "<xx> raw sentence" from scratch would have produced. This
mirrors training/build_parallel_corpus.py's own convention (tag embedded in
the source side before encoding) without needing to re-run SPM encoding at
eval time.

Output: 128k-spm/data/eval/{test-set}/{test-set}.128k.spm.{src}-{tgt}.src
(gitignored - already covered by .gitignore's `[0-9]*[a-z]*-spm/data/`
pattern - regenerable, matches 128k-spm/'s existing data/ layout).

Small and fast (1,024-1,503 lines x 44 pairs x 2 test sets, plain text
rewriting). Cheap enough to run directly, no job needed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from common.languages import LANGUAGE_CODES, LANGUAGES  # noqa: E402

TEST_DIR = ROOT / "128k-spm" / "data" / "test"
EVAL_DIR = ROOT / "128k-spm" / "data" / "eval"
TEST_SETS = ["in22-gen", "in22-conv"]

EN_SHORT = "en"
INDIC_SHORT_TAGS = [LANGUAGE_CODES[lang[:3]].strip("<>") for lang in LANGUAGES]


def build_pair(test_set: str, src_short: str, tgt_short: str) -> int:
    src_path = TEST_DIR / test_set / f"{test_set}.128k.spm.{src_short}"
    out_dir = EVAL_DIR / test_set
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{test_set}.128k.spm.{src_short}-{tgt_short}.src"
    tag = f"<{tgt_short}>"
    n = 0
    with open(src_path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(f"{tag} {line.rstrip(chr(10))}\n")
            n += 1
    return n


def main():
    if not TEST_DIR.exists():
        raise SystemExit(
            f"{TEST_DIR} does not exist - run training/build_parallel_corpus.py "
            "+ training/encode_spm.py first."
        )
    total_pairs = 0
    for test_set in TEST_SETS:
        if not (TEST_DIR / test_set).exists():
            print(f"[build_eval_pairs] skipping {test_set}: not found under {TEST_DIR}")
            continue
        for short in INDIC_SHORT_TAGS:
            n1 = build_pair(test_set, EN_SHORT, short)
            n2 = build_pair(test_set, short, EN_SHORT)
            print(f"[build_eval_pairs] {test_set}: en->{short} ({n1} lines), "
                  f"{short}->en ({n2} lines)")
            total_pairs += 2
    print(f"[build_eval_pairs] done - {total_pairs} direction files built across "
          f"{len(TEST_SETS)} test set(s)")


if __name__ == "__main__":
    main()
