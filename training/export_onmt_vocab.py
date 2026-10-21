"""Training data prep, step 3: export the frozen 128k SPM model's vocabulary
as an OpenNMT-py vocab file, instead of deriving it via `onmt_build_vocab`'s
usual frequency-count-over-the-corpus approach.

**Why not the usual onmt_build_vocab path**: that counts token occurrences in
a specific training corpus and keeps whatever appears. Our SPM model was
deliberately trained on a separate, broader corpus (tokenizer/corpus/, all 23
languages, 1M/3M sentence caps - see docs/tokenizer.md) specifically for
vocabulary *coverage*, not signal balance - some of its 128,000 pieces are
legitimately rare and might not appear at all in a single downstream split
like bpcc-human-train. Deriving the vocab from that split alone would quietly
under-cover the tokenizer's real output space: a piece the frozen SPM model
can and will produce on unseen text (mined data, eval sets, real deployment
input) would hit the model's embedding table as a gap it was never trained on
if that piece never appeared in whichever specific training split built the
vocab file. Exporting the vocab directly from the SPM model guarantees the
embedding table always covers everything the tokenizer can produce, matching
the "128k" vocabulary size this project has committed to throughout
(docs/tokenizer.md's embedding-cost accounting) exactly.

**Special tokens, handled deliberately, not left to chance**: the SPM model's
own control ids 0-3 are `<s>`, `<pad>`, `</s>`, `<unk>` (verified directly
against the model, not assumed - `<pad>` in particular does NOT match
OpenNMT-py's own default pad string, `<blank>` - see onmt/constants.py's
DefaultTokens). OpenNMT-py's vocab loader (onmt/inputters/inputter.py's
build_vocab) only injects its own default specials for strings not already
present in the supplied vocab file - so if this script included SPM's own
`<s>`/`</s>`/`<unk>` (which happen to string-match OpenNMT's own BOS/EOS/UNK
exactly) plus a mismatched `<pad>` that OpenNMT wouldn't recognize as its pad
token, the result would be a wasted, dead `<pad>` vocab entry sitting
alongside a separately-injected real `<blank>` pad token doing the actual
work - one entry over the intended 128,000, and a confusing artifact in the
checkpoint's vocab. Simplest and cleanest fix: **exclude SPM's 4 control ids
(0-3) from the exported file entirely** and let OpenNMT-py inject its own
4 defaults (`<unk>`, `<s>`, `</s>`, `<blank>`) cleanly - final vocab is
exactly 127,996 real pieces + 4 injected specials = 128,000, with no dead
entries. All 23 language tags (`<en>`..`<ur>`, SPM ids 4-26, confirmed
USER_DEFINED) are well within the exported range and
included normally.

Output format matches what OpenNMT-py's vocab loader expects either way it's
produced (by onmt_build_vocab or supplied directly): one `<piece>\t<count>`
per line. Counts are cosmetic descending placeholders (this file is never
frequency-pruned since --src_vocab_size/--tgt_vocab_size in the training yaml
is set to cover every entry) - only relative order matters for any
count-based tooling that might inspect the file.

One shared file for both sides (`share_vocab: true` in the training yaml) -
this project uses a single SPM model for source and target both
(see docs/data.md, "Language tags").
"""

import sys
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "tokenizer" / "sweep" / "mixed" / "spm_128000.model"
OUT_PATH = ROOT / "128k-spm" / "vocab.128k.shared.txt"

CONTROL_IDS = {0, 1, 2, 3}  # <s>, <pad>, </s>, <unk> - see module docstring


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} does not exist.")
    sp = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))
    n = sp.vocab_size()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_whitespace = []
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i in range(n):
            if i in CONTROL_IDS:
                continue
            piece = sp.id_to_piece(i)
            # OpenNMT-py's vocab-file reader (onmt/inputters/inputter.py's
            # _read_vocab_file) splits each line on generic whitespace
            # (`line.split(None, 1)`), not specifically on the tab this
            # script writes - a piece that is ITSELF a Unicode whitespace
            # character collapses the split and breaks the reader. This model has exactly
            # one such piece (id 125565, the C1 control character U+0085 NEL).
            # Skipping it: real text containing it (rare, likely noise
            # from the original tokenizer training corpus) falls back to
            # <unk> like any other genuinely out-of-vocabulary content -
            # acceptable, since a piece that can't even survive being
            # written to a vocab *file* isn't going to carry a machine
            # translation training signal either way.
            if piece.isspace():
                skipped_whitespace.append((i, piece))
                continue
            count = n - i  # descending placeholder, preserves SPM's own id order
            f.write(f"{piece}\t{count}\n")
            written += 1

    print(f"[export_onmt_vocab] SPM model vocab_size={n}, excluded {len(CONTROL_IDS)} "
          f"control ids and {len(skipped_whitespace)} whitespace-only piece(s) "
          f"{skipped_whitespace}, wrote {written} entries -> {OUT_PATH}")
    print(f"[export_onmt_vocab] Expected final OpenNMT vocab size after its own "
          f"4 default specials are injected: {written + 4}")


if __name__ == "__main__":
    main()
