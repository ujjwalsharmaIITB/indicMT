"""Materialize the en-xx-en parallel corpus (train + dev) and the SPM training
text, from the bpcc-human multilingual datasets built by the data pipeline.

Four outputs, all under tokenizer/corpus/ (gitignored - generated, not source):

1. train.src / train.tgt - parallel plain-text files from
   data/bpcc-human/combined_train_dataset.jsonl: train.src is the tagged
   source line (`<xx> sentence`), train.tgt is the plain target line. One
   line per row, same order in both files (OpenNMT-style parallel corpus).
   Human-only - this is pre-mixing data, not the actual OpenNMT
   training set (that comes from mixing/output/ when training starts).
2. dev.src / dev.tgt - same, from data/bpcc-human/concatenated_valid_dataset.jsonl.
3. spm_train_corpus_mixed.txt - the SentencePiece training text (the one the
   sweep actually trains on): one sentence per line, untagged, pooled across
   languages, built from BOTH bpcc-human and bpcc-mined per the capping rule
   below.
4. spm_train_corpus_human_only.txt - kept as a fixed ablation baseline (not
   rebuilt by this script once it exists) so the "does including mined data
   change tokenizer fertility" comparison has a stable human-only reference
   to compare against. See tokenizer/README.md.

## Why the SPM corpus includes mined data, capped per-language

An earlier version of this script built the SPM training corpus from
bpcc-human only. That's a real mismatch: the actual OpenNMT training data
(mixing/output/) is dominated by bpcc-mined (~210M rows vs. bpcc-human's
~6.9M, a >30x difference) - a tokenizer that never saw mined-style text
during its own training would be miscalibrated (worse fertility, more
fragmentation) on the large majority of what the MT model actually trains
on. But mined text is noisier than human text, so it shouldn't be included
uncapped either (a few dominant/noisy sources could pollute the vocabulary).

**The rule** (per-language, all 23 tags):
- **Indic languages (22): cap at 1,000,000 sentences each.** Every one of
  bpcc-human's per-language counts is already well under 1M (max ~260k for
  Hindi), so this always means "take 100% of that language's human data,
  then top up with mined data (script-filtered, deduplicated against what's
  already included) up to 1,000,000" - never *less* than the full human
  contribution for any Indic language.
- **English: cap at 3,000,000.** Naively, English looks like it should
  already exceed this from human data alone: `create_multilingual_dataset.py`
  reports ~3.43M English-side rows. But that count is *not* deduplicated -
  `ilci` is a multi-parallel corpus where the same English sentence is
  translated into many Indic languages, so it appears many times over.
  **Deduplicated, human data has only ~305k unique English sentences** -
  comfortably under the 3M cap, same as every Indic language - so English
  gets the identical top-up-from-mined treatment, just with a bigger budget
  and a bigger scan (mined has no dedicated English file - English is the
  "other side" of every mined raw file, so topping it up means scanning all
  43 files, not the 1-3 a single Indic language needs).

Mined text is pulled directly from data/bpcc-mined/raw/*.jsonl (not the
flattened data/bpcc-mined/concatenated_bpcc_mined_dataset.jsonl) because
those raw files are still organized per-language - for the 22 Indic
languages, scanning only that language's own 1-3 files is far cheaper than
scanning the full 76GB flattened/shuffled file. Four Indic languages (Dogri,
Manipuri, Santali, Sindhi) get no mined top-up at all - bpcc-mined has zero
rows for them, confirmed separately in docs/data-pipeline.md.

Intended to run as a Slurm CPU job (see
scripts/slurm/tokenizer_prepare_corpus.sh), not on the login node.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common.languages import LANGUAGE_CODES, LANGUAGES  # noqa: E402
from common.scripts import is_script_consistent, lang_side_text  # noqa: E402

HUMAN_DIR = Path(__file__).resolve().parents[1] / "data" / "bpcc-human"
MINED_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "bpcc-mined" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent / "corpus"
TRAIN_IN = HUMAN_DIR / "combined_train_dataset.jsonl"
VALID_IN = HUMAN_DIR / "concatenated_valid_dataset.jsonl"
SEED = 42

INDIC_CAP = 1_000_000
ENGLISH_CAP = 3_000_000
ENGLISH_LANG = "eng_Latn"


def write_parallel(in_path, src_out, tgt_out):
    count = 0
    with open(in_path, encoding="utf-8") as fin, \
         open(src_out, "w", encoding="utf-8") as fsrc, \
         open(tgt_out, "w", encoding="utf-8") as ftgt:
        for line in tqdm(fin, desc=f"Writing {src_out.name}/{tgt_out.name}"):
            row = json.loads(line)
            fsrc.write(row["src_modified"].replace("\n", " ").strip() + "\n")
            ftgt.write(row["tgt"].replace("\n", " ").strip() + "\n")
            count += 1
    return count


def load_human_texts_by_lang(train_path):
    """Group deduplicated tgt-field texts by tgt_lang (recovers every unique
    human sentence in every language exactly once - see module docstring in
    the original human-only version of this function for why)."""
    texts = defaultdict(list)
    seen = defaultdict(set)
    with open(train_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading human texts"):
            row = json.loads(line)
            lang, text = row["tgt_lang"], row["tgt"]
            if text in seen[lang]:
                continue
            seen[lang].add(text)
            texts[lang].append(text)
    return texts


def sample_mined_texts_for_lang(lang_code, needed, human_texts_set, rng):
    """Reservoir-sample (Algorithm R) up to `needed` script-consistent,
    non-duplicate mined sentences for this language - memory stays bounded
    at O(needed) regardless of how much of the source stream is scanned,
    which matters here: English's stream is every row of every mined raw
    file (~210M rows), so collecting-then-truncating instead of true
    reservoir sampling would hold tens of millions of strings in memory for
    no reason.

    For an Indic language, only that language's own raw files
    ({lang}_*.jsonl) are scanned - cheap, since each Indic language has at
    most 3 mined config files. For English, EVERY mined raw file is scanned
    (English is the "other side" of every file, since bpcc-mined - like
    bpcc-human - is English-centric), which is the expensive case but
    unavoidable: there's no per-language English-only file to restrict to.
    A `seen` set for dedup is still O(distinct candidates observed), which
    for English is unavoidably large, but that's just deduplication - the
    reservoir itself (what we're building `selected` into) stays O(needed).
    """
    if needed <= 0:
        return []

    glob_pattern = "*.jsonl" if lang_code == ENGLISH_LANG else f"{lang_code}_*.jsonl"
    reservoir = []
    seen = set()
    count = 0
    for path in sorted(MINED_RAW_DIR.glob(glob_pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = lang_side_text(row, lang_code)
                if not text or text in human_texts_set or text in seen:
                    continue
                if not is_script_consistent(text, lang_code):
                    continue
                seen.add(text)

                count += 1
                if len(reservoir) < needed:
                    reservoir.append(text)
                else:
                    j = rng.randint(0, count - 1)
                    if j < needed:
                        reservoir[j] = text

    return reservoir


def build_mixed_spm_corpus(train_path, out_path, seed):
    rng = random.Random(seed)
    human_texts = load_human_texts_by_lang(train_path)

    per_lang_report = {}
    with open(out_path, "w", encoding="utf-8") as fout:
        for lang in tqdm(LANGUAGES + [ENGLISH_LANG], desc="Building mixed SPM corpus"):
            cap = ENGLISH_CAP if lang == ENGLISH_LANG else INDIC_CAP
            h_texts = human_texts.get(lang, [])

            if len(h_texts) >= cap:
                rng.shuffle(h_texts)
                selected = h_texts[:cap]
                mined_added = 0
            else:
                selected = list(h_texts)
                needed = cap - len(h_texts)
                mined_texts = sample_mined_texts_for_lang(lang, needed, set(h_texts), rng)
                selected.extend(mined_texts)
                mined_added = len(mined_texts)

            rng.shuffle(selected)
            for text in selected:
                fout.write(text.replace("\n", " ").strip() + "\n")

            per_lang_report[lang] = {
                "human": len(h_texts), "mined_added": mined_added, "total": len(selected),
            }

    return per_lang_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not TRAIN_IN.exists() or not VALID_IN.exists():
        raise SystemExit(
            f"{TRAIN_IN} and/or {VALID_IN} do not exist - run "
            "data/bpcc-human/create_multilingual_dataset.py first."
        )
    if not MINED_RAW_DIR.exists():
        raise SystemExit(f"{MINED_RAW_DIR} does not exist - run download_bpcc_mined.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_count = write_parallel(TRAIN_IN, OUTPUT_DIR / "train.src", OUTPUT_DIR / "train.tgt")
    dev_count = write_parallel(VALID_IN, OUTPUT_DIR / "dev.src", OUTPUT_DIR / "dev.tgt")
    print(f"[prepare_corpus] train: {train_count} pairs -> {OUTPUT_DIR}/train.{{src,tgt}}")
    print(f"[prepare_corpus] dev:   {dev_count} pairs -> {OUTPUT_DIR}/dev.{{src,tgt}}")

    report = build_mixed_spm_corpus(TRAIN_IN, OUTPUT_DIR / "spm_train_corpus_mixed.txt", args.seed)
    total = sum(r["total"] for r in report.values())
    print(f"[prepare_corpus] spm_train_corpus_mixed.txt: {total} lines "
          f"(Indic cap={INDIC_CAP:,}, English cap={ENGLISH_CAP:,})")
    for lang in sorted(report):
        r = report[lang]
        tag = LANGUAGE_CODES.get(lang[:3], "?")
        print(f"  {lang:12s} {tag:6s} human={r['human']:>9,} mined_added={r['mined_added']:>9,} "
              f"total={r['total']:>9,}")


if __name__ == "__main__":
    main()
