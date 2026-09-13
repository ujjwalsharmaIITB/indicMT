"""Combine IN22-Gen + IN22-Conv into one multi-parallel held-out corpus for the
tokenizer fertility sweep (tokenizer/analysis/), instead of the
bpcc-human validation splits (data/bpcc-human/splits/*_valid.jsonl).

**Why IN22Gen+Conv instead of the bpcc-human valid splits**: those splits are
*different* sentences per language (each language's own 1000-sentence sample),
so a fertility difference between two languages could just reflect different
content, not a real difference in how well the tokenizer serves that language.
IN22-Gen and IN22-Conv are both multi-parallel - literally the same row's
content translated into all 23 languages - so per-language fertility numbers
become a true apples-to-apples comparison: same content, same difficulty,
only the language differs. They're also the project's actual final
test sets, i.e. genuinely never touched by training or tokenizer training, a
stronger guarantee than the bpcc-human valid split (which is training-side data,
just not sampled into train-gold/mined-mixture). Using them here is still a
test-time-only use (fertility is measured against an already-frozen SPM
model, not optimized against this data) - see download_in22.py's docstring.

Output: data/in22/combined_fertility_eval.jsonl (gitignored, matches
data/**/*.jsonl) - one row per (source dataset, original row), with a
`source` field ("gen"/"conv"), a `row_id` for traceability back to the raw
file, and all 23 language-code columns unchanged. 1,024 (IN22-Gen) + 1,503
(IN22-Conv) = 2,527 rows, each with text in every one of the 23 languages -
so fertility.py can pull a language's whole held-out corpus with a single
column read: 2,527 sentences/language, identical content across languages.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.languages import LANGUAGES  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUTPUT_FILE = Path(__file__).resolve().parent / "combined_fertility_eval.jsonl"

ENGLISH_LANG = "eng_Latn"
EXPECTED_LANG_COLUMNS = set(LANGUAGES) | {ENGLISH_LANG}

SOURCES = {
    "gen": RAW_DIR / "in22_gen.jsonl",
    "conv": RAW_DIR / "in22_conv.jsonl",
}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    for key, path in SOURCES.items():
        if not path.exists():
            raise SystemExit(f"{path} does not exist - run download_in22.py first.")

    combined = []
    per_lang_counts = {lang: 0 for lang in EXPECTED_LANG_COLUMNS}
    per_lang_empty = {lang: 0 for lang in EXPECTED_LANG_COLUMNS}

    for key, path in SOURCES.items():
        rows = load_jsonl(path)
        missing_cols = EXPECTED_LANG_COLUMNS - set(rows[0].keys())
        if missing_cols:
            raise SystemExit(f"{path}: missing expected language column(s) {sorted(missing_cols)}")

        for i, row in enumerate(rows):
            out_row = {"source": key, "row_id": i}
            for lang in EXPECTED_LANG_COLUMNS:
                text = row.get(lang)
                out_row[lang] = text
                if text:
                    per_lang_counts[lang] += 1
                else:
                    per_lang_empty[lang] += 1
            combined.append(out_row)
        print(f"[build_fertility_corpus] {key}: {len(rows)} rows from {path}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[build_fertility_corpus] Wrote {len(combined)} rows x {len(EXPECTED_LANG_COLUMNS)} "
          f"languages to {OUTPUT_FILE}")

    any_empty = {lang: n for lang, n in per_lang_empty.items() if n}
    if any_empty:
        print("[build_fertility_corpus] WARNING: some languages have empty/missing cells "
              "(breaks the 'every language has the same sentences' guarantee for those rows):")
        for lang, n in sorted(any_empty.items()):
            print(f"  {lang}: {n} empty of {len(combined)} ({per_lang_counts[lang]} non-empty)")
    else:
        print(f"[build_fertility_corpus] All {len(EXPECTED_LANG_COLUMNS)} languages have "
              f"exactly {len(combined)} non-empty sentences - fully multi-parallel, confirmed.")


if __name__ == "__main__":
    main()
