"""Generate per-language train/valid splits from the raw bpcc-human download.

This script was referenced (as a consumer, via SPLIT_DATA_DIR) by the original
create-multilingual-datasets.py draft but never actually existed - nothing
produced the `splits/` directory it expected to read from. This fills that gap.

For each of the 22 languages, the validation set is drawn **only from
bpcc-seed-latest** (not ilci - ILCI is a narrower, domain-specific corpus and
isn't representative enough for a general validation signal). Training data
is everything else: the remainder of bpcc-seed-latest plus all of ilci (and
any other bpcc-human config present). Exact-duplicate (src, tgt) pairs are
removed, and any pair pulled into validation is also excluded from training
even if it happens to reappear in another config - see the pipeline tests in
data/tests/ for the leakage check this guarantees.

Quality gate: every row is also checked with data/common/scripts.py - the
Indic-language side of the pair must actually be written in the Unicode
script implied by its BPCC language code (e.g. a "hin_Deva" row must contain
Devanagari text, not e.g. Bengali). Rows that fail this check (a known risk
in web-mined/aggregated multilingual corpora, less so in bpcc-human, but
checked here too for consistency with the mined pipeline) are dropped rather
than kept - the dominant/expected script wins, minority-script rows for that
language are discarded.

Final test-time evaluation for this project uses IN22Gen/IN22Conv (external,
untouched benchmark sets) - the "valid" split produced here is only for
early-stopping/checkpoint-selection during training, not for reporting
headline numbers.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.languages import LANGUAGES  # noqa: E402
from common.scripts import is_script_consistent, lang_side_text  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent / "raw"
SPLITS_DIR = Path(__file__).resolve().parent / "splits"
SEED = 42
DEV_SIZE_PER_LANG = 1000  # capped below by MAX_DEV_FRACTION for small languages
MAX_DEV_FRACTION = 0.02
DEV_SOURCE_CONFIG = "bpcc-seed-latest"


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[create_splits] Skipping malformed line in {path}: {exc}")
    return rows


def dedup(rows):
    seen = set()
    deduped = []
    for row in rows:
        key = (row.get("src"), row.get("tgt"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def filter_by_script(rows, lang_code):
    """Drop rows whose lang_code-side text isn't actually in that language's
    expected Unicode script. Returns (kept_rows, dropped_count)."""
    kept = []
    dropped = 0
    for row in rows:
        text = lang_side_text(row, lang_code)
        if text is None or is_script_consistent(text, lang_code):
            kept.append(row)
        else:
            dropped += 1
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-size", type=int, default=DEV_SIZE_PER_LANG)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(f"{RAW_DIR} does not exist - run download_bpcc_human.py first.")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    summary = defaultdict(dict)
    for lang in tqdm(LANGUAGES, desc="Splitting languages"):
        seed_path = RAW_DIR / f"{lang}_{DEV_SOURCE_CONFIG}_bpcc_human.jsonl"
        seed_rows, seed_dropped = filter_by_script(load_jsonl(seed_path), lang)
        seed_rows = dedup(seed_rows)

        other_rows_raw = []
        for path in sorted(RAW_DIR.glob(f"{lang}_*.jsonl")):
            if path == seed_path:
                continue
            other_rows_raw.extend(load_jsonl(path))
        other_rows, other_dropped = filter_by_script(other_rows_raw, lang)

        script_dropped = seed_dropped + other_dropped
        if script_dropped:
            print(f"[create_splits] {lang}: dropped {script_dropped} row(s) with mismatched script")

        if not seed_rows and not other_rows:
            print(f"[create_splits] No raw data for {lang}, skipping.")
            continue

        rng.shuffle(seed_rows)
        total_size = len(seed_rows) + len(other_rows)
        dev_size = min(args.dev_size, int(total_size * MAX_DEV_FRACTION) or total_size)
        dev_size = min(dev_size, len(seed_rows))

        dev_rows = seed_rows[:dev_size]
        remaining_seed_rows = seed_rows[dev_size:]

        dev_pairs = {(r.get("src"), r.get("tgt")) for r in dev_rows}
        train_rows = [
            r for r in dedup(remaining_seed_rows + other_rows)
            if (r.get("src"), r.get("tgt")) not in dev_pairs
        ]

        with open(SPLITS_DIR / f"{lang}_train.jsonl", "w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(SPLITS_DIR / f"{lang}_valid.jsonl", "w", encoding="utf-8") as f:
            for row in dev_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary[lang] = {"train": len(train_rows), "valid": len(dev_rows), "script_dropped": script_dropped}
        print(f"[create_splits] {lang}: train={len(train_rows)} valid={len(dev_rows)} "
              f"script_dropped={script_dropped} (valid from {DEV_SOURCE_CONFIG})")

    print("[create_splits] Done.", dict(summary))


if __name__ == "__main__":
    main()
