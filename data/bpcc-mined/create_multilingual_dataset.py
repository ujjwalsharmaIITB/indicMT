"""Combine all raw bpcc-mined files into one shuffled, bidirectionally-expanded
multilingual dataset. Intentionally train-only (no split): per the data
pipeline's design (docs/data.md), mined data is trusted as
already LaBSE-filtered upstream by AI4Bharat, so it is used entirely for
training, not for validation/eval - the project's validation signal during
training comes from bpcc-human's held-out split, and final scoring comes from
IN22Gen/IN22Conv.

Quality gates, applied before bidirectional expansion:
1. Unicode-script consistency (data/common/scripts.py), both sides of every
   row (src against src_lang, tgt against tgt_lang) - the pipeline's main
   defense against script/language mislabeling, a known risk in web-mined
   corpora. Mismatched rows are dropped, not corrected.
2. **Exact-duplicate removal** on the (src, tgt) pair, applied globally across
   all raw files (a pair can recur across bpcc-mined's different configs,
   e.g. the same sentence pair in both `comparable` and `nllb-filtered`) -
   this was previously only implemented for bpcc-human (in create_splits.py)
   despite docs/paper.md describing it as applying to all data; this script
   now actually does it for bpcc-mined too.

Performance note: bpcc-mined is two orders of magnitude larger than
bpcc-human (~106M raw rows at time of writing, vs. ~3.4M) - large enough that
the per-row script-consistency check (a Python-level character scan) matters.
Loading + filtering is parallelized per-file with ProcessPoolExecutor (files
are independent and vary widely in size, so this is the natural, easy
parallelism axis) before the single-process global dedup + bidirectional
expansion. Uses plain Python lists/dicts rather than pandas/HF `datasets`
Arrow tables at this row count, to keep the per-row Python-object overhead
(not Arrow's) the only cost and keep memory behavior easy to reason about.
Intended to run as a Slurm CPU-only batch job (see
scripts/slurm/data_mined_multilingual_dataset.sh), not on the login node.
"""

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.languages import tag_for  # noqa: E402
from common.tokenization import tag_source  # noqa: E402
from common.scripts import is_script_consistent  # noqa: E402

RAW_DATA_DIR = Path(__file__).resolve().parent / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKERS = 16
SEED = 42


def load_and_filter_file(file_path_str):
    """Runs in a worker process. Loads one raw JSONL file, drops rows with
    missing required fields or script-mismatched text. Returns
    (filename, kept_rows, dropped_script_count, raw_row_count); kept_rows is
    a list of (src, tgt, src_lang, tgt_lang) tuples."""
    file_path = Path(file_path_str)
    kept = []
    dropped_script = 0
    raw_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_count += 1
            row = json.loads(line)
            src, tgt = row.get("src"), row.get("tgt")
            src_lang, tgt_lang = row.get("src_lang"), row.get("tgt_lang")
            if not src or not tgt or not src_lang or not tgt_lang:
                continue
            if not (is_script_consistent(src, src_lang) and is_script_consistent(tgt, tgt_lang)):
                dropped_script += 1
                continue
            kept.append((src, tgt, src_lang, tgt_lang))
    return file_path.name, kept, dropped_script, raw_count


def bidirectional_expand(rows):
    data = []
    for src, tgt, src_lang, tgt_lang in rows:
        data.append({
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "src_modified": tag_source(tag_for(tgt_lang), src),
            "tgt": tgt,
            "dataset": "bpcc-mined",
        })
        data.append({
            "src_lang": tgt_lang,
            "tgt_lang": src_lang,
            "src_modified": tag_source(tag_for(src_lang), tgt),
            "tgt": src,
            "dataset": "bpcc-mined",
        })
    return data


def save_dataset_to_jsonl(rows, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for example in rows:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Parallel worker processes for loading+filtering raw files (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()

    if not RAW_DATA_DIR.exists():
        raise SystemExit(f"{RAW_DATA_DIR} does not exist - run download_bpcc_mined.py first.")

    raw_files = sorted(f for f in RAW_DATA_DIR.iterdir() if f.name.endswith(".jsonl"))

    total_count_before = 0
    total_dropped_script = 0
    all_rows = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(load_and_filter_file, str(p)): p for p in raw_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading+filtering raw files"):
            name, rows, dropped_script, raw_count = future.result()
            total_count_before += raw_count
            total_dropped_script += dropped_script
            if dropped_script:
                print(f"[create_multilingual_dataset] {name}: dropped {dropped_script} row(s) with mismatched script")
            all_rows.extend(rows)

    print(f"Total raw samples: {total_count_before}")
    print(f"Total rows dropped for script mismatch: {total_dropped_script}")

    seen = set()
    deduped_rows = []
    for row in all_rows:
        key = (row[0], row[1])
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)
    total_dropped_dup = len(all_rows) - len(deduped_rows)
    print(f"Total rows dropped as exact duplicates: {total_dropped_dup}")
    print(f"Rows after filtering + dedup (pre-expansion): {len(deduped_rows)}")

    bidirectional_data = bidirectional_expand(deduped_rows)
    random.Random(SEED).shuffle(bidirectional_data)

    assert len(bidirectional_data) == 2 * len(deduped_rows), (
        "Bidirectional expansion should exactly double the deduped row count - if not, rows "
        "were lost somewhere in the merge."
    )
    print(f"Total samples after processing (bidirectional): {len(bidirectional_data)}")

    out_path = OUTPUT_DIR / "concatenated_bpcc_mined_dataset.jsonl"
    save_dataset_to_jsonl(bidirectional_data, out_path)
    print(f"Concatenated BPCC Mined dataset saved to {out_path}")


if __name__ == "__main__":
    main()
