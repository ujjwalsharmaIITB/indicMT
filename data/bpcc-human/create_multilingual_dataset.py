"""Combine per-language bpcc-human splits into one multilingual train/valid
dataset, expanded bidirectionally (src->tgt and tgt->src) and covering all
44 directions (22 languages x {en->xx, xx->en}).

Details:
- SPLIT_DATA_DIR/OUTPUT_DIR are resolved relative to this file.
- Tags come from the shared data/common/languages.py (tag_for(), which includes
  "eng" and defaults to <unk>), used identically for both directions.
- Per-language datasets are combined with `concatenate_datasets()`, not
  `interleave_datasets()`: the latter defaults to `stopping_strategy=
  "first_exhausted"` and would silently drop most of every larger language's
  data. Temperature-sampled mixing is a separate step (see mixing/README.md:
  alpha=0.5, capped at 3x); this script combines everything losslessly, and
  asserts that bidirectional expansion exactly doubles the row count.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.languages import tag_for  # noqa: E402
from common.tokenization import tag_source  # noqa: E402

SPLIT_DATA_DIR = Path(__file__).resolve().parent / "splits"
OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_SUFFIX = "_train.jsonl"
VALID_SUFFIX = "_valid.jsonl"


def load_jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def process_data(df):
    bidirectional_data = []
    for row in df.itertuples(index=False):
        if not row.src or not row.tgt or not row.src_lang or not row.tgt_lang:
            continue

        bidirectional_data.append({
            "src_lang": row.src_lang,
            "tgt_lang": row.tgt_lang,
            "src_modified": tag_source(tag_for(row.tgt_lang), row.src),
            "tgt": row.tgt,
            "dataset": "bpcc-human",
        })
        bidirectional_data.append({
            "src_lang": row.tgt_lang,
            "tgt_lang": row.src_lang,
            "src_modified": tag_source(tag_for(row.src_lang), row.tgt),
            "tgt": row.src,
            "dataset": "bpcc-human",
        })
    return pd.DataFrame(bidirectional_data)


def save_dataset_to_jsonl(dataset, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    if not SPLIT_DATA_DIR.exists():
        raise SystemExit(f"{SPLIT_DATA_DIR} does not exist - run create_splits.py first.")

    train_dfs = []
    train_files = sorted(f for f in SPLIT_DATA_DIR.iterdir() if f.name.endswith(TRAIN_SUFFIX))
    train_count_before_merge = 0
    for file_path in tqdm(train_files, desc="Processing train files"):
        df = load_jsonl_to_dataframe(file_path)
        train_count_before_merge += len(df)
        df = process_data(df)
        train_dfs.append(Dataset.from_pandas(df, preserve_index=False))

    valid_dfs = []
    valid_files = sorted(f for f in SPLIT_DATA_DIR.iterdir() if f.name.endswith(VALID_SUFFIX))
    valid_count_before_merge = 0
    for file_path in tqdm(valid_files, desc="Processing validation files"):
        df = load_jsonl_to_dataframe(file_path)
        valid_count_before_merge += len(df)
        df = process_data(df)
        valid_dfs.append(Dataset.from_pandas(df, preserve_index=False))

    combined_train_dataset = concatenate_datasets(train_dfs).shuffle(seed=42)
    concatenated_valid_dataset = concatenate_datasets(valid_dfs).shuffle(seed=42)

    print(f"Train samples before merging: {train_count_before_merge}")
    print(f"Train samples after merging (bidirectional): {len(combined_train_dataset)}")
    assert len(combined_train_dataset) == 2 * train_count_before_merge, (
        "Bidirectional expansion should exactly double the row count - if not, rows were "
        "lost somewhere in the merge (see the interleave_datasets bug this replaced)."
    )
    print(f"Validation samples before merging: {valid_count_before_merge}")
    print(f"Validation samples after merging (bidirectional): {len(concatenated_valid_dataset)}")

    train_out = OUTPUT_DIR / "combined_train_dataset.jsonl"
    valid_out = OUTPUT_DIR / "concatenated_valid_dataset.jsonl"
    save_dataset_to_jsonl(combined_train_dataset, train_out)
    save_dataset_to_jsonl(concatenated_valid_dataset, valid_out)
    print(f"Combined training dataset saved to {train_out}")
    print(f"Concatenated validation dataset saved to {valid_out}")


if __name__ == "__main__":
    main()
