"""Download the IN22-Gen and IN22-Conv held-out benchmark sets from HuggingFace
(both gated - same access-request flow as ai4bharat/BPCC, see data/common/auth.py).

These are AI4Bharat's final evaluation sets (chrF++/BLEU on IN22Gen/IN22Conv),
used strictly as test data, never mixed into training or tokenizer-training
data. Reading them for the tokenizer fertility sweep (tokenizer/analysis/) is
still a test-time use - fertility is a static property of an already-trained SPM model,
not something optimized against this data - so it doesn't violate that rule; it
just means the data was inspected before model evaluation.

Both configs share the exact same 23 language-code columns (asm_Beng ... eng_Latn
... urd_Arab, matching data/common/languages.py) - i.e. every row is multi-parallel,
the same underlying sentence/turn translated into all 23 languages. They differ only
in their own metadata columns (IN22-Gen: context/source/url/domain/num_words/bucket;
IN22-Conv: doc_id/sent_id/topic/domain/prompt/scenario/speaker/turn) and row count
(IN22-Gen 1,024; IN22-Conv 1,503).

Output: data/in22/raw/in22_gen.jsonl, data/in22/raw/in22_conv.jsonl (gitignored,
matches data/**/raw/ and data/**/*.jsonl in .gitignore) - one JSON object per row,
the dataset's own columns unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.auth import ensure_hf_login  # noqa: E402

from datasets import load_dataset  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "raw"

DATASETS = {
    "gen": "ai4bharat/IN22-Gen",
    "conv": "ai4bharat/IN22-Conv",
}


def save_to_jsonl(rows, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="Re-download even if the output already exists.")
    args = parser.parse_args()

    ensure_hf_login()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for key, hf_name in DATASETS.items():
        output_file = OUTPUT_DIR / f"in22_{key}.jsonl"
        if output_file.exists() and not args.fresh:
            print(f"[download_in22] {output_file} already exists, skipping ({hf_name}). Use --fresh to redo.")
            continue

        print(f"[download_in22] Loading {hf_name} ...")
        ds = load_dataset(hf_name, "default", split="test")
        rows = [dict(row) for row in ds]
        save_to_jsonl(rows, output_file)
        print(f"[download_in22] Saved {len(rows)} rows for {hf_name} to {output_file}")


if __name__ == "__main__":
    main()
