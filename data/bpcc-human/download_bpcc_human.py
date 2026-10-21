"""Download the bpcc-human (gold, human-translated) configs of ai4bharat/BPCC.

Details:
- Output filenames include the config name (`{lang}_{config}_bpcc_human.jsonl`),
  so bpcc-seed-latest and ilci do not overwrite each other for languages present
  in both.
- Paths resolved relative to this file, not the caller's cwd.
- Explicit HF auth via data/common/auth.py, with a clear error if no token is found
  (ai4bharat/BPCC is gated).
- Wiping the output directory is opt-in (--fresh); by
  default, already-downloaded (lang, config) files are skipped so a failed run can
  be resumed without re-downloading everything.
- load_dataset() is retried with backoff on transient errors instead of aborting
  the whole multi-config/multi-language run on the first network hiccup.

Data-quality fix: BPCC's own per-row src_lang/tgt_lang metadata can disagree with
the language actually requested (verified case: every row in the sat_Olck config
split has tgt_lang="sat_Beng" in its own metadata, despite the text being
unambiguous Ol Chiki script, not Bengali - confirmed by inspecting the raw
characters, not inferred). Whichever side's declared code shares the split's
3-letter language prefix but disagrees on the script suffix is canonicalized to
the split key itself (the split we actually requested is ground truth for what
we downloaded), and the correction is logged so it stays visible rather than
silently changing the data.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.auth import ensure_hf_login  # noqa: E402
from common.languages import LANGUAGES, canonicalize_lang_code  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "raw"
CONFIGS = ["bpcc-seed-latest", "ilci"]
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 10


def save_to_jsonl(rows, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset_with_retry(config):
    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return load_dataset("ai4bharat/BPCC", config)
        except Exception as exc:
            last_exc = exc
            print(f"[download] load_dataset({config!r}) failed (attempt {attempt}/{MAX_ATTEMPTS}): {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete the existing output directory first instead of resuming/skipping existing files.",
    )
    args = parser.parse_args()

    if args.fresh and OUTPUT_DIR.exists():
        print(f"[download] --fresh set, clearing {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_hf_login()

    for config in tqdm(CONFIGS, desc="Processing configs"):
        dataset = load_dataset_with_retry(config)

        for lang in tqdm(LANGUAGES, desc=f"Processing languages for {config}"):
            output_file = OUTPUT_DIR / f"{lang}_{config}_bpcc_human.jsonl"
            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"[download] Skipping {lang}/{config}, already downloaded ({output_file}).")
                continue

            if lang not in dataset:
                print(f"[download] Skipping {lang}, not available in the {config} config.")
                continue

            lang_data = dataset[lang]
            rows = []
            corrections = 0
            for example in tqdm(lang_data, desc=f"Processing examples for {lang}", leave=False):
                if example.get("src") and example.get("tgt") and example.get("src_lang") and example.get("tgt_lang"):
                    example, changed = canonicalize_lang_code(example, lang)
                    corrections += changed
                    rows.append({
                        "src": example["src"],
                        "tgt": example["tgt"],
                        "src_lang": example["src_lang"],
                        "tgt_lang": example["tgt_lang"],
                        "meta_data": {"config": config},
                    })

            save_to_jsonl(rows, output_file)
            print(f"[download] Saved {len(rows)} rows for {lang}/{config} to {output_file}")
            if corrections:
                print(f"[download] {lang}/{config}: canonicalized src_lang/tgt_lang on "
                      f"{corrections} row(s) whose BPCC metadata disagreed with the split "
                      f"language ({lang}) - see module docstring.")


if __name__ == "__main__":
    main()
