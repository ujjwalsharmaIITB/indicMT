"""Download the bpcc-mined (synthetic/mined) configs of ai4bharat/BPCC.

Works like data/bpcc-human/download_bpcc_human.py (paths relative to this file,
explicit HF auth, resumable downloads with opt-in --fresh, retry with backoff).

Per the data pipeline's design (docs/data.md): the public
ai4bharat/BPCC dataset exposes no
LaBSE/SONAR/quality-score column for any config (verified schema check). The
"-filtered" config names (samanantar-filtered, nllb-filtered) indicate AI4Bharat
already applied LaBSE-style filtering upstream, so these configs are used as-is;
no score-threshold filtering is attempted here.

Data-quality fix (see data/common/languages.py's canonicalize_lang_code docstring
for the confirmed bpcc-human case this defends against): whichever of a row's
src_lang/tgt_lang shares the requested split's 3-letter language prefix but
disagrees on the script suffix is corrected to the split key itself. A full scan
of every bpcc-mined raw file found no such mismatches here (unlike bpcc-human's
sat_Olck split), but the fix is applied defensively for future downloads/configs.
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
CONFIGS = ["comparable", "samanantar-filtered", "nllb-filtered"]
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
            output_file = OUTPUT_DIR / f"{lang}_{config}.jsonl"
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
                      f"language ({lang}).")


if __name__ == "__main__":
    main()
