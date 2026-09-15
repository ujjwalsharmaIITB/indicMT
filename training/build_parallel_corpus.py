"""Training data prep: turn our JSONL datasets into OpenNMT-py's expected
format - line-aligned, plain-text `.src`/`.tgt` files, one sentence pair per
line, tag already embedded in the source side (`<xx> source text`).

Five datasets, each becoming one raw src/tgt file pair under
`128k-spm/data/raw/`:

- **bpcc-human-train**: `mixing/output/train_gold.jsonl` (the
  temperature-sampled, alpha=0.5/3x-capped bpcc-human mixture, "train-gold").
  Already has `src_modified` (tag-prepended source) and `tgt` fields per row -
  straight extraction, byte-chunked in parallel (the file is ~3GB/6.95M
  lines).
- **bpcc-mined-train**: `mixing/output/mined_mixture.jsonl` (same mixing
  process, mined-mixture). Same straight extraction, same parallelization -
  this one is ~72GB/213.8M lines, the reason this script parallelizes at all.
- **bpcc-human-dev**: `data/bpcc-human/concatenated_valid_dataset.jsonl`
  (held-out per-language dev draw from bpcc-seed-latest, NOT
  temperature-resampled; see mixing/README.md). Same straight extraction,
  small enough that one process is enough, but still routed through the same
  path for consistency.
- **in22gen-test** / **in22conv-test**: `data/in22/raw/in22_{gen,conv}.jsonl`.
  These are multi-parallel (one row = the same sentence in all 23 languages) -
  kept that way in the output rather than pre-flattened into src/tgt pairs, so
  any of the 44 directions can be scored later by pairing two of the 23
  per-language files at the same line index (an earlier version of this
  script pre-expanded to 44 directional src/tgt pairs; corrected - a
  pre-flattened file can't be re-sliced back into per-direction references
  for chrF++/BLEU without re-deriving which lines belong to which direction,
  which the multi-parallel form gives for free). Written as **23 line-aligned
  per-language files** under `128k-spm/data/raw/test/{in22-gen,in22-conv}/`:
  `in22-gen.<short-tag>` (e.g. `in22-gen.en`, `in22-gen.hi`, ... - short tag =
  the language's `<xx>` tag with the brackets stripped), one text file per
  language, line N of every file in a test set = the same source row's text
  in that language. No tag is prepended in this raw form (unlike the train/dev
  src side) - which direction's `<xx>` tag to prepend is an eval-time choice,
  made by whichever two files get paired up as a direction.

Output naming (matches the temp-sampled train mixtures' own naming so the
processing history is legible from the filename alone):
    <source>.<split>[.temp-sampled.0.5].raw.{src,tgt}          (train/dev)
    test/<in22-gen,in22-conv>/<in22-gen,in22-conv>.<short-tag>  (test)

Run via `scripts/slurm/training_build_dataset.sh` (SLURM, CPU-only) rather
than on a login node; bpcc-mined-train alone is 72GB and 213.8M lines.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mixing"))
sys.path.insert(0, str(ROOT / "data"))
from build_mixture import byte_chunks, iter_lines_in_range, read_shard_with_retry, unlink_with_retry  # noqa: E402
from common.languages import LANGUAGE_CODES, LANGUAGES  # noqa: E402

RAW_DIR = ROOT / "128k-spm" / "data" / "raw"
TEST_DIR = RAW_DIR / "test"

ENGLISH_LANG = "eng_Latn"
ALL_LANGS = LANGUAGES + [ENGLISH_LANG]  # 22 Indic (fixed order) then English last

# name -> (input path, output basename, kind)
#   kind "pairs": row already has src_modified/tgt - straight extraction.
#   kind "perlang": row has one column per language - keep multi-parallel,
#                   write one file per language instead of flattening to pairs.
DATASETS = {
    "bpcc-human-train": (
        ROOT / "mixing" / "output" / "train_gold.jsonl",
        "bpcc-human.train.temp-sampled.0.5.raw",
        "pairs",
    ),
    "bpcc-mined-train": (
        ROOT / "mixing" / "output" / "mined_mixture.jsonl",
        "bpcc-mined.train.temp-sampled.0.5.raw",
        "pairs",
    ),
    "bpcc-human-dev": (
        ROOT / "data" / "bpcc-human" / "concatenated_valid_dataset.jsonl",
        "bpcc-human.dev.raw",
        "pairs",
    ),
    "in22gen-test": (
        ROOT / "data" / "in22" / "raw" / "in22_gen.jsonl",
        "in22-gen",
        "perlang",
    ),
    "in22conv-test": (
        ROOT / "data" / "in22" / "raw" / "in22_conv.jsonl",
        "in22-conv",
        "perlang",
    ),
}


def tag_for(lang: str) -> str:
    return LANGUAGE_CODES[lang[:3]]


def short_tag(lang: str) -> str:
    return tag_for(lang).strip("<>")


def extract_pairs_chunk(args):
    path, start, end, shard_src, shard_tgt = args
    n = 0
    with open(shard_src, "w", encoding="utf-8") as fsrc, \
         open(shard_tgt, "w", encoding="utf-8") as ftgt:
        for line in iter_lines_in_range(path, start, end):
            row = json.loads(line)
            src = row["src_modified"].replace("\n", " ").replace("\t", " ").strip()
            tgt = row["tgt"].replace("\n", " ").replace("\t", " ").strip()
            if not src or not tgt:
                continue
            fsrc.write(src + "\n")
            ftgt.write(tgt + "\n")
            n += 1
    return shard_src, shard_tgt, n


def build_pairs(input_path: Path, out_base: str, workers: int):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_src = RAW_DIR / f"{out_base}.src"
    out_tgt = RAW_DIR / f"{out_base}.tgt"
    tmp_dir = RAW_DIR / f".{out_base}.shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunks = byte_chunks(input_path, workers)
    shard_args = [
        (input_path, s, e, tmp_dir / f"shard_{i:03d}.src", tmp_dir / f"shard_{i:03d}.tgt")
        for i, (s, e) in enumerate(chunks)
    ]
    print(f"[build_parallel_corpus] {input_path.name}: {len(chunks)} chunks, {workers} workers", flush=True)

    results = [None] * len(shard_args)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_pairs_chunk, args): i for i, args in enumerate(shard_args)}
        with tqdm(total=len(futures), desc=f"{input_path.name} chunks", file=sys.stdout) as pbar:
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                pbar.update(1)

    total = 0
    with open(out_src, "w", encoding="utf-8") as fsrc, open(out_tgt, "w", encoding="utf-8") as ftgt:
        for shard_src, shard_tgt, n in tqdm(results, desc=f"{input_path.name} concat", file=sys.stdout):
            fsrc.write(read_shard_with_retry(shard_src))
            ftgt.write(read_shard_with_retry(shard_tgt))
            total += n
            unlink_with_retry(shard_src)
            unlink_with_retry(shard_tgt)
    tmp_dir.rmdir()
    print(f"[build_parallel_corpus] Wrote {total} pairs -> {out_src.name}, {out_tgt.name}", flush=True)


def build_perlang(input_path: Path, test_name: str):
    """Keep the multi-parallel structure: one file per language, line N of
    every file = the same source row's text in that language. Never skips a
    row for a missing language's text (would desync the other 22 files'
    line counts) - writes an empty line and counts it, instead."""
    out_dir = TEST_DIR / test_name
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {lang: open(out_dir / f"{test_name}.{short_tag(lang)}", "w", encoding="utf-8")
               for lang in ALL_LANGS}

    n_rows = sum(1 for _ in open(input_path, encoding="utf-8"))
    empty_counts = {lang: 0 for lang in ALL_LANGS}
    try:
        with open(input_path, encoding="utf-8") as fin:
            for line in tqdm(fin, total=n_rows, desc=f"{input_path.name} rows", file=sys.stdout):
                row = json.loads(line)
                for lang in ALL_LANGS:
                    text = (row.get(lang) or "").replace("\n", " ").strip()
                    if not text:
                        empty_counts[lang] += 1
                    handles[lang].write(text + "\n")
    finally:
        for f in handles.values():
            f.close()

    any_empty = {lang: c for lang, c in empty_counts.items() if c}
    print(f"[build_parallel_corpus] Wrote {n_rows} lines x {len(ALL_LANGS)} languages "
          f"-> {out_dir}/{test_name}.<lang> "
          f"({'no empty lines' if not any_empty else f'empty lines: {any_empty}'})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--fresh", action="store_true", help="Rebuild even if the output already exists.")
    args = parser.parse_args()

    input_path, out_base, kind = DATASETS[args.dataset]
    if not input_path.exists():
        raise SystemExit(f"{input_path} does not exist.")

    if kind == "pairs":
        out_src = RAW_DIR / f"{out_base}.src"
        out_tgt = RAW_DIR / f"{out_base}.tgt"
        if out_src.exists() and out_tgt.exists() and not args.fresh:
            print(f"[build_parallel_corpus] {out_src.name}/{out_tgt.name} already exist, skipping "
                  f"(--fresh to rebuild).", flush=True)
            return
        build_pairs(input_path, out_base, args.workers)
    else:
        out_dir = TEST_DIR / out_base
        if out_dir.exists() and any(out_dir.iterdir()) and not args.fresh:
            print(f"[build_parallel_corpus] {out_dir} already has output, skipping "
                  f"(--fresh to rebuild).", flush=True)
            return
        build_perlang(input_path, out_base)


if __name__ == "__main__":
    main()
