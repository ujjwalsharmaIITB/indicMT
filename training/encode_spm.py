"""Training data prep, step 2: apply the frozen 128k SentencePiece model
(`tokenizer/sweep/mixed/spm_128000.model`, the vocabulary-size decision is
in docs/tokenizer.md) to the raw src/tgt files
`training/build_parallel_corpus.py` writes under `128k-spm/data/raw/`,
producing OpenNMT-py-ready, space-separated-subword-piece text under
`128k-spm/data/`.

Each raw `<name>.raw.{src,tgt}` pair becomes `<name-without-'.raw'>.128k.spm.{src,tgt}`
(e.g. `bpcc-human.train.temp-sampled.0.5.raw.src` ->
`bpcc-human.train.temp-sampled.0.5.128k.spm.src`), preserving line alignment
1:1 - SPM encoding is a per-line transform, never merges or drops lines.

The two test sets (in22gen-test, in22conv-test) are multi-parallel, kept as
one raw file per language under `128k-spm/data/raw/test/{in22-gen,in22-conv}/`
(see `build_parallel_corpus.py`) - each becomes its own tokenized file under
`128k-spm/data/test/{in22-gen,in22-conv}/<name>.128k.spm.<short-tag>`, small
enough (1,024 / 1,503 lines) that no byte-chunking is needed, one plain
`sp.encode()` call per file.

Language tags (`<xx>`) are registered as USER_DEFINED pieces in the 128k
model (verified: `tokenizer/train_spm.py`'s `user_defined_symbols=` at
training time - checked directly against the model's proto, not assumed),
so they encode as a single atomic piece each, never subword-split.

Byte-chunked in parallel like `build_parallel_corpus.py` (same
`byte_chunks`/`iter_lines_in_range` helpers, imported from
`mixing/build_mixture.py`), with each chunk's lines SPM-encoded via a single
batched `sp.encode(list_of_lines, out_type=str)` call rather than one Python
call per line, since SentencePiece's batch API amortizes the Python/C++
call overhead - meaningful at bpcc-mined-train's ~214M lines per side.

Run via `scripts/slurm/training_build_dataset.sh` (SLURM, CPU-only), same as
`build_parallel_corpus.py` - not on the login node.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mixing"))
from build_mixture import byte_chunks, iter_lines_in_range  # noqa: E402
from build_parallel_corpus import ALL_LANGS, read_shard_with_retry, short_tag, unlink_with_retry  # noqa: E402

RAW_DIR = ROOT / "128k-spm" / "data" / "raw"
OUT_DIR = ROOT / "128k-spm" / "data"
TEST_RAW_DIR = RAW_DIR / "test"
TEST_OUT_DIR = OUT_DIR / "test"
MODEL_PATH = ROOT / "tokenizer" / "sweep" / "mixed" / "spm_128000.model"

# name -> (raw/output basename, kind) - "pairs" mirrors build_parallel_corpus.py's
# src/tgt files; "perlang" mirrors its per-language multi-parallel test output.
DATASETS = {
    "bpcc-human-train": ("bpcc-human.train.temp-sampled.0.5", "pairs"),
    "bpcc-mined-train": ("bpcc-mined.train.temp-sampled.0.5", "pairs"),
    "bpcc-human-dev": ("bpcc-human.dev", "pairs"),
    "in22gen-test": ("in22-gen", "perlang"),
    "in22conv-test": ("in22-conv", "perlang"),
}


ENCODE_BATCH_SIZE = 100_000  # sub-batch within a chunk, bounds peak memory per worker


def encode_chunk(args):
    """Streams through its byte range in ENCODE_BATCH_SIZE-line sub-batches rather
    than materializing the whole chunk's lines + encoded pieces at once. A chunk
    of a ~213M-line file with workers=32 is ~6.7M lines - loading that (and its
    full encoded output) into memory all at once, x32 concurrent processes,
    is what OOM-killed a worker mid-run on bpcc-mined-train's tgt side (Indic
    text -> more subwords -> more memory than the same file's mostly-English
    src side, which is why src finished fine and tgt didn't). Peak memory per
    worker is now bounded by ENCODE_BATCH_SIZE regardless of chunk or corpus size."""
    model_path, path, start, end, shard_out = args
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    n = 0
    batch = []
    with open(shard_out, "w", encoding="utf-8") as f:
        for line in iter_lines_in_range(path, start, end):
            batch.append(line.decode("utf-8").rstrip("\n"))
            if len(batch) >= ENCODE_BATCH_SIZE:
                for toks in sp.encode(batch, out_type=str):
                    f.write(" ".join(toks) + "\n")
                n += len(batch)
                batch = []
        if batch:
            for toks in sp.encode(batch, out_type=str):
                f.write(" ".join(toks) + "\n")
            n += len(batch)
    return shard_out, n


def encode_file(model_path: Path, in_path: Path, out_path: Path, workers: int):
    tmp_dir = out_path.parent / f".{out_path.name}.shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunks = byte_chunks(in_path, workers)
    shard_args = [
        (model_path, in_path, s, e, tmp_dir / f"shard_{i:03d}.txt")
        for i, (s, e) in enumerate(chunks)
    ]
    results = [None] * len(shard_args)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(encode_chunk, args): i for i, args in enumerate(shard_args)}
        with tqdm(total=len(futures), desc=f"{in_path.name} chunks", file=sys.stdout) as pbar:
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
                pbar.update(1)

    total = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for shard_out, n in tqdm(results, desc=f"{in_path.name} concat", file=sys.stdout):
            fout.write(read_shard_with_retry(shard_out))
            total += n
            unlink_with_retry(shard_out)
    tmp_dir.rmdir()
    print(f"[encode_spm] Wrote {total} lines -> {out_path.name}", flush=True)


def encode_test_set(model_path: Path, test_name: str, fresh: bool):
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    in_dir = TEST_RAW_DIR / test_name
    out_dir = TEST_OUT_DIR / test_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in tqdm(ALL_LANGS, desc=f"{test_name} languages", file=sys.stdout):
        tag = short_tag(lang)
        in_path = in_dir / f"{test_name}.{tag}"
        out_path = out_dir / f"{test_name}.128k.spm.{tag}"
        if not in_path.exists():
            raise SystemExit(f"{in_path} does not exist - run build_parallel_corpus.py first.")
        if out_path.exists() and not fresh:
            continue
        with open(in_path, encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]
        pieces = sp.encode(lines, out_type=str)
        with open(out_path, "w", encoding="utf-8") as f:
            for toks in pieces:
                f.write(" ".join(toks) + "\n")
    print(f"[encode_spm] Wrote {out_dir}/{test_name}.128k.spm.<lang> for all {len(ALL_LANGS)} languages",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--fresh", action="store_true", help="Re-encode even if the output already exists.")
    args = parser.parse_args()

    name, kind = DATASETS[args.dataset]

    if kind == "perlang":
        encode_test_set(args.model, name, args.fresh)
        return

    raw_base = name + ".raw"
    out_base = name + ".128k.spm"

    for side in ("src", "tgt"):
        in_path = RAW_DIR / f"{raw_base}.{side}"
        out_path = OUT_DIR / f"{out_base}.{side}"
        if not in_path.exists():
            raise SystemExit(f"{in_path} does not exist - run build_parallel_corpus.py first.")
        if out_path.exists() and not args.fresh:
            print(f"[encode_spm] {out_path.name} already exists, skipping (--fresh to redo).", flush=True)
            continue
        print(f"[encode_spm] Encoding {in_path} -> {out_path}", flush=True)
        encode_file(args.model, in_path, out_path, args.workers)


if __name__ == "__main__":
    main()
