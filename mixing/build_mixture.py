"""Temperature-sampled data mixing.

Builds a temperature-sampled multilingual mixture from one source
(bpcc-human -> "train-gold", or bpcc-mined -> "mined-mixture").

**Decision 1: train-gold mixes only the 22 Indic languages; English is not a
weighted "language" in that formula at all.** mined-mixture keeps the
original design (all 23 languages, including English, compete for weight).
See docs/mixing.md for the full rationale; short version: temperature
sampling by `tgt_lang` alone treats a row's *direction* as its identity, so
English silently got its own ratio (`ratio_en`) independent of whichever
Indic language it was paired with. Because English is the pivot side of
every bpcc-human pair, that ratio governed the *whole* xx->en direction
(the encoder's exposure to Indic text as input) for all 22 languages at
once, uniformly - a low-resource language's xx->en rows were being
downsampled by English's ratio, not boosted by its own. For train-gold,
`INDIC_ONLY_SOURCES` below resolves every row's sampling key to whichever
side is the *Indic* language, regardless of direction - so a language's
en->xx rows (decoder generates it) and xx->en rows (encoder reads it) both
inherit that language's own ratio. mined-mixture is left as before
(English keeps its own ratio, `<en>` ratio=0.441 etc.) - see the "we will
have both variations" decision recorded in docs/mixing.md.

**Decision 2 (a consequence of decision 1): train-gold's post-mixing row
count for a language is, in expectation, exactly 2x its single-direction
sampled count** - since both directions for language i are scaled by the
same ratio_i, drawn from the same per-language token budget. This
deliberately gives the encoder more Indic-as-input exposure (better
representation/understanding) at the same time it gives the decoder more
Indic-as-output exposure (better generation), for exactly the languages
that most need the boost - rather than the old behavior where xx->en
exposure for every Indic language was capped by English's single ratio.

Formula: for language i with token count n_i (i ranges over the 22 Indic
languages for train-gold; over all 23, including English, for mined-mixture),
    weight_i = n_i ^ alpha         (alpha = 0.5)
    p_i      = weight_i / sum_j(weight_j)
    target_tokens_i = p_i * total_target_tokens
    target_tokens_i = min(target_tokens_i, 3 * n_i)     (3x cap, in TOKENS)
    ratio_i  = target_tokens_i / n_i

`total_target_tokens` defaults to the sum of n_i over exactly the languages
being weighted (Indic-only for train-gold, all 23 for mined-mixture) - i.e.
the mixture redistributes that pool, it doesn't grow or shrink it. Pass
--total-target-tokens to override.

Sampling mechanism (single pass, no need to hold the source file in memory):
for a row keyed to language i with ratio_i:
  - ratio_i <= 1 (undersample): keep the row with probability ratio_i.
  - ratio_i >  1 (oversample, capped at 3x): always keep the row
    floor(ratio_i) times, plus one more copy with probability
    (ratio_i - floor(ratio_i)) - this hits the target token budget in
    expectation without needing to index/re-read specific rows.

This is a Bernoulli/replication scheme, not exact reservoir sampling - the
realized token count per language will vary a little around the target,
which is fine for a mixing ratio (the point is the *proportions*, not
hitting an exact count) - and for train-gold, the two directions of a given
language are sampled with independent random draws at the *same* ratio, so
their row counts match closely (exactly 2x in expectation) rather than
being deterministically bound to each other.

Runs in two passes, both parallelized by splitting the input file into
byte-range chunks (works uniformly for bpcc-human's ~3GB file and
bpcc-mined's ~76GB file):
  Pass 1 - count total tokens per language (cheap, embarrassingly parallel).
  Pass 2 - given the target ratios from pass 1, stream again and
           keep/replicate rows per the scheme above, each chunk writing its
           own output shard; shards are concatenated (not merged/shuffled -
           OpenNMT/training-time shuffling handles final ordering).

Intended to run as a Slurm CPU job (see scripts/slurm/mixing_build_mixture.sh),
not on the login node, given bpcc-mined's size.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common.languages import LANGUAGE_CODES  # noqa: E402

ALPHA = 0.5
CAP = 3.0
SEED = 42
ENGLISH_LANG = "eng_Latn"

SOURCES = {
    "human": {
        "input": Path(__file__).resolve().parents[1] / "data" / "bpcc-human" / "combined_train_dataset.jsonl",
        "output_name": "train_gold.jsonl",
        "indic_only": True,
    },
    "mined": {
        "input": Path(__file__).resolve().parents[1] / "data" / "bpcc-mined" / "concatenated_bpcc_mined_dataset.jsonl",
        "output_name": "mined_mixture.jsonl",
        "indic_only": False,
    },
}
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def word_count(text: str) -> int:
    return len(text.split())


def sampling_key(row, indic_only: bool) -> str:
    """The language identity a row is sampled under. Normally that's just
    tgt_lang (what the decoder produces). Under indic_only mode, an
    xx->en row (tgt_lang=en) resolves to its Indic src_lang instead, so
    both directions of language xx share one ratio - see decisions 1/2 in
    the module docstring."""
    if indic_only and row["tgt_lang"] == ENGLISH_LANG:
        return row["src_lang"]
    return row["tgt_lang"]


def byte_chunks(path: Path, n_chunks: int):
    size = path.stat().st_size
    step = size // n_chunks
    bounds = [i * step for i in range(n_chunks)] + [size]
    return list(zip(bounds[:-1], bounds[1:]))


def iter_lines_in_range(path, start, end):
    """Yields whole lines starting at or after byte offset `start`, stopping
    once the read position reaches `end` (the boundary lines from byte_chunks
    - contiguous, not necessarily line-aligned).

    `start` almost always lands mid-line, since byte_chunks splits purely by
    byte offset - the caller's job is to skip that partial line (the previous
    chunk's own scan already consumed it in full, since a chunk's own loop
    below doesn't stop until it reads a complete line whose end is >= its
    `end` boundary). But when `start` happens to land EXACTLY at a line
    boundary (the byte right before it is '\\n', or start==0), there is no
    partial line to skip - the previous chunk's last full line ended exactly
    at `start`. Unconditionally skipping a line in that case silently drops
    one real line per such boundary - confirmed as a real, reproducible bug
    (not hypothetical): on a 213.8M-line/~21GB file split into 32 chunks, one
    of the 31 internal boundaries happened to land exactly on a line break,
    and the old unconditional-skip version dropped that line - a
    single-file line-count check wouldn't have caught it by luck alone here,
    but a cross-file alignment check did (two independently-chunked files of
    different byte sizes hit this edge case at different, uncorrelated
    boundaries, desyncing their line counts by one from that point on)."""
    with open(path, "rb") as f:
        if start != 0:
            f.seek(start - 1)
            if f.read(1) == b"\n":
                pass  # start already lands exactly at a line boundary - nothing to skip
            else:
                f.readline()  # discard the genuine partial line
        else:
            f.seek(0)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            yield line


def read_shard_with_retry(path: Path, retries=6, base_delay=0.5) -> str:
    """Read a shard file's full content, retrying on FileNotFoundError/OSError.

    Real, observed failure mode on a shared network filesystem: with many
    worker processes writing shard files and the main process then reading
    and unlinking them right after, the main process occasionally sees a
    file as not-yet-existing (or gets a stale-handle error on read/unlink)
    for a shard that the writer process already finished and closed -
    metadata-cache staleness under heavy concurrent I/O, not a logic bug
    (confirmed: the "missing" shard file was present on disk seconds later,
    unchanged). Retries with exponential backoff rather than failing a
    multi-hour job over a transient race. Shared by every script that does
    this chunk-then-concat pattern (this module and training/*.py)."""
    for attempt in range(retries):
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def unlink_with_retry(path: Path, retries=6, base_delay=0.5):
    for attempt in range(retries):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return  # already gone - fine, that's the goal
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def count_chunk(args):
    path, start, end, indic_only = args
    counts = Counter()
    for line in iter_lines_in_range(path, start, end):
        row = json.loads(line)
        if indic_only and row["tgt_lang"] == ENGLISH_LANG:
            # This row's Indic content (the same sentence, tagged) is already
            # counted once via its mirrored en->xx row's tgt field - counting
            # it again here would double-count, and English itself isn't a
            # weighted language under indic_only mode.
            continue
        counts[row["tgt_lang"]] += word_count(row["tgt"])
    return counts


def sample_chunk(args):
    path, start, end, ratios, indic_only, out_path, seed = args
    rng = random.Random(seed)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for line in iter_lines_in_range(path, start, end):
            row = json.loads(line)
            ratio = ratios.get(sampling_key(row, indic_only), 1.0)
            floor_copies = int(ratio)
            frac = ratio - floor_copies
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            for _ in range(floor_copies):
                fout.write(text)
                written += 1
            if rng.random() < frac:
                fout.write(text)
                written += 1
    return out_path, written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES.keys(), required=True)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--cap", type=float, default=CAP)
    parser.add_argument("--total-target-tokens", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cfg = SOURCES[args.source]
    input_path = cfg["input"]
    indic_only = cfg["indic_only"]
    if not input_path.exists():
        raise SystemExit(f"{input_path} does not exist - build it first (see docs/data-pipeline.md).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = byte_chunks(input_path, args.workers)

    mode = "indic-only (English excluded from weighting)" if indic_only else "all 23 languages"
    print(f"[build_mixture] {args.source}: mode={mode}")
    print(f"[build_mixture] Pass 1/2: counting tokens per language across {len(chunks)} chunks of {input_path}")
    with mp.Pool(args.workers) as pool:
        partials = pool.map(count_chunk, [(input_path, s, e, indic_only) for s, e in chunks])
    total_counts = Counter()
    for c in partials:
        total_counts.update(c)

    total_target_tokens = args.total_target_tokens or sum(total_counts.values())

    weights = {lang: n ** args.alpha for lang, n in total_counts.items()}
    weight_sum = sum(weights.values())
    ratios = {}
    print(f"[build_mixture] {args.source}: alpha={args.alpha} cap={args.cap}x total_target_tokens={total_target_tokens:,}")
    print(f"{'lang':12s} {'tag':6s} {'raw_tokens':>14s} {'target_tokens':>14s} {'ratio':>7s} {'capped':>7s}")
    for lang, n in sorted(total_counts.items()):
        p = weights[lang] / weight_sum
        target = p * total_target_tokens
        capped = target > args.cap * n
        target = min(target, args.cap * n)
        ratio = target / n if n else 0.0
        ratios[lang] = ratio
        tag = LANGUAGE_CODES.get(lang[:3], "?")
        print(f"{lang:12s} {tag:6s} {n:>14,} {int(target):>14,} {ratio:>7.3f} {'yes' if capped else 'no':>7s}")

    print(f"[build_mixture] Pass 2/2: sampling rows across {len(chunks)} chunks")
    shard_args = [
        (input_path, s, e, ratios, indic_only, OUTPUT_DIR / f".shard_{args.source}_{i}.jsonl", args.seed + i)
        for i, (s, e) in enumerate(chunks)
    ]
    with mp.Pool(args.workers) as pool:
        results = pool.map(sample_chunk, shard_args)

    out_path = OUTPUT_DIR / cfg["output_name"]
    total_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for shard_path, written in results:
            # Streamed (not read_shard_with_retry's whole-file read) - a mined-mixture
            # shard after 3x oversampling can be large; retry only the open, which is
            # where the filesystem staleness race (see read_shard_with_retry) manifests.
            for attempt in range(6):
                try:
                    with open(shard_path, encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
                    break
                except (FileNotFoundError, OSError):
                    if attempt == 5:
                        raise
                    time.sleep(0.5 * (2 ** attempt))
            unlink_with_retry(shard_path)
            total_written += written

    print(f"[build_mixture] Wrote {total_written:,} rows to {out_path}")


if __name__ == "__main__":
    main()
