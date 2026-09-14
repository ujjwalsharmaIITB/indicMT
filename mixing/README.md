# mixing: temperature-sampled data mixtures

This step rebalances the training data so that small languages are not drowned out by large ones. It reads the
multilingual datasets from [`../data/`](../data/README.md) and writes two mixtures. The full explanation, with the
per-language results, is in [`../docs/mixing.md`](../docs/mixing.md).

## The idea

BPCC's languages differ a lot in size. **Temperature sampling** gives each language a share proportional to
`n_i ^ alpha` instead of `n_i`. With `alpha = 0.5` that boosts small languages and shrinks large ones, without going to
equal shares.

```
weight_i = n_i ^ alpha          (alpha = 0.5)
p_i      = weight_i / sum_j(weight_j)
target_i = min(p_i * total, 3 * n_i)       (3x cap on repetition)
ratio_i  = target_i / n_i
```

Key decisions:

- **`n_i` is a word count, not a sentence count.** Languages with short sentences carry less text per example, so
  counting sentences under-weights them. Counting words moves some ratios by 0.1 to 0.5.
- **Cap of 3x.** A language is never repeated more than three times.
- **Two separate mixtures,** not one merged file: `train-gold` (from bpcc-human) and `mined-mixture` (from bpcc-mined). The
  human-to-mined ratio can then be chosen at training time without redoing this step.
- **English handling.** In `train-gold`, only the 22 Indic languages are weighted, and both directions of a language
  share its ratio. `mined-mixture` keeps English as a weighted language.

## How rows are sampled

`build_mixture.py` makes one streaming pass, split into byte ranges that run in parallel:

- `ratio <= 1`: keep each row with probability `ratio`.
- `ratio > 1`: keep the row `floor(ratio)` times, plus one more copy with probability `ratio - floor(ratio)`.

## Results in short

| Mixture | Rows | Notes |
|---|---:|---|
| `train-gold` (from 6,866,292 raw rows) | 6,953,605 | No language hits the 3x cap. Highest ratio: Sindhi, 1.625. |
| `mined-mixture` (from 210,695,714 raw rows) | 213,807,105 | Seven languages hit the cap: Assamese, Bodo, Kashmiri, Maithili, Nepali, Odia, Sanskrit. |

Five languages (Dogri, Konkani, Manipuri, Santali, Sindhi) have no mined data, so `train-gold` is their only source.

## Run

Input files must exist first (see [`../data/README.md`](../data/README.md)). Job4 is an array of two tasks, one per source:

```bash
sbatch -p <partition> scripts/slurm/mixing_build_mixture.sh
# or directly:
python mixing/build_mixture.py --source human --workers 16
python mixing/build_mixture.py --source mined --workers 16
```

Options: `--alpha` (default 0.5), `--cap` (default 3.0), `--total-target-tokens`, `--workers`, `--seed`.

## Outputs

Written to `mixing/output/` (not tracked by git; the mined file is tens of GB):

- `train_gold.jsonl`
- `mined_mixture.jsonl`
- a printed per-language table of raw tokens, target tokens and ratio (keep the job log if you want it)

Next: [`../tokenizer/README.md`](../tokenizer/README.md) (the tokenizer has its own corpus) and
[`../training/README.md`](../training/README.md) (which consumes these mixtures).
