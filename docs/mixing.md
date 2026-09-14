# Data mixing

This page explains how the 23-language corpus (22 Indic languages and English) is combined into balanced training
mixtures, and why the balance is computed the way it is.

Code: [`../mixing/build_mixture.py`](../mixing/build_mixture.py). Run it as a batch job:
`scripts/slurm/mixing_build_mixture.sh` (Job4, an array of two tasks). Outputs go to `mixing/output/` and are not
tracked by git.

## 1. The problem

BPCC's languages differ a lot in size. Hindi has over five times Bodo's raw sentence count, and English (the shared
partner of every pair) is larger than any single Indic language. If we trained on the raw data, the model would
get very good at the three or four biggest languages and see little of the smallest ones. Those small languages
(Sindhi, Santali, Bodo, Dogri and others) are exactly what this project is for.

**Temperature sampling** is the standard fix. It boosts small languages and shrinks big ones, without going all the
way to equal shares. Equal shares would over-represent tiny languages relative to how much distinct content they have.

## 2. The formula

For each language `i` with a measured size `n_i`:

```
weight_i  = n_i ^ alpha              (alpha = 0.5)
p_i       = weight_i / sum_j(weight_j)
target_i  = p_i * total_target_size
target_i  = min(target_i, 3 * n_i)   (capped at 3x that language's own raw size)
ratio_i   = target_i / n_i
```

`total_target_size` defaults to the sum of all languages' raw sizes. So the mixture redistributes the corpus. It does
not grow or shrink it. A language with `ratio_i < 1` is undersampled: some of its data is left out. A language with
`ratio_i > 1` is oversampled: its sentences are repeated, up to 3 times.

## 3. The key choice: size is counted in words, not sentences

Most descriptions of temperature sampling count `n_i` in examples (sentence pairs). We count it in **tokens**
(whitespace-separated words) instead.

A language with fewer but longer sentences carries more training signal per example than one with more but shorter
sentences. Counting examples cannot see that. Counting words weighs languages by how much text the model has to
learn from. It is also nearly free to compute (`str.split()`).

On the real bpcc-human data, the choice changes the ratios like this:

| Language | Avg. words per sentence | Example-based ratio | Token-based ratio | Difference |
|---|---:|---:|---:|---:|
| Sanskrit | 10.7 | 2.116 | 2.610 | **+0.494** |
| Malayalam | 10.9 | 1.548 | 1.886 | **+0.338** |
| Kannada | 12.1 | 1.593 | 1.843 | +0.251 |
| Telugu | 12.3 | 1.602 | 1.838 | +0.236 |
| Nepali | 13.2 | 1.718 | 1.904 | +0.187 |
| Konkani | 13.2 | 1.606 | 1.784 | +0.178 |
| ... | ... | ... | ... | ... |
| Sindhi | 20.0 | 3.000 (capped) | 2.780 | **-0.220** |
| Santali | 18.5 | 2.355 | 2.206 | -0.148 |
| Urdu | 19.8 | 1.484 | 1.342 | **-0.141** |
| Hindi | 19.6 | 1.296 | 1.180 | -0.116 |

Short-sentence languages (Sanskrit, Malayalam, Kannada, Telugu) are pushed up. Long-sentence languages (Sindhi,
Urdu, Hindi, Santali) are pulled down. Sindhi is the sharpest case. It hits the 3x cap when counting examples, but
not when counting words, because its sentences are the longest in the corpus. The effect is modest (0.1 to 0.5 in
ratio) but systematic, so it is worth correcting.

## 4. How the sampling is done

The mixed file is made in one streaming pass. The source file is split into byte ranges so the work runs in parallel.
It works the same for bpcc-human (about 3 GB) and bpcc-mined (about 76 GB).

- If `ratio_i <= 1`: keep each row with probability `ratio_i`.
- If `ratio_i > 1`: keep the row `floor(ratio_i)` times, plus one more copy with probability
  `ratio_i - floor(ratio_i)`.

This reaches each language's target in expectation, with ordinary statistical noise. We checked it on a small
synthetic two-language test: a language capped at 3x landed at exactly 3.00x, and an uncapped one landed within 0.1%
of its predicted count. That is precise enough, because the proportions between languages are what matter.

One bug is worth knowing about. The function that reads a byte range of lines (`iter_lines_in_range`) could drop a
line at a chunk boundary. This is fixed. If you have mixtures built by an older version, rebuild them (the data
preparation job takes `FRESH=1`).

## 5. Two separate mixtures

- **`train-gold`** (`mixing/output/train_gold.jsonl`) is temperature-sampled from bpcc-human only.
- **`mined-mixture`** (`mixing/output/mined_mixture.jsonl`) is temperature-sampled from bpcc-mined only, with its own
  per-language counts and ratios. The mined size distribution is very different from the human one.
- **Dev** is not resampled. It is the fixed validation set from the data pipeline (up to 1,000 sentences per language
  from `bpcc-seed-latest`).

The two mixtures are kept separate, not merged. That way the human-to-mined ratio can still be chosen at training
time (with OpenNMT-py's weighted corpora) without redoing this step. The data study, which compares human-only,
mined-only and combined training, uses exactly these two files.

## 6. How English takes part in the formula

If every row is keyed on its own target language, English becomes a full competitor in the formula, and its ratio
applies to every xx-to-en row. That is wrong in spirit. English is the partner in every pair. It is not a language
that needs balancing, and one ratio would then control how much Indic text the encoder sees as input for all 22
languages at once.

**Decision 1: `train-gold` weights only the 22 Indic languages.** Each row is sampled under whichever side is the
Indic language, whatever the direction. So a language's en-to-xx rows and its xx-to-en rows get the *same* ratio,
that language's own. English has no ratio of its own. The target budget is the sum of the 22 Indic languages' raw
tokens only.

**Decision 2 (a consequence): in `train-gold`, both directions of a language get the same number of rows, in
expectation.** This gives the encoder more Indic input and the decoder more Indic output for exactly the languages
that need it.

**`mined-mixture` keeps the original design on purpose.** English stays a weighted contender with its own ratio, so
both variants exist. The `SOURCES` setting in `mixing/build_mixture.py` carries this per source
(`indic_only: True` for human, `False` for mined).

## 7. Results

### train-gold (bpcc-human, Indic languages only)

`alpha = 0.5`, cap 3.0x, target 51,133,939 tokens (the sum of the 22 Indic languages' raw tokens). No language hits
the cap. Even Sindhi, the most under-resourced, needs only 1.625x.

| Language | Raw tokens | Target tokens | Ratio |
|---|---:|---:|---:|
| `<as>` | 2,580,063 | 2,502,518 | 0.970 |
| `<bn>` | 3,354,241 | 2,853,377 | 0.851 |
| `<br>` | 2,518,776 | 2,472,616 | 0.982 |
| `<doi>` | 1,594,666 | 1,967,419 | 1.234 |
| `<gom>` | 2,233,237 | 2,328,249 | 1.043 |
| `<gu>` | 3,231,339 | 2,800,614 | 0.867 |
| `<hi>` | 5,100,076 | 3,518,442 | 0.690 |
| `<kn>` | 2,091,567 | 2,253,190 | 1.077 |
| `<ks>` | 1,654,408 | 2,003,933 | 1.211 |
| `<mai>` | 1,484,123 | 1,898,003 | 1.279 |
| `<ml>` | 1,998,457 | 2,202,467 | 1.102 |
| `<mr>` | 3,239,674 | 2,804,224 | 0.866 |
| `<mni>` | 1,567,885 | 1,950,828 | 1.244 |
| `<ne>` | 1,958,989 | 2,180,610 | 1.113 |
| `<or>` | 1,363,318 | 1,819,117 | 1.334 |
| `<pa>` | 2,868,356 | 2,638,630 | 0.920 |
| `<sa>` | 1,042,774 | 1,590,952 | 1.526 |
| `<sat>` | 1,459,725 | 1,882,337 | 1.290 |
| `<sd>` | 919,349 | 1,493,833 | 1.625 |
| `<ta>` | 2,825,124 | 2,618,670 | 0.927 |
| `<te>` | 2,104,217 | 2,259,994 | 1.074 |
| `<ur>` | 3,943,575 | 3,093,906 | 0.785 |

`<en>` has no row because it is not weighted. The output is **6,953,605 rows** (from 6,866,292 raw bidirectional
rows), a net change of about 1.3%. Decision 2 holds in practice: of those rows, 3,477,128 have English as target
and 3,476,477 do not, a difference of 0.019%.

### mined-mixture (bpcc-mined, all 23 languages)

`alpha = 0.5`, cap 3.0x, target 2,390,937,444 tokens (all 23 languages, English included). **Seven languages hit the
3x cap**: Assamese, Bodo, Kashmiri, Maithili, Nepali, Odia and Sanskrit. Mined data is much more skewed than human
data. Bodo has only 5,365 raw mined tokens. Dogri, Konkani, Manipuri, Santali and Sindhi have no mined rows, so
they do not appear below.

| Language | Raw tokens | Target tokens | Ratio |
|---|---:|---:|---:|
| `<as>` | 4,737,440 | 14,212,320 | 3.000 (capped) |
| `<bn>` | 147,853,281 | 190,279,602 | 1.287 |
| `<br>` | 5,365 | 16,095 | 3.000 (capped) |
| `<en>` | 1,256,702,154 | 554,744,407 | 0.441 |
| `<gu>` | 123,632,781 | 173,997,725 | 1.407 |
| `<hi>` | 191,043,877 | 216,293,423 | 1.132 |
| `<kn>` | 106,682,154 | 161,630,248 | 1.515 |
| `<ks>` | 1,445,940 | 4,337,820 | 3.000 (capped) |
| `<mai>` | 593,771 | 1,781,313 | 3.000 (capped) |
| `<ml>` | 80,587,560 | 140,478,789 | 1.743 |
| `<mr>` | 69,312,294 | 130,281,241 | 1.880 |
| `<ne>` | 19,332,405 | 57,997,215 | 3.000 (capped) |
| `<or>` | 27,206,379 | 81,619,137 | 3.000 (capped) |
| `<pa>` | 56,064,891 | 117,171,568 | 2.090 |
| `<sa>` | 1,250,528 | 3,751,584 | 3.000 (capped) |
| `<ta>` | 87,869,076 | 146,688,065 | 1.669 |
| `<te>` | 118,371,641 | 170,255,279 | 1.438 |
| `<ur>` | 98,245,907 | 155,107,924 | 1.579 |

The output is **213,807,105 rows** (from 210,695,714 raw bidirectional rows).

## Next step

Tokenizer: [`tokenizer.md`](tokenizer.md). The mixtures are used again when the training files are built
([`training.md`](training.md)).
