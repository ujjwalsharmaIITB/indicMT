# Data

This page explains where the training data comes from, how it is cleaned and split, and how it is turned into
one multilingual dataset. The code is in [`../data/`](../data/); the commands are in
[`../data/README.md`](../data/README.md).

IndicMT covers English and the 22 Scheduled Indian languages. That gives **44 translation directions**
(22 languages, each to and from English).

## 1. Where the data comes from

All training data comes from **BPCC** (Bharat Parallel Corpus Collection), published by AI4Bharat as
`ai4bharat/BPCC` on Hugging Face. It is a gated dataset: you need an account with access approved, and a token.

BPCC has several named subsets ("configs"). They fall into two kinds:

- **bpcc-human** is the gold subset. People wrote or translated these sentence pairs. Configs used:
  `bpcc-seed-latest` and `ilci`.
- **bpcc-mined** is the synthetic subset. These pairs were found automatically by mining comparable web text.
  They are noisier. Configs used: `comparable`, `samanantar-filtered` and `nllb-filtered`.

Every row in every config has the same four fields: `src`, `tgt`, `src_lang`, `tgt_lang`.

**There is no quality score in the data.** We checked the live dataset schema for all five configs. None has a
LaBSE or SONAR similarity column. The `-filtered` names show that AI4Bharat already filtered the mined data
before publishing it. So we trust that upstream filtering. We still apply the checks we can compute ourselves
(section 3).

### Corpus size

| | bpcc-human | bpcc-mined |
|---|---:|---:|
| Raw sentence pairs | 3,433,146 | 106,097,276 |
| After the script-consistency filter | see section 3 | 105,633,796 |
| After exact-duplicate removal | (done at split time) | 105,347,857 |
| After bidirectional expansion (44 directions) | 6,866,292 | 210,695,714 |

bpcc-human covers all 22 languages in `bpcc-seed-latest`. `ilci` covers 15 of the 22 (it lacks Dogri, Kashmiri,
Maithili, Manipuri, Odia, Sanskrit, Sindhi and Santali).

bpcc-mined has **no rows at all** for five languages: Dogri, Konkani, Manipuri, Santali and Sindhi. For these five,
the human data is the only training source anywhere in the pipeline.

## 2. Downloading

`data/bpcc-human/download_bpcc_human.py` and `data/bpcc-mined/download_bpcc_mined.py` download every language
of every config. They write one JSONL file per (language, config) pair under each folder's `raw/` directory.
These files are generated data and are not tracked by git.

If a download stops, run the script again. It skips files it already saved. Use `--fresh` to start over.

The scripts read the token from the `HF_TOKEN` environment variable, or from the cache made by
`huggingface-cli login`. Nothing else is read, and no token is ever stored in the repository.

## 3. Quality filtering

Every sentence pair goes through two checks before it can reach training.

1. **Exact-duplicate removal** on the `(src, tgt)` pair. For bpcc-mined this is applied across all raw files,
   because the same pair can appear in several configs.
2. **Unicode-script consistency** (`data/common/scripts.py`). The language-tagged side of a pair must be written in
   the script that its BPCC language code implies. A row labelled `hin_Deva` (Hindi) must contain Devanagari text.
   If it is really in Bengali or Latin script, something upstream mislabelled it, and we drop the row. We do not
   try to correct it.

What the script check removed:

- **bpcc-human:** between 0 and 2,783 rows per language. The largest losses were Sindhi (2,783 of about 49k raw
  rows, 5.7%), Manipuri (2,576 of about 112k, 2.3%) and Malayalam (2,199 of about 185k, 1.2%). Most languages lost
  fewer than 100 rows.
- **bpcc-mined:** 463,480 of 106,097,276 raw rows (0.44%), concentrated in a few files. For example
  `ben_Beng_nllb-filtered` lost 112,267 rows and `hin_Deva_nllb-filtered` lost 59,257.
- **Exact duplicates in bpcc-mined:** 285,939 of the 105,633,796 rows that passed the script check (0.27%).

A further check found a real upstream bug in the Santali data. See [`data-issues.md`](data-issues.md).

## 4. Splitting (bpcc-human only)

`data/bpcc-human/create_splits.py` works per language. It draws a small **validation set** of up to 1,000
sentences, capped at 2% of that language's data. The validation sentences come only from `bpcc-seed-latest`.
We avoid `ilci` because it covers a narrow domain and is not a fair general validation signal. All of `ilci` and
the rest of `bpcc-seed-latest` form the training split. Output goes to
`data/bpcc-human/splits/{lang}_train.jsonl` and `{lang}_valid.jsonl`.

**bpcc-mined is not split.** All of it is training data.

The validation set is only used while training, to watch loss and pick checkpoints. Final quality is reported on
**IN22-Gen** and **IN22-Conv**, AI4Bharat's held-out test sets (1,024 and 1,503 sentences). They never touch
training, tokenizer training or the validation split.

## 5. Building the multilingual dataset

`data/bpcc-human/create_multilingual_dataset.py` and `data/bpcc-mined/create_multilingual_dataset.py` turn the
per-language files into one shuffled multilingual dataset each. They do two things.

1. **Bidirectional expansion.** Every `src -> tgt` pair also produces the reverse `tgt -> src` pair. One
   English-Hindi pair becomes both an English-to-Hindi and a Hindi-to-English example.
2. **Language tagging.** The target language is marked by a tag at the start of the *source* sentence. To translate
   into Hindi, the input is `<hi> the source sentence`. To translate into English it is `<en> the source sentence`.
   This is the scheme from Johnson et al. (2016). It lets one model serve all directions.
   The tagged line is built by `tag_source` in `data/common/tokenization.py`. Evaluation and serving encode their sources through the same module, so the model always sees `['▁', '<xx>', ...]`, the format it was trained on.

### Language tags

| Language | BPCC code | Tag | | Language | BPCC code | Tag |
|---|---|---|---|---|---|---|
| English | `eng` | `<en>` | | Manipuri (Meitei) | `mni_Mtei` | `<mni>` |
| Assamese | `asm_Beng` | `<as>` | | Nepali | `npi_Deva` | `<ne>` |
| Bengali | `ben_Beng` | `<bn>` | | Odia | `ory_Orya` | `<or>` |
| Bodo | `brx_Deva` | `<br>` | | Punjabi | `pan_Guru` | `<pa>` |
| Dogri | `doi_Deva` | `<doi>` | | Sanskrit | `san_Deva` | `<sa>` |
| Konkani | `gom_Deva` | `<gom>` | | Sindhi | `snd_Deva` | `<sd>` |
| Gujarati | `guj_Gujr` | `<gu>` | | Santali | `sat_Olck` | `<sat>` |
| Hindi | `hin_Deva` | `<hi>` | | Tamil | `tam_Taml` | `<ta>` |
| Kannada | `kan_Knda` | `<kn>` | | Telugu | `tel_Telu` | `<te>` |
| Kashmiri | `kas_Arab` | `<ks>` | | Urdu | `urd_Arab` | `<ur>` |
| Maithili | `mai_Deva` | `<mai>` | | | | |
| Malayalam | `mal_Mlym` | `<ml>` | | | | |
| Marathi | `mar_Deva` | `<mr>` | | | | |

There are 23 tags (22 Indic and English). They are registered as reserved SentencePiece symbols, so they are never
split into pieces. The tag always names the language of the **target**, never the language the source is written in.

### Examples

One row per direction (from the combined human training file):

```
en -> hi
SRC: <hi> The buyer may not be able to inspect the property before buying it.
TGT: हो सकता है कि खरीदार प्रॉपर्टी खरीदने से पहले उसका निरीक्षण न कर सके।

hi -> en
SRC: <en> यह अब बेसिक प्रोग्रामिंग भाषा से मिलता-जुलता है।
TGT: It is now similar to the BASIC programming language.
```

### Combining without loss

The per-language datasets are combined with `concatenate_datasets`, not `interleave_datasets`. The latter's default
behaviour stops as soon as the *smallest* language runs out, which would silently discard most of every larger
language's data. Bidirectional expansion must exactly double the row count; an assertion in the script and an
automated test check this.

## 6. Running at scale: job scripts and array jobs

Every step that builds a dataset is CPU heavy. Run it as a batch job (see [`../scripts/README.md`](../scripts/README.md))
and not on a shared login node.

A **Slurm array job** runs the same script several times as separate jobs, each with its own index. It suits a fixed
list of independent tasks. In the script, `#SBATCH --array=0-N` declares it, and `$SLURM_ARRAY_TASK_ID` gives the
index:

```bash
#SBATCH --array=0-1
SOURCES=(human mined)
SOURCE=${SOURCES[$SLURM_ARRAY_TASK_ID]}
python mixing/build_mixture.py --source "$SOURCE"
```

One caution. Some clusters limit how many jobs one user can have submitted at once, and **each array task counts
as one submission** even before it runs. A 7-task array for the tokenizer sweep was rejected for this reason.
`--array=0-6%3` only limits how many run at the same time, so it does not help. In that case, loop inside one job
instead. `scripts/slurm/tokenizer_spm_sweep.sh` does this.

## 7. Dataset statistics

`data/dataset_report.py` walks the output of each stage and writes `data/dataset_statistics.md`. It lists, per
stage, the total samples and the counts per language and per language pair. The file is generated and not tracked
by git.

## 8. Automated checks

Run these before using the data downstream:

```
pytest data/tests/ -v
```

They check, among other things, that:

- every one of the 22 languages and English has a tag, and no two share a tag;
- a row whose target is English really starts with `<en> ` (and likewise for every other language);
- the training and validation splits of a language never share an identical sentence pair.

Tests whose input data does not exist yet are skipped, not failed, so you can run the suite at any stage.

## Next step

Mixing: [`mixing.md`](mixing.md).
