# data: download, clean and split BPCC and IN22

This directory downloads and prepares `ai4bharat/BPCC` (a gated Hugging Face dataset) for English and the 22
Scheduled Indian languages, and downloads the IN22 test sets. The plain-language write-up is
[`../docs/data.md`](../docs/data.md). This file covers the commands.

## Setup

Install the project environment once (see the top-level [README](../README.md)):

```bash
pip install -r requirements-train.txt
```

BPCC and IN22 are gated. Request access on their Hugging Face pages, then authenticate with either:

```bash
export HF_TOKEN=<your token>
# or
huggingface-cli login
```

`data/common/auth.py` reads `HF_TOKEN` first, then the cache from `huggingface-cli login`. No token is ever stored in
the repository.

## Layout

```
data/
  common/languages.py    the 22-language list and the tag dictionary (including <en>)
  common/scripts.py      Unicode script ranges used by the script-consistency filter
  common/auth.py         Hugging Face token resolution
  bpcc-human/            gold, human-translated data (configs: bpcc-seed-latest, ilci)
  bpcc-mined/            mined data (configs: comparable, samanantar-filtered, nllb-filtered)
  in22/                  IN22-Gen and IN22-Conv (test sets) and the fertility corpus
  dataset_report.py      writes data/dataset_statistics.md
  tests/                 pytest checks of the pipeline outputs
```

## Run order

Downloads need internet access, so run them where you have it (for example a login node). The other steps are CPU
heavy. Run them as batch jobs ([`../scripts/README.md`](../scripts/README.md)).

```bash
# 1. Download (re-runnable: finished files are skipped; use --fresh to start over)
python data/bpcc-human/download_bpcc_human.py
python data/bpcc-mined/download_bpcc_mined.py

# 2. Split bpcc-human into train and validation, with quality filtering
python data/bpcc-human/create_splits.py

# 3. Build the multilingual datasets (tag, bidirectional expansion)
sbatch -p <partition> scripts/slurm/data_human_multilingual_dataset.sh   # Job1
sbatch -p <partition> scripts/slurm/data_mined_multilingual_dataset.sh   # Job2

# 4. Statistics
sbatch -p <partition> scripts/slurm/data_dataset_report.sh               # Job3

# 5. IN22 test sets (small; no job needed)
python data/in22/download_in22.py
python data/in22/build_fertility_corpus.py
```

## What each step writes

All outputs are generated, large, and **not tracked by git**.

| Step | Output |
|---|---|
| `download_bpcc_human.py` | `data/bpcc-human/raw/{lang}_{config}_bpcc_human.jsonl` |
| `download_bpcc_mined.py` | `data/bpcc-mined/raw/` (one file per language and config) |
| `create_splits.py` | `data/bpcc-human/splits/{lang}_train.jsonl` and `{lang}_valid.jsonl` |
| `bpcc-human/create_multilingual_dataset.py` | `data/bpcc-human/combined_train_dataset.jsonl` and `concatenated_valid_dataset.jsonl` |
| `bpcc-mined/create_multilingual_dataset.py` | `data/bpcc-mined/concatenated_bpcc_mined_dataset.jsonl` |
| `dataset_report.py` | `data/dataset_statistics.md` |
| `in22/download_in22.py` | `data/in22/raw/in22_gen.jsonl` and `in22_conv.jsonl` |
| `in22/build_fertility_corpus.py` | `data/in22/combined_fertility_eval.jsonl` (IN22-Gen and Conv combined, for the tokenizer study) |

## Quality filtering

BPCC has no similarity-score column in any config (we checked the schema). The `-filtered` mined configs are trusted
as already filtered upstream. The pipeline applies only what it can compute:

- exact-duplicate removal on the `(src, tgt)` pair;
- a Unicode-script consistency check (`data/common/scripts.py`): the language-tagged side of a pair must be written in the
  script implied by its BPCC language code.

For bpcc-human the duplicate and script checks run in `create_splits.py`. For bpcc-mined they run inside its
`create_multilingual_dataset.py`. Details and numbers are in [`../docs/data.md`](../docs/data.md). An upstream Santali
metadata bug and our workaround are in [`../docs/data-issues.md`](../docs/data-issues.md).

## Splits

Per language, a validation set of up to 1,000 sentences (at most 2% of the language's data) is drawn only from
`bpcc-seed-latest`. Everything else is training data. bpcc-mined is training only. IN22 is never used for training.

## Tests

```bash
pytest data/tests/ -v
```

The tests check the language tags (including `<en>`), the script detection, and that train and validation splits do
not overlap. Tests whose input data is missing are skipped, so you can run them at any stage.
