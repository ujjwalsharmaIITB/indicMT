# tokenizer/analysis: corpus laws and fertility

Scripts, config and figures behind the vocabulary-size decision. The results are explained in
[`../../docs/tokenizer.md`](../../docs/tokenizer.md).

```
analysis/
  scripts/corpus_laws.py     Zipf and Heaps fits per language (and for the union)
  scripts/fertility.py       fertility of our 7 models and of external tokenizers
  scripts/plot_fertility.py  plots from fertility.csv
  configs/spm_sweep.yaml     the exact SentencePiece flags and the 7 vocabulary sizes
  plots/                     three figures used in the docs
```

## 1. Corpus laws (`corpus_laws.py`, Job7)

- **Input:** `data/bpcc-human/combined_train_dataset.jsonl`, target-side text, deduplicated per language.
- **Words:** whitespace-separated tokens after NFKC normalization; punctuation-only and digit-only tokens are dropped.
- **Zipf fit:** least squares on log-log ranks 10 to 10,000. It tells how skewed word frequencies are.
- **Heaps fit:** least squares on log-spaced subsamples (0.1% to 100%), averaged over 3 seeds. It tells how fast the
  vocabulary grows with corpus size, and gives the extrapolated vocabulary at 10 times the corpus.
- **Output:** `results/corpus_laws.csv` and per-language plots in `results/plots/`.

Headline: every language has a hapax rate above 48% and a Zipf fit R-squared of at least 0.982, so subwords are
justified. Agglutinative languages grow vocabulary fastest (mean Heaps beta 0.716), then Indo-Aryan (0.646), then English
(0.609). Sanskrit is the outlier at 0.797. The union of all 23 languages extrapolates to about 14.0 million word forms.

## 2. Fertility (`fertility.py`)

- **Held-out text:** `data/in22/combined_fertility_eval.jsonl`, IN22-Gen and IN22-Conv combined (2,527 multi-parallel
  sentences, the same content in all 23 languages, so languages compare fairly).
- **Fertility:** subwords produced divided by words consumed, using the same word definition as `corpus_laws.py`.
- **Case 1:** our seven models `tokenizer/sweep/mixed/spm_<size>.model`.
- **Case 2:** external tokenizers: IndicTrans2 (its two SentencePiece models, told apart by vocabulary size), Gemma 3,
  Gemma 4, Sarvam-Translate, Indic-Translate and Sarvam-30B. These are downloaded from Hugging Face.
- **Output:** `results/fertility.csv` (one row per tokenizer and language).

`plot_fertility.py` turns that file into the per-language figures. Run it after `fertility.py`.

## Run

```bash
sbatch -p <partition> scripts/slurm/tokenizer_corpus_laws.sh      # Job7
python tokenizer/analysis/scripts/fertility.py                     # needs the trained models and internet
python tokenizer/analysis/scripts/plot_fertility.py
```

## Outputs

Everything under `tokenizer/analysis/results/` is generated and **not tracked by git**. Only the three figures in
`plots/` are kept, as documentation:

- `plots/union_heaps.png`: Heaps fit for the union of all 23 languages.
- `plots/fertility_sweep_per_language.png`: fertility per language for each vocabulary size.
- `plots/fertility_external_per_language.png`: our tokenizers against external ones.
