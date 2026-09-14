# tokenizer: one SentencePiece model for 23 languages

This step builds the corpus the tokenizer learns from and trains SentencePiece models of seven sizes. The chosen model
has a **128k** vocabulary. The plain-language write-up, with all numbers and plots, is
[`../docs/tokenizer.md`](../docs/tokenizer.md). The analysis code is in [`analysis/`](analysis/README.md).

## Design

- **Algorithm:** SentencePiece **unigram** (not BPE), a better fit for morphologically rich Indic languages.
- **One joint vocabulary** for all 23 languages, shared by source and target.
- **`character_coverage=1.0`:** with 8 or more scripts, dropping rare characters could silently gut a script.
- **23 reserved symbols:** the language tags `<en>`, `<hi>`, ... They are never split.
- **Normalization:** `nmt_nfkc`, the same in the corpus-law and fertility scripts.
- **Tokenizer corpus** (`prepare_corpus.py`): per Indic language, 100% of bpcc-human plus a script-filtered,
  deduplicated bpcc-mined top-up to **1,000,000** sentences. English uses the same rule with a **3,000,000** cap. After
  deduplication, bpcc-human has only about 305,000 unique English sentences, because `ilci` repeats the same English
  sentence for many languages. So English needs the mined top-up too.

This corpus is deliberately different from the translation model's training mixtures. A tokenizer needs coverage,
not training-signal balance. IN22 never touches tokenizer training.

## Run

```bash
sbatch -p <partition> scripts/slurm/tokenizer_prepare_corpus.sh   # Job5: build the corpus
sbatch -p <partition> scripts/slurm/tokenizer_spm_sweep.sh        # Job6: train 7 vocabulary sizes
# one size directly:
python tokenizer/train_spm.py --vocab-size 128000
```

Job6 trains 16k, 32k, 64k, 96k, 128k, 192k and 256k one after another in a single job. It needs a lot of RAM (unigram
training is memory-bound) and no GPU.

## Outputs (not tracked by git)

| Path | Contents |
|---|---|
| `tokenizer/corpus/train.src`, `train.tgt`, `dev.src`, `dev.tgt` | tagged parallel text from bpcc-human |
| `tokenizer/corpus/spm_train_corpus_mixed.txt` | the SentencePiece training text, one sentence per line |
| `tokenizer/sweep/mixed/spm_<size>.model` and `.vocab` | the trained models; `spm_128000.model` is the one used everywhere |

## The vocabulary-size decision

Fertility is the average number of subwords per word (lower is better). Measured on IN22-Gen and IN22-Conv combined:

| Vocabulary | Macro fertility | Worst case | Embedding cost at d=512 |
|---|---:|---|---:|
| 16k | 2.927 | Malayalam 4.261 | 4% |
| 32k | 2.417 | Sanskrit 3.556 | 8% |
| 64k | 2.061 | Sanskrit 3.155 | 17% |
| 96k | 1.905 | Sanskrit 2.925 | 25% |
| **128k** | **1.815** | Sanskrit 2.789 | **34%** |
| 192k | 1.717 | Sanskrit 2.611 | 50% |
| 256k | 1.675 | Sanskrit 2.506 | 67% |

The gain flattens past 128k (96k to 128k: 4.7%; 128k to 192k: 5.4%; 192k to 256k: 2.4%), while the embedding table keeps
growing. Our 128k tokenizer also beats every external tokenizer compared (IndicTrans2 3.897, Gemma family 3.268,
Sarvam-30B 2.375). See [`../docs/tokenizer.md`](../docs/tokenizer.md).

Next: [`../training/README.md`](../training/README.md).
