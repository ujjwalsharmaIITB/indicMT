# IndicMT documentation

Plain-language write-ups of each part of the pipeline. Each page is self-contained. For commands, see the README in
each directory (linked from the [top-level README](../README.md)).

| Page | What it covers |
|---|---|
| [data.md](data.md) | Where the data comes from (BPCC), quality filtering, splits, language tags, the multilingual dataset. |
| [data-issues.md](data-issues.md) | An upstream data bug we found in `ai4bharat/BPCC` (Santali metadata), with a reproduction. |
| [mixing.md](mixing.md) | Temperature-sampled mixing: the formula, why size is counted in words, the two mixtures, per-language results. |
| [tokenizer.md](tokenizer.md) | The SentencePiece corpus, Zipf and Heaps analysis, the vocabulary-size sweep (why 128k), external tokenizers. |
| [training.md](training.md) | The model, the three data arms, stopping by hand, checkpoint selection, the source-encoding fix (and the English-to-Indic decline it caused), fine-tuning, the architecture study. |
| [evaluation.md](evaluation.md) | How scores are computed, the source-encoding fix, chrF++ and BLEU results, comparison with 12 systems (IndicTrans2, Indic Translate and Sarvam Translate run under the same protocol), per-language results, fine-tuning and architecture results, the LLM judge. |
| [deployment.md](deployment.md) | CTranslate2 conversion, quantization (float32, int16, int8) with a significance test, the API and frontend, the container and the load test. |

## Conventions

- **Job names.** Batch jobs are called Job1, Job2 and so on. The table is in [`../scripts/README.md`](../scripts/README.md).
- **The release model** is the 12+12 model pre-trained on the combined 8:2 (mined to human) data (step 13,600) and
  fine-tuned on human data without replay (step 1,800).
- **Single seed.** Every result comes from one training run per configuration. Two extra seeds of the human-only arm
  reproduce the seed-42 chrF++ curve; the seed variance of the release model itself is not measured.
- **Selection bias.** Checkpoints were selected on the IN22 test sets, so the reported IN22 scores are slightly optimistic.
- **Not tracked.** Data, tokenizer models, checkpoints, translations and score files are generated and are not stored in
  git. Each page and README says where its outputs are written.
