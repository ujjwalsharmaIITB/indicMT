# llm_judge: LLM-as-judge scoring

chrF++ and BLEU compare surface text. This judge adds what they cannot see: fluency and adequacy scored separately, wrong
language or script, hallucination and negation errors. A strong language model reads each source sentence and
translation and scores it. The results are in [`../../docs/evaluation.md`](../../docs/evaluation.md).

## Protocol

- **Judge:** Gemma 4 31B-it, run offline with vLLM, tensor-parallel over two 48 GB GPUs (Job19). Temperature 0, seed 42.
- **Items:** all **55,594** English-to-Indic sentences of IN22-Gen and IN22-Conv, for one checkpoint.
- **Scores (1 to 5):** *fluency* (does the target read correctly on its own?), *adequacy* (is the meaning there?) and
  *overall* (what a professional reviewer would give). The judge also flags whether the output is in the right language.
- **Two variants:**
  - **V2** (the headline) sees the human reference.
  - **V1** is reference-free. It is a control that checks whether V2 just rewards overlap with the reference.
- **Perfect** means fluency = adequacy = overall = 5.
- **Prompt:** [`PROMPT.md`](PROMPT.md) holds the rubrics, the system prompts, the user templates, the output contract and
  the bias controls. `judge.py` reads it at run time and records its SHA-256 in every result, so a run cannot silently
  change prompts. Resuming with a different prompt, model or seed is refused.
- **Failures:** items whose output cannot be parsed are retried once with a repair message. Failed items are never imputed
  and never enter a mean. An empty hypothesis is scored automatically.
- **Confidence cap:** for languages the judge reads poorly (Santali, Manipuri, Kashmiri, Sindhi, Dogri, Bodo, Konkani,
  Maithili) the judge must not claim high confidence.

## Sanity checks

Run these before trusting any number (`--sanity`):

- **Mismatch:** pair each source with another sentence's translation. Adequacy must be at most 2. Result: adequacy 1.00
  (pass).
- **Identity:** judge the human reference itself, reference-free. Expected overall 4.7 or higher. Result: overall 4.62, a
  perfect 5/5/5 on about 79% of items. It is below the expected value because the judge flags real errors in the IN22
  references. Santali (identity overall 3.88) and Manipuri (4.02) are unreliable for this judge.

## Results

Being re-run. The results published earlier were computed on translations produced with an older source encoding
that differed from training (see [`../../docs/evaluation.md`](../../docs/evaluation.md), section 2). They are
withdrawn, and the judge is being re-run on the release model's corrected translations. The sanity checks above test
the judge itself and still stand.

## Run

The judge needs its own environment (vLLM and a recent `transformers`) and a local copy of the model.

```bash
# 1. environment (Job16) and model download (Job17), then a checksum check (Job18)
sbatch -p <partition> scripts/slurm/llm_judge_setup_env.sh
sbatch -p <partition> --export=ALL,REPO=google/gemma-4-31B-it,JUDGE_MODEL_DIR=/path/to/gemma-4-31B-it \
  scripts/slurm/llm_judge_download_model.sh
sbatch -p <partition> --export=ALL,JUDGE_MODEL_DIR=/path/to/gemma-4-31B-it scripts/slurm/llm_judge_verify_model.sh

# 2. judge one evaluated checkpoint (Job19); evaluation/results/<arm>/model_step_<N>/ must exist
sbatch -p <partition> --export=ALL,JUDGE_MODEL_DIR=/path/to/gemma-4-31B-it scripts/slurm/llm_judge.sh \
  --backend vllm --model /path/to/gemma-4-31B-it --arm bpcc-ft-combined-human --step 1800 --variant both
```

Set `JUDGE_MODEL_DIR` to the model directory. Compute nodes often have no internet, so the job runs offline. Resubmitting
the same command resumes: finished items are skipped. Useful options: `--limit-per-direction N` (sample), `--langs hi ta`,
`--variant V1|V2|both`, `--sanity`, `--retry-failed`. `--backend echo` is an offline stub that tests the whole pipeline
without a GPU. `--backend openai` talks to any OpenAI-compatible endpoint.

Tests (no GPU needed): `pytest evaluation/llm_judge/test_judge.py`.

## Outputs

`evaluation/llm_judge/results/<judge>/<arm>/model_step_<N>/` (not tracked by git; one run is about 95 MB):

- `results.jsonl`: one line per item per variant, appended after each chunk.
- `summary.json` and `summary.md`: aggregates, rewritten after each chunk.
- `run_meta.json`: judge model, backend, seed, prompt hash, options, timestamps.
- `sanity/`: with `--sanity`, the item files, `sanity.json` and `sanity.md`.

## Why two GPUs: tensor parallelism in brief

The judge's weights (62.6 GB in 16-bit) do not fit on one 48 GB GPU. **Tensor parallelism** cuts every big weight matrix
into pieces and gives one piece to each GPU. Both GPUs see the same input and work on every layer at the same time. This
differs from putting some layers on one GPU and the rest on another. Where partial results must be combined, the GPUs add
them in one fast exchange (an *all-reduce*).

- In a feed-forward block, the first matrix is split by columns and the second by rows. Each GPU computes whole neurons,
  applies the nonlinearity locally, and the all-reduce *sums* the partial outputs. The result is exact, because a
  nonlinearity is never applied to a partial sum.
- Attention splits by heads. Each head is computed entirely on one GPU, so softmax needs no communication, and the key-value
  cache is halved per GPU too. The output matrix is split by rows and one all-reduce rebuilds the answer.
- With Gemma 4 31B-it (60 layers, hidden size 5,376, 32 query heads) at two GPUs, each GPU holds about 31.3 GB of weights
  and 16 query heads. vLLM is asked to use 90% of each card, leaving roughly 12 GB for cache and activations.

These per-GPU numbers are estimates from the model configuration, not measurements.
