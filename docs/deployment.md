# Deployment: quantization and serving

This page explains how a trained checkpoint becomes a small CPU-served model: conversion to CTranslate2,
quantization, the API and frontend, and what has and has not been verified. Code and commands are in
[`../deployment/`](../deployment/README.md).

The model served is the **current best model** (12+12, combined pre-training, then human fine-tuning without replay,
step 1,800). The release model will be chosen after the deeper architectures finish fine-tuning, and the numbers
below will be updated then. Weights will be linked from `<WEIGHTS_URL>`.

## 1. Why CPU serving

The goal is a model that runs without dedicated GPU infrastructure. Training produces float32 weights (4 bytes each).
The question is how much of that can be given up on a CPU, in size and latency, before translation quality moves.

## 2. Background: what quantization is

A trained weight is a real number. How it is stored changes the space it takes and how the processor computes with it.

- **float32** uses 32 bits per number (1 sign, 8 exponent, 23 mantissa bits). It has a wide range and about 7 digits.
- **float16** keeps the same structure in 16 bits. It is called reduced precision, not quantization.
- **int8** is a plain 8-bit integer with *no exponent*. A value only means something together with a separate scale
  factor for its tensor. Converting float32 weights to int8 plus a scale is what we mean by **quantization**.
  **int16** is the same idea on a finer grid (about 65,000 steps instead of 254).

For one tensor spanning -2.4 to +2.4, the scale is `max(|x|) / 127`, about 0.0189. Each weight is stored as
`q = round(x / scale)` and recovered as `q * scale`. The small gap is the quantization error. Matrix multiplies then run
in cheap integer arithmetic, and the scale is applied afterwards. Network weights are individually redundant, so a
small error per weight rarely changes the output. That is why quantization tends to be cheap in practice.

What we did **not** test: float16 and int8_float16 (the converter supports them, but our CPUs do not; the supported CPU
types were `float32`, `int16`, `int8_float32` and `int8`); quantization-aware training (all quantization here is
applied after training); and per-channel against per-tensor scales (we use the converter's default).

## 3. Conversion to CTranslate2

`deployment/convert_to_ct2.py` converts one OpenNMT-py checkpoint into three CTranslate2 model directories
(`float32`, `int16`, `int8`) with CTranslate2's `OpenNMTPyConverter`. We found and fixed two compatibility issues:

1. **`weights_only` loading.** OpenNMT-py's release path calls `torch.load(..., weights_only=True)`. That rejects the
   checkpoint's `opt` field (a standard `argparse.Namespace`, present in every OpenNMT-py checkpoint). We use
   CTranslate2's own `unsafe_deserialization=True` flag. That is legitimate here because the checkpoint is
   self-trained and trusted. Do not use it on untrusted downloads.
2. **Flash attention label.** OpenNMT-py 3.5.1 chose flash attention automatically during training
   (`self_attn_type == "scaled-dot-flash"`, never set in a config). CTranslate2's converter only accepts
   `"scaled-dot"`. We checked in OpenNMT-py's attention module that this label only selects the attention *kernel*
   and does not change the learned weights. So the script relabels it on a patched temporary copy before conversion.
   The original checkpoint is never changed.

`deployment/ct2_eval.py` then runs the full 44-direction IN22-Gen evaluation directly through
`ctranslate2.Translator.translate_batch` for each variant. That is the same code path the API uses. It reports
corpus-level chrF++ and BLEU, p50 and p95 single-sentence latency, batch-32 throughput, and model size. Job20 runs the
whole sequence.

## 4. Results for the current best model

CPU only, 8 intra-op threads, beam 5, IN22-Gen:

| Variant | en→xx chrF++ | en→xx BLEU | xx→en chrF++ | xx→en BLEU | p50 latency (ms) | p95 latency (ms) | Throughput (sent/s, batch 32) | Size |
|---|---|---|---|---|---|---|---|---|
| float32 | 46.04 | 16.11 | 53.87 | 28.82 | 399.2 | 743.8 | 5.9 | 630.4 MB |
| int16 | 45.91 | 15.91 | 53.79 | 28.71 | 367.1 | 690.3 | 8.4 | 323.2 MB |
| int8 | 45.90 | 15.89 | 53.77 | 28.72 | 374.2 | 687.8 | 8.4 | 170.6 MB |

CTranslate2 scores this model 0.68 chrF++ higher on en→xx than `onmt_translate` does (46.04 against 45.36; xx→en 53.87
against 53.73). The cause has not been isolated, so the headline numbers in these docs stay the `onmt_translate` ones.
Sizes are byte-identical across checkpoints, because quantized size depends on the architecture, not the weights.

### Is the int8 loss real? A significance test

We regenerated the translations with each variant, saved them, and compared with sacreBLEU's **paired bootstrap**
(1,000 resamples, chrF++, float32 as baseline). Job21 runs this (`evaluation/quant_significance.py`). A paired
bootstrap resamples sentences and asks how often the difference would appear by chance. Pooled scores are corpus-level
chrF++ over the concatenated directions of a bucket, so they differ slightly from the macro averages. **Bonferroni**
correction divides the 0.05 threshold by 22, because 22 directions are tested at once.

| Bucket | Variant | Pooled chrF++, float32 → variant | p | Macro Δ | Directions p < 0.05 | After Bonferroni |
|---|---|---|---|---|---|---|
| en→xx | int16 | 46.19 → 46.19 | 0.210 | −0.006 | 3/22 | 1/22 |
| en→xx | int8 | 46.19 → 46.06 | 0.001 | −0.134 | 16/22 | 1/22 |
| xx→en | int16 | 53.87 → 53.87 | 0.421 | +0.000 | 3/22 | 1/22 |
| xx→en | int8 | 53.87 → 53.81 | 0.004 | −0.061 | 1/22 | 0/22 |

**int16 is indistinguishable from float32 in both directions. int8 costs a small but statistically significant amount
in both:** −0.13 chrF++ English to Indic (16 of 22 directions individually significant, all deltas between −0.26 and
+0.01) and −0.06 Indic to English. This cost is far below the gap to any neighbouring system in
[`evaluation.md`](evaluation.md), so **int8 remains the serving choice**. Compared with float32 it is 3.7 times smaller,
with 6% lower p50 and 8% lower p95 latency and 42% higher batch throughput. int16 is the lossless option at half the
size, with the same throughput.

### What earlier checkpoints showed

Quantization robustness depends on the weight distribution, and that changes with training. We therefore repeated the
test as training went on:

- An early mined-only checkpoint (step 8,184, about 30% of one mined epoch): all three variants within 0.03 chrF++.
  int8 had 30.6% lower p50 latency and 3.7 times smaller size.
- The human-only checkpoint at step 11,400: int8 cost 0.15 chrF++ only on Indic to English (significant pooled, but no
  single direction survived Bonferroni). int8's latency advantage shrank to about 5%, and int16 was slower than float32.
  That ordering change is observed, not explained. It was one measurement on a shared node.
- The current best model: int8 costs about 0.1 in both directions (above).

So int8's cost grows a little as the model trains, which is what a more differentiated weight distribution would
predict. The int16 and int8 scores also moved by up to 0.06 between two runs on different nodes, so the quantized CPU
kernels are apparently not bit-identical across machines (not investigated).

## 5. API and frontend

A FastAPI service (`deployment/api/`) exposes:

- `POST /api/translate`: routes a request as direct en→xx, direct xx→en, or an **xx→xx pivot**. A pivot is literally an
  xx→en translation followed by an en→xx translation through the same model. There is no separate pivot model. The
  response says whether it pivoted and gives the intermediate English text.
- `GET /api/languages`: the supported languages.
- `GET /api/health`: model identity (checkpoint name, quantization) and a notice string.

A plain HTML, CSS and JS frontend (`deployment/frontend/`) calls the API over real HTTP and has two views: a Translate
view and a Benchmarks view with the published baselines. The Benchmarks view reads `baselines.json` (published
numbers, kept in the repository because the view needs it) and, if present, a `benchmarks.json` file with IndicMT's own
scores, which you can copy in from `evaluation/results/.../scores.json`. The API and frontend run and have been
verified directly on a node, outside a container.

Settings are environment variables: `MODEL_DIR`, `SPM_PATH`, `CHECKPOINT_NAME` (default `indicmt-final-int8`),
`QUANTIZATION`, `INTER_THREADS`, `INTRA_THREADS`, `BEAM_SIZE` (default 2 for latency) and `DISCLAIMER` (default
"Research model; outputs may contain errors.").

## 6. Container build

The `Dockerfile` is ready (Python 3.11 slim, no PyTorch in the image, CTranslate2 and FastAPI only). **It has not been
built.** On the development cluster, rootless `podman` cannot build any real image, because the user has no entry in
`/etc/subuid` and `/etc/subgid`. Rootless builds need those ranges to remap the user IDs in any base image layer. We
tried three ways (the default `overlay` driver, the `vfs` driver with a fresh storage root, and `--userns=keep-id`).
All failed the same way. Fixing it takes an administrator: add a subuid and subgid range for the user and run
`podman system migrate`. Once that is done, Job23 runs the full staged check.

## 7. Stress test

`deployment/stress_test.py` is an asyncio and httpx load generator. It reports p50, p95 and p99 latency and requests
per second by direction class (en→xx, xx→en, xx→xx pivot), payload size and concurrency. Job22 starts the API, waits
for it to be healthy, runs the test and stops the server. **It has been written but not run yet.** No stress-test
numbers are reported.

## 8. Open points

- The CTranslate2 and `onmt_translate` score difference is unexplained (section 4).
- The latency ordering of the variants changes between checkpoints and nodes. Treat single latency numbers as
  indicative.
- float16 on CPU is untested because the hardware does not support it.
- The CPU model and thread settings behind the latency numbers on earlier checkpoints were not recorded. The conversion
  job now records them.
