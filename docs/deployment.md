# Deployment: quantization and serving

This page explains how a trained checkpoint becomes a small CPU-served model: conversion to CTranslate2,
quantization, the API and frontend, and the limitations. Code and commands are in
[`../deployment/`](../deployment/README.md).

The model served is the **release model** (12+12, combined pre-training, then human fine-tuning without replay,
step 1,800). Weights: [ujjwalsharmaIITB/indicMT](https://huggingface.co/ujjwalsharmaIITB/indicMT) on Hugging Face.

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
(`float32`, `int16`, `int8`) with CTranslate2's `OpenNMTPyConverter`. The script handles two compatibility issues:

1. **`weights_only` loading.** OpenNMT-py's release path calls `torch.load(..., weights_only=True)`. That rejects the
   checkpoint's `opt` field (a standard `argparse.Namespace`, present in every OpenNMT-py checkpoint). We use
   CTranslate2's own `unsafe_deserialization=True` flag. That is legitimate here because the checkpoint is
   self-trained and trusted. Do not use it on untrusted downloads.
2. **Flash attention label.** OpenNMT-py 3.5.1 chose flash attention automatically during training
   (`self_attn_type == "scaled-dot-flash"`, never set in a config). CTranslate2's converter only accepts
   `"scaled-dot"`. We checked in OpenNMT-py's attention module that this label only selects the attention *kernel*
   and does not change the learned weights. So the script relabels it on a patched temporary copy before conversion.
   The original checkpoint is never changed.

`deployment/ct2_eval.py` then runs the full 44-direction IN22 evaluation (IN22-Gen by default, `--test-set in22-conv`
for IN22-Conv) directly through `ctranslate2.Translator.translate_batch` for each variant. It reads the source files
built by `evaluation/build_eval_pairs.py`, and the API encodes live requests with the same function
(`data/common/tokenization.py`), so training, evaluation and serving see identical source pieces. It reports
corpus-level chrF++ and BLEU, p50 and p95 single-sentence latency, batch-32 throughput, and model size. Job20 runs the
whole sequence.

## 4. Results for the release model

CPU only, 8 intra-op threads, beam 5, IN22-Gen:

| Variant | en→xx chrF++ | en→xx BLEU | xx→en chrF++ | xx→en BLEU | p50 latency (ms) | p95 latency (ms) | Throughput (sent/s, batch 32) | Size |
|---|---|---|---|---|---|---|---|---|
| float32 | 47.15 | 19.19 | 53.87 | 29.19 | 399.2 | 743.8 | 5.9 | 630.4 MB |
| int16 | 47.15 | 19.18 | 53.86 | 29.20 | 367.1 | 690.3 | 8.4 | 323.2 MB |
| int8 | 47.16 | 19.19 | 53.85 | 29.14 | 374.2 | 687.8 | 8.4 | 170.6 MB |

Scores use the training-format source encoding (see [`evaluation.md`](evaluation.md), section 2). Latency and
throughput are from the first conversion run. They were not re-measured after the encoding fix, which adds one piece
to each source.

CTranslate2 float32 scores this model 0.20 chrF++ higher on en→xx than `onmt_translate` does (47.15 against 46.95;
xx→en 53.87 against 53.69). The cause has not been isolated. **The reported results are the CTranslate2 float32
ones.** Sizes are byte-identical across checkpoints, because quantized size depends on the
architecture, not the weights.

### Does quantization change the translations? A significance test

We saved each variant's translations and compared them with sacreBLEU's **paired bootstrap** (1,000 resamples, chrF++,
float32 as baseline). Job21 runs this (`evaluation/quant_significance.py`). A paired bootstrap resamples sentences and
asks how often the difference would appear by chance. Pooled scores are corpus-level chrF++ over the concatenated
directions of a bucket, so they differ slightly from the macro averages. **Bonferroni** correction divides the 0.05
threshold by 22, because 22 directions are tested at once.

| Bucket | Variant | Pooled chrF++, float32 → variant | p | Macro Δ | Directions p < 0.05 | After Bonferroni |
|---|---|---|---|---|---|---|
| en→xx | int16 | 47.31 → 47.31 | 0.199 | +0.005 | 3/22 | 0/22 |
| en→xx | int8 | 47.31 → 47.32 | 0.181 | +0.009 | 1/22 | 0/22 |
| xx→en | int16 | 53.88 → 53.87 | 0.212 | −0.006 | 1/22 | 0/22 |
| xx→en | int8 | 53.88 → 53.86 | 0.149 | −0.017 | 4/22 | 0/22 |

**Neither int16 nor int8 is significantly different from float32 in either direction.** Pooled p values lie between
0.149 and 0.212. No direction survives Bonferroni correction, and the 1 to 4 directions per row under 0.05 are about
what 22 tests at that threshold produce by chance (1.1 expected). Per-direction int8 deltas range from −0.125 to
+0.164 (en→xx) and from −0.200 to +0.185 (xx→en). **int8 is the serving choice.** Compared with float32 it is 3.7 times
smaller, with 6% lower p50 and 8% lower p95 latency and 44% higher batch throughput. int16 is the middle option at half
the size, with the same throughput.

## 5. API and frontend

A FastAPI service (`deployment/api/`) exposes:

- `POST /api/translate`: routes a request as direct en→xx, direct xx→en, or an **xx→xx pivot**. A pivot is literally an
  xx→en translation followed by an en→xx translation through the same model. There is no separate pivot model. The
  response says whether it pivoted and gives the intermediate English text.
- `GET /api/languages`: the supported languages.
- `GET /api/health`: model identity (checkpoint name, quantization) and a notice string.

A plain HTML, CSS and JS frontend (`deployment/frontend/`) calls the API over real HTTP and has two views: a Translate
view and a Benchmarks view comparing IndicMT with other systems. The Benchmarks view reads `baselines.json` (kept in
the repository because the view needs it: IndicTrans2 from our own run, the other systems from the
`bodhan-ai/indic-translate` model page) and, if present, a `benchmarks.json` file with IndicMT's own scores: the
CTranslate2 float32 reports written by `deployment/ct2_eval.py` (IN22-Gen and IN22-Conv merged into one file). The API and frontend run and have been
verified directly on a node, outside a container.

Settings are environment variables: `MODEL_DIR`, `SPM_PATH`, `CHECKPOINT_NAME` (default `indicmt-final-int8`),
`QUANTIZATION`, `INTER_THREADS`, `INTRA_THREADS`, `BEAM_SIZE` (default 2 for latency) and `DISCLAIMER` (default
"Research model; outputs may contain errors.").

## 6. Container build

The `Dockerfile` builds a small serving image (Python 3.11 slim, no PyTorch, CTranslate2 and FastAPI only). Rootless
`podman` needs a subuid and subgid range for the user (in `/etc/subuid` and `/etc/subgid`) to remap the user IDs in the
base image layers. Job23 builds the image, starts it and checks the health and translate endpoints. See
[`../deployment/README.md`](../deployment/README.md).

## 7. Load test

`deployment/stress_test.py` is an asyncio and httpx load generator. It reports p50, p95 and p99 latency and requests
per second by direction class (en→xx, xx→en, xx→xx pivot), payload size and concurrency. Job22 starts the API, waits
for it to be healthy, runs the test and stops the server. It can also be pointed at any running instance of the API.

## 8. Limitations

- The CTranslate2 and `onmt_translate` score difference (0.20 en→xx and 0.18 xx→en for float32) is unexplained (section 4).
- The latency ordering of the variants changes between checkpoints and nodes. Treat single latency numbers as
  indicative.
- int16 and int8 scores can move by up to 0.06 chrF++ between runs on different machines: the quantized CPU kernels are
  not bit-identical across CPUs.
- float16 on CPU is untested because the hardware does not support it.
