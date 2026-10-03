# Training

This page describes the model, the three pre-training runs ("arms"), why we stopped them by hand, how we picked
checkpoints, the fine-tuning runs, the study of encoder and decoder depth, and what we learned about English-to-Indic
quality falling late in training. Configs are in [`../training/configs/`](../training/configs/).
Training uses OpenNMT-py 3.5.1.

"Job9" and similar names refer to the batch-job templates in [`../scripts/README.md`](../scripts/README.md).

## 1. The model

- A Transformer encoder-decoder with **12 encoder and 12 decoder layers**, 16 attention heads, `d_model` 512 and
  feed-forward size 2048.
- One shared **128k SentencePiece vocabulary**. A single embedding matrix is tied across the source embedding, the
  target embedding and the output layer.
- **153,807,872 parameters** (encoder 103,310,336 and decoder 50,497,536), checked with an `onmt_train` dry run
  against the real vocabulary.
- The vocabulary file is exported directly from the frozen SentencePiece model (`training/export_onmt_vocab.py`), not
  counted from one training split. That way every piece the tokenizer can produce has a row in the table.

The first plan was 18+18 layers with 8 heads (197,878,784 parameters). Its training loss was judged to fall too
slowly, so it was changed to 12+12 with 16 heads. Two points go against the "just make it bigger" instinct. Head count
does not change the parameter count (attention projections are `d_model x d_model` whatever the head split), so the
real change is the depth cut from 18 to 12. And `d_model` stayed at 512: the vocabulary's Zipfian tail (981 pieces
occur fewer than 10 times and 3,754 never occur in the human training data) means a wider embedding would add
dimensions the model has too little data to train, and the model has to stay small enough to deploy.

## 2. From mixtures to training files

1. `training/build_parallel_corpus.py` turns the mixtures into line-aligned `.src` and `.tgt` files. The source side
   already carries the `<xx>` tag. IN22 test sets stay multi-parallel (one file per language).
2. `training/encode_spm.py` encodes those files with the 128k SentencePiece model.
3. `training/export_onmt_vocab.py` writes the OpenNMT-py vocabulary file.

Job8 runs steps 1 and 2 for all five datasets (human train, mined train, human dev and the two test sets). All
outputs go under `128k-spm/` and are not tracked by git. If the mixtures are rebuilt, rebuild these files too (pass
`FRESH=1`).

## 3. The three data arms

Three arms train from scratch with seed 42, `train_steps: 100000`, the same architecture, vocabulary, batching and
schedule. They run at matched compute (the same effective batch of 262,144 tokens per step), so the data is the only
variable.

- **Human-only** (`train-gold`, 6,953,603 sentences): `pretrain_human.2gpu.yaml`. Job9, 2 GPUs.
- **Mined-only** (`mined-mixture`, 213,807,105 sentences): `pretrain_mined.4gpu.yaml`. Job10, 4 GPUs.
- **Combined, 8:2 mined to human**: `pretrain_combined.4gpu.yaml`. Job11, 4 GPUs. The two corpora are mixed on the
  fly with OpenNMT-py's `WeightedMixer`.

Plain concatenation would let mined data's 30.75 times larger size dilute the human data to about 3.15% of what the
model sees. That is far below the intended 20%.

**A pitfall.** In OpenNMT-py 3.5.1 the corpus `weight` is an *integer* repeat count per cycle (the code casts it with
`int(...)`), not a probability. A weight such as `0.9` silently becomes `0`, and that corpus is never used, with no
error. For an 8:2 mix the correct values are `weight: 8` and `weight: 2`: eight mined sentences, then two human
sentences, repeated.

### Settings

| | human | mined | combined |
|---|---|---|---|
| GPUs x gradient accumulation | 2 x 8 | 4 x 4 | 4 x 4 |
| `max_grad_norm` | 10 | 5 | 5 |
| Checkpoint and validation | every 100 steps | every 100 steps | every 100 steps |
| `keep_checkpoint` | -1 (keep all) | -1 | -1 |
| `early_stopping` (validations) | 32 (3,200 steps, 2.53 epochs) | 273 (27,300 steps, 1.0 epoch) | 273 |
| Warmup | 1,500 steps | 1,500 steps | 1,500 steps |
| `average_decay` | 0 | 0 | 0 |
| Time limit | 6 days | 6 days | 6 days |

- **Learning rate.** OpenNMT's `noam` schedule: `lr = learning_rate * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)`
  with `learning_rate: 1` and 1,500 warmup steps. The peak is 1.14e-3 at step 1,500, then the rate falls as
  `step^-0.5`. 1,500 steps is 1.19 human epochs and 5.5% of a mined epoch.
- **Raw weights.** `average_decay: 0`, so every checkpoint and every validation uses the raw weights of that step, not
  a moving average. Because all checkpoints are kept, you can still average the last few later.
- **Why every 100 steps and keep all.** Most learning happens early. Dense checkpoints let us track training dynamics
  and pick the best step.
- **Early stopping** counts validations, so patience in steps is `early_stopping x 100`. Human has 1,262.5 steps per
  epoch and mined has 27,282.1. Combined has no single epoch (its two corpora cycle at different rates), so it reuses
  mined's patience.
- **Why human's `max_grad_norm` is 10.** OpenNMT-py 3.5.1 divides gradients by the number of GPUs after the
  all-reduce. A 2-GPU gradient is therefore exactly twice a 4-GPU one for the same data. Adam cancels a constant
  scale, but clipping happens after the division. A value of 5 would clip the 4-GPU runs at a true norm of 20 and the
  2-GPU run at 10. Setting 10 clips both at the same true norm.
- **Validation** is loss and perplexity on the human dev set, not IN22. It takes about 60 s per pass (43,878
  sentences, 4 GPUs). With the checkpoint write, that is about 15% of wall-clock time. We kept it for a live
  validation curve at every checkpoint.
- **Data order.** Both training files are globally shuffled. OpenNMT-py streams them in order and reshuffles only
  inside a 262,144-example bucket, so every epoch sees the same global order with local reshuffling.
- **Storage.** 2.40 GB per checkpoint (it includes the Adam state).

### Operational lessons

- **DDP hangs, only mitigated.** Several 4-GPU attempts on the human data hung completely (GPUs idle at 0%) after
  about 150 steps, on more than one node. An identical mined-only job trained for hours on another node, and a 2-GPU
  variant passed step 150. This rules out both a bad node and a universal config bug as the only cause. We never
  found the root cause.
- **NCCL timeouts.** Two long jobs (a 2-GPU human job after 15h44m and a 4-GPU combined job after about 3.5h) died to
  NCCL `ALLREDUCE` watchdog timeouts at the default 300 s. We set `timeout: 1800` in every config. That widens the
  window, but it is not a confirmed fix.
- **Memory with a 128k vocabulary.** The generator's logits tensor scales with vocabulary size. The values that fit
  are `batch_size: 16384` and `max_generator_batches: 32`, with `accum_count` raised to keep 262,144 tokens per step.
  OpenNMT-py's own recovery from out-of-memory crashes in DDP, so memory has to be avoided.
- **Nodes.** A node with a stalled filesystem read, and a node with heavy contention from other users' jobs, were
  excluded from submission (`sbatch --exclude=<node>`). Neither cause was fully understood.
- **CPU allocation.** Request CPUs explicitly. A job without `--cpus-per-task` got 2 CPUs, which on these nodes are
  the two hyperthreads of one core, shared by all GPU ranks. The GPUs waited on the CPU (step times of 9 to 11 s,
  against about 5.6 to 6.4 s after the fix). The job scripts now request 16 CPUs (2 GPUs) or 32 (4 GPUs) and pin
  `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 4.
- **Resume is opt-in.** The scripts start from step 0 unless `RESUME=1` is set. A resume restores weights, optimizer
  state and step, but restarts the data iterator at the start of the file and resets the early-stopping counter. For a
  clean record, start fresh in an empty checkpoint directory.

## 4. Stopping by hand

None of the arms converged by validation loss. We stopped all three by hand, because **IN22 English-to-Indic quality
had been falling for many checkpoints while validation perplexity kept improving**.

| Arm | Stopped at | Notes |
|---|---|---|
| human | step 20,800 (16.5 epochs) | plateau 44.1 to 44.3 chrF++ between steps 7,600 and 11,400; 43.09 at 19,900 |
| mined | step 35,000 (1.28 mined epochs) | stuck at 30 to 32 after step 3,000; fell after 21,800 (29.34 at 32,700) |
| combined | step 37,700 (1.38 mined epochs) | peak 44.32 at step 10,900; fell at every checkpoint after 13,600, to 39.08 at 35,500 |

Peak IN22-Gen English-to-Indic chrF++: combined 44.32 (step 10,900) and human 44.33 (step 10,100). Indic-to-English
stayed flat or rose slightly in every arm. This stop replaced the planned rule (early stopping). A longer
low-learning-rate tail was not tested.

**chrF++** is a character n-gram F-score (with word n-grams of order 2). It is the main automatic metric here. See
[`evaluation.md`](evaluation.md).

## 5. Selecting checkpoints

One fixed rule picks the checkpoint to carry forward: **the highest mean chrF++ over IN22-Gen and IN22-Conv, in both
directions.** For the combined arm that is step 13,600 (Gen 43.84 / 53.80, Conv 39.56 / 49.10). The best by
English-to-Indic alone would have been step 10,900. For the mined arm it is step 24,600 (Gen 30.45 / 44.24, Conv
26.89 / 38.79). Its English-to-Indic-only best, step 4,100, lies in a noisy early band and is much weaker in the other
direction.

This rule uses the IN22 test sets for model selection, so IN22 scores of selected checkpoints carry a **small
optimistic bias**. Every table in these docs is a single seed.

## 6. Why English-to-Indic declines after about step 13,000

We compared the English-to-Indic outputs of the best checkpoint and the one-epoch checkpoint (27,000, or 27,300 for
12+12) for the three combined depths, and of the human-only checkpoints 11,700 and 19,900.

- **It is not a decoding artifact.** OpenNMT-py's default length penalty favours longer outputs. Re-decoding with no
  length penalty changed English-to-Indic chrF++ by at most 0.16, and the outputs stayed as long. The change is in the
  model.
- **Outputs get longer and loop more.** The hypothesis-to-reference character ratio (macro over 22 languages,
  IN22-Gen) rises from 1.04 to 1.14 (12+12), 1.02 to 1.10 (12+18) and 1.01 to 1.10 (18+18). On IN22-Conv, sentences
  with a repetition loop (fewer than 60% distinct word trigrams, 8 or more words) rise from about 1% to 4.4 to 7.5%.
  Output in the wrong script stays under 1%, and source copying stays at 0.1%.
- **Lengthening alone does not explain it.** Human-only training lengthens just as much (1.01 to 1.10) but loses only
  1.3 chrF++ on Gen English-to-Indic.
- **The extra loss sits on languages with little or no mined data.** Mean English-to-Indic chrF++ change from the best
  to the one-epoch checkpoint:

| Mined data per language | 12+12 Gen / Conv | 12+18 Gen / Conv | 18+18 Gen / Conv | Human-only Gen / Conv |
|---|---|---|---|---|
| none (doi, gom, mni, sat, sd) | -3.20 / -5.51 | -2.59 / -3.69 | -3.73 / -3.57 | -1.55 / -0.64 |
| under 6M words (as, br, ks, mai, sa) | -3.23 / -4.99 | -3.00 / -3.57 | -3.05 / -3.43 | -1.31 / -0.50 |
| 19 to 27M words (ne, or) | -2.61 / -3.44 | -2.98 / -2.60 | -2.68 / -1.57 | -1.02 / -0.43 |
| over 56M words (bn, gu, hi, kn, ml, mr, pa, ta, te, ur) | -1.94 / -1.85 | -1.71 / -1.40 | -2.06 / -1.72 | -1.32 / -0.66 |

In the combined runs, languages without mined data lose about twice as much as the large ones. In human-only training
the loss is small and the same for every group.

Our reading, which is **not yet tested**: in the 8:2 mix, languages without mined data learn English-to-Indic only
from the 20% human stream, which by step 27,000 has been seen about 4.3 times (2.1 times at step 13,000). Meanwhile
80% of updates go to mined data of the large languages. Continued training moves the shared decoder toward the mined
data and away from what the human stream taught for the small languages. For all languages it also learns to emit
longer outputs. Validation perplexity is measured with teacher forcing, so it does not see length or loops in free
decoding, which is why it kept falling. The peak step is not tied to an epoch: all combined depths and the human-only
run peak between steps 10,000 and 14,000.

## 7. Fine-tuning on human data

The best pre-trained checkpoint of the mined and combined arms is fine-tuned on human data. We test two ways:

- **Without replay:** human data only.
- **With replay:** human data plus **20% mined data** (human to mined 8:2 by sentence, OpenNMT-py weights 4 and 1).
  *Replay* means mixing some of the old training data back in, to reduce forgetting.

Setup, identical for all runs: a fresh optimizer (`--reset_optim all`), 1,500 warmup steps (about one human epoch) to
a peak learning rate of 1e-4, 6,000 steps, and a checkpoint and validation every 200 steps. Under noam the warmup length
also sets the peak rate, so `learning_rate` is 0.0876 to keep the 1e-4 peak. Job13 runs any of the fine-tuning configs.

Start points: combined from step 13,600, mined from step 24,600. Validation perplexity on the human dev set fell from
51.25 to 12.72 (mined to human, with replay) and from 12.26 to 10.76 (combined to human, with replay). That is the
fine-tuning domain, so it is not evidence of better IN22 quality. We scored IN22 chrF++ at every 200 steps.

| Run | Best step | Gen en-xx | Gen xx-en | Conv en-xx | Conv xx-en | Gen 44-direction mean |
|---|---|---|---|---|---|---|
| combined, start (step 13,600) | - | 43.84 | 53.80 | 39.56 | 49.10 | 48.82 |
| **combined to human, no replay (current best)** | **1,800** | **45.36** | 53.73 | **41.06** | **50.04** | **49.54** |
| combined to human + 20% replay | 2,000 | 44.89 | 53.89 | 40.46 | 49.66 | 49.39 |
| mined, start (step 24,600) | - | 30.45 | 44.24 | 26.89 | 38.79 | 37.35 |
| mined to human, no replay | 4,400 | 39.73 | 53.16 | 36.29 | 49.24 | 46.45 |
| mined to human + 20% replay | 5,800 | 39.53 | 53.84 | 36.20 | 49.45 | 46.69 |
| human-only pre-training, step 11,400 (reference) | - | 44.05 | 44.08 | | | 44.06 |

Findings:

- **Combined to human without replay, step 1,800 is the current best model.** It is best of the four runs on every
  IN22 bucket except IN22-Gen Indic-to-English, where both replay runs are slightly higher.
- **For the combined start, replay does not help.** Without replay the peak is +0.47 English-to-Indic on IN22-Gen and
  +0.60 / +0.38 on IN22-Conv. The combined model already saw mined data for 13,600 steps.
- **Replay protects Indic-to-English late in fine-tuning.** Without replay, combined Indic-to-English falls after its
  peak (53.73 at step 1,800 to 52.67 at 6,000). With replay it stays near 53.5.
- **Combined fine-tuning peaks early,** around steps 1,800 to 2,000 (about 1.4 human epochs). The 6,000-step budget
  was about three times what was needed. Mined fine-tuning keeps improving until about step 3,400.
- **Mined to human gains about 9 points in both directions** but stays about 5.6 English-to-Indic below the combined
  route.
- **Indic-to-English is set by the mined data.** Every model that saw mined data scores 53 to 54 on IN22-Gen. The
  human-only arm scores 44.

### Why mined to human stays behind

Indic-to-English is equal for the two routes, so the gap is in *writing* Indic, not reading it. The English-to-Indic
gap is largest where the mined model had little or no data. The five languages without mined data average a +9.6 gap
(Manipuri +16.0, Santali +12.3), the 17 with mined data +4.5, and across those 17 the gap falls as mined data grows
(Pearson r = -0.81 against log10 of the mined-to-human ratio). The mined model could barely write Manipuri or Santali
before fine-tuning (0.9 and 0.8 chrF++), and the gentle fine-tune does not fully teach them. The main cause is missing
exposure to writing low-resource languages, not noisy mined targets. A residual gap of +2.4 to +4 on well-covered
languages is unexplained. The per-language table is in [`evaluation.md`](evaluation.md).

## 8. Architecture study

The best recipe (combined pre-training, then human fine-tuning without replay) is repeated with two deeper shapes at
the same width: **12 encoder + 18 decoder layers** (178,992,128 parameters) and **18 + 18** (197,878,784). Job12 runs
the pre-training stage. The two configs differ from `pretrain_combined.4gpu.yaml` only in layer counts, output paths,
master port and checkpoint schedule: they checkpoint and validate every 1,000 steps, keep the latest 40 and stop after
one epoch without validation improvement.

**Stage 1** (combined pre-training) ended when the cluster administrators cancelled both jobs, so early stopping
never triggered: 12+18 at step 49,700 and 18+18 at step 45,670. Both are past the 12+12 baseline's checkpoint, so
stage 1 is treated as complete. All 40 checkpoints of each still on disk were scored on IN22. The best checkpoint by
the selection rule:

| Shape | Parameters | Best step | Mean | Gen en-xx | Gen xx-en | Conv en-xx | Conv xx-en | Mean at step ~27,000 |
|---|---|---|---|---|---|---|---|---|
| 12+12 | 153.8M | 13,600 | 46.57 | 43.84 | 53.80 | 39.56 | 49.10 | 45.32 |
| 12+18 | 179.0M | 13,000 | 46.77 | 44.38 | 53.41 | 40.24 | 49.04 | 45.45 |
| 18+18 | 197.9M | 13,000 | 47.08 | 44.33 | 54.00 | 40.29 | 49.69 | 45.82 |
| *human-only 12+12* | *153.8M* | *11,700* | *42.85* | *44.43* | *44.31* | *40.48* | *42.17* | - |

A deeper decoder (12+12 to 12+18) adds +0.20 on the mean, mostly English-to-Indic. A deeper encoder on top (12+18 to
18+18) adds +0.31, mostly Indic-to-English. These are pre-fine-tuning numbers without a significance test, and
checkpoint selection used the test sets. All three shapes peak at the same point (13,000 to 13,600) and lose
English-to-Indic afterwards in the same way (section 6), so depth does not change when to stop. Against human-only
pre-training, the combined shapes are about 4 chrF++ higher on the mean, all from Indic-to-English.

**Stage 2** (human fine-tuning without replay from step 13,000 of each deeper run, configs
`finetune_combined_12e18d_human.4gpu.yaml` and `finetune_combined_18e18d_human.4gpu.yaml`) is **queued, not yet
finished**. The release model will be chosen after it finishes. Wider depth sweeps are planned future work.

## 9. Checkpoints and outputs

Checkpoints land in `128k-spm/opennmt/<run>/checkpoints/model_step_<N>.pt`. They are large and not tracked by git.
TensorBoard logs go beside them.

## 10. Compute

One node per job with up to 4 NVIDIA A100 80 GB GPUs (CUDA 12.4 driver), `torch==2.2.0+cu121`, DistributedDataParallel.
The cluster caps a job at 4 GPUs and at 6 days. The human arm uses 2 GPUs. The mined and combined arms use 4.
