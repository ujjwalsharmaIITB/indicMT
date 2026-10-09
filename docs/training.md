# Training

This page describes the model, the three pre-training runs ("arms"), why we stopped them by hand, how we picked
checkpoints, a source-encoding mismatch that made English-to-Indic quality appear to fall late in training, the
fine-tuning runs and the study of encoder and decoder depth. Configs are in [`../training/configs/`](../training/configs/).
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

None of the arms converged by validation loss. We stopped all three by hand, because IN22 English-to-Indic chrF++
appeared to fall for many checkpoints while validation perplexity kept improving.

| Arm | Stopped at |
|---|---|
| human | step 20,800 (16.5 epochs) |
| mined | step 35,000 (1.28 mined epochs) |
| combined | step 37,700 (1.38 mined epochs) |

**That fall was an artifact of the evaluation.** The evaluation built its sources with a slightly different
SentencePiece encoding from training (section 6). Re-scored with the training encoding, English-to-Indic keeps rising
to the end of training: the combined arm at step 37,700 scores 46.4 chrF++ on IN22-Gen English-to-Indic, not 36.1, and
the human arm at step 20,800 scores 45.3, not 42.9. For the mined and combined arms the best checkpoint by the
selection rule is now the last one (steps 35,000 and 37,700), so both were still improving when stopped. This stop
replaced the planned rule (early stopping). A longer run was not tested.

**chrF++** is a character n-gram F-score (with word n-grams of order 2). It is the main automatic metric here. See
[`evaluation.md`](evaluation.md).

## 5. Selecting checkpoints

One fixed rule picks the checkpoint to carry forward: **the highest mean chrF++ over IN22-Gen and IN22-Conv, in both
directions** (the "mean of four"). The fine-tunes in section 7 start from combined step 13,600 and mined step 24,600,
which this rule picked under the old, mismatched scores. Under the corrected scores, the rule picks step 37,700 for the
combined arm (48.44), step 35,000 for the mined arm (35.96) and step 13,900 for the human-only arm (43.00). Fine-tunes
from those later checkpoints have not been run.

This rule uses the IN22 test sets for model selection, so IN22 scores of selected checkpoints carry a **small
optimistic bias**. Every table in these docs is a single seed. Two extra seeds of the human-only arm reproduce its
chrF++ curve (38.3, 38.3 and 38.5 English-to-Indic on IN22-Gen at step 2,000).

## 6. Source encoding, and the English-to-Indic decline that was not

Training sources are written as `<xx> text` and SentencePiece-encoded as one string. With SentencePiece's default
dummy prefix and the tags registered as user-defined pieces, that gives `['▁', '<xx>', '▁word', ...]`: a lone `▁`
piece, then the tag. Evaluation, CTranslate2 serving and our analyses instead encoded the sentence and then put the tag
in front (`['<xx>', '▁word', ...]`), which drops the leading `▁`. The model never saw that format in training.

The fix is one shared module, `data/common/tokenization.py`. The data pipeline, `evaluation/build_eval_pairs.py`,
`deployment/ct2_infer.py` and `deployment/ct2_eval.py` now all use it. It reproduces 20,000 lines each of the encoded
human training, mined training and dev files exactly, and it rejects any source that does not start with
`['▁', '<xx>']`.

We re-scored 564 checkpoints. On average, English-to-Indic rose by 1.96 chrF++ on IN22-Gen and 2.22 on IN22-Conv;
Indic-to-English rose by 0.02 and 0.37. The effect grows with training length:

| Checkpoint | IN22-Gen en→xx, old encoding | IN22-Gen en→xx, training encoding |
|---|---|---|
| combined 12+12, step 37,700 | 36.1 | 46.4 |
| combined 12+18, step 49,000 | 34.8 | 46.6 |
| human-only, step 20,800 | 42.9 | 45.3 |

So the earlier finding that English-to-Indic declines late in pre-training was an artifact of the mismatch. The
analysis built on it (output length, repetition loops and per-language losses) used outputs from the mismatched inputs
and has been withdrawn. In the training encoding, English-to-Indic keeps rising to the end of training. The lesson:
one function owns the input format, and every entry point checks it.

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

Best checkpoint of each run by the selection rule, and the release checkpoint:

| Run | Step | Gen en-xx | Gen xx-en | Conv en-xx | Conv xx-en | Mean of four |
|---|---|---|---|---|---|---|
| combined to human, no replay | 1,000 | 46.80 | 54.04 | 42.80 | 50.66 | 48.58 |
| **combined to human, no replay (release model)** | **1,800** | **46.95** | **53.69** | **42.97** | **50.38** | **48.50** |
| combined to human + 20% replay | 2,800 | 46.64 | 53.85 | 42.32 | 50.09 | 48.22 |
| mined to human, no replay | 5,800 | 47.04 | 54.13 | 43.00 | 50.61 | 48.70 |
| mined to human + 20% replay | 5,400 | 46.65 | 54.47 | 42.03 | 50.28 | 48.36 |
| human-only pre-training (reference) | 13,900 | 45.21 | 43.80 | 41.67 | 41.31 | 43.00 |

Findings:

- **Combined to human without replay, step 1,800, is the release model.** Under the corrected scores it is not the
  top checkpoint by the selection rule. Step 1,000 of the same run (48.58), mined to human at step 5,800 (48.70),
  12+18 to human at step 1,400 (48.59, section 8) and 18+18 combined pre-training at step 38,000 (48.70) are all
  within 0.2 of it on the mean of four. These are single-seed differences without a significance test, and we kept
  the release model.
- **Replay does not help.** Without replay the mean of four is higher for both starts (combined 48.58 against 48.22,
  mined 48.70 against 48.36). Only for the mined start does replay give a higher IN22-Gen Indic-to-English score
  (54.47 against 54.13).
- **Combined fine-tuning peaks early** (step 1,000 without replay, 2,800 with replay). **Mined fine-tuning peaks
  late** (steps 5,800 and 5,400 of the 6,000-step budget).
- **Mined to human ends level with combined to human** (IN22-Gen English-to-Indic 47.04 against 46.80). An earlier
  version of this page found the mined route well behind on English-to-Indic and analysed why, per language. That gap
  came from the source-encoding mismatch (section 6), and the analysis has been withdrawn.
- **Indic-to-English needs both data sources.** Every model trained on mined and human data scores 53.69 to 55.07 on
  IN22-Gen. The human-only arm scores 43.80 and the mined-only arm 44.20.
- **Pre-training alone gets as far.** The last combined pre-training checkpoint (step 37,700, 48.44) is level with the
  fine-tuned models. The fine-tunes started from step 13,600, chosen under the old encoding.

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

| Shape | Parameters | Best step | Mean of four | Gen en-xx | Gen xx-en | Conv en-xx | Conv xx-en |
|---|---|---|---|---|---|---|---|
| 12+12 | 153.8M | 37,700 | 48.44 | 46.36 | 55.05 | 41.81 | 50.55 |
| 12+18 | 179.0M | 38,000 | 48.38 | 46.65 | 54.51 | 42.23 | 50.13 |
| 18+18 | 197.9M | 38,000 | 48.70 | 46.79 | 55.07 | 42.32 | 50.62 |
| *human-only 12+12* | *153.8M* | *13,900* | *43.00* | *45.21* | *43.80* | *41.67* | *41.31* |

The 12+12 run was stopped at step 37,700, so it cannot be compared beyond that point. A deeper decoder (12+12 to
12+18) does not help on the mean (−0.06): it gains English-to-Indic (+0.29 Gen, +0.42 Conv) and loses Indic-to-English
(−0.54 Gen, −0.42 Conv). 18+18 is the best shape, +0.26 over 12+12 on the mean. These are pre-fine-tuning numbers
without a significance test, and checkpoint selection used the test sets. An earlier version of this section said all
three shapes peak at steps 13,000 to 13,600 and lose English-to-Indic afterwards. That came from the source-encoding
mismatch (section 6). Against human-only pre-training, the combined shapes are 5.4 to 5.7 higher on the mean, mostly
from Indic-to-English.

**Stage 2** is human fine-tuning without replay from step 13,000 of each deeper run (configs
`finetune_combined_12e18d_human.4gpu.yaml` and `finetune_combined_18e18d_human.4gpu.yaml`). For 12+18 it is complete:
the best checkpoint is step 1,400, mean of four 48.59 (47.08 / 53.73 on IN22-Gen, 43.15 / 50.39 on IN22-Conv). That is
level with 12+12 to human (48.58 at step 1,000). Stage 2 results for 18+18 are not reported yet. The release model
stays the 12+12 model. Wider depth sweeps are planned future work.

## 9. Checkpoints and outputs

Checkpoints land in `128k-spm/opennmt/<run>/checkpoints/model_step_<N>.pt`. They are large and not tracked by git.
TensorBoard logs go beside them.

## 10. Compute

One node per job with up to 4 NVIDIA A100 80 GB GPUs (CUDA 12.4 driver), `torch==2.2.0+cu121`, DistributedDataParallel.
The cluster caps a job at 4 GPUs and at 6 days. The human arm uses 2 GPUs. The mined and combined arms use 4.
