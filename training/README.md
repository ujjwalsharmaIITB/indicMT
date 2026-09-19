# training: data preparation, pre-training and fine-tuning

This directory prepares the OpenNMT-py training files, holds all training configs, and describes how the model was
trained. The full discussion (arms, stopping, the English-to-Indic decline, fine-tuning, the architecture study) is in
[`../docs/training.md`](../docs/training.md).

The model is a **153,807,872-parameter Transformer** (12 encoder + 12 decoder layers, `d_model` 512, `d_ff` 2048,
16 heads, one tied 128k SentencePiece vocabulary), trained from scratch with OpenNMT-py 3.5.1.

## Step 1: build the training files

Three scripts, in order. Job8 runs the first two for all five datasets (human train, mined train, human dev, IN22-Gen
and IN22-Conv test) as an array job.

| Script | What it does | Writes (not tracked by git) |
|---|---|---|
| `build_parallel_corpus.py` | Turns the mixtures into line-aligned `.src` and `.tgt` files. The source side already has its `<xx>` tag. IN22 sets are written as 23 per-language files, so any direction can be built later. | `128k-spm/data/raw/` |
| `encode_spm.py` | Encodes those files with the frozen 128k SentencePiece model (`tokenizer/sweep/mixed/spm_128000.model`). | `128k-spm/data/` |
| `export_onmt_vocab.py` | Exports the SentencePiece vocabulary as an OpenNMT-py vocab file. | `128k-spm/vocab.128k.shared.txt` |

```bash
sbatch -p <partition> scripts/slurm/training_build_dataset.sh      # Job8 (array of 5)
python training/export_onmt_vocab.py
```

The vocabulary comes straight from the tokenizer, not from counting one training split, so every piece the tokenizer
can produce has a row in the embedding table. Both build scripts skip finished outputs. Pass `FRESH=1` to rebuild
(do this if the mixtures were rebuilt).

## Step 2: pre-training (three data arms)

Same architecture, vocabulary, batching and schedule in every arm, at matched compute (262,144 tokens per step), so the
data is the only difference.

| Arm | Config | Job | Data |
|---|---|---|---|
| human-only | `configs/pretrain_human.2gpu.yaml` | Job9 (2 GPUs) | `train-gold`, 6,953,603 sentences |
| mined-only | `configs/pretrain_mined.4gpu.yaml` | Job10 (4 GPUs) | `mined-mixture`, 213,807,105 sentences |
| combined | `configs/pretrain_combined.4gpu.yaml` | Job11 (4 GPUs) | 8:2 mined to human, mixed on the fly |

```bash
sbatch -p <partition> scripts/slurm/training_pretrain_human.sh
sbatch -p <partition> scripts/slurm/training_pretrain_mined.sh
sbatch -p <partition> scripts/slurm/training_pretrain_combined.sh
```

### Reading the configs

All configs are plain OpenNMT-py YAML with comments. The ones that matter:

- **Naming.** `pretrain_*` start from scratch. `finetune_*` start from a pre-trained checkpoint. The `.2gpu` and `.4gpu`
  suffix is the number of GPUs the config was written for.
- **Output directories.** Run directories look like `128k-spm/opennmt/bpcc-combined/`. Checkpoints go in
  `.../checkpoints/model_step_<N>.pt` (about 2.4 GB each, all kept).
- **`pretrain_human.2gpu.yaml`** is the reference. Its header explains the architecture check, the vocabulary export, the
  length limit and the batching fix for the 128k vocabulary. The other configs point to it.
- **The 8:2 mix.** `pretrain_combined.4gpu.yaml` lists two corpora with `weight: 8` (mined) and `weight: 2` (human). In
  OpenNMT-py 3.5.1 the weight is an **integer repeat count per cycle**, not a probability. A fractional weight such as
  `0.9` silently becomes 0 and drops that corpus with no error.
- **Learning rate.** `noam` schedule, 1,500 warmup steps, peak 1.14e-3.
- **Checkpointing.** Validation and checkpoint every 100 steps, all kept (`keep_checkpoint: -1`), raw weights
  (`average_decay: 0`). This lets us follow training dynamics and pick the best step.
- **Gradient clipping.** Human (2 GPUs) uses `max_grad_norm: 10`. The 4-GPU arms use 5. This clips at the same true norm,
  because OpenNMT-py divides gradients by the GPU count after the all-reduce.
- **NCCL.** `timeout: 1800` seconds in every config, after long jobs died at the default 300 s. A mitigation, not a
  confirmed fix.
- **Batching.** `batch_size: 16384`, `max_generator_batches: 32`, with gradient accumulation to reach 262,144 tokens per step.

Resume is opt-in (`sbatch --export=ALL,RESUME=1 ...`). A resumed run restarts the data iterator and resets
early stopping, so start fresh when you need a clean record.

### Stopping and selecting

None of the arms converged by validation loss. We **stopped all three by hand**, because IN22 English-to-Indic quality had
been falling for many checkpoints while validation perplexity kept improving.

| Arm | Stopped at | Peak IN22-Gen en→xx chrF++ |
|---|---|---|
| human | step 20,800 (16.5 epochs) | 44.33 at step 10,100 |
| mined | step 35,000 (1.28 mined epochs) | stuck at 30 to 32 after step 3,000 |
| combined | step 37,700 (1.38 mined epochs) | 44.32 at step 10,900; 39.08 at 35,500 |

Checkpoints are selected by one rule: the highest mean chrF++ over IN22-Gen and IN22-Conv, in both directions. For the
combined arm that is step 13,600, and for the mined arm step 24,600. This uses the test sets for selection, so later IN22
numbers carry a small optimistic bias.

**Why English-to-Indic declines.** In the 8:2 mix, languages without mined data learn English-to-Indic only from the 20%
human stream. Continued training moves the shared decoder toward the mined data of the large languages and makes outputs
longer, with more repetition loops. It is not a decoding artifact (re-decoding without the length penalty changes
chrF++ by at most 0.16). Validation perplexity does not see it, because it is measured with teacher forcing. This is our
reading, not a tested conclusion. The analysis is in [`../docs/training.md`](../docs/training.md).

## Step 3: fine-tuning on human data

The best pre-trained checkpoint of the combined and mined arms is fine-tuned on human data, with and without **replay**
(mixing 20% of the old, mined data back in to reduce forgetting). Fresh optimizer, 1,500 warmup steps to a peak learning
rate of 1e-4, 6,000 steps, checkpoint every 200 steps.

| Config | Starts from | Data |
|---|---|---|
| `finetune_combined_human.4gpu.yaml` | combined, step 13,600 | human only (**current best model**, best step 1,800) |
| `finetune_combined_human_replay.4gpu.yaml` | combined, step 13,600 | human + 20% mined |
| `finetune_mined_human.4gpu.yaml` | mined, step 24,600 | human only |
| `finetune_mined_human_replay.4gpu.yaml` | mined, step 24,600 | human + 20% mined |

```bash
sbatch -p <partition> \
  --export=ALL,CFG=finetune_combined_human,SRC=128k-spm/opennmt/bpcc-combined/checkpoints/model_step_13600.pt \
  scripts/slurm/training_finetune.sh                                # Job13
```

The script loads the weights with `--reset_optim all` (new optimizer, step counter back to 0) and checks that a
checkpoint was written by the run. IN22-Gen chrF++ (en→xx / xx→en) of the best checkpoint of each run:

| Run | Best step | en→xx / xx→en |
|---|---|---|
| combined → human, no replay | 1,800 | **45.36** / 53.73 |
| combined → human + replay | 2,000 | 44.89 / 53.89 |
| mined → human, no replay | 4,400 | 39.73 / 53.16 |
| mined → human + replay | 5,800 | 39.53 / 53.84 |

Replay does not help the combined start. Combined fine-tuning peaks early (steps 1,800 to 2,000). Mined fine-tuning gains
about 9 points but stays about 5.6 en→xx behind, mostly in languages with no mined data.

## Step 4: architecture study

The current best recipe is repeated with **12 encoder + 18 decoder** layers (178,992,128 parameters) and **18 + 18**
(197,878,784), same width.

| Stage | Configs | Job | Status |
|---|---|---|---|
| 1. Combined pre-training | `pretrain_combined_12e18d.4gpu.yaml`, `pretrain_combined_18e18d.4gpu.yaml` | Job12 | done |
| 2. Human fine-tune, no replay | `finetune_combined_12e18d_human.4gpu.yaml`, `finetune_combined_18e18d_human.4gpu.yaml` | Job13 | queued |

```bash
sbatch -p <partition> --export=ALL,CFG=pretrain_combined_12e18d,RUN=bpcc-combined-12e18d \
  scripts/slurm/training_pretrain_arch.sh
```

Stage 1 results (best checkpoint by mean IN22 chrF++): 12+12 46.57 (step 13,600), 12+18 46.77 (step 13,000), 18+18 47.08
(step 13,000). Depth does not change when to stop. The release model will be chosen after stage 2.

Outputs of all training steps are written under `128k-spm/` and are not tracked by git. Next:
[`../evaluation/README.md`](../evaluation/README.md).
