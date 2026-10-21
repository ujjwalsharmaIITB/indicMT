# scripts: batch-job templates

`scripts/slurm/` holds SLURM job templates for every heavy step of the pipeline. They are templates: adapt the
environment lines to your cluster. In the docs, each job is called **Job1, Job2, ...** (the SLURM job name is `job01`,
`job02`, ...).

## Using the templates

Run from the repository root, and create the log directory first:

```bash
mkdir -p logs
sbatch -p <partition> scripts/slurm/<script>.sh [args]
```

Things the scripts assume or leave to you:

- **Repository path.** Each script does `cd "$REPO_DIR"`. `REPO_DIR` defaults to the directory you ran `sbatch` from
  (`SLURM_SUBMIT_DIR`), and otherwise to the repository root next to the script. Override it with
  `sbatch --export=ALL,REPO_DIR=/path/to/indicMT ...`.
- **Environment.** Each script runs `source activate "${CONDA_ENV:-indicmt}"`. Change this line to match your setup. The LLM
  judge scripts use `JUDGE_CONDA_ENV` (default `indicmt-judge`) instead.
- **Partition and QOS** are not set in the scripts. Pass them at submission (`-p`, plus a QOS option if your cluster needs one).
  Node exclusions work the same way (`--exclude=<node>`).
- **Resources** (`--gres`, `--cpus-per-task`, `--mem`, `--time`) are kept, because they document the hardware each step
  needs. Hardware used: 4x A100 80 GB nodes for the 4-GPU training jobs, 2 GPUs for the human arm, 48 GB GPUs for evaluation
  and the judge.
- **Logs** go to `logs/%x-%j.out` and `.err`.
- Array jobs count each task as one submission against per-user submit limits on some clusters.

## Job table

| Job | Script | Stage | Resources | Inputs | Outputs (not tracked by git) |
|---|---|---|---|---|---|
| Job1 | `data_human_multilingual_dataset.sh` | Data | 4 CPUs, 2 h | `data/bpcc-human/splits/` | `data/bpcc-human/combined_train_dataset.jsonl`, `concatenated_valid_dataset.jsonl` |
| Job2 | `data_mined_multilingual_dataset.sh` | Data | 16 CPUs, 24 h | `data/bpcc-mined/raw/` | `data/bpcc-mined/concatenated_bpcc_mined_dataset.jsonl` |
| Job3 | `data_dataset_report.sh` | Data | 4 CPUs, 6 h | raw and split data | `data/dataset_statistics.md` |
| Job4 | `mixing_build_mixture.sh` | Mixing | array of 2, 32 CPUs each, 24 h | Job1 and Job2 outputs | `mixing/output/train_gold.jsonl`, `mined_mixture.jsonl` |
| Job5 | `tokenizer_prepare_corpus.sh` | Tokenizer | 4 CPUs, 8 h | Job1 and Job2 outputs, `data/bpcc-mined/raw/` | `tokenizer/corpus/` |
| Job6 | `tokenizer_spm_sweep.sh` | Tokenizer | 8 CPUs, high memory, 3 days | Job5 corpus | `tokenizer/sweep/mixed/spm_<size>.model` (7 sizes) |
| Job7 | `tokenizer_corpus_laws.sh` | Tokenizer analysis | 4 CPUs, 6 h | Job1 output | `tokenizer/analysis/results/` |
| Job8 | `training_build_dataset.sh` | Training data | array of 5, 32 CPUs each, 24 h | Job4 output, IN22, Job6 model | `128k-spm/data/` (raw and encoded) |
| Job9 | `training_pretrain_human.sh` | Pre-training, human-only arm | 2 GPUs, 16 CPUs, 6 days | Job8 output, vocab | `128k-spm/opennmt/bpcc-human/` |
| Job10 | `training_pretrain_mined.sh` | Pre-training, mined-only arm | 4 GPUs, 32 CPUs, 6 days | Job8 output, vocab | `128k-spm/opennmt/bpcc-mined/` |
| Job11 | `training_pretrain_combined.sh` | Pre-training, combined 8:2 arm | 4 GPUs, 32 CPUs, 6 days | Job8 output, vocab | `128k-spm/opennmt/bpcc-combined/` |
| Job12 | `training_pretrain_arch.sh` | Architecture study, stage 1 (`CFG`, `RUN`) | 4 GPUs, 32 CPUs, 6 days | Job8 output, vocab | `128k-spm/opennmt/bpcc-combined-{12e18d,18e18d}/` |
| Job13 | `training_finetune.sh` | Fine-tuning (`CFG`, `SRC`) | 4 GPUs, 32 CPUs, 2 days | a pre-trained checkpoint | `128k-spm/opennmt/bpcc-ft-*/` |
| Job14 | `evaluation_run.sh` | Evaluation of one checkpoint | 1 GPU, 32 GB, 4 h | a checkpoint, IN22 files | `evaluation/results/<arm>/<checkpoint>/` |
| Job15 | `evaluation_batch_run.sh` | Evaluation of many checkpoints | 2 GPUs, 24 CPUs, 64 GB, 12 h | checkpoints, IN22 files | `evaluation/results/...` |
| Job16 | `llm_judge_setup_env.sh` | LLM judge: build environment | 8 CPUs, 4 h | none | a conda environment |
| Job17 | `llm_judge_download_model.sh` | LLM judge: download model (`REPO`, `JUDGE_MODEL_DIR`) | 8 CPUs, 12 h | Hugging Face | `${JUDGE_MODEL_DIR}` |
| Job18 | `llm_judge_verify_model.sh` | LLM judge: checksum model | 4 CPUs, 2 h | `${JUDGE_MODEL_DIR}` | log only |
| Job19 | `llm_judge.sh` | LLM judge: score one checkpoint | 2 GPUs, 16 CPUs, 128 GB, 12 h | Job14 output, judge model | `evaluation/llm_judge/results/` |
| Job20 | `deployment_convert_ct2.sh` | CTranslate2 conversion and quantization comparison | 8 CPUs, 32 GB, 12 h | a checkpoint | `.../ct2/<checkpoint>/{float32,int16,int8}/` |
| Job21 | `deployment_quant_significance.sh` | Quantization significance test | 24 CPUs, 48 GB, 12 h | Job20 output | `.../ct2/<checkpoint>/significance/` |
| Job22 | `deployment_stress_test.sh` | API load test | 8 CPUs, 16 GB, 3 h | staged model in `deployment/model/` | `logs/stress_test_results.json` |
| Job23 | `deployment_container_check.sh` | Container build and run check | 4 CPUs, 8 GB, 1 h | staged model in `deployment/model/` | log only |
| Job24 | `evaluation_indictrans2.sh` | IndicTrans2 1B baseline on IN22, scored like our own checkpoints | 2 GPUs, 16 CPUs, 64 GB, 12 h | IN22 raw files (Job8 output), gated IndicTrans2 models, a separate environment | `evaluation/results/indictrans2-1B/` |

## Notes

- **Job23** needs `podman`. Rootless `podman` needs a subuid and subgid range for the user. See
  [`../deployment/README.md`](../deployment/README.md).
- Training jobs resume only when asked (`--export=ALL,RESUME=1`). See [`../training/README.md`](../training/README.md).
- Steps that need the internet (downloads) run outside these templates, for example on a login node. See
  [`../data/README.md`](../data/README.md).
