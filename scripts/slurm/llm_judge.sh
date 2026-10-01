#!/bin/bash
#SBATCH --job-name=job19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Job19: LLM-as-judge over one evaluated checkpoint (spec: evaluation/llm_judge/PROMPT.md, code:
# evaluation/llm_judge/judge.py). vLLM offline batch inference, tensor parallel over 2 GPUs of 48 GB.
# Compute nodes may have no internet, so the model must already be on disk and the HF libraries
# run offline. Set JUDGE_MODEL_DIR to the local model directory.
#
# Usage: sbatch -p <partition> --export=ALL,JUDGE_MODEL_DIR=/path/to/gemma-4-31B-it scripts/slurm/llm_judge.sh [judge.py args ...]
# Default (no args): --backend vllm --model ${JUDGE_MODEL_DIR} --arm bpcc-ft-combined-human --step 1800
# Any args replace the default entirely, e.g.
#   sbatch -p <partition> --export=ALL,JUDGE_MODEL_DIR=/path/to/gemma-4-31B-it scripts/slurm/llm_judge.sh \
#       --backend vllm --model /path/to/gemma-4-31B-it --arm bpcc-ft-combined-human --step 1800 --variant both
#   ... --sanity       (identity and mismatch sanity checks)
# Re-submitting the same command resumes: finished items are skipped (results.jsonl is appended per chunk).

# --- Environment (adapt to your cluster) ---
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}"; cd "$REPO_DIR"
source activate "${JUDGE_CONDA_ENV:-indicmt-judge}"  # the judge environment (vLLM); adapt to your environment

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
# FlashInfer JIT-compiles its sampling kernels on first use and needs nvcc, which compute nodes
# often do not have; fall back to the PyTorch top-k/top-p sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8

mkdir -p logs

if [ "$#" -eq 0 ]; then
    : "${JUDGE_MODEL_DIR:?set JUDGE_MODEL_DIR or pass judge.py args}"
    set -- --backend vllm --model "$JUDGE_MODEL_DIR" --arm bpcc-ft-combined-human --step 1800
fi

echo "judge job $SLURM_JOB_ID: python evaluation/llm_judge/judge.py $* ($(date))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python evaluation/llm_judge/judge.py "$@"
rc=$?
echo "judge exit code $rc ($(date))"
exit $rc
