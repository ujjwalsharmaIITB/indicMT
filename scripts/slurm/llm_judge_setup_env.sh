#!/bin/bash
#SBATCH --job-name=job16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Job16: build a separate conda environment (python 3.11) for running the LLM judge: vLLM plus a
# transformers version recent enough for the Gemma 4 architecture (the model config asks for
# transformers 5.5). Reports what installed and whether this vLLM knows the Gemma4 architecture.
# Package installs are CPU and network work, so this runs as a job.
# See evaluation/llm_judge/README.md.
JUDGE_ENV="${JUDGE_CONDA_ENV:-indicmt-judge}"
if ! conda env list | grep -q "^$JUDGE_ENV "; then conda create -y -n "$JUDGE_ENV" python=3.11; fi
source activate "$JUDGE_ENV"  # adapt to your environment
python -m pip install --upgrade pip
python -m pip install vllm
python -m pip install --upgrade "transformers>=5.5" || python -m pip install --upgrade --pre transformers
python -m pip list 2>/dev/null | grep -i "^vllm \|^torch \|^transformers \|^accelerate \|^flash"
python - <<'PY'
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    print("vLLM knows Gemma4ForConditionalGeneration:", "Gemma4ForConditionalGeneration" in archs)
    print("gemma archs:", sorted(a for a in archs if "emma" in a))
except Exception as e:
    print("vLLM registry check failed:", e)
PY
echo "Done ($(date))"
