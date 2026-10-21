# deployment: CTranslate2 conversion, quantization, API and frontend

This directory turns a trained checkpoint into a small CPU-served model with a FastAPI backend and a plain HTML/JS
frontend. The explanation and all numbers are in [`../docs/deployment.md`](../docs/deployment.md). The model served is the
**release model** (12+12, combined pre-training then human fine-tuning without replay, step 1,800), converted to
**int8** (170.6 MB, runs on CPU). Weights: [ujjwalsharmaIITB/indicMT](https://huggingface.co/ujjwalsharmaIITB/indicMT) on Hugging Face.

## Files

| File | Purpose |
|---|---|
| `convert_to_ct2.py` | Converts a checkpoint to CTranslate2 in three CPU-usable variants: `float32`, `int16`, `int8`. |
| `ct2_eval.py` | Runs the full 44-direction IN22 evaluation (IN22-Gen by default, `--test-set in22-conv`) directly through CTranslate2 for one variant: chrF++, BLEU, latency and throughput. |
| `ct2_infer.py` | The one shared inference path (tag and SentencePiece-encode with `data/common/tokenization.py`, `translate_batch`, decode), used by the API and as a command-line translator. `ct2_eval.py` uses the same encoding function. |
| `stage_common.py` | Copies `data/common/languages.py` and `data/common/tokenization.py` into `deployment/api/common/` for the container build. |
| `api/` | FastAPI app: `main.py`, `schemas.py`, the direct/pivot router `translator.py`, `languages.py`. |
| `frontend/` | Plain HTML, CSS and JS, served by the same process. No build step. |
| `stress_test.py` | asyncio and httpx load generator (p50, p95, p99 and requests per second). |
| `Dockerfile`, `.dockerignore`, `requirements.txt` | Container image (Python 3.11 slim, no PyTorch). `requirements.txt` here has the same pins as the top-level `requirements-deploy.txt`; the Docker build context is `deployment/`, so it needs its own copy. |

## 1. Convert and compare quantization variants

```bash
sbatch -p <partition> scripts/slurm/deployment_convert_ct2.sh                 # Job20, CPU only
# defaults to the release model; override with CHECKPOINT=... CT2_ROOT=...
```

This converts the checkpoint, evaluates float32, int16 and int8 on IN22-Gen through CTranslate2, and writes
`quantization_report.md`. Outputs go beside the checkpoint, in
`128k-spm/opennmt/<arm>/ct2/<checkpoint>/{float32,int16,int8}/`, and are not tracked by git.

Two compatibility problems were found and fixed in the converter:

1. OpenNMT-py's default `torch.load(..., weights_only=True)` rejects the checkpoint's `opt` field. We pass CTranslate2's
   `unsafe_deserialization=True`, which is fine for a self-trained, trusted checkpoint. Do not use it on untrusted files.
2. CTranslate2 rejects `self_attn_type == "scaled-dot-flash"`, which OpenNMT-py 3.5.1 chose automatically. The script
   relabels it `"scaled-dot"` in a patched temporary copy. This only names the attention kernel; the learned weights are the
   same. The original checkpoint is not modified.

Quantization on the release model (CPU, 8 threads, beam 5, IN22-Gen):

| Variant | en→xx chrF++ | xx→en chrF++ | p50 latency | Size |
|---|---|---|---|---|
| float32 | 47.15 | 53.87 | 399.2 ms | 630.4 MB |
| int16 | 47.15 | 53.86 | 367.1 ms | 323.2 MB |
| int8 | 47.16 | 53.85 | 374.2 ms | 170.6 MB |

With training-format inputs, neither int16 nor int8 is significantly different from float32 (paired bootstrap, pooled
p = 0.18 for int8 en→xx and 0.15 for xx→en, no direction significant after Bonferroni correction), so int8 is the
serving choice. Job21 (`scripts/slurm/deployment_quant_significance.sh`) runs the significance test.

## 2. Run the API (no container needed)

The API takes the int8 model and tokenizer from [Hugging Face](https://huggingface.co/ujjwalsharmaIITB/indicMT) (its model card is [`../huggingface/README.md`](../huggingface/README.md)). At startup it uses the local Hugging Face cache if the model is already there, and downloads it into that cache otherwise, so only the first start downloads. To serve your own conversion (step 1) instead, set `MODEL_DIR` and `SPM_PATH`; when both exist they are used and nothing is downloaded.

```bash
pip install -r requirements-deploy.txt   # minimal serving environment (CPU, no PyTorch)
python deployment/stage_common.py
# while the Hugging Face repository is private: huggingface-cli login (or export HF_TOKEN=...)

# quick command-line check of the raw CTranslate2 model
# (needs local files: huggingface-cli download ujjwalsharmaIITB/indicMT --include "indicMT/*" "indicMT.spm.model" --local-dir hf-indicmt)
python deployment/ct2_infer.py --model-dir hf-indicmt/indicMT \
  --spm hf-indicmt/indicMT.spm.model --text "Hello, how are you?" --target-lang hi

# serve (open http://localhost:8000/ for the frontend)
uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000
```

`deployment/model/` is not tracked by git.

Endpoints:

- `POST /api/translate`, body `{"text": "...", "source_lang": "en", "target_lang": "hi"}` (short language tags). The
  response has `translation`, `pivoted` and `pivot_intermediate`. Indic to Indic requests are **pivoted through English**:
  an xx→en translation followed by an en→xx translation with the same model.
- `GET /api/languages`: supported languages.
- `GET /api/health`: checkpoint name, quantization and notice.

```bash
curl -X POST localhost:8000/api/translate -H "Content-Type: application/json" \
  -d '{"text":"Hello, how are you?","source_lang":"en","target_lang":"hi"}'
curl -X POST localhost:8000/api/translate -H "Content-Type: application/json" \
  -d '{"text":"नमस्ते, आप कैसे हैं?","source_lang":"hi","target_lang":"ta"}'   # pivot: "pivoted": true
```

Settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `deployment/model/ct2_model` | CTranslate2 model directory |
| `SPM_PATH` | `deployment/model/spm_128000.model` | SentencePiece model |
| `HF_REPO` | `ujjwalsharmaIITB/indicMT` | Hugging Face repository used when `MODEL_DIR`/`SPM_PATH` are absent |
| `HF_REVISION` | `main` | Branch, tag or commit of `HF_REPO` |
| `CHECKPOINT_NAME` | `indicmt-final-int8` | label shown by `/api/health` |
| `QUANTIZATION` | `int8` | label shown by `/api/health` |
| `INTER_THREADS` / `INTRA_THREADS` | 4 / 1 | CTranslate2 parallelism |
| `BEAM_SIZE` | 2 | interactive beam size (lower than the 5 used offline, for latency) |
| `DISCLAIMER` | "Research model; outputs may contain errors." | notice shown in the UI |

## 3. Frontend

`frontend/` has two tabs: **Translate** (searchable language pickers, swap, copy, character counter, light, dark and system
themes, direct-or-pivot route badge, recent translations kept in browser storage) and **Benchmarks** (sortable and
filterable charts against other systems). It uses system fonts only and works offline.

The Benchmarks tab reads `frontend/baselines.json`, a small file of baseline scores. **The frontend needs it to draw
the comparison charts, so it is kept in the repository.** IndicTrans2 is our own run; the other baselines are taken from
the [`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate) model page. The tab also reads an
optional `frontend/benchmarks.json` with IndicMT's own scores: the CTranslate2 float32 reports from
`deployment/ct2_eval.py`, with IN22-Gen and IN22-Conv merged into one file. It is not tracked. Without it the tab shows
only the baselines.

## 4. Container build

The `Dockerfile` is ready: Python 3.11 slim, no PyTorch, one uvicorn worker (a second worker would load a second copy of
the model; CTranslate2 already parallelizes inside the process). The build context is `deployment/` itself, so stage the
model, the tokenizer and `python deployment/stage_common.py` first.

```bash
podman build -f deployment/Dockerfile -t indicmt-deploy:latest deployment/
podman run -d -p 8000:8000 indicmt-deploy:latest
```

Rootless `podman` needs a `/etc/subuid` and `/etc/subgid` range for the user to remap user IDs in base image layers. If
builds fail with a user-namespace error, an administrator must add a range (`usermod --add-subuids ... --add-subgids ...`)
and you then run `podman system migrate`. Docker works with the same commands.

Job23 (`scripts/slurm/deployment_container_check.sh`) builds the image, reports its size, starts it and checks
`/api/health` and `/api/translate` with `curl`.

## 5. Load test

```bash
sbatch -p <partition> scripts/slurm/deployment_stress_test.sh       # Job22
```

Job22 starts the API, waits for `/api/health`, runs `stress_test.py` (by direction class, payload size and concurrency) and
stops the server. The results go to `logs/stress_test_results.json`. To test an API that is already running, call the
script directly: `python deployment/stress_test.py --base-url http://<host>:8000 --output results.json`.

Next: [`../docs/deployment.md`](../docs/deployment.md).
