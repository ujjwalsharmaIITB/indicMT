# Build context is deployment/ itself, not the repo root (data and training
# outputs are very large - a root-context build would send that whole tree
# to the build daemon on every build). This means deployment/api/common/
# (the mechanical copies of data/common/languages.py and tokenization.py) must be
# populated BEFORE running `podman build` - see deployment/stage_common.py.
#
# The model is not baked into the image: at startup the API serves the int8 model
# from the Hugging Face cache, downloading it once if it is not there. Mount a volume
# at /root/.cache/huggingface to keep the cache across restarts, and pass HF_TOKEN
# while the repository is private.
#
# python:3.11-slim, not Alpine: CTranslate2 ships manylinux/glibc wheels
# only, no musl wheel exists - Alpine would force a slow/unreliable
# from-source build. No PyTorch in this image - CTranslate2's CPU inference
# has no PyTorch runtime dependency, which is most of the "minimal" size win
# by itself.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY frontend ./frontend
COPY ct2_infer.py .

ENV HF_REPO=ujjwalsharmaIITB/indicMT
ENV CHECKPOINT_NAME=indicmt-final-int8
ENV QUANTIZATION=int8

EXPOSE 8000

# Single uvicorn worker: a second worker would load a second full copy of
# the CT2 model into memory; CTranslate2's own inter_threads already
# parallelizes within one process. Revisit only if the stress test shows
# Python-side request handling (not CT2 itself) is the bottleneck.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
