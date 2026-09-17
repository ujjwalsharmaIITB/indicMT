"""FastAPI app: loads the CTranslate2 translator once at startup (lifespan),
exposes /api/health, /api/languages, /api/translate, and mounts the static
frontend at "/" - one process, one port, no compose file needed.

Config via env vars (all have sensible defaults for local/dev use; the
Dockerfile sets MODEL_DIR/SPM_PATH to the staged deployment/model/ paths):
  MODEL_DIR       - CTranslate2 model directory (default: deployment/model/ct2_model)
  SPM_PATH        - SentencePiece .model file (default: deployment/model/spm_128000.model)
  CHECKPOINT_NAME - identity string for /api/health (default: indicmt-final-int8)
  QUANTIZATION    - identity string for /api/health (default: int8)
  INTER_THREADS   - CTranslate2 parallel-translation thread count (default: 4)
  INTRA_THREADS   - CTranslate2 per-translation OpenMP threads (default: 1)
  DISCLAIMER      - notice shown in the UI (default: "Research model; outputs may contain errors.")
  BEAM_SIZE       - interactive beam size, deliberately low for latency, NOT the
                     beam=5 used for offline evaluation (default: 2)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .schemas import DISCLAIMER, HealthResponse, LanguageInfo, TranslateRequest, TranslateResponse
from .languages import VALID_CODES, language_list
from .translator import RoutedTranslator

DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.environ.get("MODEL_DIR", str(DEPLOYMENT_DIR / "model" / "ct2_model"))
SPM_PATH = os.environ.get("SPM_PATH", str(DEPLOYMENT_DIR / "model" / "spm_128000.model"))
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "indicmt-final-int8")
QUANTIZATION = os.environ.get("QUANTIZATION", "int8")
INTER_THREADS = int(os.environ.get("INTER_THREADS", "4"))
INTRA_THREADS = int(os.environ.get("INTRA_THREADS", "1"))
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "2"))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(MODEL_DIR).exists():
        raise RuntimeError(
            f"MODEL_DIR not found: {MODEL_DIR} - run deployment/convert_to_ct2.py "
            "and stage the chosen variant into deployment/model/ct2_model first."
        )
    state["translator"] = RoutedTranslator(
        MODEL_DIR, SPM_PATH, inter_threads=INTER_THREADS, intra_threads=INTRA_THREADS,
        beam_size=BEAM_SIZE,
    )
    yield
    state.clear()


app = FastAPI(title="IndicMT", lifespan=lifespan)


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if "translator" in state else "loading",
                           checkpoint=CHECKPOINT_NAME, quantization=QUANTIZATION)


@app.get("/api/languages", response_model=list[LanguageInfo])
def languages():
    return language_list()


@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if req.source_lang not in VALID_CODES:
        raise HTTPException(400, f"unknown source_lang: {req.source_lang!r}")
    if req.target_lang not in VALID_CODES:
        raise HTTPException(400, f"unknown target_lang: {req.target_lang!r}")
    if req.source_lang == req.target_lang:
        raise HTTPException(400, "source_lang and target_lang must differ")
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")

    result = state["translator"].translate(req.text, req.source_lang, req.target_lang)
    return TranslateResponse(
        translation=result["translation"], source_lang=req.source_lang,
        target_lang=req.target_lang, pivoted=result["pivoted"],
        pivot_intermediate=result["pivot_intermediate"], disclaimer=DISCLAIMER,
    )


# Static frontend, mounted last so it doesn't shadow the /api routes above.
_frontend_dir = DEPLOYMENT_DIR / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
