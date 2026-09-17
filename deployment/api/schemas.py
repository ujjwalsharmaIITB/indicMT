import os

from pydantic import BaseModel

# Shown in the UI. Override with the DISCLAIMER environment variable.
DISCLAIMER = os.environ.get("DISCLAIMER", "Research model; outputs may contain errors.")


class TranslateRequest(BaseModel):
    text: str
    source_lang: str  # short tag, e.g. "hi", "en"
    target_lang: str


class TranslateResponse(BaseModel):
    translation: str
    source_lang: str
    target_lang: str
    pivoted: bool
    pivot_intermediate: str | None = None
    disclaimer: str = DISCLAIMER


class LanguageInfo(BaseModel):
    code: str
    name: str


class HealthResponse(BaseModel):
    status: str
    checkpoint: str
    quantization: str
    disclaimer: str = DISCLAIMER
