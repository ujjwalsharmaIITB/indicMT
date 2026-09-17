"""Thin routing layer over deployment/ct2_infer.py's Ct2Translator: decides
direct vs. pivot translation from the source/target language codes alone,
no per-language branches (generic over all 44 directions plus the xx->xx
pivot)."""

import sys
from pathlib import Path

# ct2_infer.py sits at deployment/, a sibling of this api/ package - make
# this import work whether uvicorn is started with cwd=repo-root
# ("deployment.api.main:app", bare-metal dev) or cwd=/app inside the
# container ("api.main:app", where deployment/ct2_infer.py was COPY'd to
# /app/ct2_infer.py) - both cases resolve correctly via this explicit path
# insert rather than relying on whichever cwd happens to already be on
# sys.path.
_DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent
if str(_DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOYMENT_DIR))

from ct2_infer import Ct2Translator  # noqa: E402

from .languages import tag_for_short  # noqa: E402


class RoutedTranslator:
    def __init__(self, model_dir: str, spm_path: str, inter_threads: int = 1,
                 intra_threads: int = 0, beam_size: int = 2):
        self.ct2 = Ct2Translator(model_dir, spm_path, inter_threads=inter_threads,
                                  intra_threads=intra_threads)
        self.beam_size = beam_size

    def translate(self, text: str, source_lang: str, target_lang: str) -> dict:
        target_tag = tag_for_short(target_lang)

        if source_lang == "en" or target_lang == "en":
            translation = self.ct2.translate_one(text, target_tag, beam_size=self.beam_size)
            return {"translation": translation, "pivoted": False, "pivot_intermediate": None}

        translation, english = self.ct2.translate_pivot(text, target_tag, beam_size=self.beam_size)
        return {"translation": translation, "pivoted": True, "pivot_intermediate": english}
