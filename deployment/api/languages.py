"""Wire-format language codes (short tags, e.g. "hi") + display names for
the API/frontend, built on top of the mechanically-copied
deployment/api/common/languages.py (see deployment/stage_common.py).
Display names are new here (data/common/languages.py has no display-name
mapping - it only needs the BPCC lang_Script <-> tag mapping for the data
pipeline)."""

from .common.languages import LANGUAGE_CODES

# short tag (bracket-stripped LANGUAGE_CODES value) -> human-readable name
DISPLAY_NAMES = {
    "en": "English",
    "as": "Assamese", "bn": "Bengali", "br": "Bodo", "doi": "Dogri",
    "gom": "Konkani", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
    "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mr": "Marathi",
    "mni": "Manipuri (Meitei)", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
    "sa": "Sanskrit", "sd": "Sindhi", "sat": "Santali", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}

# short tag -> full "<xx>" tag, derived from LANGUAGE_CODES (3-letter code keys)
SHORT_TO_TAG = {v.strip("<>"): v for v in LANGUAGE_CODES.values()}
VALID_CODES = set(SHORT_TO_TAG)


def tag_for_short(code: str) -> str:
    """"hi" -> "<hi>". Raises KeyError for an unknown code - callers should
    validate against VALID_CODES first for a clean 400, not a 500."""
    return SHORT_TO_TAG[code]


def language_list() -> list:
    return [{"code": code, "name": DISPLAY_NAMES.get(code, code)} for code in sorted(VALID_CODES)]
