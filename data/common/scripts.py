"""Unicode-script consistency checks, per language.

BPCC's per-language HF config keys already commit to one script per language
(e.g. "kas_Arab" only ever pulls Perso-Arabic Kashmiri - there is no separate
"kas_Deva" config in our LANGUAGES list). But the *rows themselves* inside a
config are not guaranteed to actually be in that script: script/language
mislabeling is a known issue in web-mined multilingual corpora, and is the
main quality risk in bpcc-mined's comparable / samanantar-filtered /
nllb-filtered configs in particular.

This module detects which Unicode script a piece of text actually uses and
checks it against the script implied by a language's BPCC code (e.g.
"hin_Deva" -> Devanagari), so mismatched rows can be dropped instead of
silently poisoning training/eval data with the wrong script. This is the
"flag inconsistencies, drop the minority script, keep the dominant one" data
quality gate applied in create_splits.py and the multilingual dataset
builders.
"""

# BPCC script tag (the suffix of a lang_Script code) -> Unicode codepoint
# ranges that count as "written in this script". Deliberately covers only the
# core block for each script - good enough to catch gross mislabeling
# (e.g. Bengali text under a "hin_Deva" label) without needing a full
# language-ID model.
SCRIPT_RANGES = {
    "Beng": [(0x0980, 0x09FF)],                          # Assamese, Bengali
    "Deva": [(0x0900, 0x097F)],                           # Hindi, Marathi, Nepali, Sanskrit, Bodo, Dogri, Konkani, Maithili, Sindhi(Deva variant)
    "Gujr": [(0x0A80, 0x0AFF)],                            # Gujarati
    "Guru": [(0x0A00, 0x0A7F)],                            # Punjabi
    "Knda": [(0x0C80, 0x0CFF)],                            # Kannada
    "Mlym": [(0x0D00, 0x0D7F)],                            # Malayalam
    "Mtei": [(0xAAE0, 0xAAFF), (0xABC0, 0xABFF)],          # Manipuri/Meetei Mayek
    "Olck": [(0x1C50, 0x1C7F)],                            # Santali/Ol Chiki
    "Orya": [(0x0B00, 0x0B7F)],                            # Odia
    "Taml": [(0x0B80, 0x0BFF)],                            # Tamil
    "Telu": [(0x0C00, 0x0C7F)],                            # Telugu
    "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],  # Kashmiri, Sindhi(Arab variant), Urdu
    "Latn": [(0x0041, 0x005A), (0x0061, 0x007A)],          # English
}

DEFAULT_THRESHOLD = 0.5


def expected_script(lang_code: str) -> str:
    """e.g. "hin_Deva" -> "Deva", "eng_Latn" -> "Latn"."""
    return lang_code.rsplit("_", 1)[-1] if lang_code and "_" in lang_code else None


def script_match_fraction(text: str, script_tag: str):
    """Fraction of alphabetic characters in `text` that fall inside
    `script_tag`'s Unicode ranges. None if the script is unknown or the text
    has no alphabetic characters to judge (empty/whitespace/digits/punct)."""
    ranges = SCRIPT_RANGES.get(script_tag)
    if ranges is None or not text:
        return None

    total = 0
    matched = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in ranges):
            matched += 1

    if total == 0:
        return None
    return matched / total


def is_script_consistent(text: str, lang_code: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True if `text` is judged to actually be written in the script implied
    by `lang_code`. Text with nothing alphabetic to judge is treated as
    consistent (nothing there to contradict the label)."""
    frac = script_match_fraction(text, expected_script(lang_code))
    return frac is None or frac >= threshold


def lang_side_text(row: dict, lang_code: str):
    """Pick out whichever of a BPCC row's src/tgt corresponds to `lang_code`
    (BPCC rows are English-centric: one side is English, the other is the
    Indic language), by matching the 3-letter ISO prefix. Returns None if
    neither side matches."""
    prefix = lang_code[:3]
    if row.get("src_lang", "")[:3] == prefix:
        return row.get("src")
    if row.get("tgt_lang", "")[:3] == prefix:
        return row.get("tgt")
    return None
