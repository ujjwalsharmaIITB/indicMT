"""Single source of truth for the 22-language + English tag scheme used across
every data-pipeline script (bpcc-human and bpcc-mined), including the "eng"
entry for English.
"""

# ai4bharat/BPCC config keys for the 22 Scheduled Indian languages (lang_Script).
LANGUAGES = [
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr", "hin_Deva",
    "kan_Knda", "kas_Arab", "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Mtei", "npi_Deva",
    "ory_Orya", "pan_Guru", "san_Deva", "snd_Deva", "sat_Olck", "tam_Taml", "tel_Telu", "urd_Arab",
]

# 3-letter ISO code (as sliced from a BPCC src_lang/tgt_lang value, e.g. "hin_Deva"[:3])
# -> language tag prepended to the source sentence during multilingual dataset creation.
# Includes "eng" for English, which BPCC's src_lang/tgt_lang columns use constantly
# since BPCC is an English-centric corpus.
LANGUAGE_CODES = {
    "eng": "<en>",
    "asm": "<as>", "ben": "<bn>", "brx": "<br>", "doi": "<doi>", "gom": "<gom>",
    "guj": "<gu>", "hin": "<hi>", "kan": "<kn>", "kas": "<ks>", "mai": "<mai>",
    "mal": "<ml>", "mar": "<mr>", "mni": "<mni>", "npi": "<ne>", "ory": "<or>",
    "pan": "<pa>", "san": "<sa>", "snd": "<sd>", "sat": "<sat>", "tam": "<ta>",
    "tel": "<te>", "urd": "<ur>",
}

UNK_TAG = "<unk>"


def tag_for(lang_code: str) -> str:
    """Look up the language tag for a full BPCC lang code (e.g. "hin_Deva" or "eng_Latn"),
    falling back to UNK_TAG for anything unrecognized instead of raising."""
    return LANGUAGE_CODES.get(lang_code[:3], UNK_TAG)


def canonicalize_lang_code(example: dict, lang: str):
    """Fix whichever of src_lang/tgt_lang shares `lang`'s 3-letter prefix but
    disagrees on the script suffix.

    BPCC's own per-row src_lang/tgt_lang metadata can disagree with the
    language actually requested from a config split. Confirmed case (reported
    upstream to AI4Bharat): every one of the 79,968 rows in the bpcc-human
    sat_Olck/bpcc-seed-latest split has tgt_lang="sat_Beng" in its own
    metadata, despite the text being unambiguous Ol Chiki script, not
    Bengali (verified against the actual Unicode characters, not inferred
    from the label). Since the config split we actually requested (`lang`) is
    ground truth for what we downloaded, whichever side's declared code
    shares its 3-letter prefix but not its script suffix gets corrected to
    `lang` itself.

    Returns (corrected_example, correction_made: bool).
    """
    corrected = dict(example)
    changed = False
    for side in ("src_lang", "tgt_lang"):
        val = corrected.get(side) or ""
        if val[:3] == lang[:3] and val != lang:
            corrected[side] = lang
            changed = True
    return corrected, changed
