"""Baseline scores this project's own evaluation.run_eval.py output is
compared against, taken from a published model card rather than re-run
ourselves.

Source: https://huggingface.co/bodhan-ai/indic-translate model card, parsed
from the public model page's tables on 2026-09-19. It reports chrF++ on
**IN22-Gen only** (1,024 sentences per direction, 22 languages - matches this
project's own in22-gen.128k.spm.<short-tag> test files) for 12 models per
direction, in two separately-ranked tables (English -> Indic, Indic -> English),
plus a per-language table for three of them. No IN22-Conv numbers exist on the
card, so this project's own IN22-Conv scores have no external baseline; that
comparison is left blank rather than substituting the IN22-Gen numbers.

Protocol differences to keep in mind when comparing with this project's scores:
the card states its numbers are *sentence-level* chrF++ with greedy decoding;
this project reports sacrebleu's *corpus-level* chrF++ (word_order=2) with beam
size 5. Both are chrF++ on the same 1,024 sentences per direction, but the
aggregation differs. Rows marked in PARTIAL_LANGUAGE_MODELS cover only 16 of
the 22 languages (translategemma has no code path for Bodo, Dogri, Konkani,
Maithili, Manipuri or Santali) and are not comparable to the 22-language
aggregates. The by-language table reproduces the ranking tables' macro numbers
to within 0.2 (Sarvam Translate en->xx: 48.72 mean of its 22 languages against
48.9 in the ranking table); all numbers are used as published, not re-verified
by running the models.
"""

# {model_name: (en->xx macro chrF++, xx->en macro chrF++)}, in the card's English -> Indic
# ranking order.
INDIC_TRANSLATE_IN22_GEN_CHRF = {
    "Sarvam Translate": (48.9, 52.4),
    "IndicTrans2": (48.6, 62.8),
    "Indic Translate": (47.7, 60.4),
    "gemini-3.5-flash": (40.3, 66.6),
    "translategemma-27b-it": (39.6, 58.5),
    "gemma-4-31B-it": (39.2, 59.6),
    "translategemma-12b-it": (37.1, 55.0),
    "gemma-4-26B-a4b-it": (36.2, 56.6),
    "gemma-3-27b-it": (31.9, 55.5),
    "translategemma-4b-it": (31.2, 49.2),
    "gemma-3-12b-it": (28.8, 53.2),
    "gemma-3-4b-it": (22.9, 47.1),
}

# Models whose scores cover only 16 of the 22 IN22-Gen languages.
PARTIAL_LANGUAGE_MODELS = {'translategemma-12b-it', 'translategemma-4b-it', 'translategemma-27b-it'}

# {model_name: {language short tag: (en->xx chrF++, xx->en chrF++)}} - the card's by-language
# table, for the three models it reports per language. 'Indic Translate' is the card's own
# model (labelled v4 there); its values equal the card's other by-language table.
IN22_GEN_PER_LANGUAGE_CHRF = {
    "Indic Translate": {
        "as": (45.5, 64.5),
        "bn": (48.9, 62.1),
        "br": (48.5, 54.2),
        "doi": (54.5, 66.9),
        "gu": (52.3, 64.9),
        "hi": (55.5, 64.1),
        "kn": (51.1, 63.2),
        "ks": (34.4, 56.7),
        "gom": (43.8, 54.6),
        "mai": (50.2, 63.0),
        "ml": (47.1, 61.8),
        "mni": (44.3, 52.7),
        "mr": (50.2, 62.3),
        "ne": (49.3, 66.1),
        "or": (47.3, 64.8),
        "pa": (48.4, 59.2),
        "sa": (36.0, 53.6),
        "sat": (39.2, 47.3),
        "sd": (37.5, 55.5),
        "ta": (50.2, 58.4),
        "te": (51.3, 62.7),
        "ur": (64.0, 69.6),
    },
    "Sarvam Translate": {
        "as": (46.4, 54.1),
        "bn": (49.4, 53.1),
        "br": (48.9, 48.9),
        "doi": (58.2, 56.3),
        "gu": (53.1, 55.2),
        "hi": (56.4, 56.1),
        "kn": (52.2, 53.4),
        "ks": (35.1, 50.1),
        "gom": (44.9, 48.2),
        "mai": (50.5, 55.2),
        "ml": (47.7, 52.5),
        "mni": (45.5, 47.3),
        "mr": (51.3, 54.3),
        "ne": (49.9, 56.9),
        "or": (48.2, 54.6),
        "pa": (48.3, 53.6),
        "sa": (38.1, 47.5),
        "sat": (40.9, 43.4),
        "sd": (39.1, 49.0),
        "ta": (50.9, 50.1),
        "te": (51.6, 53.1),
        "ur": (65.3, 58.6),
    },
    "IndicTrans2": {
        "as": (47.1, 65.8),
        "bn": (51.8, 63.2),
        "br": (47.8, 62.1),
        "doi": (57.8, 72.6),
        "gu": (53.5, 66.5),
        "hi": (56.7, 65.4),
        "kn": (51.0, 64.2),
        "ks": (40.2, 60.4),
        "gom": (45.2, 59.2),
        "mai": (48.7, 64.8),
        "ml": (50.9, 64.5),
        "mni": (44.6, 57.9),
        "mr": (51.0, 63.7),
        "ne": (49.0, 67.7),
        "or": (43.9, 66.2),
        "pa": (50.6, 63.4),
        "sa": (38.8, 54.8),
        "sat": (33.4, 45.3),
        "sd": (36.6, 57.3),
        "ta": (49.5, 59.8),
        "te": (52.4, 64.8),
        "ur": (68.2, 73.0),
    },
}

# Parameter counts in billions: {model: (total, active-if-different, basis, note)}. Gathered from each
# model's public page or report on 2026-09-19, not measured here. 'nominal' = from the model name,
# 'rounded' = a rounded figure on the page, 'variant' = which variant was ranked is not stated,
# 'undisclosed' = no official count (None). This project's own model: 153,807,872 parameters.
MODEL_PARAMS_B = {
    "Sarvam Translate": (4.0, None, 'rounded', "Hugging Face model page shows '4B params'; fine-tuned from Gemma 3 4B"),
    "IndicTrans2": (1.1, None, 'variant', 'The 1.1B variant; the model card does not say which variant was ranked (a distilled variant is about 0.3B)'),
    "Indic Translate": (7.94, None, 'card', 'Total parameters as stated on its own model card; built on Gemma 4 E4B'),
    "gemini-3.5-flash": (None, None, 'undisclosed', 'Google does not publish a parameter count'),
    "translategemma-27b-it": (27.0, None, 'nominal', 'Size from the model name; built on Gemma 3 27B'),
    "gemma-4-31B-it": (30.7, None, 'primary', 'Total parameters on the Hugging Face model card'),
    "translategemma-12b-it": (12.0, None, 'nominal', 'Size from the model name; built on Gemma 3 12B'),
    "gemma-4-26B-a4b-it": (25.2, 3.8, 'primary', 'Mixture of experts: 25.2B total, 3.8B active per token (model card)'),
    "gemma-3-27b-it": (27.43, None, 'primary', 'Gemma 3 technical report parameter table, including the vision encoder'),
    "translategemma-4b-it": (4.0, None, 'nominal', 'Size from the model name; built on Gemma 3 4B'),
    "gemma-3-12b-it": (12.19, None, 'primary', 'Gemma 3 technical report parameter table, including the vision encoder'),
    "gemma-3-4b-it": (4.3, None, 'primary', 'Gemma 3 technical report parameter table, including the vision encoder'),
}
OUR_PARAMS = 153_807_872
