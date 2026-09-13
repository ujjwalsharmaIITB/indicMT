"""Unit tests for data/common/languages.py — no downloaded data required, always runnable.

Milestone gate: language tag scheme is internally consistent (every BPCC
language has a tag, English is covered, no duplicate tags) before it's relied
on by any downstream script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.languages import LANGUAGES, LANGUAGE_CODES, UNK_TAG, tag_for  # noqa: E402


def test_all_22_languages_present():
    assert len(LANGUAGES) == 22
    assert len(set(LANGUAGES)) == 22


def test_english_tag_present():
    assert LANGUAGE_CODES.get("eng") == "<en>"


def test_every_bpcc_language_has_a_tag():
    missing = [lang for lang in LANGUAGES if lang[:3] not in LANGUAGE_CODES]
    assert not missing, f"Languages missing a LANGUAGE_CODES entry: {missing}"


def test_tag_for_known_languages_is_not_unk():
    for lang in LANGUAGES:
        tag = tag_for(lang)
        assert tag != UNK_TAG, f"{lang} resolved to {UNK_TAG}, expected a real tag"
        assert tag.startswith("<") and tag.endswith(">")


def test_tag_for_english():
    assert tag_for("eng_Latn") == "<en>"


def test_tag_for_unknown_language_falls_back_to_unk():
    assert tag_for("xyz_Zzzz") == UNK_TAG


def test_tags_are_unique():
    tags = list(LANGUAGE_CODES.values())
    assert len(tags) == len(set(tags)), "Duplicate language tags found"
