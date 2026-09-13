"""Unit tests for data/common/scripts.py - the Unicode script-consistency
quality gate. No downloaded data required, always runnable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.scripts import (  # noqa: E402
    expected_script,
    is_script_consistent,
    lang_side_text,
    script_match_fraction,
)

HINDI_TEXT = "मुझे हिंदी आती है"  # Devanagari
BENGALI_TEXT = "আমি বাংলা জানি"  # Bengali
ENGLISH_TEXT = "I know Hindi well"


def test_expected_script():
    assert expected_script("hin_Deva") == "Deva"
    assert expected_script("eng_Latn") == "Latn"
    assert expected_script("kas_Arab") == "Arab"
    assert expected_script(None) is None


def test_matching_script_scores_high():
    frac = script_match_fraction(HINDI_TEXT, "Deva")
    assert frac is not None and frac > 0.9


def test_mismatched_script_scores_low():
    frac = script_match_fraction(BENGALI_TEXT, "Deva")
    assert frac is not None and frac < 0.5


def test_empty_text_is_none():
    assert script_match_fraction("", "Deva") is None
    assert script_match_fraction("   123 ", "Deva") is None


def test_unknown_script_tag_is_none():
    assert script_match_fraction(HINDI_TEXT, "NotAScript") is None


def test_is_script_consistent_true_for_matching_text():
    assert is_script_consistent(HINDI_TEXT, "hin_Deva") is True


def test_is_script_consistent_false_for_mismatched_text():
    assert is_script_consistent(BENGALI_TEXT, "hin_Deva") is False


def test_is_script_consistent_true_when_nothing_to_judge():
    # Digits/punctuation-only text can't contradict the label.
    assert is_script_consistent("123 !!!", "hin_Deva") is True


def test_lang_side_text_picks_the_correct_side():
    row = {"src": ENGLISH_TEXT, "tgt": HINDI_TEXT, "src_lang": "eng_Latn", "tgt_lang": "hin_Deva"}
    assert lang_side_text(row, "hin_Deva") == HINDI_TEXT
    assert lang_side_text(row, "eng_Latn") == ENGLISH_TEXT


def test_lang_side_text_none_when_no_match():
    row = {"src": ENGLISH_TEXT, "tgt": HINDI_TEXT, "src_lang": "eng_Latn", "tgt_lang": "hin_Deva"}
    assert lang_side_text(row, "ben_Beng") is None
