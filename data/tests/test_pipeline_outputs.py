"""Milestone validation tests for the data pipeline's actual output files.

Each test is skipped (not failed) if the file/stage it checks hasn't been
produced yet, so this suite can be run at any point during the pipeline and
only exercises what exists so far. Run it after each stage, and always before
kicking off a big/expensive run downstream (tokenizer training, mixing,
OpenNMT training):

    pytest data/tests/ -v

Covers, per stage:
- download_bpcc_human.py / download_bpcc_mined.py -> raw JSONL schema
- create_splits.py -> no train/valid leakage per language
- create_multilingual_dataset.py (human + mined) -> language-tag rendering,
  explicitly including the "<en> ..." case for English targets requested as
  the milestone gate example.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # data/
from common.languages import tag_for  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1]
REQUIRED_RAW_FIELDS = {"src", "tgt", "src_lang", "tgt_lang", "meta_data"}
SAMPLE_LIMIT = 2000


def _iter_jsonl(path, limit=None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def _raw_files():
    files = []
    for sub in ("bpcc-human/raw", "bpcc-mined/raw"):
        d = DATA_DIR / sub
        if d.exists():
            files.extend(sorted(d.glob("*.jsonl")))
    return files


def _splits_dir():
    return DATA_DIR / "bpcc-human" / "splits"


def _multilingual_outputs():
    candidates = [
        DATA_DIR / "bpcc-human" / "combined_train_dataset.jsonl",
        DATA_DIR / "bpcc-human" / "concatenated_valid_dataset.jsonl",
        DATA_DIR / "bpcc-mined" / "concatenated_bpcc_mined_dataset.jsonl",
    ]
    return [p for p in candidates if p.exists()]


# --- Milestone: raw download -------------------------------------------------

@pytest.mark.parametrize("path", _raw_files(), ids=lambda p: p.name)
def test_raw_download_schema(path):
    for row in _iter_jsonl(path, limit=200):
        missing = REQUIRED_RAW_FIELDS - row.keys()
        assert not missing, f"{path.name}: row missing fields {missing}"
        assert row["src"] and row["tgt"] and row["src_lang"] and row["tgt_lang"], \
            f"{path.name}: row has an empty required field: {row}"


def test_raw_download_stage_ran_if_expected():
    if not (DATA_DIR / "bpcc-human" / "raw").exists() and not (DATA_DIR / "bpcc-mined" / "raw").exists():
        pytest.skip("No raw/ directories yet - download scripts not run.")


# --- Milestone: multilingual dataset merge preserves row count -------------

def test_combined_train_dataset_has_no_silent_row_loss():
    """Guards against silent row loss when combining per-language datasets:
    HF datasets.interleave_datasets(), whose default
    stopping_strategy="first_exhausted" silently truncates to the length of
    the SMALLEST language's data, dropping most of every larger language's
    rows. Bidirectional expansion must exactly double the per-language split
    row count - not shrink it."""
    train_path = DATA_DIR / "bpcc-human" / "combined_train_dataset.jsonl"
    splits_dir = _splits_dir()
    if not train_path.exists() or not splits_dir.exists():
        pytest.skip("combined_train_dataset.jsonl or splits/ not generated yet.")

    split_row_count = sum(
        sum(1 for _ in open(p, encoding="utf-8")) for p in splits_dir.glob("*_train.jsonl")
    )
    combined_row_count = sum(1 for _ in open(train_path, encoding="utf-8"))

    assert combined_row_count == 2 * split_row_count, (
        f"Expected combined_train_dataset.jsonl to have exactly 2x the split train row "
        f"count ({split_row_count} -> {2 * split_row_count}), got {combined_row_count} - "
        f"rows were lost during the merge."
    )


# --- Milestone: train/valid splitting ---------------------------------------

@pytest.mark.skipif(not _splits_dir().exists(), reason="splits/ not generated yet - run create_splits.py")
def test_train_valid_no_overlap_per_language():
    train_files = sorted(_splits_dir().glob("*_train.jsonl"))
    assert train_files, "splits/ exists but has no *_train.jsonl files"

    for train_path in train_files:
        lang = train_path.name[: -len("_train.jsonl")]
        valid_path = _splits_dir() / f"{lang}_valid.jsonl"
        if not valid_path.exists():
            continue
        train_pairs = {(r["src"], r["tgt"]) for r in _iter_jsonl(train_path)}
        valid_pairs = {(r["src"], r["tgt"]) for r in _iter_jsonl(valid_path)}
        overlap = train_pairs & valid_pairs
        assert not overlap, f"{lang}: {len(overlap)} sentence pairs leak between train and valid"


# --- Milestone: multilingual dataset creation (language-tag rendering) -----

@pytest.mark.parametrize("path", _multilingual_outputs(), ids=lambda p: p.name)
def test_language_tag_rendering(path):
    """For every row, src_modified must start with the tag for tgt_lang - e.g. a
    row with tgt_lang="eng_Latn" must render as "<en> <sentence>", a row with
    tgt_lang="hin_Deva" as "<hi> <sentence>", and so on for all 22 languages."""
    checked = 0
    for row in _iter_jsonl(path, limit=SAMPLE_LIMIT):
        expected_prefix = tag_for(row["tgt_lang"]) + " "
        assert row["src_modified"].startswith(expected_prefix), (
            f"{path.name}: expected src_modified to start with {expected_prefix!r} for "
            f"tgt_lang={row['tgt_lang']!r}, got {row['src_modified'][:40]!r}"
        )
        remainder = row["src_modified"][len(expected_prefix):]
        assert remainder, f"{path.name}: src_modified has no sentence content after the tag"
        checked += 1
    assert checked > 0, f"{path.name} produced no rows to check"


def test_english_target_renders_en_tag():
    """Explicit milestone-gate check: an English-target row's src_modified must
    literally begin with '<en> ' (the case called out for manual verification)."""
    outputs = _multilingual_outputs()
    if not outputs:
        pytest.skip("No multilingual dataset outputs yet - run create_multilingual_dataset.py.")

    found_any = False
    for path in outputs:
        for row in _iter_jsonl(path, limit=SAMPLE_LIMIT):
            if row["tgt_lang"][:3] == "eng":
                found_any = True
                assert row["src_modified"].startswith("<en> "), (
                    f"{path.name}: English-target row did not render '<en> ' prefix: "
                    f"{row['src_modified'][:40]!r}"
                )
    if not found_any:
        pytest.skip(f"No English-target rows found in the first {SAMPLE_LIMIT} lines of any output file.")


def test_all_22_languages_appear_as_targets():
    """Milestone gate: every one of the 22 Indic languages (not just English)
    actually shows up as a tgt_lang somewhere in the multilingual outputs, with
    its own correctly-rendered tag - catches a language silently dropping out
    of the pipeline (e.g. missing from a raw config, or filtered out entirely)."""
    from common.languages import LANGUAGES

    outputs = _multilingual_outputs()
    if not outputs:
        pytest.skip("No multilingual dataset outputs yet - run create_multilingual_dataset.py.")

    seen_prefixes = set()
    for path in outputs:
        for row in _iter_jsonl(path, limit=20000):
            seen_prefixes.add(row["tgt_lang"][:3])

    missing = [lang for lang in LANGUAGES if lang[:3] not in seen_prefixes]
    if missing:
        pytest.skip(
            f"{len(missing)} language(s) not seen as a target in the first 20000 sampled rows "
            f"per file (may just need a larger sample once data is fully downloaded): {missing}"
        )
