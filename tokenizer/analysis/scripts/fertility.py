"""Experiment A2 (in-house sweep) + A3 (external tokenizer comparison): fertility.

Fertility = subwords produced / whitespace words consumed, measured on held-out
text the tokenizer never trained on. See tokenizer/README.md and
tokenizer/analysis/README.md for the full methodology; this script
implements both in one pass since they share the same held-out corpus and word
definition.

**Held-out corpus**: data/in22/combined_fertility_eval.jsonl (IN22-Gen +
IN22-Conv combined, 2,527 multi-parallel rows across all 23 languages - see
tokenizer/README.md for why this replaced the
original bpcc-human-valid-split plan).

**Word definition**: identical to A1 (corpus_laws.py's `tokenize()` - NFKC
normalization, whitespace split, strip punctuation-only and digit-only
tokens) - reused directly, not reimplemented, so word counts are comparable
across A1/A2/A3. Note this undercounts slightly against a tokenizer's raw
subword count for punctuation-heavy text (SPM/BPE still emit pieces for
punctuation, which A1's word definition strips from the denominator) - this
affects every tokenizer measured here equally, so relative comparisons
remain fair even though the absolute number isn't a "pure" per-word figure.

**Case 1 - in-house sweep**: all 7 tokenizer/sweep/mixed/spm_{size}.model
files from the A2 sweep (see tokenizer/README.md).

**Case 2 - external tokenizers** (see tokenizer/README.md):
- IndicTrans2 (ai4bharat/indictrans2-indic-en-1B): two separate SentencePiece models, not
  one joint vocab - model.SRC and model.TGT, disambiguated by vocab size (128K = Indic,
  32K = English) rather than hardcoded per repo, since which one is "SRC" vs "TGT" depends
  on the repo's own translation direction (an indic-en repo has SRC=Indic, TGT=English -
  the opposite of what an en-indic repo would have). An earlier version of this script
  pointed at a different repo (`indictrans2-en-indic-dist-200M`) with a hardcoded
  SRC=English/TGT=Indic assumption; that repo's `model.SRC`/`model.TGT` files were never
  actually fetched successfully in this project's one completed pre-fix run (A2/A3's
  `fertility.csv` at commit a2ba875 has zero IndicTrans2 rows - a gated-repo/download
  failure, not a scored-then-discarded result), so **there is no real "before" number to
  compare against** - the only IndicTrans2 fertility numbers this project has ever produced
  are from the corrected code below. The vocab-size disambiguation is still the right fix
  (a fixed SRC=English/TGT=Indic assumption is genuinely wrong for an indic-en repo and
  would silently swap which sub-tokenizer each language uses if pointed at one), but do not
  restate a specific "X was inflated from A to B" claim anywhere else without an actual
  completed buggy run backing it - an earlier draft of this docstring, this project's internal
  notes, and the
  commit message for this fix did exactly that (a specific but unverified "Malayalam read
  10.7 instead of ~2.8" figure), which was a fabricated number, not a measurement; corrected
  in a follow-up commit. Loaded directly via `sentencepiece`, bypassing the custom
  transformers wrapper (IndicTransToolkit) since only raw piece counts are needed here.
- Gemma 3 (google/gemma-3-27b-it), Gemma 4 (google/gemma-4-31B-it), Sarvam-30B
  (sarvamai/sarvam-30b), Sarvam-Translate (sarvamai/sarvam-translate), and Bodhan AI's
  Indic-Translate (bodhan-ai/indic-translate): loaded via `transformers.AutoTokenizer`, one
  joint vocab each, same as this
  project's own tokenizer. **Verified finding, not a bug**: Sarvam-Translate's
  tokenizer.json is byte-identical (matching sha256) to Gemma 3's, and Bodhan AI's
  Indic-Translate's is byte-identical to Gemma 4's - both are fine-tunes that reused
  their base model's tokenizer unchanged, confirmed via `HfApi.model_info(...,
  files_metadata=True)` file hashes, not inferred from matching fertility numbers alone
  (which would be circular). Sarvam-30B has its own genuinely distinct tokenizer.
- Gated repos (varies by which the HF token in use has been granted) are skipped
  with a clear message rather than failing the whole run - re-run once granted,
  no code changes needed.

Output: results/fertility.csv (long format: tokenizer, variant, lang, tag,
n_words, n_subwords, fertility, unk_rate) - one row per (tokenizer, language).
"""

import csv
import json
import sys
from pathlib import Path

import sentencepiece as spm
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data"))
from common.auth import ensure_hf_login  # noqa: E402
from common.languages import LANGUAGE_CODES, LANGUAGES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_laws import tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
IN22_PATH = ROOT / "data" / "in22" / "combined_fertility_eval.jsonl"
SWEEP_DIR = ROOT / "tokenizer" / "sweep" / "mixed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
VOCAB_SIZES = [16000, 32000, 64000, 96000, 128000, 192000, 256000]

ENGLISH_LANG = "eng_Latn"
ALL_LANGS = LANGUAGES + [ENGLISH_LANG]


def load_in22_by_lang():
    texts = {lang: [] for lang in ALL_LANGS}
    if not IN22_PATH.exists():
        raise SystemExit(f"{IN22_PATH} does not exist - run data/in22/download_in22.py and "
                          "data/in22/build_fertility_corpus.py first.")
    with open(IN22_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for lang in ALL_LANGS:
                text = row.get(lang)
                if text:
                    texts[lang].append(text)
    return texts


def tag_for_lang(lang: str) -> str:
    return LANGUAGE_CODES.get(lang[:3], "?")


def fertility_rows(encode_fn, texts_by_lang, unk_check_fn, tokenizer_name, variant):
    """encode_fn(lang, sentence) -> list of piece ids/strings.
    unk_check_fn(piece) -> bool, whether this piece is that tokenizer's <unk>."""
    rows = []
    for lang, sentences in texts_by_lang.items():
        n_words = sum(len(tokenize(s)) for s in sentences)
        n_subwords = 0
        n_unk = 0
        for s in sentences:
            pieces = encode_fn(lang, s)
            n_subwords += len(pieces)
            n_unk += sum(1 for p in pieces if unk_check_fn(p))
        fertility = n_subwords / n_words if n_words else float("nan")
        unk_rate = n_unk / n_subwords if n_subwords else float("nan")
        rows.append({
            "tokenizer": tokenizer_name, "variant": variant, "lang": lang, "tag": tag_for_lang(lang),
            "n_words": n_words, "n_subwords": n_subwords,
            "fertility": round(fertility, 4), "unk_rate": round(unk_rate, 6),
        })
    return rows


def run_in_house_sweep(texts_by_lang):
    all_rows = []
    for vs in VOCAB_SIZES:
        model_path = SWEEP_DIR / f"spm_{vs}.model"
        if not model_path.exists():
            print(f"[fertility] SKIP in-house vocab={vs}: {model_path} not found")
            continue
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        unk_id = sp.unk_id()

        def encode_fn(lang, s, sp=sp):
            return sp.encode(s, out_type=int)

        rows = fertility_rows(encode_fn, texts_by_lang, lambda p, unk_id=unk_id: p == unk_id,
                               "in-house", str(vs))
        all_rows.extend(rows)
        macro = sum(r["fertility"] for r in rows) / len(rows)
        print(f"[fertility] in-house vocab={vs}: macro fertility={macro:.4f}")
    return all_rows


def try_download(repo_id, filename):
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except (GatedRepoError, HfHubHTTPError) as e:
        print(f"[fertility] SKIP {repo_id}: not accessible yet ({type(e).__name__}) - "
              f"request access at https://huggingface.co/{repo_id} and re-run.")
        return None


def run_indictrans2(texts_by_lang):
    # model.SRC/model.TGT name the model's OWN source/target for whichever direction this
    # repo is - NOT fixed "English"/"Indic" labels. indic-en-1B has model.SRC=Indic (source,
    # 128K vocab), model.TGT=English (target, 32K) - the opposite of what an en-indic repo
    # would have. Disambiguate by vocab size (128K vs 32K) rather than hardcoding per repo,
    # so this doesn't silently mis-map again if the repo changes.
    repo = "ai4bharat/indictrans2-indic-en-1B"
    src_path = try_download(repo, "model.SRC")
    tgt_path = try_download(repo, "model.TGT")
    if not src_path or not tgt_path:
        return []
    sp_src = spm.SentencePieceProcessor(model_file=src_path)
    sp_tgt = spm.SentencePieceProcessor(model_file=tgt_path)
    if sp_src.vocab_size() > sp_tgt.vocab_size():
        sp_indic, sp_en = sp_src, sp_tgt
    else:
        sp_indic, sp_en = sp_tgt, sp_src
    print(f"[fertility] indictrans2: English tokenizer vocab={sp_en.vocab_size()}, "
          f"Indic tokenizer vocab={sp_indic.vocab_size()} (from {repo})")
    unk_en, unk_indic = sp_en.unk_id(), sp_indic.unk_id()

    # Two different sub-tokenizers depending on language, so fertility_rows'
    # single encode_fn/unk_check_fn signature doesn't fit cleanly - inlined
    # instead of forcing it through that helper.
    rows = []
    for lang, sentences in texts_by_lang.items():
        sp = sp_en if lang == ENGLISH_LANG else sp_indic
        unk_id = unk_en if lang == ENGLISH_LANG else unk_indic
        n_words = sum(len(tokenize(s)) for s in sentences)
        n_subwords = 0
        n_unk = 0
        for s in sentences:
            ids = sp.encode(s, out_type=int)
            n_subwords += len(ids)
            n_unk += sum(1 for i in ids if i == unk_id)
        fertility = n_subwords / n_words if n_words else float("nan")
        unk_rate = n_unk / n_subwords if n_subwords else float("nan")
        rows.append({
            "tokenizer": "indictrans2", "variant": "en" if lang == ENGLISH_LANG else "indic",
            "lang": lang, "tag": tag_for_lang(lang), "n_words": n_words, "n_subwords": n_subwords,
            "fertility": round(fertility, 4), "unk_rate": round(unk_rate, 6),
        })
    macro = sum(r["fertility"] for r in rows) / len(rows)
    print(f"[fertility] indictrans2: macro fertility={macro:.4f}")
    return rows


def run_hf_tokenizer(texts_by_lang, tokenizer_name, repo_id):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo_id)
    except Exception as e:
        print(f"[fertility] SKIP {tokenizer_name} ({repo_id}): not accessible yet "
              f"({type(e).__name__}) - request access at https://huggingface.co/{repo_id} and re-run.")
        return []

    unk_str = tok.unk_token

    def encode_fn(lang, s):
        return tok.tokenize(s)

    rows = fertility_rows(encode_fn, texts_by_lang, lambda p: unk_str is not None and p == unk_str,
                           tokenizer_name, repo_id.split("/")[-1])
    macro = sum(r["fertility"] for r in rows) / len(rows)
    print(f"[fertility] {tokenizer_name}: macro fertility={macro:.4f}")
    return rows


def main():
    ensure_hf_login()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    texts_by_lang = load_in22_by_lang()
    print(f"[fertility] Loaded held-out corpus: {sum(len(v) for v in texts_by_lang.values())} "
          f"sentences across {len(texts_by_lang)} languages")

    all_rows = []
    all_rows += run_in_house_sweep(texts_by_lang)
    all_rows += run_indictrans2(texts_by_lang)
    all_rows += run_hf_tokenizer(texts_by_lang, "gemma-3-27b-it", "google/gemma-3-27b-it")
    all_rows += run_hf_tokenizer(texts_by_lang, "gemma-4-31b-it", "google/gemma-4-31B-it")
    all_rows += run_hf_tokenizer(texts_by_lang, "sarvam-30b", "sarvamai/sarvam-30b")
    all_rows += run_hf_tokenizer(texts_by_lang, "sarvam-translate", "sarvamai/sarvam-translate")
    all_rows += run_hf_tokenizer(texts_by_lang, "indic-translate-bodhan", "bodhan-ai/indic-translate")

    out_path = RESULTS_DIR / "fertility.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tokenizer", "variant", "lang", "tag", "n_words", "n_subwords", "fertility", "unk_rate",
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[fertility] Wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
