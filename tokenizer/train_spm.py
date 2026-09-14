"""Train a single SentencePiece unigram model at one vocabulary size.

Flags match tokenizer/analysis/configs/spm_sweep.yaml exactly - keep it
that way; if a flag changes here, update that config and vice versa, since
the fertility sweep depends on every run in the sweep using identical flags
except vocab_size.

Usage: python tokenizer/train_spm.py --vocab-size 64000
"""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from common.languages import LANGUAGE_CODES  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus" / "spm_train_corpus_mixed.txt"
OUTPUT_DIR = Path(__file__).resolve().parent / "sweep" / "mixed"

# Tags in a fixed order for reproducibility (dict insertion order in
# data/common/languages.py, "eng" first).
USER_DEFINED_SYMBOLS = list(LANGUAGE_CODES.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"{args.corpus} does not exist - run tokenizer/prepare_corpus.py first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = args.output_dir / f"spm_{args.vocab_size}"

    print(f"[train_spm] Training vocab_size={args.vocab_size} on {args.corpus} "
          f"-> {model_prefix}.{{model,vocab}}")

    spm.SentencePieceTrainer.Train(
        input=str(args.corpus),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        train_extremely_large_corpus=True,
        normalization_rule_name="nmt_nfkc",
        bos_id=0,
        pad_id=1,
        eos_id=2,
        unk_id=3,
        byte_fallback=False,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
    )

    print(f"[train_spm] Done: {model_prefix}.model / {model_prefix}.vocab")


if __name__ == "__main__":
    main()
