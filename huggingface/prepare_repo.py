"""Stage the Hugging Face model repository in huggingface/.

huggingface/ is the content of the Hugging Face model repository
(ujjwalsharmaIITB/IndicMT): the model card (README.md), a small manifest
(indicmt_config.json) and, after this script runs, the model files. Only the
card, the manifest and this script are tracked by git. The copied model files
are ignored (see .gitignore), so they cannot be committed by accident.

This script copies the converted CTranslate2 model directories (float32 and
int8, written by deployment/convert_to_ct2.py) and the SentencePiece model into
huggingface/. It then checks that the tokenizer encodes a tagged source with
the training prefix ['▁', '<xx>'] (data/common/tokenization.py). Run it from
the repository root:

    python deployment/convert_to_ct2.py \
        --checkpoint 128k-spm/opennmt/bpcc-ft-combined-human/checkpoints/model_step_1800.pt
    python huggingface/prepare_repo.py
    huggingface-cli upload ujjwalsharmaIITB/IndicMT huggingface . \
        --exclude "prepare_repo.py" "__pycache__/*"
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / "huggingface"
DEFAULT_CT2_ROOT = ROOT / "128k-spm" / "opennmt" / "bpcc-ft-combined-human" / "ct2" / "model_step_1800"
DEFAULT_SPM = ROOT / "tokenizer" / "sweep" / "mixed" / "spm_128000.model"
SPM_NAME = "spm_128000.model"
VARIANTS = ["float32", "int8"]
CT2_FILES = ["config.json", "model.bin", "shared_vocabulary.json"]
TRACKED = ["README.md", "indicmt_config.json"]


def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def copy_variant(src: Path, dst: Path, force: bool) -> None:
    missing = [name for name in CT2_FILES if not (src / name).is_file()]
    if missing:
        raise SystemExit(f"{src} is missing {missing} - run deployment/convert_to_ct2.py first")
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} already exists - pass --force to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in CT2_FILES:
        shutil.copy2(src / name, dst / name)


def check_tokenizer(spm_path: Path) -> None:
    """A tagged source must encode as ['▁', '<xx>', ...], the format of every training source."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("[prepare_repo] sentencepiece is not installed - skipping the tokenizer check")
        return
    sys.path.insert(0, str(ROOT / "data"))
    from common.tokenization import encode_source

    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    for tag in ("<hi>", "<en>"):
        encode_source(sp, tag, "Hello, how are you?")  # raises ValueError if the prefix is wrong
    print("[prepare_repo] tokenizer check passed: tagged sources start with ['▁', tag]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct2-root", type=Path, default=DEFAULT_CT2_ROOT,
                    help="directory holding the float32/ and int8/ CTranslate2 models")
    ap.add_argument("--spm", type=Path, default=DEFAULT_SPM, help="SentencePiece .model file")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--out", type=Path, default=HF_DIR, help="default: huggingface/ (this folder)")
    ap.add_argument("--force", action="store_true", help="replace model files that are already staged")
    ap.add_argument("--skip-tokenizer-check", action="store_true")
    args = ap.parse_args()

    out = args.out
    if out.resolve() != HF_DIR.resolve():
        out.mkdir(parents=True, exist_ok=True)
        for name in TRACKED:
            shutil.copy2(HF_DIR / name, out / name)
    if not args.spm.is_file():
        raise SystemExit(f"SentencePiece model not found: {args.spm}")

    for variant in args.variants:
        copy_variant(args.ct2_root / variant, out / variant, args.force)
        print(f"[prepare_repo] {variant}: {size_mb(out / variant):.1f} MB -> {out / variant}")
    spm_dst = out / SPM_NAME
    if spm_dst.exists() and not args.force:
        raise SystemExit(f"{spm_dst} already exists - pass --force to replace it")
    shutil.copy2(args.spm, spm_dst)
    print(f"[prepare_repo] tokenizer: {size_mb(spm_dst):.1f} MB -> {spm_dst}")
    if not args.skip_tokenizer_check:
        check_tokenizer(spm_dst)
    print(f"[prepare_repo] ready: {out}\n"
          f"  upload: huggingface-cli upload ujjwalsharmaIITB/IndicMT {out} . "
          f"--exclude prepare_repo.py '__pycache__/*'")


if __name__ == "__main__":
    main()
