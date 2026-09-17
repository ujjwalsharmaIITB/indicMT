"""Convert an OpenNMT-py checkpoint to CTranslate2, in the real,
CPU-usable variants on this hardware. Verified directly, not assumed:
`onmt_release_model --quantization` supports {int8, int16, float16,
int8_float16}, but `ctranslate2.get_supported_compute_types("cpu")` on this
machine returns {float32, int16, int8_float32, int8} - float16/int8_float16
are not real CPU-usable options here. So this converts exactly three
variants: float32 (baseline, no quantization), int16, int8.

**Calls ctranslate2's OpenNMTPyConverter directly, not the `onmt_release_model`
CLI** - a real bug found running this for the first time: the CLI (and the
Python API by default) calls `torch.load(..., weights_only=True)`, which
rejects this checkpoint's `opt` field (a standard `argparse.Namespace`,
present in every OpenNMT-py checkpoint) with
`_pickle.UnpicklingError: ... Unsupported class argparse.Namespace`. CTranslate2
itself provides the fix for exactly this case: `OpenNMTPyConverter(...,
unsafe_deserialization=True)` ("Allow unsafe pickle deserialization when
loading trusted legacy checkpoints" - its own docstring's exact wording),
which sets `weights_only=False`. Legitimate here, not a security shortcut:
this is our own self-trained checkpoint, not an untrusted download. The CLI
doesn't expose this flag at all, hence calling the converter class directly.

**A second real incompatibility, also found running this for the first time**:
CTranslate2's converter rejects `opt.self_attn_type == "scaled-dot-flash"`
("Option --self_attn_type scaled-dot-flash is not supported (supported
values are: scaled-dot)") - our OpenNMT-py 3.5.1 install auto-selected flash
attention during training (this project never explicitly set
`self_attn_type` in any training config). Verified directly in OpenNMT-py's
own source (`onmt/modules/multi_headed_attn.py`) before working around it,
not assumed safe: `self_attn_type` only selects which attention KERNEL runs
(`flash_attn_func`/`flash_attn_with_kvcache` vs. the standard PyTorch path)
- the linear_keys/linear_values/linear_query/final_linear projections and
the actual learned weights are identical either way, same
softmax(QK^T/sqrt(d))V computation, just a different, numerically
equivalent implementation for speed/memory. Safe to relabel for conversion:
this script loads the checkpoint itself, overwrites
`checkpoint["opt"].self_attn_type` to `"scaled-dot"` if it's
`"scaled-dot-flash"`, and writes a patched copy to a temp file that the
converter reads instead - the original training checkpoint is never
modified.

Output travels beside the checkpoint that produced it (an evidence trail,
not eval output - this location convention is deliberately different from
evaluation/results/):
128k-spm/opennmt/<arm>/ct2/<checkpoint-stem>/{float32,int16,int8}/
"""

import argparse
import tempfile
import time
from pathlib import Path

import torch
from ctranslate2.converters.opennmt_py import OpenNMTPyConverter

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "float32": None,
    "int16": "int16",
    "int8": "int8",
}


def patched_checkpoint_path(checkpoint: Path) -> Path:
    """Load the checkpoint, relabel self_attn_type=scaled-dot-flash to
    scaled-dot (see module docstring - a kernel choice, not a parameter
    change), and write the patched copy to a temp file for the converter to
    read. Returns the original path unchanged if no patch was needed."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    opt = ckpt.get("opt")
    if getattr(opt, "self_attn_type", None) != "scaled-dot-flash":
        return checkpoint
    print("[convert_to_ct2] patching opt.self_attn_type: scaled-dot-flash -> scaled-dot "
          "(kernel choice only, weights unaffected - see module docstring)", flush=True)
    opt.self_attn_type = "scaled-dot"
    tmp = Path(tempfile.mkdtemp(prefix="ct2_convert_")) / checkpoint.name
    torch.save(ckpt, tmp)
    return tmp


def convert(checkpoint: Path, out_root: Path) -> dict:
    convert_from = patched_checkpoint_path(checkpoint)
    timings = {}
    for name, quant in VARIANTS.items():
        out_dir = out_root / name
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[convert_to_ct2] {name}: already exists at {out_dir}, skipping")
            continue
        print(f"[convert_to_ct2] {name}: converting (quantization={quant}) -> {out_dir}", flush=True)
        t0 = time.time()
        converter = OpenNMTPyConverter(str(convert_from), unsafe_deserialization=True)
        converter.convert(str(out_dir), quantization=quant, force=True)
        timings[name] = time.time() - t0
        size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
        print(f"[convert_to_ct2] {name}: done in {timings[name]:.1f}s, {size_mb:.1f} MB", flush=True)
    return timings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-root", default=None,
                     help="default: 128k-spm/opennmt/<arm>/ct2/<checkpoint-stem>/")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    if args.output_root:
        out_root = Path(args.output_root)
    elif checkpoint.parent.name == "checkpoints":
        arm_dir = checkpoint.parent.parent
        out_root = arm_dir / "ct2" / checkpoint.stem
    else:
        raise SystemExit("--output-root required when checkpoint isn't under a checkpoints/ folder")

    out_root.mkdir(parents=True, exist_ok=True)
    timings = convert(checkpoint, out_root)
    print(f"[convert_to_ct2] all variants ready under {out_root}: {list(VARIANTS)}")
    print(f"[convert_to_ct2] conversion timings: {timings}")


if __name__ == "__main__":
    main()
