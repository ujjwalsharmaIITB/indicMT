"""Mechanically copies data/common/languages.py and tokenization.py into
deployment/api/common/, so the Docker build context (deployment/, not the repo
root - see deployment/Dockerfile's header) has the language-tag and
source-encoding source of truth without needing a huge build context (data/
and training outputs) or fragile .dockerignore negation rules. Run this before
every deployment build; the copies are never hand-edited (see the generated
header it writes).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# data/common/<name> -> deployment/api/common/<name>
FILES = ["languages.py", "tokenization.py"]

GENERATED_HEADER = (
    '"""GENERATED FILE - do not hand-edit.\n\n'
    "Mechanical copy of data/common/{name}, made by deployment/stage_common.py\n"
    "so the Docker build context (deployment/) doesn't need the full repo root.\n"
    "Re-run that script (or the Dockerfile's staging step) if the source changes.\n"
    '"""\n\n'
)


def main():
    dst_dir = ROOT / "deployment" / "api" / "common"
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "__init__.py").touch()
    for name in FILES:
        src = ROOT / "data" / "common" / name
        dst = dst_dir / name
        if not src.exists():
            raise SystemExit(f"source not found: {src}")
        content = src.read_text(encoding="utf-8")
        # Strip the original module docstring (first triple-quoted block) and
        # replace it with the generated-file header, keeping the rest intact.
        if content.startswith('"""'):
            end = content.find('"""', 3) + 3
            content = content[end:].lstrip("\n")
        dst.write_text(GENERATED_HEADER.format(name=name) + content, encoding="utf-8")
        print(f"[stage_common] copied {src} -> {dst}")


if __name__ == "__main__":
    main()
