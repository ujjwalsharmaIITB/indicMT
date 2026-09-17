"""Package a checkpoint's IN22 translations into one .tar.gz release asset.

Input: an evaluation results directory written by evaluation/run_eval.py, e.g.
    evaluation/results/bpcc-ft-combined-human/model_step_1800/

Output: <name>.tar.gz holding
  - the 88 detokenized translation files `<test-set>.<src>-<tgt>.hyp.txt`
    (44 directions x IN22-Gen and IN22-Conv), one translation per line;
  - scores.json (the corpus-level chrF++ and BLEU of every direction, plus macro averages);
  - README.txt (generated: checkpoint name, file naming, how the scores were computed).
The SentencePiece-level `*.hyp.spm` files are not included.

The script fails with a clear message if any of the 88 translation files or scores.json
is missing. Standard library only.

Usage:
    python evaluation/package_translations.py \
        --results-dir evaluation/results/bpcc-ft-combined-human/model_step_1800 \
        --out indicmt-in22-translations.tar.gz
"""

import argparse
import io
import sys
import tarfile
from pathlib import Path

TEST_SETS = ["in22-gen", "in22-conv"]
# Short language tags used in the file names (English plus the 22 Scheduled languages).
INDIC_TAGS = ["as", "bn", "br", "doi", "gom", "gu", "hi", "kn", "ks", "mai", "ml", "mni",
              "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]
EN = "en"

README_TEMPLATE = """IndicMT: IN22 translations
==========================

Checkpoint: {checkpoint}
Run: {run}

Contents
--------
- {n_files} translation files, one per direction and test set, named
      <test-set>.<src>-<tgt>.hyp.txt
  <test-set> is in22-gen (1,024 sentences) or in22-conv (1,503 sentences).
  <src> and <tgt> are short language tags: en plus {indic}.
  There are 44 directions (en->xx and xx->en for 22 languages) x 2 test sets = 88 files.
  The files are detokenized plain text, one translation per line, line-aligned with the
  IN22 source and reference files (line N is the translation of sentence N).
- scores.json: corpus-level chrF++ and BLEU for every direction, plus macro averages
  (unweighted mean over the 22 languages per direction bucket).

How the scores were computed
----------------------------
Beam size 5 with OpenNMT-py's default length penalty. Corpus-level chrF++ (word order 2) and
BLEU with sacreBLEU, against the untokenized IN22 references.
Note: the checkpoint was selected using these test sets, so the scores carry a small
optimistic bias.

The IN22 sets themselves are not included. They are available from AI4Bharat after an
access request on Hugging Face.
"""


def expected_files():
    for test_set in TEST_SETS:
        for tag in INDIC_TAGS:
            yield f"{test_set}.{EN}-{tag}.hyp.txt"
            yield f"{test_set}.{tag}-{EN}.hyp.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="evaluation/results/<run>/<checkpoint> written by run_eval.py")
    ap.add_argument("--out", required=True, type=Path, help="output .tar.gz path")
    args = ap.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        sys.exit(f"ERROR: {results_dir} is not a directory")

    names = list(expected_files())
    missing = [n for n in names if not (results_dir / n).is_file()]
    if not (results_dir / "scores.json").is_file():
        missing.append("scores.json")
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        sys.exit(f"ERROR: {len(missing)} required file(s) missing in {results_dir}: {shown}")

    checkpoint = results_dir.name
    run = results_dir.parent.name
    readme = README_TEMPLATE.format(checkpoint=checkpoint, run=run, n_files=len(names),
                                    indic=", ".join(INDIC_TAGS)).encode("utf-8")

    prefix = "indicmt-in22-translations"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        info = tarfile.TarInfo(f"{prefix}/README.txt")
        info.size = len(readme)
        tar.addfile(info, io.BytesIO(readme))
        tar.add(results_dir / "scores.json", arcname=f"{prefix}/scores.json")
        for n in sorted(names):
            tar.add(results_dir / n, arcname=f"{prefix}/{n}")
    print(f"Wrote {args.out} ({len(names)} translation files + scores.json + README.txt)")


if __name__ == "__main__":
    main()
