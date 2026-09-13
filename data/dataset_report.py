"""Generate dataset_statistics.md covering bpcc-human (raw + splits) and
bpcc-mined (raw). Moved here from an earlier script and extended to
also cover bpcc-mined (see docs/data-pipeline.md).

Fixes vs. the original draft:
- Paths resolved relative to this file, not the caller's cwd.
- A section whose directory doesn't exist yet (e.g. bpcc-mined/raw before it's
  been downloaded) is skipped with a note in the report instead of crashing
  the whole run with FileNotFoundError.
- Malformed/incomplete JSONL lines are now caught and counted per file instead
  of aborting the entire report on the first bad line.
"""

import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent
REPORT_FILE = DATA_DIR / "dataset_statistics.md"

SECTIONS = [
    ("bpcc-human / raw", DATA_DIR / "bpcc-human" / "raw"),
    ("bpcc-human / splits", DATA_DIR / "bpcc-human" / "splits"),
    ("bpcc-mined / raw", DATA_DIR / "bpcc-mined" / "raw"),
]


def process_directory(directory):
    dataset_stats = {}
    src_counter = Counter()
    tgt_counter = Counter()
    pair_counter = Counter()
    total_examples = 0
    error_count = 0

    files = sorted(f for f in directory.iterdir() if f.name.endswith(".jsonl"))

    for file_path in tqdm(files, desc=f"Processing {directory}"):
        count = 0
        src_langs = set()
        tgt_langs = set()

        with open(file_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    src = ex["src_lang"]
                    tgt = ex["tgt_lang"]
                except (json.JSONDecodeError, KeyError) as exc:
                    error_count += 1
                    print(f"[dataset_report] Skipping bad line {file_path}:{line_no} ({exc})")
                    continue

                count += 1
                total_examples += 1
                src_counter[src] += 1
                tgt_counter[tgt] += 1
                pair_counter[(src, tgt)] += 1
                src_langs.add(src)
                tgt_langs.add(tgt)

        dataset_stats[file_path.name] = {
            "count": count,
            "src_langs": sorted(src_langs),
            "tgt_langs": sorted(tgt_langs),
        }

    return dataset_stats, src_counter, tgt_counter, pair_counter, total_examples, error_count


def markdown_dataset_table(dataset_stats, total):
    lines = ["| Dataset | Samples | % of Total | Src Langs | Tgt Langs |",
             "|---------|--------:|-----------:|-----------|-----------|"]
    for dataset, stats in sorted(dataset_stats.items()):
        pct = stats["count"] * 100 / total if total else 0
        lines.append(
            f"| {dataset} | {stats['count']:,} | {pct:.2f}% | "
            f"{', '.join(stats['src_langs'])} | {', '.join(stats['tgt_langs'])} |"
        )
    return "\n".join(lines)


def markdown_language_table(src_counter, tgt_counter, total):
    languages = sorted(set(src_counter) | set(tgt_counter))
    lines = ["| Language | Source Count | Target Count | Total | % of Dataset |",
              "|----------|-------------:|-------------:|------:|-------------:|"]
    for lang in languages:
        src = src_counter[lang]
        tgt = tgt_counter[lang]
        total_lang = src + tgt
        pct = total_lang * 100 / (2 * total) if total else 0
        lines.append(f"| {lang} | {src:,} | {tgt:,} | {total_lang:,} | {pct:.2f}% |")
    return "\n".join(lines)


def markdown_pair_table(pair_counter):
    lines = ["| Source | Target | Samples |", "|--------|--------|--------:|"]
    for (src, tgt), count in sorted(pair_counter.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {src} | {tgt} | {count:,} |")
    return "\n".join(lines)


def write_section(name, directory, report):
    if not directory.exists():
        report.append(f"# {name}\n\n_Directory {directory} does not exist yet - skipped._\n")
        return

    dataset_stats, src_counter, tgt_counter, pair_counter, total, error_count = process_directory(directory)

    report.append(f"# {name}")
    report.append("")
    report.append(f"**Total Samples:** {total:,}")
    report.append(f"**Datasets:** {len(dataset_stats)}")
    report.append(f"**Languages:** {len(set(src_counter) | set(tgt_counter))}")
    if error_count:
        report.append(f"**Malformed lines skipped:** {error_count:,}")
    report.append("")

    report.append("## Dataset-wise Statistics")
    report.append("")
    report.append(markdown_dataset_table(dataset_stats, total))
    report.append("")

    report.append("## Language-wise Statistics")
    report.append("")
    report.append(markdown_language_table(src_counter, tgt_counter, total))
    report.append("")

    report.append("## Language Pair Statistics")
    report.append("")
    report.append(markdown_pair_table(pair_counter))
    report.append("")


def generate_report():
    report = []
    for name, directory in SECTIONS:
        write_section(name, directory, report)
        report.append("\n---\n")

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to {REPORT_FILE}")


if __name__ == "__main__":
    generate_report()
