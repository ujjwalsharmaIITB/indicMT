"""Plots for the A2 sweep and A3 external-tokenizer comparison, both driven by
results/fertility.csv (written by fertility.py). Re-run any time that CSV is
regenerated - tokenizers/vocab sizes are read dynamically, no hardcoded list.

Both graphs are grouped bar charts: one group per language (fixed x-axis
order, LANGUAGE_CODES' own order, matching every other table in this
project), one bar per model within each group, plus one dotted horizontal
line per series at that series' own macro average (mean over the languages
it has a value for), same color as its bars - not a single line averaged
across series per language. The macro value is also appended to that
series' legend label so it can be read off directly. Figures are large
(wide enough that bars never overlap even with ~10 models/group) and saved
at high DPI so they're readable zoomed in.

Graph 1 (fertility_sweep_per_language.png): the in-house A2 sweep - bars are
the 7 vocab sizes (16k-256k), one consistent color per size across every
language group.

Graph 2 (fertility_external_per_language.png): the A3 comparison - bars are
the external tokenizers actually available in the CSV, plus the in-house
64k/128k/192k/256k models included alongside them for a direct baseline
(not just vocab-size-only, so the "how do we compare" question has an
answer right on the same chart).
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data"))
from common.languages import LANGUAGE_CODES  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CSV_PATH = RESULTS_DIR / "fertility.csv"

LANG_TAG_ORDER = list(LANGUAGE_CODES.values())

# Cosmetic relabeling only (raw `tokenizer` values in the CSV stay as the
# stable identifiers) - these two are fine-tunes of Gemma checkpoints, per
# 03-external-tokenizer-comparison.md's notes, worth surfacing on the chart
# rather than leaving the reader to infer it from matching numbers.
DISPLAY_LABEL = {
    "sarvam-translate": "sarvam-translate (Gemma 3)",
    "indic-translate-bodhan": "indic-translate (Gemma 4)",
    "gemma-3-27b-it": "gemma-3-27b-it",
    "gemma-4-31b-it": "gemma-4-31b-it",
    "sarvam-30b": "sarvam-30b",
    "indictrans2": "indictrans2",
}

IN_HOUSE_SERIES_FOR_GRAPH2 = [64000, 128000, 192000]  # the 3 candidate sizes under consideration

BAR_GROUP_WIDTH = 0.82


def load_rows():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grouped_bar_plot(series_names, series_display, data_by_series_lang, present_tags, title, out_path):
    """series_names: stable ids, in the order bars are drawn.
    data_by_series_lang: {series_name: {tag: fertility}}."""
    n_series = len(series_names)
    n_groups = len(present_tags)
    bar_w = BAR_GROUP_WIDTH / n_series

    fig_w = max(14, n_groups * (0.55 + 0.16 * n_series))
    fig, ax = plt.subplots(figsize=(fig_w, 10))

    cmap = plt.get_cmap("tab10" if n_series <= 10 else "tab20")
    x = np.arange(n_groups)

    for i, name in enumerate(series_names):
        offset = (i - (n_series - 1) / 2) * bar_w
        ys = [data_by_series_lang[name].get(tag) for tag in present_tags]
        ys_plot = [y if y is not None else 0 for y in ys]
        color = cmap(i % cmap.N)

        # Macro average for THIS series (mean over the languages it actually has
        # a value for, out of the 23) - a horizontal dotted line at that series'
        # own color, not a single line averaged across series per language.
        # The number itself goes right in the legend label so it can be read
        # off directly instead of estimated from the line's height.
        vals = list(data_by_series_lang[name].values())
        macro = sum(vals) / len(vals) if vals else None
        label = series_display.get(name, name)
        if macro is not None:
            label = f"{label} (macro {macro:.3f})"

        ax.bar(x + offset, ys_plot, width=bar_w * 0.95, color=color, label=label)
        if macro is not None:
            ax.axhline(macro, color=color, linestyle=":", linewidth=1.8, alpha=0.9, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(present_tags, fontsize=11)
    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Fertility (subwords / word)", fontsize=12)
    ax.set_title(title + "\n(dotted horizontal line = that series' own macro average"
                 f" over its {n_groups} languages, same color as its bars)", fontsize=14)
    ax.legend(fontsize=10, ncol=min(4, n_series), loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.6, n_groups - 0.4)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_fertility] Wrote {out_path} ({fig_w:.1f}x10in, {n_series} bars/group)")


def plot_sweep(rows):
    sweep_rows = [r for r in rows if r["tokenizer"] == "in-house"]
    if not sweep_rows:
        print("[plot_fertility] No in-house sweep rows found, skipping graph 1.")
        return

    vocab_sizes = sorted({int(r["variant"]) for r in sweep_rows})
    series_names = [str(v) for v in vocab_sizes]
    series_display = {str(v): f"{v // 1000}k" for v in vocab_sizes}
    data = {str(v): {} for v in vocab_sizes}
    for r in sweep_rows:
        data[r["variant"]][r["tag"]] = float(r["fertility"])

    present_tags = [t for t in LANG_TAG_ORDER if any(t in data[s] for s in series_names)]
    grouped_bar_plot(
        series_names, series_display, data, present_tags,
        "A2 - in-house SPM sweep: fertility per language, by vocabulary size\n"
        "(held out: IN22-Gen + IN22-Conv combined, 2,527 multi-parallel sentences)",
        PLOTS_DIR / "fertility_sweep_per_language.png",
    )


def plot_external(rows):
    ext_rows = [r for r in rows if r["tokenizer"] != "in-house"]
    sweep_rows = {int(r["variant"]): r for r in rows if r["tokenizer"] == "in-house"}
    if not ext_rows:
        print("[plot_fertility] No external tokenizer rows found, skipping graph 2.")
        return

    ext_names = sorted({r["tokenizer"] for r in ext_rows})
    in_house_names = [f"in-house-{v}" for v in IN_HOUSE_SERIES_FOR_GRAPH2
                       if any(int(r["variant"]) == v for r in rows if r["tokenizer"] == "in-house")]
    series_names = in_house_names + ext_names

    series_display = {f"in-house-{v}": f"in-house {v // 1000}k" for v in IN_HOUSE_SERIES_FOR_GRAPH2}
    series_display.update(DISPLAY_LABEL)

    data = {name: {} for name in series_names}
    for v in IN_HOUSE_SERIES_FOR_GRAPH2:
        name = f"in-house-{v}"
        if name not in data:
            continue
        for r in rows:
            if r["tokenizer"] == "in-house" and int(r["variant"]) == v:
                data[name][r["tag"]] = float(r["fertility"])
    for r in ext_rows:
        data[r["tokenizer"]][r["tag"]] = float(r["fertility"])

    present_tags = [t for t in LANG_TAG_ORDER if any(t in data[s] for s in series_names)]
    grouped_bar_plot(
        series_names, series_display, data, present_tags,
        "A3 - external tokenizers vs. in-house candidates: fertility per language\n"
        "(held out: IN22-Gen + IN22-Conv combined, 2,527 multi-parallel sentences)",
        PLOTS_DIR / "fertility_external_per_language.png",
    )


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} does not exist - run fertility.py first.")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plot_sweep(rows)
    plot_external(rows)


if __name__ == "__main__":
    main()
