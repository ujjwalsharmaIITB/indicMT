"""Experiment A1: Zipf's law and Heaps' law on the bpcc-human corpus.

See tokenizer/analysis/README.md for the full methodology and why these are two
different questions (frequency-distribution shape vs. vocabulary-growth
bound). This script implements that spec exactly:

- Input: data/bpcc-human/combined_train_dataset.jsonl, target-side text
  deduplicated per language (tgt_lang).
- Word = whitespace-delimited token after NFKC normalization, stripping
  Unicode punctuation and standalone-digit tokens.
- Zipf fit: log-log least squares over ranks 10..10,000.
- Heaps fit: log-spaced subsample fractions (0.1%, 0.3%, 1%, 3%, 10%, 30%,
  100%), averaged over 3 seeds, log-log least squares.
- Repeats both for the union of all 23 languages (the joint-vocabulary case).
- Writes results/corpus_laws.csv and a rank-frequency + vocabulary-growth
  plot per language (plus the union) into results/plots/.

Intended to run as a Slurm CPU job (see
scripts/slurm/tokenizer_corpus_laws.sh), rather than on a login node - the full
23-language pass + subsampling is not fast on ~7M rows.
"""

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data"))
from common.languages import LANGUAGE_CODES  # noqa: E402

TRAIN_IN = Path(__file__).resolve().parents[3] / "data" / "bpcc-human" / "combined_train_dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

ZIPF_RANK_LO, ZIPF_RANK_HI = 10, 10_000
HEAPS_FRACTIONS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
HEAPS_SEEDS = [0, 1, 2]

# Matches a run of Unicode punctuation/symbol characters, and a standalone
# digit-only token, to strip per the word definition in tokenizer/analysis/README.md.
_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)
_DIGIT_TOKEN_RE = re.compile(r"^\d+$")


def tokenize(text: str):
    text = unicodedata.normalize("NFKC", text)
    words = []
    for tok in text.split():
        tok = _PUNCT_RE.sub("", tok)
        if not tok or _DIGIT_TOKEN_RE.match(tok):
            continue
        words.append(tok)
    return words


def load_sentences_by_lang(path):
    """Deduplicated target-side sentences per tgt_lang."""
    sentences = defaultdict(list)
    seen = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading + deduping sentences"):
            row = json.loads(line)
            lang, text = row["tgt_lang"], row["tgt"]
            if text in seen[lang]:
                continue
            seen[lang].add(text)
            sentences[lang].append(text)
    return sentences


def zipf_fit(word_counts: Counter):
    freqs = np.array(sorted(word_counts.values(), reverse=True), dtype=np.float64)
    n = len(freqs)
    hi = min(ZIPF_RANK_HI, n)
    if n < ZIPF_RANK_LO + 10:
        return None, None
    ranks = np.arange(ZIPF_RANK_LO, hi + 1, dtype=np.float64)
    f = freqs[ZIPF_RANK_LO - 1:hi]
    log_r, log_f = np.log(ranks), np.log(f)
    slope, intercept = np.polyfit(log_r, log_f, 1)
    pred = slope * log_r + intercept
    ss_res = np.sum((log_f - pred) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha = -slope
    # Linear-scale curve for plotting: f(r) = C * r^slope, C = exp(intercept).
    fit_c = np.exp(intercept)
    plot_ranks = np.linspace(ranks.min(), ranks.max(), 400)
    plot_fit = fit_c * plot_ranks ** slope
    return alpha, r2, (ranks, f, plot_ranks, plot_fit)


def heaps_fit(sentences, seeds=HEAPS_SEEDS, fractions=HEAPS_FRACTIONS):
    all_words_per_sentence = [tokenize(s) for s in sentences]
    total_n = sum(len(w) for w in all_words_per_sentence)
    if total_n == 0:
        return None

    points_by_fraction = defaultdict(list)  # fraction -> list of (n, V) across seeds
    for seed in seeds:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(all_words_per_sentence))
        cum_n, cum_types = 0, set()
        idx = 0
        sorted_fractions = sorted(fractions)
        for frac in sorted_fractions:
            target_count = max(1, int(round(frac * len(all_words_per_sentence))))
            while idx < target_count:
                words = all_words_per_sentence[order[idx]]
                cum_types.update(words)
                cum_n += len(words)
                idx += 1
            points_by_fraction[frac].append((cum_n, len(cum_types)))

    avg_points = []
    for frac in sorted(fractions):
        pts = points_by_fraction[frac]
        avg_n = np.mean([p[0] for p in pts])
        avg_v = np.mean([p[1] for p in pts])
        if avg_n > 0 and avg_v > 0:
            avg_points.append((avg_n, avg_v))

    if len(avg_points) < 3:
        return None

    ns = np.array([p[0] for p in avg_points], dtype=np.float64)
    vs = np.array([p[1] for p in avg_points], dtype=np.float64)
    log_n, log_v = np.log(ns), np.log(vs)
    beta, log_k = np.polyfit(log_n, log_v, 1)
    k = np.exp(log_k)
    pred = beta * log_n + log_k
    ss_res = np.sum((log_v - pred) ** 2)
    ss_tot = np.sum((log_v - log_v.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    v_at_2x = k * (total_n * 2) ** beta
    v_at_10x = k * (total_n * 10) ** beta

    # Linear-scale curve for plotting: V(n) = K * n^beta, evaluated smoothly from 0 to total_n.
    plot_ns = np.linspace(0, ns.max(), 400)
    plot_ns[0] = plot_ns[1] / 10  # avoid n=0 (0^beta is fine, but keep it a hair off zero)
    plot_vs = k * plot_ns ** beta

    return {
        "K": k, "beta": beta, "R2": r2, "total_n": total_n,
        "V_at_2x": v_at_2x, "V_at_10x": v_at_10x,
        "points": (ns, vs, plot_ns, plot_vs),
    }


def plot_zipf(tag, zipf_data, out_path):
    ranks, freqs, plot_ranks, plot_fit = zipf_data
    plt.figure(figsize=(5, 4))
    plt.scatter(ranks, freqs, s=4, alpha=0.5, label="observed")
    plt.plot(plot_ranks, plot_fit, color="red", label="fit")
    plt.xlabel("rank")
    plt.ylabel("frequency")
    plt.title(f"Zipf: {tag} (linear scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def plot_heaps(tag, heaps_data, out_path):
    ns, vs, plot_ns, plot_vs = heaps_data["points"]
    plt.figure(figsize=(5, 4))
    plt.scatter(ns, vs, s=30, zorder=3, label="observed (subsample points)")
    plt.plot(plot_ns, plot_vs, color="red", label="fit")
    plt.xlabel("tokens N")
    plt.ylabel("vocabulary V")
    plt.title(f"Heaps: {tag} (beta={heaps_data['beta']:.3f}, linear scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def analyze_language(tag_label, sentences):
    words = [w for s in sentences for w in tokenize(s)]
    word_counts = Counter(words)
    n_tokens = len(words)
    n_types = len(word_counts)
    hapax = sum(1 for c in word_counts.values() if c == 1)
    hapax_pct = 100 * hapax / n_types if n_types else 0.0

    zipf_result = zipf_fit(word_counts)
    heaps_result = heaps_fit(sentences)

    row = {
        "language": tag_label, "sentences": len(sentences), "tokens_N": n_tokens,
        "types_V": n_types, "hapax_pct": hapax_pct,
    }
    file_stem = tag_label.strip("<>")  # filesystem/URL-safe; e.g. "<as>" -> "as", "union" -> "union"
    if zipf_result and zipf_result[0] is not None:
        alpha, r2, plot_data = zipf_result
        row["zipf_alpha"] = alpha
        row["zipf_R2"] = r2
        plot_zipf(tag_label, plot_data, PLOTS_DIR / f"{file_stem}_zipf.png")
    else:
        row["zipf_alpha"] = row["zipf_R2"] = None

    if heaps_result:
        row["heaps_K"] = heaps_result["K"]
        row["heaps_beta"] = heaps_result["beta"]
        row["heaps_R2"] = heaps_result["R2"]
        row["V_at_10x_N"] = heaps_result["V_at_10x"]
        plot_heaps(tag_label, heaps_result, PLOTS_DIR / f"{file_stem}_heaps.png")
    else:
        row["heaps_K"] = row["heaps_beta"] = row["heaps_R2"] = row["V_at_10x_N"] = None

    return row


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_IN.exists():
        raise SystemExit(f"{TRAIN_IN} does not exist - run data/bpcc-human/create_multilingual_dataset.py first.")

    sentences_by_lang = load_sentences_by_lang(TRAIN_IN)

    rows = []
    all_sentences = []
    for lang in tqdm(sorted(sentences_by_lang), desc="Per-language analysis"):
        tag = LANGUAGE_CODES.get(lang[:3], lang)
        sentences = sentences_by_lang[lang]
        all_sentences.extend(sentences)
        row = analyze_language(tag, sentences)
        row["bpcc_lang"] = lang
        rows.append(row)
        print(f"[corpus_laws] {tag} ({lang}): sentences={row['sentences']:,} tokens={row['tokens_N']:,} "
              f"types={row['types_V']:,} hapax%={row['hapax_pct']:.1f} "
              f"zipf_alpha={row['zipf_alpha']} heaps_beta={row['heaps_beta']}")

    print("[corpus_laws] Analyzing union of all languages...")
    union_row = analyze_language("union", all_sentences)
    union_row["bpcc_lang"] = "ALL"
    rows.append(union_row)
    print(f"[corpus_laws] union: sentences={union_row['sentences']:,} tokens={union_row['tokens_N']:,} "
          f"types={union_row['types_V']:,} zipf_alpha={union_row['zipf_alpha']} "
          f"heaps_beta={union_row['heaps_beta']}")

    csv_path = RESULTS_DIR / "corpus_laws.csv"
    fieldnames = ["language", "bpcc_lang", "sentences", "tokens_N", "types_V", "hapax_pct",
                  "zipf_alpha", "zipf_R2", "heaps_K", "heaps_beta", "heaps_R2", "V_at_10x_N"]
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[corpus_laws] Wrote {csv_path}")


if __name__ == "__main__":
    main()
