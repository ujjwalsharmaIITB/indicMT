# Evaluation

This page reports how IndicMT is scored and how it compares with published systems. The code is in
[`../evaluation/`](../evaluation/). The LLM-judge protocol is in
[`../evaluation/llm_judge/README.md`](../evaluation/llm_judge/README.md).

**Reading the numbers.** The checkpoint reported as the *current best model* was selected using the IN22 test sets
(see "Selection" below), so its scores carry a small optimistic bias. Every result is a **single seed**. The release
model will be chosen after the deeper architectures (12+18 and 18+18) finish fine-tuning, and the tables will be
updated then.

## 1. Method

- **Metrics:** chrF++ and BLEU, computed with `sacrebleu`. **chrF++** is a character n-gram F-score that also uses
  word n-grams up to order 2 (`corpus_chrf` with `word_order=2`). It works better than BLEU for languages with rich
  word forms. **BLEU** is the word n-gram precision score.
- **Test sets:** **IN22-Gen** (1,024 sentences per direction) and **IN22-Conv** (1,503), AI4Bharat's held-out
  benchmarks. Training never touches them.
- **Directions:** all 44 per checkpoint (22 languages, to and from English), beam size 5.
- **Pipeline:** `evaluation/build_eval_pairs.py` builds each direction's source file from the multi-parallel
  per-language test files (target tag prepended, frozen 128k SentencePiece encoding). `evaluation/run_eval.py` decodes
  with `onmt_translate` (beam 5, OpenNMT-py's default length penalty), detokenizes with the same SentencePiece model
  and scores against the untokenized references. Scores are **corpus-level**: one score over the whole test set.
- **Macro average:** the unweighted mean of the 22 per-language scores in a direction bucket (en to xx, or xx to en).
  Small languages count as much as large ones.
- **Raw weights:** every checkpoint holds raw training weights (no moving average), so a score at step N is the model
  at step N.
- **Separate inference path:** the CTranslate2 evaluation used for deployment ([`deployment.md`](deployment.md)) is
  never merged with these numbers.
- Outputs go to `evaluation/results/<run>/<checkpoint>/` and are not tracked by git.

## 2. What was scored

Every pre-training arm on a dense grid (every 100 steps to about step 5,000, then every 25% of a human epoch or 10%
of a mined epoch); every checkpoint of the four fine-tunes (every 200 steps); and every checkpoint still on disk of
the two deeper architectures (every 1,000 steps).

**Selection.** The checkpoint to carry forward is chosen by one rule: the highest mean chrF++ over IN22-Gen and
IN22-Conv, in both directions. That is selection on the test sets, hence the optimistic bias.

## 3. Current best model

The current best model has the 12+12 architecture. It was pre-trained on the combined 8:2 (mined to human) data, best
checkpoint at step 13,600, then fine-tuned on human data without replay, best checkpoint at step 1,800.

Macro over 22 languages:

| Test set | en→xx chrF++ | en→xx BLEU | xx→en chrF++ | xx→en BLEU |
|---|---|---|---|---|
| IN22-Gen | 45.36 | 16.39 | 53.73 | 28.71 |
| IN22-Conv | 41.06 | 13.01 | 50.04 | 27.62 |

## 4. Against published systems (IN22-Gen only)

Baselines are the published chrF++ scores from the `bodhan-ai/indic-translate` model card (read on 2026-09-19). They
are taken as published and not re-run. The card's scores are **sentence-level with greedy decoding**. Ours are
**corpus-level with beam 5**. So the comparison is indicative, not exact. No IN22-Conv baseline exists. The table is
sorted by the mean of the two directions. A star marks a system that covers only 16 of the 22 languages.

| Rank | Model | en→xx | xx→en | Mean | Parameters |
|---|---|---|---|---|---|
| 1 | IndicTrans2 | 48.60 | 62.80 | 55.70 | 1.1B |
| 2 | Indic Translate | 47.70 | 60.40 | 54.05 | 7.94B |
| 3 | gemini-3.5-flash | 40.30 | 66.60 | 53.45 | undisclosed |
| 4 | Sarvam Translate | 48.90 | 52.40 | 50.65 | 4B |
| 5 | **IndicMT (current best model)** | 45.36 | 53.73 | 49.54 | 0.154B |
| 6 | gemma-4-31B-it | 39.20 | 59.60 | 49.40 | 30.7B |
| 7 | translategemma-27b-it * | 39.60 | 58.50 | 49.05 | 27B |
| 8 | gemma-4-26B-a4b-it | 36.20 | 56.60 | 46.40 | 25.2B |
| 9 | translategemma-12b-it * | 37.10 | 55.00 | 46.05 | 12B |
| 10 | gemma-3-27b-it | 31.90 | 55.50 | 43.70 | 27.43B |
| 11 | gemma-3-12b-it | 28.80 | 53.20 | 41.00 | 12.19B |
| 12 | translategemma-4b-it * | 31.20 | 49.20 | 40.20 | 4B |
| 13 | gemma-3-4b-it | 22.90 | 47.10 | 35.00 | 4.3B |

IndicMT ranks 5th of 13 on the mean, 0.14 above gemma-4-31B-it and 1.11 below Sarvam Translate. It is **4th on
en→xx**, behind Sarvam Translate (48.9, +3.5 higher), IndicTrans2 (48.6) and Indic Translate (47.7). It is 9th on
xx→en, ahead of Sarvam Translate (52.4). It has 7 times fewer parameters than the 1.1B IndicTrans2 variant and 26 times
fewer than Sarvam Translate. The human-only pre-training checkpoint at step 11,400 (44.05 / 44.08) ranked 9th on the
mean.

### Model sizes

Parameter counts (in billions) come from each model's public page or report. They were not measured here. Ours is
measured: 153,807,872 (encoder 103,310,336 and decoder 50,497,536). "nominal" means taken from the model name,
"rounded" a rounded figure on the page, and "variant" that the card does not say which variant was ranked.

| Model | Parameters | Basis | Note |
|---|---|---|---|
| Sarvam Translate | 4.0B | rounded | Hugging Face model page shows '4B params'; fine-tuned from Gemma 3 4B |
| IndicTrans2 | 1.1B | variant | The 1.1B variant; the model card does not say which variant was ranked (a distilled variant is about 0.3B) |
| Indic Translate | 7.94B | card | Total parameters as stated on its own model card; built on Gemma 4 E4B |
| gemini-3.5-flash | not disclosed | undisclosed | Google does not publish a parameter count |
| translategemma-27b-it | 27.0B | nominal | Size from the model name; built on Gemma 3 27B |
| gemma-4-31B-it | 30.7B | primary | Total parameters on the Hugging Face model card |
| translategemma-12b-it | 12.0B | nominal | Size from the model name; built on Gemma 3 12B |
| gemma-4-26B-a4b-it | 25.2B total, 3.8B active | primary | Mixture of experts: 25.2B total, 3.8B active per token (model card) |
| gemma-3-27b-it | 27.43B | primary | Gemma 3 technical report parameter table, including the vision encoder |
| translategemma-4b-it | 4.0B | nominal | Size from the model name; built on Gemma 3 4B |
| gemma-3-12b-it | 12.19B | primary | Gemma 3 technical report parameter table, including the vision encoder |
| gemma-3-4b-it | 4.3B | primary | Gemma 3 technical report parameter table, including the vision encoder |

## 5. Results by language

chrF++ of the current best model on both test sets, with the best published IN22-Gen score among the three systems
whose card lists per-language numbers (Indic Translate, Sarvam Translate, IndicTrans2):

| Language | en→xx Gen | en→xx Conv | xx→en Gen | xx→en Conv | en→xx best published (Gen) | xx→en best published (Gen) |
|---|---|---|---|---|---|---|
| Assamese | 42.98 | 42.45 | 56.14 | 54.91 | 47.1 | 65.8 |
| Bengali | 46.31 | 43.64 | 54.30 | 53.35 | 51.8 | 63.2 |
| Bodo | 46.92 | 43.67 | 52.23 | 48.39 | 48.9 | 62.1 |
| Dogri | 52.93 | 46.57 | 60.39 | 57.62 | 58.2 | 72.6 |
| Gujarati | 49.52 | 48.60 | 57.70 | 56.28 | 53.5 | 66.5 |
| Hindi | 51.56 | 45.48 | 56.76 | 54.42 | 56.7 | 65.4 |
| Kannada | 47.96 | 31.08 | 55.09 | 42.93 | 52.2 | 64.2 |
| Kashmiri | 31.31 | 24.21 | 50.30 | 46.42 | 40.2 | 60.4 |
| Konkani | 42.01 | 38.66 | 49.55 | 44.61 | 45.2 | 59.2 |
| Maithili | 48.40 | 41.78 | 56.33 | 51.29 | 50.5 | 64.8 |
| Malayalam | 43.37 | 36.91 | 53.37 | 48.11 | 50.9 | 64.5 |
| Manipuri | 40.64 | 37.49 | 50.44 | 46.01 | 45.5 | 57.9 |
| Marathi | 47.40 | 44.56 | 54.88 | 52.57 | 51.3 | 63.7 |
| Nepali | 48.28 | 47.15 | 58.59 | 56.78 | 49.9 | 67.7 |
| Odia | 44.95 | 40.17 | 56.87 | 53.93 | 48.2 | 66.2 |
| Punjabi | 45.99 | 50.98 | 53.76 | 55.71 | 50.6 | 63.4 |
| Sanskrit | 35.02 | 32.37 | 46.03 | 41.33 | 38.8 | 54.8 |
| Santali | 38.68 | 40.63 | 43.47 | 44.15 | 40.9 | 47.3 |
| Sindhi | 37.93 | 33.81 | 50.15 | 45.16 | 39.1 | 57.3 |
| Tamil | 46.63 | 37.01 | 50.33 | 41.41 | 50.9 | 59.8 |
| Telugu | 48.35 | 41.93 | 54.12 | 47.39 | 52.4 | 64.8 |
| Urdu | 60.75 | 54.23 | 61.26 | 58.09 | 68.2 | 73.0 |

- **en→xx is close to the specialists in several low-resource languages.** The model trails the best published system
  by a median of 4.1 chrF++ on IN22-Gen en→xx, but by only 1.2 to 2.2 for Sindhi, Nepali, Bodo, Maithili and Santali.
  The largest gaps are Kashmiri (8.9), Malayalam (7.5) and Urdu (7.5). It does not beat the best published system in
  any language.
- **xx→en trails further,** by a median of 9.2 (Santali 3.8 is the smallest; Dogri 12.2 the largest). Every model of ours
  that saw mined data reaches 53 to 54 here, whatever the fine-tuning, so the gap is not specific to this checkpoint.
- **Weakest directions:** English to Kashmiri (31.31 Gen, 24.21 Conv), English to Sanskrit (35.02) and Santali to
  English (43.47). Sanskrit is the second-weakest en→xx direction, but its gap to the best published system (3.8) is
  close to the median, so the evidence does not single out the 128k vocabulary.
- **Conversational en→xx is uneven.** IN22-Conv is lower than IN22-Gen for most languages, by 16.9 for Kannada and
  9.6 for Tamil, but higher for Punjabi (+5.0) and Santali (+2.0). The causes have not been examined.

## 6. Fine-tuning results

Four fine-tuning runs, 30 checkpoints each (every 200 steps), scored on both test sets, all 44 directions, beam 5. The
best checkpoint of each run is chosen by the rule in section 2. "Replay" means 20% mined data was mixed back in. See
[`training.md`](training.md) for the setup and the findings.

| Run | Best step | IN22-Gen en→xx | IN22-Gen xx→en | IN22-Conv en→xx | IN22-Conv xx→en | IN22-Gen 44-direction mean |
|---|---|---|---|---|---|---|
| combined, start (pre-training step 13,600) | – | 43.84 | 53.80 | 39.56 | 49.10 | 48.82 |
| **combined → human, no replay (current best)** | **1,800** | **45.36** | 53.73 | **41.06** | **50.04** | **49.54** |
| combined → human + 20% replay | 2,000 | 44.89 | **53.89** | 40.46 | 49.66 | 49.39 |
| mined, start (pre-training step 24,600) | – | 30.45 | 44.24 | 26.89 | 38.79 | 37.35 |
| mined → human, no replay | 4,400 | 39.73 | 53.16 | 36.29 | 49.24 | 46.45 |
| mined → human + 20% replay | 5,800 | 39.53 | 53.84 | 36.20 | 49.45 | 46.69 |
| human-only pre-training, step 11,400 (reference) | – | 44.05 | 44.08 | | | 44.06 |

IN22-Gen en→xx / xx→en chrF++ along each run (selected steps):

| Step | combined, no replay | combined + replay | mined, no replay | mined + replay |
|---|---|---|---|---|
| 200 | not reported here | 44.31 / 53.86 | 33.31 / 45.99 | 33.22 / 45.96 |
| 1,000 | not reported here | 44.37 / 54.04 | 35.26 / 51.12 | 34.69 / 51.10 |
| 1,800 | **45.36 / 53.73** | 44.59 / 53.80 | 38.93 / 52.37 | 37.10 / 52.58 |
| 2,000 | 44.97 / 53.61 | 44.89 / 53.89 | 38.67 / 52.47 | 37.59 / 52.87 |
| 3,000 | 44.80 / 53.35 | 44.48 / 53.80 | 38.97 / 53.01 | 37.86 / 53.46 |
| 4,000 | 44.45 / 53.25 | 44.68 / 53.54 | 37.96 / 53.15 | 38.72 / 53.53 |
| 5,000 | 44.58 / 53.02 | 44.39 / 53.52 | 38.58 / 53.40 | 38.65 / 53.64 |
| 6,000 | 44.75 / 52.67 | 44.06 / 53.54 | 38.69 / 53.11 | 38.65 / 53.86 |

Per-checkpoint scores (chrF++ and BLEU for every step) are written to `evaluation/results/` by the evaluation scripts
and are not tracked in git.

### Per-language gap: mined → human against combined → human

IN22-Gen chrF++, sorted by post-mixing mined-to-human token ratio. The gap is largest where the mined model had little
or no data. See [`training.md`](training.md), section 7.

| Language | Mined : human tokens | en→xx: mined start | en→xx: mined → human | en→xx: combined → human | en→xx gap | xx→en: mined → human | xx→en: combined → human | xx→en gap |
|---|---|---|---|---|---|---|---|---|
| Dogri | none | 11.2 | 43.9 | 52.9 | **+9.0** | 59.7 | 60.4 | +0.7 |
| Konkani | none | 8.2 | 37.3 | 42.0 | **+4.7** | 49.3 | 49.5 | +0.3 |
| Manipuri | none | 0.9 | 24.7 | 40.6 | **+16.0** | 49.1 | 50.4 | +1.3 |
| Santali | none | 0.8 | 26.4 | 38.7 | **+12.3** | 41.1 | 43.5 | +2.4 |
| Sindhi | none | 8.5 | 32.0 | 37.9 | **+5.9** | 48.4 | 50.1 | +1.8 |
| Bodo | <0.01× | 7.7 | 38.7 | 46.9 | **+8.2** | 50.9 | 52.2 | +1.3 |
| Maithili | 0.9× | 27.4 | 42.3 | 48.4 | **+6.0** | 56.3 | 56.3 | +0.0 |
| Kashmiri | 2.2× | 21.7 | 26.0 | 31.3 | **+5.3** | 48.7 | 50.3 | +1.6 |
| Sanskrit | 2.4× | 18.2 | 26.6 | 35.0 | **+8.4** | 45.1 | 46.0 | +0.9 |
| Assamese | 5.7× | 34.4 | 37.6 | 43.0 | **+5.4** | 56.5 | 56.1 | -0.3 |
| Nepali | 26.6× | 40.1 | 43.5 | 48.3 | **+4.7** | 58.4 | 58.6 | +0.1 |
| Punjabi | 44.4× | 44.2 | 43.6 | 46.0 | **+2.4** | 53.1 | 53.8 | +0.6 |
| Odia | 44.9× | 39.3 | 41.7 | 44.9 | **+3.2** | 56.3 | 56.9 | +0.6 |
| Marathi | 46.5× | 41.4 | 43.5 | 47.4 | **+3.9** | 54.8 | 54.9 | +0.1 |
| Urdu | 50.1× | 53.8 | 54.9 | 60.7 | **+5.8** | 60.6 | 61.3 | +0.7 |
| Tamil | 56.0× | 44.8 | 43.9 | 46.6 | **+2.8** | 49.8 | 50.3 | +0.5 |
| Hindi | 61.5× | 49.9 | 47.5 | 51.6 | **+4.1** | 56.6 | 56.8 | +0.2 |
| Gujarati | 62.1× | 46.1 | 45.8 | 49.5 | **+3.7** | 57.6 | 57.7 | +0.1 |
| Malayalam | 63.8× | 40.1 | 40.4 | 43.4 | **+3.0** | 53.6 | 53.4 | -0.2 |
| Bengali | 66.7× | 42.0 | 43.7 | 46.3 | **+2.6** | 54.2 | 54.3 | +0.1 |
| Kannada | 71.7× | 43.3 | 45.3 | 48.0 | **+2.7** | 55.3 | 55.1 | -0.2 |
| Telugu | 75.3× | 45.8 | 44.8 | 48.4 | **+3.6** | 54.1 | 54.1 | -0.0 |

## 7. LLM-as-judge (Gemma 4 31B-it, en→xx)

chrF++ and BLEU compare surface text. We also used a language model as a judge, on all **55,594** English-to-Indic
sentences of IN22-Gen and IN22-Conv. The judge, Gemma 4 31B-it, scores **fluency**, **adequacy** (is the meaning
there?) and **overall** from 1 to 5, at temperature 0 with seed 42. It runs in two variants:

- **V2** (the headline) sees the human reference.
- **V1** is reference-free. It is a control against rewarding mere overlap with the reference.

"Perfect" means fluency = adequacy = overall = 5. Means are over every scored item. Failed items are excluded, and the
one empty output was auto-scored. Protocol details: [`../evaluation/llm_judge/README.md`](../evaluation/llm_judge/README.md).

**Sanity checks.** A deliberately mismatched translation gets adequacy 1.00 (pass). The human reference itself, judged
reference-free, gets overall 4.62 and a perfect 5/5/5 on about 79% of items. That is below the expected 4.7, because
the judge flags real errors in the IN22 references (for example a tense error in an Assamese reference, and a Hindi
reference that renders "dharmic action" as religious struggles). Santali (identity overall 3.88) and Manipuri (4.02)
are unreliable for this judge.

| Model | Variant | Perfect 5/5/5 | Mean of the three ≥ 4 | Fluency | Adequacy | Overall | Failed items |
|---|---|---|---|---|---|---|---|
| **Current best (step 1,800)** | V2, with reference | **46.7%** | **57.5%** | 4.04 | 4.16 | 3.80 | 5 |
| **Current best (step 1,800)** | V1, reference-free | **50.1%** | **59.8%** | 4.10 | 4.22 | 3.87 | 20 |
| Replay (step 2,000) | V2, with reference | 44.9% | 56.1% | 4.02 | 4.13 | 3.77 | 13 |
| Replay (step 2,000) | V1, reference-free | 48.5% | 58.7% | 4.07 | 4.19 | 3.84 | 20 |

By test set (current best, V2): IN22-Conv 54.7% perfect (overall 4.01) and IN22-Gen 34.8% (overall 3.51). Under V1:
58.5% and 37.8%. Both models and both variants score clearly higher on the conversational set.

**Against the replay checkpoint.** Paired over the same items, the current best is ahead under both variants: +1.76
points perfect under V2 (95% bootstrap interval 1.43 to 2.09, 1,000 resamples of items) and +1.61 under V1 (1.30 to
1.91). Mean overall is +0.04 and +0.03. Per item, 8,597 improve and 7,364 get worse under V2 (39,617 unchanged). The
judge agrees with the chrF++-based choice, though the margin is small, as the chrF++ difference was.

V1 and V2 agree within about 3.5 points for both models, so the reference-aided scores are not driven by overlap with
the reference.

**Per language (current best, V2; overall / perfect):**

| Language | Overall | Perfect | Language | Overall | Perfect |
|---|---|---|---|---|---|
| Punjabi | 4.19 | 60.3% | Urdu | 3.91 | 49.3% |
| Bengali | 4.16 | 59.4% | Maithili | 3.90 | 48.8% |
| Gujarati | 4.14 | 58.2% | Marathi | 3.88 | 48.8% |
| Telugu | 4.06 | 55.4% | Assamese | 3.83 | 47.7% |
| Hindi | 4.06 | 56.1% | Konkani | 3.73 | 41.9% |
| Kannada | 4.03 | 54.1% | Bodo | 3.59 | 38.0% |
| Odia | 4.02 | 55.4% | Dogri | 3.58 | 37.1% |
| Malayalam | 3.99 | 53.6% | Santali | 3.47 | 33.7% |
| Tamil | 3.93 | 51.5% | Sindhi | 3.39 | 33.6% |
| Nepali | 3.93 | 49.7% | Sanskrit | 3.36 | 31.0% |
| | | | Kashmiri | 3.28 | 31.7% |
| | | | Manipuri | 3.25 | 31.1% |

The ordering matches the replay checkpoint's closely (same top four and bottom six). Hindi is the one language that
drops (4.15 to 4.06), while Telugu, Nepali and Urdu gain the most. Excluding Santali and Manipuri, 48.1% are perfect.
Compared with the 79% the references themselves reach, 46.7% is about 59% of the human ceiling.

The judge's per-item outputs and summaries are written to `evaluation/llm_judge/results/` and are not tracked in git.
