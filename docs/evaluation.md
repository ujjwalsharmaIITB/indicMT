# Evaluation

This page reports how IndicMT is scored and how it compares with published systems. The code is in
[`../evaluation/`](../evaluation/). The LLM-judge protocol is in
[`../evaluation/llm_judge/README.md`](../evaluation/llm_judge/README.md).

**Reading the numbers.** The *release model* was selected using the IN22 test sets (see "Selection" below), so its
scores carry a small optimistic bias. It is a **single training run**. Two extra seeds of the human-only arm reproduce
that arm's chrF++ curve (38.3, 38.3 and 38.5 en→xx on IN22-Gen at step 2,000). All scores on this page use the
training-format source encoding described in section 2. Scores published before that fix are superseded.

## 1. Method

- **Metrics:** chrF++ and BLEU, computed with `sacrebleu`. **chrF++** is a character n-gram F-score that also uses
  word n-grams up to order 2 (`corpus_chrf` with `word_order=2`). It works better than BLEU for languages with rich
  word forms. **BLEU** is the word n-gram precision score.
- **Test sets:** **IN22-Gen** (1,024 sentences per direction) and **IN22-Conv** (1,503), AI4Bharat's held-out
  benchmarks. Training never touches them.
- **Directions:** all 44 per checkpoint (22 languages, to and from English), beam size 5.
- **Pipeline:** `evaluation/build_eval_pairs.py` builds each direction's source file from the untokenized
  multi-parallel per-language test files: the target tag and the raw sentence, `<xx> sentence`, encoded in one call
  with the frozen 128k SentencePiece model, exactly as the training sources were (`data/common/tokenization.py`).
  `evaluation/run_eval.py` decodes with `onmt_translate` (beam 5, OpenNMT-py's default length penalty), detokenizes
  with the same SentencePiece model and scores against the untokenized references. Scores are **corpus-level**: one
  score over the whole test set.
- **Macro average:** the unweighted mean of the 22 per-language scores in a direction bucket (en to xx, or xx to en).
  Small languages count as much as large ones.
- **Raw weights:** every checkpoint holds raw training weights (no moving average), so a score at step N is the model
  at step N.
- **Separate inference path:** the CTranslate2 evaluation used for deployment ([`deployment.md`](deployment.md)) is
  reported separately and never merged with these numbers.
- Outputs go to `evaluation/results/<run>/<checkpoint>/` and are not tracked by git.

## 2. Source encoding

Every training source is written as `<xx> sentence` and encoded by SentencePiece as one string. With SentencePiece's
default dummy prefix and the 23 tags registered as user-defined pieces, that always gives
`['▁', '<xx>', '▁word', ...]`: a lone `▁` piece first, then the tag. Our evaluation, the CTranslate2 path and our
analyses had instead encoded the sentence and then put the tag in front (`['<xx>', '▁word', ...]`). That drops the
leading `▁`, so the model was evaluated on a source format it never saw in training.

**What changed.** One shared module, `data/common/tokenization.py`, now builds and encodes every source. The data
pipeline (`tag_source`), `evaluation/build_eval_pairs.py`, `deployment/ct2_eval.py` and `deployment/ct2_infer.py` (and
so the API) all use it. It reproduces 20,000 lines each of the encoded human training, mined training and dev files
exactly, and it rejects any source that does not start with `['▁', '<xx>']`.

**Effect on scores.** We re-scored 564 checkpoints. On average, English-to-Indic chrF++ rose by 1.96 on IN22-Gen and
2.22 on IN22-Conv; Indic-to-English rose by 0.02 and 0.37. The effect grows with training length:

| Checkpoint | IN22-Gen en→xx, old encoding | IN22-Gen en→xx, training encoding |
|---|---|---|
| combined pre-training (12+12), step 37,700 | 36.1 | 46.4 |
| combined pre-training (12+18), step 49,000 | 34.8 | 46.6 |
| human-only pre-training, step 20,800 | 42.9 | 45.3 |

For the release model, the macro scores moved from 45.36 / 53.73 (IN22-Gen) and 41.06 / 50.04 (IN22-Conv) to the
numbers in section 4. The earlier finding that English-to-Indic quality declines late in pre-training was an artifact
of this mismatch. With the training encoding, English-to-Indic keeps rising to the end of training (see
[`training.md`](training.md), section 6). The lesson we took: one function owns the input format, and every entry
point checks it.

## 3. What was scored

Every pre-training arm on a dense grid (every 100 steps to about step 5,000, then every 25% of a human epoch or 10%
of a mined epoch); every checkpoint of the four fine-tunes and of the 12+18 fine-tune (every 200 steps); and every
checkpoint still on disk of the two deeper architectures (every 1,000 steps).

**Selection.** The checkpoint to carry forward is chosen by one rule: the highest mean chrF++ over IN22-Gen and
IN22-Conv, in both directions (the "mean of four" in section 8). That is selection on the test sets, hence the
optimistic bias.

## 4. Release model

The release model has the 12+12 architecture. It was pre-trained on the combined 8:2 (mined to human) data to step
13,600, then fine-tuned on human data without replay. The release checkpoint is step 1,800.

Macro over 22 languages, OpenNMT-py (`onmt_translate`, beam 5):

| Test set | en→xx chrF++ | en→xx BLEU | xx→en chrF++ | xx→en BLEU |
|---|---|---|---|---|
| IN22-Gen | 46.95 | 19.11 | 53.69 | 29.00 |
| IN22-Conv | 42.97 | 15.80 | 50.38 | 28.29 |

Through CTranslate2 on CPU (the served path, IN22-Gen), the same model scores 47.15 / 53.87 in float32 and
47.16 / 53.85 in int8 ([`deployment.md`](deployment.md)).

**Model selection.** Under the corrected scores, several checkpoints are within 0.2 chrF++ of the release model on the
mean of four (48.50): 18+18 combined pre-training at step 38,000 (48.70), mined → human at step 5,800 (48.70),
12+18 → human at step 1,400 (48.59) and step 1,000 of the release run (48.58). These differences are small, come from a
single seed and have not been significance-tested. The release model was kept.

## 5. Against published systems (IN22-Gen only)

IndicTrans2, Indic Translate and Sarvam Translate are **our own runs** on both test sets (section 6 for IndicTrans2;
Indic Translate and Sarvam Translate with vLLM, greedy decoding and each model's documented prompt), scored exactly like
IndicMT: corpus-level chrF++, the same references, macro over 22 languages. **All other baselines are the published
chrF++ scores from the `bodhan-ai/indic-translate` model page** (read on 2026-09-19), which are **sentence-level with
greedy decoding** on IN22-Gen only, so the comparison with those rows is indicative, not exact. IndicMT is the
CTranslate2 float32 model (beam 5). The table is sorted by the mean of the two directions. A star marks a system that
covers only 16 of the 22 languages.

| Rank | Model | en→xx | xx→en | Mean | Parameters | Source |
|---|---|---|---|---|---|---|
| 1 | IndicTrans2 1B | 46.39 | 62.48 | 54.44 | 1.1B | our run |
| 2 | Indic Translate | 47.73 | 60.33 | 54.03 | 7.94B | our run |
| 3 | gemini-3.5-flash | 40.30 | 66.60 | 53.45 | undisclosed | model page |
| 4 | **IndicMT (release model)** | **47.15** | **53.87** | **50.51** | 0.154B | our run |
| 5 | Sarvam Translate | 48.74 | 51.82 | 50.28 | 4B | our run |
| 6 | gemma-4-31B-it | 39.20 | 59.60 | 49.40 | 30.7B | model page |
| 7 | translategemma-27b-it * | 39.60 | 58.50 | 49.05 | 27B | model page |
| 8 | gemma-4-26B-a4b-it | 36.20 | 56.60 | 46.40 | 25.2B | model page |
| 9 | translategemma-12b-it * | 37.10 | 55.00 | 46.05 | 12B | model page |
| 10 | gemma-3-27b-it | 31.90 | 55.50 | 43.70 | 27.43B | model page |
| 11 | gemma-3-12b-it | 28.80 | 53.20 | 41.00 | 12.19B | model page |
| 12 | translategemma-4b-it * | 31.20 | 49.20 | 40.20 | 4B | model page |
| 13 | gemma-3-4b-it | 22.90 | 47.10 | 35.00 | 4.3B | model page |

The model page lists 48.6 / 62.8 for IndicTrans2, 47.7 / 60.4 for Indic Translate and 48.9 / 52.4 for Sarvam
Translate; under our protocol these are 46.39 / 62.48, 47.73 / 60.33 and 48.74 / 51.82.

IndicMT ranks 4th of 13 on the mean, 0.23 above Sarvam Translate. It is **3rd on en→xx**, behind Sarvam Translate
(48.74, 1.59 higher) and Indic Translate (47.73, 0.58 higher), and 0.76 above IndicTrans2. It is 9th on xx→en, ahead of
Sarvam Translate (51.82). It has 7.2 times fewer parameters than the 1.1B IndicTrans2, 26 times fewer than Sarvam
Translate and 51.6 times fewer than Indic Translate.

**IN22-Conv and BLEU (our runs).**

| Model | Gen en→xx chrF++ / BLEU | Gen xx→en chrF++ / BLEU | Conv en→xx chrF++ / BLEU | Conv xx→en chrF++ / BLEU |
|---|---|---|---|---|
| IndicMT (CTranslate2 float32) | 47.15 / 19.19 | 53.87 / 29.19 | 43.15 / 15.73 | 50.59 / 28.36 |
| IndicTrans2 1B | 46.39 / 17.88 | 62.48 / 39.12 | 42.61 / 15.22 | 55.63 / 34.02 |
| Indic Translate | 47.73 / 19.37 | 60.33 / 36.43 | 45.04 / 17.30 | 55.10 / 33.37 |
| Sarvam Translate | 48.74 / 20.67 | 51.82 / 28.32 | 45.71 / 17.95 | 49.77 / 27.51 |

On IN22-Conv en→xx IndicMT is 1.89 chrF++ below Indic Translate and 2.56 below Sarvam Translate; on xx→en it is ahead
of Sarvam Translate on both test sets.

### Model sizes

Parameter counts (in billions) come from each model's public page or report. They were not measured here. Ours is
measured: 153,807,872 (encoder 103,310,336 and decoder 50,497,536). "nominal" means taken from the model name,
"rounded" a rounded figure on the page, and "variant" that the card does not say which variant was ranked.

| Model | Parameters | Basis | Note |
|---|---|---|---|
| Sarvam Translate | 4.0B | rounded | Hugging Face model page shows '4B params'; fine-tuned from Gemma 3 4B |
| IndicTrans2 | 1.1B | variant | The 1.1B `indictrans2-{en-indic,indic-en}-1B` models, which we re-ran (section 6); the model card does not say which variant it ranked (a distilled variant is about 0.3B) |
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

## 6. IndicTrans2 under the same protocol

The model-card scores above are sentence-level with greedy decoding. To remove that difference for the strongest open
baseline, we ran `ai4bharat/indictrans2-en-indic-1B` (en→xx) and `ai4bharat/indictrans2-indic-en-1B` (xx→en) on the raw
IN22 text of both test sets. We used IndicTransToolkit pre- and post-processing as on the model cards and beam 5, and
scored the outputs exactly like IndicMT (corpus-level sacreBLEU chrF++, same references, macro over 22 languages).

| Test set | Direction | IndicTrans2 1B | IndicMT |
|---|---|---|---|
| IN22-Gen | en→xx | 46.39 | 46.95 |
| IN22-Gen | xx→en | 62.48 | 53.69 |
| IN22-Conv | en→xx | 42.61 | 42.97 |
| IN22-Conv | xx→en | 55.63 | 50.38 |

Under one protocol, IndicTrans2's en→xx IN22-Gen score is 46.39, against 48.6 on its model card. Its xx→en score
hardly moves (62.48 against 62.8).

**Significance (IN22-Gen).** We used `evaluation/paired_significance.py`: sacreBLEU paired bootstrap, 1,000 resamples,
chrF++, IndicTrans2 as the baseline, with the 22 directions of each bucket pooled.

- **en→xx:** IndicMT is slightly but significantly ahead when the directions are pooled (47.11 against 46.53,
  p = 0.001). Per direction it is split: 11 directions significantly better and 10 significantly worse (p < 0.05).
- **xx→en:** IndicTrans2 is ahead in all 22 directions (pooled p = 0.001).

IndicTrans2 1B has 1.1B parameters, 7.2 times more than IndicMT. The IN22-Conv comparison was not significance-tested.

## 7. Results by language

chrF++ of the release model (CTranslate2 float32, beam 5) on both test sets, with the best IN22-Gen score among our
runs of the three baselines with per-language numbers (Indic Translate, Sarvam Translate, IndicTrans2 1B), all scored
with the same protocol:

| Language | en→xx Gen | en→xx Conv | xx→en Gen | xx→en Conv | en→xx best baseline (Gen) | xx→en best baseline (Gen) |
|---|---|---|---|---|---|---|
| Assamese | 44.74 | 44.58 | 56.02 | 55.39 | 46.40 | 65.13 |
| Bengali | 47.24 | 45.01 | 54.39 | 53.75 | 49.47 | 62.04 |
| Bodo | 49.30 | 46.39 | 52.34 | 49.18 | 48.81 | 60.81 |
| Dogri | 55.86 | 50.64 | 61.04 | 58.29 | 58.26 | 71.97 |
| Gujarati | 50.88 | 50.21 | 57.92 | 56.42 | 53.09 | 66.42 |
| Hindi | 53.19 | 47.14 | 56.99 | 55.34 | 56.56 | 64.51 |
| Kannada | 49.06 | 31.70 | 55.31 | 43.40 | 52.07 | 64.50 |
| Kashmiri | 32.99 | 27.55 | 50.47 | 47.18 | 35.12 | 60.44 |
| Konkani | 43.51 | 41.41 | 49.67 | 45.31 | 44.98 | 57.92 |
| Maithili | 49.69 | 44.08 | 56.72 | 51.92 | 50.45 | 65.86 |
| Malayalam | 44.83 | 38.19 | 53.35 | 48.49 | 47.69 | 63.98 |
| Manipuri | 46.26 | 41.64 | 50.29 | 46.54 | 45.29 | 59.95 |
| Marathi | 48.63 | 46.08 | 54.94 | 52.94 | 51.41 | 64.68 |
| Nepali | 49.68 | 49.60 | 59.00 | 57.08 | 50.19 | 69.28 |
| Odia | 46.23 | 40.87 | 57.14 | 54.57 | 48.11 | 65.88 |
| Punjabi | 47.12 | 52.55 | 53.73 | 56.25 | 48.36 | 59.66 |
| Sanskrit | 37.26 | 34.35 | 45.86 | 41.76 | 38.07 | 54.40 |
| Santali | 41.11 | 42.99 | 43.72 | 45.11 | 40.87 | 47.61 |
| Sindhi | 39.73 | 37.11 | 50.54 | 46.16 | 38.91 | 55.58 |
| Tamil | 47.63 | 38.19 | 50.60 | 41.90 | 51.05 | 59.79 |
| Telugu | 49.58 | 42.56 | 54.03 | 47.61 | 51.77 | 64.75 |
| Urdu | 62.73 | 56.44 | 61.04 | 58.34 | 66.69 | 73.04 |

- **en→xx is at or above the best baseline in several low-resource languages.** On IN22-Gen en→xx the model trails
  the best of the three baselines by a median of 2.00 chrF++. It is above it for Manipuri (+0.96), Sindhi (+0.82),
  Bodo (+0.48) and Santali (+0.24), and within 1 for Nepali (0.51), Maithili (0.76) and Sanskrit (0.82). The largest
  gaps are Urdu (3.95), Tamil (3.42) and Hindi (3.37).
- **xx→en trails further,** by a median of 9.12 (Santali 3.90 is the smallest; Urdu 11.99 the largest). Every model of
  ours trained on both mined and human data reaches 53.69 to 55.07 here (OpenNMT-py scores, section 8), whatever the
  fine-tuning, so the gap is not specific to this checkpoint.
- **Weakest directions:** English to Kashmiri (32.99 Gen, 27.55 Conv), English to Sanskrit (37.26) and Santali to
  English (43.72). Sanskrit is the second-weakest en→xx direction, but its gap to the best baseline (0.82) is well below
  the median, so the evidence does not single out the 128k vocabulary.
- **Conversational en→xx is uneven.** IN22-Conv is lower than IN22-Gen for most languages, by 17.35 for Kannada and
  9.45 for Tamil, but higher for Punjabi (+5.43) and Santali (+1.88). The causes have not been examined.

## 8. Pre-training, fine-tuning and architecture runs

The table gives the best checkpoint of each run by the selection rule (section 3), plus the release checkpoint.
"Replay" means 20% mined data was mixed back in. The fine-tunes start from combined step 13,600 and mined step 24,600,
and the 12+18 fine-tune starts from step 13,000. These starting points were chosen with the old encoding (section 2).
See [`training.md`](training.md) for the setup.

| Run | Step | IN22-Gen en→xx | IN22-Gen xx→en | IN22-Conv en→xx | IN22-Conv xx→en | Mean of four |
|---|---|---|---|---|---|---|
| human-only pre-training | 13,900 | 45.21 | 43.80 | 41.67 | 41.31 | 43.00 |
| mined-only pre-training | 35,000 | 32.68 | 44.20 | 28.13 | 38.85 | 35.96 |
| combined pre-training, 12+12 | 37,700 | 46.36 | 55.05 | 41.81 | 50.55 | 48.44 |
| combined pre-training, 12+18 | 38,000 | 46.65 | 54.51 | 42.23 | 50.13 | 48.38 |
| combined pre-training, 18+18 | 38,000 | 46.79 | 55.07 | 42.32 | 50.62 | 48.70 |
| combined → human, no replay | 1,000 | 46.80 | 54.04 | 42.80 | 50.66 | 48.58 |
| **combined → human, no replay (release model)** | **1,800** | **46.95** | **53.69** | **42.97** | **50.38** | **48.50** |
| combined → human + 20% replay | 2,800 | 46.64 | 53.85 | 42.32 | 50.09 | 48.22 |
| mined → human, no replay | 5,800 | 47.04 | 54.13 | 43.00 | 50.61 | 48.70 |
| mined → human + 20% replay | 5,400 | 46.65 | 54.47 | 42.03 | 50.28 | 48.36 |
| 12+18 combined → human, no replay | 1,400 | 47.08 | 53.73 | 43.15 | 50.39 | 48.59 |

- **Long combined pre-training is as good as fine-tuning.** The last combined pre-training checkpoints (48.44 for
  12+12, 48.70 for 18+18) are level with the best fine-tuned models (48.58 to 48.70). The fine-tunes started from step
  13,600 because the old encoding made later checkpoints look worse.
- **Replay does not help** either start on the mean of four (combined 48.58 without against 48.22 with; mined 48.70
  against 48.36).
- **The mined route is level with the combined route** after human fine-tuning (IN22-Gen en→xx 47.04 against 46.80).
  An earlier version of this page reported a large English-to-Indic gap between the two routes, with a per-language
  analysis. That gap came from the old encoding, and the analysis has been withdrawn.
- **Indic-to-English needs both data sources.** Every model trained on mined and human data scores 53.69 to 55.07 on
  IN22-Gen xx→en, against 43.80 for human-only and 44.20 for mined-only pre-training.
- **Depth** is discussed in [`training.md`](training.md), section 8.

Per-checkpoint scores (chrF++ and BLEU for every step) are written to `evaluation/results/` by the evaluation scripts
and are not tracked in git.

## 9. LLM-as-judge (Gemma 4 31B-it, en→xx)

chrF++ and BLEU compare surface text. We also use a language model as a judge, on all **22,528** English-to-Indic
sentences of IN22-Gen (22 directions × 1,024). The judge, Gemma 4 31B-it, scores **fluency**, **adequacy** (is the
meaning there?) and **overall** from 1 to 5, at temperature 0 with seed 42. It runs in two variants:

- **V2** (the headline) sees the human reference.
- **V1** is reference-free. It is a control against rewarding mere overlap with the reference.

"Perfect" means fluency = adequacy = overall = 5. Protocol details:
[`../evaluation/llm_judge/README.md`](../evaluation/llm_judge/README.md).

**Results.** Three systems were judged on the same sentences: the release model through CTranslate2 float32 (the
reported model), the same checkpoint through OpenNMT-py, and IndicTrans2 1B (our run, section 6). Means and
percentages are over every scored item; items the judge failed to score (at most 4 of 22,528 per system and variant)
are excluded. "Mean ≥ 4" is the share of items whose mean of the three scores is at least 4.

| System | Variant | Perfect 5/5/5 | Mean ≥ 4 | Fluency | Adequacy | Overall |
|---|---|---:|---:|---:|---:|---:|
| **IndicMT, CTranslate2 float32** | V2 (reference) | **63.8%** | 77.0% | 4.62 | 4.48 | 4.33 |
| IndicMT, OpenNMT-py | V2 (reference) | 61.7% | 74.7% | 4.57 | 4.41 | 4.27 |
| IndicTrans2 1B | V2 (reference) | 70.3% | 83.0% | 4.65 | 4.64 | 4.48 |
| **IndicMT, CTranslate2 float32** | V1 (no reference) | **66.9%** | 78.4% | 4.67 | 4.50 | 4.38 |
| IndicMT, OpenNMT-py | V1 (no reference) | 65.2% | 76.3% | 4.63 | 4.44 | 4.33 |
| IndicTrans2 1B | V1 (no reference) | 73.8% | 84.0% | 4.71 | 4.64 | 4.52 |

The judge prefers IndicTrans2 on English→Indic by 6.5 (V2) and 6.9 (V1) points of perfect translations, although
IndicMT is slightly ahead on chrF++ for the same sentences (section 6). The gap is mostly adequacy (4.48 against 4.64
under V2); fluency is close (4.62 against 4.65). The CTranslate2 float32 model is ahead of the OpenNMT-py decoding of
the same weights by 2.1 points (V2), consistent with its 0.2 chrF++ lead. The judge marks its own confidence as low for
most items in Bodo, Dogri, Konkani, Maithili, Manipuri and Santali (by protocol a low-confidence item is never scored
5), so the per-language scores for those languages should not be relied on; they lower all three systems alike. The judge results published before the source-encoding fix (section 2) are withdrawn.

**Sanity checks of the judge itself.** A deliberately mismatched translation gets adequacy 1.00 (pass). The human
reference itself, judged reference-free on IN22-Gen English→Indic (6,599 items), gets overall 4.53 and a perfect 5/5/5 on 73.0% of items. That is below
the expected 4.7, because the judge flags real errors in the IN22 references (for example a tense error in an
Assamese reference, and a Hindi reference that renders "dharmic action" as religious struggles). Santali (identity
overall 3.50) and Manipuri (3.98) are unreliable for this judge.

The judge's per-item outputs and summaries are written to `evaluation/llm_judge/results/` and are not tracked in git.
