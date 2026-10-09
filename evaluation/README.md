# evaluation: chrF++ and BLEU on IN22

This directory scores a trained checkpoint on **IN22-Gen** and **IN22-Conv** in all 44 directions, compares it with
published baselines, and tests whether quantization changes translations. The full results and tables are in
[`../docs/evaluation.md`](../docs/evaluation.md). The LLM judge is in [`llm_judge/`](llm_judge/README.md).

## What is measured

- **chrF++:** a character n-gram F-score that also uses word n-grams up to order 2 (`sacrebleu` `corpus_chrf` with
  `word_order=2`). It is the main metric.
- **BLEU:** word n-gram precision, via `sacrebleu`.
- **Test sets:** IN22-Gen (1,024 sentences per direction) and IN22-Conv (1,503). They never touch training.
- **Directions:** 22 languages, each to and from English (44 per test set), beam size 5.
- **Corpus-level scores,** one score per direction over the whole set. The **macro** average is the unweighted mean of the
  22 languages in a direction bucket (en→xx or xx→en).

## Files

| File | Purpose |
|---|---|
| `build_eval_pairs.py` | Builds the 44 direction-specific source files per test set: each raw test sentence is tagged and SentencePiece-encoded exactly as the training sources were (`data/common/tokenization.py`). Run once, and again if your files were built before this encoding fix. |
| `run_eval.py` | Evaluates one OpenNMT-py checkpoint: `onmt_translate`, detokenize with the 128k SentencePiece model, score against the untokenized references, write results. |
| `baselines.py` | Published baseline scores (IN22-Gen, chrF++) used for comparison. |
| `quant_significance.py` | Paired bootstrap test of int16 and int8 against float32 (see [`../docs/deployment.md`](../docs/deployment.md)). |
| `paired_significance.py` | Paired bootstrap test between any two systems' IN22 translations, per direction and pooled per bucket (e.g. IndicMT against IndicTrans2). |
| `run_eval_indictrans2.py` | Evaluates AI4Bharat's IndicTrans2 1B models with the same scoring as `run_eval.py`, so the comparison in the docs can be reproduced (see "Comparing two systems"). Needs its own environment (Job24). |
| `package_translations.py` | Packages a checkpoint's IN22 translations into one `.tar.gz` (see "Translations"). |
| `llm_judge/` | LLM-as-judge scoring. |

## Run

```bash
# once: build the tagged source files (needs the 128k SentencePiece model and the raw IN22 files from Job8)
python evaluation/build_eval_pairs.py

# one checkpoint (Job14: 1 GPU)
sbatch -p <partition> scripts/slurm/evaluation_run.sh \
  128k-spm/opennmt/bpcc-ft-combined-human/checkpoints/model_step_1800.pt bpcc-ft-combined-human-step1800

# many checkpoints on one 2-GPU allocation (Job15)
sbatch -p <partition> scripts/slurm/evaluation_batch_run.sh <ckpt1.pt> <run-name1> <ckpt2.pt> <run-name2> ...
```

Job15 packs three evaluations per GPU. It sets `EVAL_BATCH_SIZE` (default 2,048 tokens) and retries a failed checkpoint
at 1,024. Lessons from running it at scale:

- Evaluation is **CPU-bound**: with the default 2 CPUs for 6 processes, the job used about 1.9 CPUs. The script requests
  24 CPUs and pins threads to 4 per process.
- Peak GPU memory is set by sentence count times beam size, so it spikes on short-sentence directions. A batch of 4,096
  tokens with 6 processes per GPU ran out of memory.

## Outputs

Written to `evaluation/results/<arm>/<checkpoint>/` (the arm is the checkpoint's run directory name). Not tracked by git.

| File | Contents |
|---|---|
| `<test-set>.<src>-<tgt>.hyp.spm` | raw SentencePiece-level translations |
| `<test-set>.<src>-<tgt>.hyp.txt` | detokenized translations, one per line, aligned with the reference file |
| `scores.json` | chrF++ and BLEU for every direction, plus macro averages |
| `report.md` | a readable table with baseline comparison |

For a checkpoint stored outside `.../checkpoints/`, pass `--output-dir`, or the results land in the wrong place.

## Results in short (release model)

12+12, combined pre-training (step 13,600) then human fine-tuning without replay (step 1,800). Macro chrF++ / BLEU:

| Test set | en→xx chrF++ | en→xx BLEU | xx→en chrF++ | xx→en BLEU |
|---|---|---|---|---|
| IN22-Gen | 46.95 | 19.11 | 53.69 | 29.00 |
| IN22-Conv | 42.97 | 15.80 | 50.38 | 28.29 |

On IN22-Gen it ranks **3rd of 13** systems on en→xx and 5th of 13 on the two-direction mean (50.32), with 0.154B
parameters. Most baselines are published sentence-level, greedy-decoding scores, while ours (and IndicTrans2's, which we
re-ran) are corpus-level with beam 5, so the comparison is indicative. Under the same protocol, IndicMT is slightly but
significantly ahead of IndicTrans2 1B on en→xx when the 22 directions are pooled (47.11 against 46.53, p = 0.001), and
IndicTrans2 is clearly ahead on xx→en (62.48 against 53.69). The checkpoint was chosen on these test sets, so the
scores are slightly optimistic. Single seed. Full ranking and per-language tables:
[`../docs/evaluation.md`](../docs/evaluation.md).

## Baselines

`baselines.py` holds the chrF++ scores from the `bodhan-ai/indic-translate` model card (read on 2026-09-19): IN22-Gen only,
12 systems, plus a per-language table for three of them. They are taken as published and not re-run. No IN22-Conv baseline
exists. Systems marked partial (translategemma) cover only 16 of the 22 languages.

The docs' ranking makes one exception: we re-ran IndicTrans2's 1B models with our protocol (beam 5, corpus-level
chrF++) and report those scores, 46.39 / 62.48, instead of the card's 48.6 / 62.8. `baselines.py` itself keeps the
card values.

## Quantization significance

`quant_significance.py` uses sacreBLEU's paired bootstrap on chrF++ with float32 as the baseline, per direction and pooled
per bucket, with a Bonferroni correction. Job21 runs it after re-translating IN22-Gen with each quantized variant. On the
release model, with training-format inputs, neither int16 nor int8 is significantly different from float32 (pooled p = 0.18 for int8 en→xx and 0.15 for xx→en). See
[`../docs/deployment.md`](../docs/deployment.md).

## Comparing two systems

`paired_significance.py` runs the same paired bootstrap between any two systems whose detokenized translations follow
`run_eval.py`'s naming (`<test-set>.<src>-<tgt>.hyp.txt`):

```bash
python evaluation/paired_significance.py \
  --baseline other=evaluation/results/<other-system> \
  --system indicmt=evaluation/results/bpcc-ft-combined-human/model_step_1800 \
  --test-set in22-gen --out evaluation/results/significance/indicmt-vs-other
```

It writes `significance.json` and `significance.md`: pooled scores and p values, the number of directions where each
system is significantly better, and the count after Bonferroni correction.

### Reproducing the IndicTrans2 scores

The docs report IndicTrans2 1B under our protocol (beam 5, corpus-level chrF++), not the model-card values.
`run_eval_indictrans2.py` produces those scores from the raw IN22 text. It runs `ai4bharat/indictrans2-en-indic-1B`
for en→xx and `ai4bharat/indictrans2-indic-en-1B` for xx→en with IndicTransToolkit pre- and post-processing, then scores
the outputs with `run_eval.py`'s own functions. It writes to `evaluation/results/indictrans2-1B/` in the same layout, so
the output can be the `--baseline` above.

It needs a **separate environment** from OpenNMT-py: `transformers==4.46.3` and `IndicTransToolkit==1.1.1` on top of
torch, sacrebleu and sentencepiece, plus a GPU. The two models are gated, so accept their terms on Hugging Face first.

```bash
python evaluation/run_eval_indictrans2.py --bucket "en->xx" --gpu 0
python evaluation/run_eval_indictrans2.py --bucket "xx->en" --gpu 1
python evaluation/run_eval_indictrans2.py --merge        # scores.json and report.md
```

`scripts/slurm/evaluation_indictrans2.sh` (Job24) runs both buckets on two GPUs and merges them.

## Translations

The model's IN22 translations are published as a **release asset** (a file attached to a GitHub release), not stored in
git. The archive `<TRANSLATIONS_URL>` contains:

- 88 detokenized translation files, `<test-set>.<src>-<tgt>.hyp.txt`: 44 directions times IN22-Gen and IN22-Conv. Each
  has one translation per line, aligned with the IN22 source and reference files.
- `scores.json`: corpus-level chrF++ and BLEU for every direction.
- `README.txt`: the checkpoint name, the file naming scheme and the scoring method (corpus-level chrF++ and BLEU with
  sacreBLEU, beam 5).

The SentencePiece-level `*.hyp.spm` files are left out. To build the archive from your own evaluation results:

```bash
python evaluation/package_translations.py \
  --results-dir evaluation/results/bpcc-ft-combined-human/model_step_1800 \
  --out indicmt-in22-translations.tar.gz
```

The script uses only the standard library. It stops with a clear message if any of the 88 files or `scores.json` is missing.
The IN22 source and reference files are not included. Get them from AI4Bharat after an access request.
