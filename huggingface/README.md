---
license: apache-2.0
language:
- en
- as
- bn
- brx
- doi
- gom
- gu
- hi
- kn
- ks
- mai
- ml
- mr
- mni
- ne
- or
- pa
- sa
- sd
- sat
- ta
- te
- ur
library_name: ctranslate2
pipeline_tag: translation
tags:
- translation
- ctranslate2
- opennmt
- indic
datasets:
- ai4bharat/BPCC
metrics:
- chrf
model-index:
- name: IndicMT
  results:
  - task:
      type: translation
      name: English to Indic translation
    dataset:
      name: IN22-Gen
      type: ai4bharat/IN22-Gen
      split: test
    metrics:
    - type: chrf
      value: 46.95
      name: chrF++ en-xx (macro over 22 languages)
  - task:
      type: translation
      name: Indic to English translation
    dataset:
      name: IN22-Gen
      type: ai4bharat/IN22-Gen
      split: test
    metrics:
    - type: chrf
      value: 53.69
      name: chrF++ xx-en (macro over 22 languages)
  - task:
      type: translation
      name: English to Indic translation
    dataset:
      name: IN22-Conv
      type: ai4bharat/IN22-Conv
      split: test
    metrics:
    - type: chrf
      value: 42.97
      name: chrF++ en-xx (macro over 22 languages)
  - task:
      type: translation
      name: Indic to English translation
    dataset:
      name: IN22-Conv
      type: ai4bharat/IN22-Conv
      split: test
    metrics:
    - type: chrf
      value: 50.38
      name: chrF++ xx-en (macro over 22 languages)
---

# IndicMT

Compact multilingual machine translation between English and the 22 Scheduled Indian languages (44 directions), in
CTranslate2 format for CPU inference.

- **Model:** Transformer encoder-decoder, 12 encoder and 12 decoder layers, `d_model` 512, feed-forward 2048,
  16 heads, **153,807,872 parameters**, one shared 128k SentencePiece (unigram) vocabulary with tied embeddings.
- **Training:** from scratch with OpenNMT-py 3.5.1 on AI4Bharat's BPCC. Pre-trained on an 8:2 mix of mined and
  human-translated data (step 13,600), then fine-tuned on human data only (step 1,800).
- **Code, data pipeline and write-ups:** `<CODE_URL>`. **Technical report:** `<REPORT_URL>`.

## Files

| Path | Contents | Size |
|---|---|---|
| `int8/` | CTranslate2 model, int8 weights (recommended on CPU) | 170.6 MB |
| `float32/` | CTranslate2 model, float32 weights | 630.4 MB |
| `spm_128000.model` | SentencePiece model, shared by both variants | |
| `indicmt_config.json` | Variants, language tags and the source format, machine-readable | |

Each model directory holds `model.bin`, `config.json` and `shared_vocabulary.json`. The OpenNMT-py training checkpoint
is not included.

## Usage

```bash
pip install ctranslate2==4.8.2 sentencepiece==0.2.2 huggingface_hub
```

```python
import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

path = snapshot_download("ujjwalsharmaIITB/IndicMT", allow_patterns=["int8/*", "spm_128000.model"])
sp = spm.SentencePieceProcessor(model_file=f"{path}/spm_128000.model")
translator = ctranslate2.Translator(f"{path}/int8", device="cpu")


def translate(texts, tgt, beam_size=5):
    """texts: raw sentences in one source language; tgt: target tag without brackets, e.g. "hi" or "en"."""
    tag = f"<{tgt}>"
    # Encode exactly as in training: the tag and the raw sentence as ONE string.
    sources = sp.encode([f"{tag} {t}" for t in texts], out_type=str)
    assert all(s[:2] == ["▁", tag] for s in sources), "every source must start with ['▁', tag]"
    results = translator.translate_batch(sources, beam_size=beam_size)
    return [sp.decode(r.hypotheses[0]) for r in results]


print(translate(["Hello, how are you?"], "hi"))     # English -> Hindi
print(translate(["नमस्ते, आप कैसे हैं?"], "en"))      # Hindi -> English
print(translate(translate(["नमस्ते, आप कैसे हैं?"], "en"), "ta"))  # Hindi -> Tamil, pivoted through English
```

> **Encode the source exactly as in training.** Call `sp.encode(f"<xx> {text}")` on the whole string. That gives
> `['▁', '<xx>', '▁Hello', ...]`: a lone `▁` piece, then the tag. Every training source starts that way. Do **not**
> encode the sentence and then prepend the tag (`['<xx>'] + sp.encode(text)`). That drops the leading `▁`, a format the
> model never saw. Across our checkpoints it cost about 2 chrF++ on English-to-Indic on average, and it was the cause
> of a scoring error in an earlier version of our evaluation.

The tag names the **target** language. The output carries no tag; detokenize it with `sp.decode`. Translation between
two Indian languages goes through English (two calls), as in the last example.

## Language tags

| Language | Tag | Code | Script |
|---|---|---|---|
| English | `<en>` | `eng_Latn` | Latin |
| Assamese | `<as>` | `asm_Beng` | Bengali |
| Bengali | `<bn>` | `ben_Beng` | Bengali |
| Bodo | `<br>` | `brx_Deva` | Devanagari |
| Dogri | `<doi>` | `doi_Deva` | Devanagari |
| Gujarati | `<gu>` | `guj_Gujr` | Gujarati |
| Hindi | `<hi>` | `hin_Deva` | Devanagari |
| Kannada | `<kn>` | `kan_Knda` | Kannada |
| Kashmiri | `<ks>` | `kas_Arab` | Perso-Arabic |
| Konkani | `<gom>` | `gom_Deva` | Devanagari |
| Maithili | `<mai>` | `mai_Deva` | Devanagari |
| Malayalam | `<ml>` | `mal_Mlym` | Malayalam |
| Manipuri | `<mni>` | `mni_Mtei` | Meitei (Meetei Mayek) |
| Marathi | `<mr>` | `mar_Deva` | Devanagari |
| Nepali | `<ne>` | `npi_Deva` | Devanagari |
| Odia | `<or>` | `ory_Orya` | Odia |
| Punjabi | `<pa>` | `pan_Guru` | Gurmukhi |
| Sanskrit | `<sa>` | `san_Deva` | Devanagari |
| Santali | `<sat>` | `sat_Olck` | Ol Chiki |
| Sindhi | `<sd>` | `snd_Deva` | Devanagari |
| Tamil | `<ta>` | `tam_Taml` | Tamil |
| Telugu | `<te>` | `tel_Telu` | Telugu |
| Urdu | `<ur>` | `urd_Arab` | Perso-Arabic |

The Bodo tag is `<br>` (ISO 639-3 `brx`). It is not Breton.

## Results

Macro chrF++ over the 22 languages, beam 5, corpus-level sacreBLEU (`word_order=2`). The headline is the OpenNMT-py
model; the CTranslate2 variants in this repository score as shown on IN22-Gen.

| Test set | en→xx | xx→en |
|---|---:|---:|
| IN22-Gen | 46.95 | 53.69 |
| IN22-Conv | 42.97 | 50.38 |

| Variant (IN22-Gen) | en→xx | xx→en | Size |
|---|---:|---:|---:|
| CTranslate2 float32 | 47.15 | 53.87 | 630.4 MB |
| CTranslate2 int8 | 47.16 | 53.85 | 170.6 MB |

int8 is not significantly different from float32 (paired bootstrap, pooled p = 0.18 en→xx and 0.15 xx→en). Against
IndicTrans2 1B run with the same protocol, IndicMT is slightly but significantly ahead on en→xx when the 22 directions
are pooled (47.11 against 46.53, p = 0.001; 11 directions better and 10 worse at p < 0.05). IndicTrans2 is clearly
ahead on xx→en (62.48 against 53.69) and has 7.2 times more parameters.

<details>
<summary>Per-language chrF++ (OpenNMT-py, beam 5)</summary>

| Language | IN22-Gen en→xx | IN22-Gen xx→en | IN22-Conv en→xx | IN22-Conv xx→en |
|---|---:|---:|---:|---:|
| Assamese | 44.53 | 55.78 | 44.20 | 55.41 |
| Bengali | 47.18 | 54.23 | 44.68 | 53.41 |
| Bodo | 49.15 | 52.29 | 46.17 | 49.08 |
| Dogri | 55.55 | 60.83 | 50.46 | 57.88 |
| Gujarati | 50.77 | 57.84 | 50.09 | 56.40 |
| Hindi | 53.12 | 56.77 | 47.18 | 55.13 |
| Kannada | 48.96 | 55.17 | 31.70 | 43.00 |
| Kashmiri | 32.89 | 50.20 | 27.30 | 46.91 |
| Konkani | 43.40 | 49.60 | 41.24 | 45.23 |
| Maithili | 49.57 | 56.51 | 43.85 | 51.88 |
| Malayalam | 44.65 | 53.28 | 37.80 | 48.08 |
| Manipuri | 45.60 | 50.13 | 41.48 | 46.29 |
| Marathi | 48.47 | 54.77 | 45.74 | 52.68 |
| Nepali | 49.53 | 58.69 | 49.54 | 57.01 |
| Odia | 46.05 | 56.99 | 40.73 | 54.33 |
| Punjabi | 47.11 | 53.66 | 52.44 | 56.02 |
| Sanskrit | 36.82 | 45.59 | 34.23 | 41.76 |
| Santali | 40.65 | 43.43 | 42.67 | 44.90 |
| Sindhi | 39.47 | 50.26 | 37.03 | 45.81 |
| Tamil | 47.43 | 50.23 | 37.91 | 41.65 |
| Telugu | 49.42 | 53.87 | 42.64 | 47.24 |
| Urdu | 62.61 | 60.99 | 56.33 | 58.29 |

</details>

## Limitations

- **Sentence-level.** Trained on sentences of at most 200 SentencePiece pieces. Split paragraphs and documents into
  sentences first.
- **Indic to Indic pivots through English:** two passes, so errors compound and latency doubles.
- **One script per language** (table above). For example, Sindhi is Devanagari only, Kashmiri Perso-Arabic only,
  Santali Ol Chiki only and Manipuri Meetei Mayek only. Text in other scripts is out of domain. Characters outside the
  vocabulary become `<unk>` (no byte fallback).
- **Uneven quality.** The weakest directions are English to Kashmiri (32.89 IN22-Gen) and English to Sanskrit (36.82).
  Conversational English-to-Kannada (31.70 IN22-Conv) and English-to-Tamil (37.91) are much lower than on IN22-Gen.
  Indic-to-English trails IndicTrans2 1B by 8.79 chrF++ on average.
- **Evaluation caveats.** A single training run. The checkpoint was selected on the IN22 test sets, so IN22 scores are
  slightly optimistic. Scores are automatic metrics only; an LLM-judge evaluation is being re-run, and there is no
  human evaluation.
- **Data.** BPCC includes mined web data, and the model can reproduce its errors and biases. This is a research model:
  outputs may contain errors, so review them before any high-stakes use.

## Training data and evaluation

- **Training:** AI4Bharat BPCC, human-translated (`bpcc-seed-latest`, `ilci`) and mined (`comparable`,
  `samanantar-filtered`, `nllb-filtered`) configs, with script-consistency filtering and temperature-sampled mixing.
- **Evaluation:** IN22-Gen (1,024 sentences per direction) and IN22-Conv (1,503). They were never used for training,
  but they were used to select the checkpoint.

## Citation

```bibtex
@techreport{sharma2026indicmt,
  author = {Sharma, Ujjwal},
  title  = {{IndicMT}: Compact Multilingual Machine Translation for 22 Indian Languages},
  year   = {2026},
  note   = {Technical report}
}
```

## License

Apache License 2.0. Copyright 2026 Ujjwal Sharma. The training data (BPCC) and test sets (IN22) are AI4Bharat's and
have their own terms.
