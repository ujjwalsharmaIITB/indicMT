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
- pytorch
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
      value: 47.15
      name: chrF++ en-xx (macro over 22 languages, CTranslate2 float32)
  - task:
      type: translation
      name: Indic to English translation
    dataset:
      name: IN22-Gen
      type: ai4bharat/IN22-Gen
      split: test
    metrics:
    - type: chrf
      value: 53.87
      name: chrF++ xx-en (macro over 22 languages, CTranslate2 float32)
  - task:
      type: translation
      name: English to Indic translation
    dataset:
      name: IN22-Conv
      type: ai4bharat/IN22-Conv
      split: test
    metrics:
    - type: chrf
      value: 43.15
      name: chrF++ en-xx (macro over 22 languages, CTranslate2 float32)
  - task:
      type: translation
      name: Indic to English translation
    dataset:
      name: IN22-Conv
      type: ai4bharat/IN22-Conv
      split: test
    metrics:
    - type: chrf
      value: 50.59
      name: chrF++ xx-en (macro over 22 languages, CTranslate2 float32)
---

<div align="center">

<img src="assets/banner.svg" alt="IndicMT: English ↔ 22 Indian languages · 153.8M parameters" width="100%"/>

<h1 style="color:#7C3AED">IndicMT</h1>

<b>Compact multilingual machine translation for English and the 22 Scheduled Indian languages.</b>

<br/><br/>

<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:4px; margin:6px 0">
<a href="https://github.com/ujjwalsharmaIITB/indicMT"><img src="https://img.shields.io/badge/GitHub-indicMT-4F46E5?logo=github&logoColor=white" alt="GitHub: indicMT" style="display:inline-block; margin:2px"/></a>
<img src="https://img.shields.io/badge/license-Apache--2.0-4F46E5" alt="License: Apache-2.0" style="display:inline-block; margin:2px"/>
<img src="https://img.shields.io/badge/CTranslate2-4.8.2-7C3AED" alt="CTranslate2 4.8.2" style="display:inline-block; margin:2px"/>
<img src="https://img.shields.io/badge/OpenNMT--py-3.5.1-7C3AED" alt="OpenNMT-py 3.5.1" style="display:inline-block; margin:2px"/>
<a href="MEDIUM_BLOG_URL"><img src="https://img.shields.io/badge/Medium-Blog-7C3AED?logo=medium&logoColor=white" alt="Medium Blog" style="display:inline-block; margin:2px"/></a>
<a href="TECHNICAL_REPORT_URL"><img src="https://img.shields.io/badge/Technical-Report-EC4899?logo=readthedocs&logoColor=white" alt="Technical Report" style="display:inline-block; margin:2px"/></a>
</div>
<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:4px; margin:6px 0">
<img src="https://img.shields.io/badge/languages-23-A855F7" alt="Languages: 23" style="display:inline-block; margin:2px"/>
<img src="https://img.shields.io/badge/directions-44-A855F7" alt="Directions: 44" style="display:inline-block; margin:2px"/>
<img src="https://img.shields.io/badge/parameters-153.8M-EC4899" alt="Parameters: 153.8M" style="display:inline-block; margin:2px"/>
<img src="https://img.shields.io/badge/int8%20model-170.6%20MB-EC4899" alt="int8 model: 170.6 MB" style="display:inline-block; margin:2px"/>
</div>

<br/><br/>

<table>
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/IN22--Gen%20en%E2%86%92xx-47.15%20chrF%2B%2B-7C3AED?style=for-the-badge" alt="IN22-Gen en→xx 47.15 chrF++" style="display:inline-block; margin:2px"/></td>
    <td align="center"><img src="https://img.shields.io/badge/IN22--Gen%20xx%E2%86%92en-53.87%20chrF%2B%2B-7C3AED?style=for-the-badge" alt="IN22-Gen xx→en 53.87 chrF++" style="display:inline-block; margin:2px"/></td>
  </tr>
  <tr>
    <td align="center"><img src="https://img.shields.io/badge/IN22--Conv%20en%E2%86%92xx-43.15%20chrF%2B%2B-A855F7?style=for-the-badge" alt="IN22-Conv en→xx 43.15 chrF++" style="display:inline-block; margin:2px"/></td>
    <td align="center"><img src="https://img.shields.io/badge/IN22--Conv%20xx%E2%86%92en-50.59%20chrF%2B%2B-A855F7?style=for-the-badge" alt="IN22-Conv xx→en 50.59 chrF++" style="display:inline-block; margin:2px"/></td>
  </tr>
</table>

<a href="#usage">Usage</a> ·
<a href="#language-tags">Languages</a> ·
<a href="#results">Results</a> ·
<a href="#limitations">Limitations</a> ·
<a href="#citation">Citation</a>

</div>

---

Compact multilingual machine translation between English and the 22 Scheduled Indian languages (44 directions). This
repository holds the OpenNMT-py training checkpoint and an int8 CTranslate2 conversion of it for fast CPU inference.

- **Model:** Transformer encoder-decoder, 12 encoder and 12 decoder layers, `d_model` 512, feed-forward 2048,
  16 heads, **153,807,872 parameters**, one shared 128k SentencePiece (unigram) vocabulary with tied embeddings.
- **Training:** from scratch with OpenNMT-py 3.5.1 on AI4Bharat's BPCC. Pre-trained on an 8:2 mix of mined and
  human-translated data (step 13,600), then fine-tuned on human data only (step 1,800).
- **Code, data pipeline, evaluation and serving:** [github.com/ujjwalsharmaIITB/indicMT](https://github.com/ujjwalsharmaIITB/indicMT). **Technical report:** [Technical report](TECHNICAL_REPORT_URL). **Blog:** [Medium blog](MEDIUM_BLOG_URL).

<h2 id="files" style="color:#7C3AED">Files</h2>

| Path | Contents | Size |
|---|---|---|
| `indicMT/` | CTranslate2 model, int8 weights (recommended for CPU inference) | 170.6 MB |
| `indicMT.pt` | OpenNMT-py 3.5.1 checkpoint, float32, as saved at the end of training | 2.4 GB |
| `indicMT.spm.model` | SentencePiece model (128k unigram), shared by both | 3.5 MB |
| `indicmt_config.json` | Variants, language tags and the source format, machine-readable | |

`indicMT/` holds `model.bin`, `config.json` and `shared_vocabulary.json`. The checkpoint is the unmodified training file:
model weights, vocabulary, training options and the Adam optimizer state, so fine-tuning can continue from it.

<h2 id="usage" style="color:#7C3AED">Usage</h2>

<h3 id="ctranslate2-int8-cpu" style="color:#A855F7">CTranslate2 (int8, CPU)</h3>

```bash
pip install ctranslate2==4.8.2 sentencepiece==0.2.2 huggingface_hub
```

```python
import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

path = snapshot_download("ujjwalsharmaIITB/indicMT", allow_patterns=["indicMT/*", "indicMT.spm.model"])
sp = spm.SentencePieceProcessor(model_file=f"{path}/indicMT.spm.model")
translator = ctranslate2.Translator(f"{path}/indicMT", device="cpu")


def translate(texts, tgt, beam_size=5):
    """texts: raw sentences in one source language; tgt: target tag without brackets, e.g. "hi" or "en"."""
    tag = f"<{tgt}>"
    sources = sp.encode([f"{tag} {t}" for t in texts], out_type=str)  # tag and sentence as one string
    results = translator.translate_batch(sources, beam_size=beam_size)
    return [sp.decode(r.hypotheses[0]) for r in results]


print(translate(["Hello, how are you?"], "hi"))     # English -> Hindi
print(translate(["नमस्ते, आप कैसे हैं?"], "en"))      # Hindi -> English
print(translate(translate(["नमस्ते, आप कैसे हैं?"], "en"), "ta"))  # Hindi -> Tamil, pivoted through English
```

> **Encoding the input.** Use `sp.encode(f"<xx> {text}")` on the whole string. Do **not** use
> `['<xx>'] + sp.encode(text)`.

The tag names the **target** language. The output carries no tag; detokenize it with `sp.decode`. Translation between
two Indian languages goes through English (two calls), as in the last example.

<h3 id="opennmt-py-checkpoint-gpu-or-cpu" style="color:#A855F7">OpenNMT-py checkpoint (GPU or CPU)</h3>

```bash
pip install torch==2.2.0 OpenNMT-py==3.5.1 sentencepiece==0.2.2 huggingface_hub
```

```python
import subprocess
import sentencepiece as spm
from huggingface_hub import snapshot_download

path = snapshot_download("ujjwalsharmaIITB/indicMT", allow_patterns=["indicMT.pt", "indicMT.spm.model"])
sp = spm.SentencePieceProcessor(model_file=f"{path}/indicMT.spm.model")

texts, tag = ["Hello, how are you?"], "<hi>"
with open("src.spm", "w", encoding="utf-8") as f:
    for pieces in sp.encode([f"{tag} {t}" for t in texts], out_type=str):  # same source format as above
        f.write(" ".join(pieces) + "\n")

subprocess.run(["onmt_translate", "-model", f"{path}/indicMT.pt", "-src", "src.spm",
                "-output", "hyp.spm", "-beam_size", "5", "-gpu", "0"], check=True)  # drop "-gpu 0" on CPU
with open("hyp.spm", encoding="utf-8") as f:
    print([sp.decode(line.split()) for line in f])
```

The checkpoint stores its training options as a Python object, so it loads with `torch.load(..., weights_only=False)`.
That is the default in torch 2.2.0, which OpenNMT-py 3.5.1 pins. From torch 2.6 the default changed and loading fails.
Load only files you trust this way.

<h3 id="batch-inference-and-serving" style="color:#A855F7">Batch inference and serving</h3>

The [GitHub repository](https://github.com/ujjwalsharmaIITB/indicMT) has the code for larger jobs:

- **Batch translation and scoring of test sets** with the int8 model: [`deployment/ct2_eval.py`](https://github.com/ujjwalsharmaIITB/indicMT/blob/main/deployment/ct2_eval.py), built on the shared inference module [`deployment/ct2_infer.py`](https://github.com/ujjwalsharmaIITB/indicMT/blob/main/deployment/ct2_infer.py).
- **Batch translation with the OpenNMT-py checkpoint:** [`evaluation/run_eval.py`](https://github.com/ujjwalsharmaIITB/indicMT/blob/main/evaluation/run_eval.py).
- **REST API and web interface** (FastAPI, CPU, Docker): [`deployment/api/main.py`](https://github.com/ujjwalsharmaIITB/indicMT/blob/main/deployment/api/main.py); setup in [`deployment/README.md`](https://github.com/ujjwalsharmaIITB/indicMT/blob/main/deployment/README.md).

<h2 id="language-tags" style="color:#7C3AED">Language tags</h2>

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

<h2 id="results" style="color:#7C3AED">Results</h2>

<img src="assets/in22-gen-comparision-chart.png" alt="IN22-Gen English to Indic chrF++: IndicMT 47.15 against other systems" width="100%"/>

Macro chrF++ over the 22 languages, beam 5, corpus-level sacreBLEU (`word_order=2`). **All reported results are from
the CTranslate2 float32 model**, the checkpoint in this repository converted to CTranslate2 without quantization.

| Test set | en→xx | xx→en |
|---|---:|---:|
| IN22-Gen | 47.15 | 53.87 |
| IN22-Conv | 43.15 | 50.59 |

Against IndicTrans2 1B run with the same protocol (paired bootstrap, 22 directions pooled), IndicMT is slightly but
significantly ahead on en→xx on IN22-Gen (47.31 against 46.53, p = 0.001; 11 directions better and 8 worse at
p < 0.05) and on IN22-Conv (43.23 against 42.60, p = 0.001; 8 better, 13 worse). IndicTrans2 is clearly ahead on xx→en (62.48 against 53.87 on IN22-Gen, 55.63
against 50.59 on IN22-Conv) and has 7.2 times more parameters.

<h3 id="quantization-and-the-files-in-this-repository" style="color:#A855F7">Quantization and the files in this repository</h3>

This repository ships the OpenNMT-py checkpoint (`indicMT.pt`) and the int8 CTranslate2 model (`indicMT/`); the reported
results are from the CTranslate2 float32 conversion of the same checkpoint. chrF++, beam 5; latency is the median for one sentence on CPU (8 threads), IN22-Gen.

| Model | IN22-Gen en→xx | IN22-Gen xx→en | IN22-Conv en→xx | IN22-Conv xx→en | p50 latency | Size |
|---|---:|---:|---:|---:|---:|---:|
| OpenNMT-py checkpoint (`indicMT.pt`) | 46.95 | 53.69 | 42.97 | 50.38 | | 2.4 GB |
| **CTranslate2 float32 (reported)** | **47.15** | **53.87** | **43.15** | **50.59** | 399.2 ms | 630.4 MB |
| CTranslate2 int16 | 47.15 | 53.86 | | | 367.1 ms | 323.2 MB |
| CTranslate2 int8 (`indicMT/`) | 47.16 | 53.85 | | | 374.2 ms | 170.6 MB |

int16 and int8 are **not significantly different from float32** on IN22-Gen, and int8 is 3.7 times smaller.

<h3 id="compared-with-other-systems" style="color:#A855F7">Compared with other systems</h3>

**IndicMT (CTranslate2 float32), IndicTrans2, Indic Translate and Sarvam Translate are from our own runs** (corpus-level chrF++; beam 5 for
IndicMT and IndicTrans2, greedy decoding for the two LLMs). Our runs of Indic Translate and Sarvam Translate are in
progress; until they finish, their rows show the scores from the Indic Translate page. **All other baselines are taken
from the
[`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate) model page** (read on 2026-09-19),
which reports sentence-level chrF++ with greedy decoding on IN22-Gen only, so those rows are indicative. A star marks a
system that covers only 16 of the 22 languages.

| Model | Parameters | IN22-Gen en→xx | IN22-Gen xx→en | IN22-Conv en→xx | IN22-Conv xx→en | Source |
|---|---:|---:|---:|---:|---:|---|
| **IndicMT** | **0.154B** | **47.15** | **53.87** | **43.15** | **50.59** | our run |
| IndicTrans2 1B | 1.1B | 46.39 | 62.48 | 42.61 | 55.63 | our run |
| Indic Translate | 7.94B | 47.70 | 60.40 | | | Indic Translate page (our run in progress) |
| Sarvam Translate | 4B | 48.90 | 52.40 | | | Indic Translate page (our run in progress) |
| gemini-3.5-flash | undisclosed | 40.30 | 66.60 | | | Indic Translate page |
| gemma-4-31B-it | 30.7B | 39.20 | 59.60 | | | Indic Translate page |
| translategemma-27b-it * | 27B | 39.60 | 58.50 | | | Indic Translate page |
| gemma-4-26B-a4b-it | 25.2B | 36.20 | 56.60 | | | Indic Translate page |
| translategemma-12b-it * | 12B | 37.10 | 55.00 | | | Indic Translate page |
| gemma-3-27b-it | 27.43B | 31.90 | 55.50 | | | Indic Translate page |
| gemma-3-12b-it | 12.19B | 28.80 | 53.20 | | | Indic Translate page |
| translategemma-4b-it * | 4B | 31.20 | 49.20 | | | Indic Translate page |
| gemma-3-4b-it | 4.3B | 22.90 | 47.10 | | | Indic Translate page |

<details>
<summary>Per-language chrF++ (CTranslate2 float32, beam 5)</summary>

| Language | IN22-Gen en→xx | IN22-Gen xx→en | IN22-Conv en→xx | IN22-Conv xx→en |
|---|---:|---:|---:|---:|
| Assamese | 44.74 | 56.02 | 44.58 | 55.39 |
| Bengali | 47.24 | 54.39 | 45.01 | 53.75 |
| Bodo | 49.30 | 52.34 | 46.39 | 49.18 |
| Dogri | 55.86 | 61.04 | 50.64 | 58.29 |
| Gujarati | 50.88 | 57.92 | 50.21 | 56.42 |
| Hindi | 53.19 | 56.99 | 47.14 | 55.34 |
| Kannada | 49.06 | 55.31 | 31.70 | 43.40 |
| Kashmiri | 32.99 | 50.47 | 27.55 | 47.18 |
| Konkani | 43.51 | 49.67 | 41.41 | 45.31 |
| Maithili | 49.69 | 56.72 | 44.08 | 51.92 |
| Malayalam | 44.83 | 53.35 | 38.19 | 48.49 |
| Manipuri | 46.26 | 50.29 | 41.64 | 46.54 |
| Marathi | 48.63 | 54.94 | 46.08 | 52.94 |
| Nepali | 49.68 | 59.00 | 49.60 | 57.08 |
| Odia | 46.23 | 57.14 | 40.87 | 54.57 |
| Punjabi | 47.12 | 53.73 | 52.55 | 56.25 |
| Sanskrit | 37.26 | 45.86 | 34.35 | 41.76 |
| Santali | 41.11 | 43.72 | 42.99 | 45.11 |
| Sindhi | 39.73 | 50.54 | 37.11 | 46.16 |
| Tamil | 47.63 | 50.60 | 38.19 | 41.90 |
| Telugu | 49.58 | 54.03 | 42.56 | 47.61 |
| Urdu | 62.73 | 61.04 | 56.44 | 58.34 |

</details>

<h2 id="limitations" style="color:#7C3AED">Limitations</h2>

- **Sentence-level.** Trained on sentences of at most 200 SentencePiece pieces. Split paragraphs and documents into
  sentences first.
- **Indic to Indic pivots through English:** two passes, so errors compound and latency doubles.
- **One script per language** (table above). For example, Sindhi is Devanagari only, Kashmiri Perso-Arabic only,
  Santali Ol Chiki only and Manipuri Meetei Mayek only. Text in other scripts is out of domain. Characters outside the
  vocabulary become `<unk>` (no byte fallback).
- **Uneven quality.** The weakest directions are English to Kashmiri (32.99 IN22-Gen) and English to Sanskrit (37.26).
  Conversational English-to-Kannada (31.70 IN22-Conv) and English-to-Tamil (38.19) are much lower than on IN22-Gen.
  Indic-to-English trails IndicTrans2 1B by 8.61 chrF++ on average (IN22-Gen).
- **Automatic metrics only.** Scores are chrF++ and BLEU; there is no human evaluation.
- **Data.** BPCC includes mined web data, and the model can reproduce its errors and biases. This is a research model:
  outputs may contain errors, so review them before any high-stakes use.

<h2 id="training-data-and-evaluation" style="color:#7C3AED">Training data and evaluation</h2>

- **Training:** AI4Bharat BPCC, human-translated (`bpcc-seed-latest`, `ilci`) and mined (`comparable`,
  `samanantar-filtered`, `nllb-filtered`) configs, with script-consistency filtering and temperature-sampled mixing.
- **Evaluation:** IN22-Gen (1,024 sentences per direction) and IN22-Conv (1,503).

<h2 id="citation" style="color:#7C3AED">Citation</h2>

```bibtex
@techreport{sharma2026indicmt,
  author = {Sharma, Ujjwal},
  title  = {{IndicMT}: Compact Multilingual Machine Translation for 22 Indian Languages},
  year   = {2026},
  note   = {Technical report}
}
```

<h2 id="license" style="color:#7C3AED">License</h2>

Apache License 2.0. Copyright 2026 Ujjwal Sharma. The training data (BPCC) and test sets (IN22) are AI4Bharat's and
have their own terms.
