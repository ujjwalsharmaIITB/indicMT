/* Numbers shown in the IndicMT trailer, from the project's evaluation results. */
window.TRAILER_DATA = {
 "ours": {
  "model": "IndicMT",
  "params": 153807872
 },
 "ours_quality": {
  "checkpoint": "final model (combined pre-training step 13,600 + human fine-tune without replay, step 1,800; selected on IN22), OpenNMT-py checkpoint (.pt)",
  "runtime": "onmt_translate (PyTorch .pt checkpoint), beam 5, GPU",
  "input_format": "sources encoded as in training: SentencePiece over \"<xx> text\" (data/common/tokenization.py), from 2026-10-08",
  "en_xx_chrf": 46.95,
  "xx_en_chrf": 53.69,
  "en_xx_chrf_ct2_float32": 47.15,
  "xx_en_chrf_ct2_float32": 53.87,
  "previous_format_en_xx_xx_en": [
   45.36,
   53.73
  ]
 },
 "languages": {
  "indic": 22,
  "total_incl_english": 23,
  "directions": 44,
  "tags": [
   "as",
   "bn",
   "br",
   "doi",
   "gom",
   "gu",
   "hi",
   "kn",
   "ks",
   "mai",
   "ml",
   "mr",
   "mni",
   "ne",
   "or",
   "pa",
   "sa",
   "sd",
   "sat",
   "ta",
   "te",
   "ur"
  ],
  "autonyms": {
   "as": "অসমীয়া",
   "bn": "বাংলা",
   "br": "बड़ो",
   "doi": "डोगरी",
   "gom": "कोंकणी",
   "gu": "ગુજરાતી",
   "hi": "हिन्दी",
   "kn": "ಕನ್ನಡ",
   "ks": "کٲشُر",
   "mai": "मैथिली",
   "ml": "മലയാളം",
   "mr": "मराठी",
   "mni": "ꯃꯩꯇꯩꯂꯣꯟ",
   "ne": "नेपाली",
   "or": "ଓଡ଼ିଆ",
   "pa": "ਪੰਜਾਬੀ",
   "sa": "संस्कृतम्",
   "sd": "सिन्धी",
   "sat": "ᱥᱟᱱᱛᱟᱲᱤ",
   "ta": "தமிழ்",
   "te": "తెలుగు",
   "ur": "اردو"
  }
 },
 "sizes_b": [
  {
   "model": "IndicMT (ours)",
   "params_b": 0.153807872,
   "kind": "measured"
  },
  {
   "model": "IndicTrans2",
   "params_b": 1.1,
   "kind": "variant"
  },
  {
   "model": "Sarvam Translate",
   "params_b": 4.0,
   "kind": "rounded"
  },
  {
   "model": "translategemma-4b-it",
   "params_b": 4.0,
   "kind": "nominal"
  },
  {
   "model": "gemma-3-4b-it",
   "params_b": 4.3,
   "kind": "primary"
  },
  {
   "model": "Indic Translate",
   "params_b": 7.94,
   "kind": "card"
  },
  {
   "model": "translategemma-12b-it",
   "params_b": 12.0,
   "kind": "nominal"
  },
  {
   "model": "gemma-3-12b-it",
   "params_b": 12.19,
   "kind": "primary"
  },
  {
   "model": "gemma-4-26B-a4b-it",
   "params_b": 25.2,
   "kind": "primary"
  },
  {
   "model": "translategemma-27b-it",
   "params_b": 27.0,
   "kind": "nominal"
  },
  {
   "model": "gemma-3-27b-it",
   "params_b": 27.43,
   "kind": "primary"
  },
  {
   "model": "gemma-4-31B-it",
   "params_b": 30.7,
   "kind": "primary"
  }
 ],
 "size_ratios_vs_ours": {
  "IndicTrans2": 7.2,
  "Sarvam Translate": 26.0,
  "translategemma-4b-it": 26.0,
  "gemma-3-4b-it": 28.0,
  "Indic Translate": 51.6,
  "translategemma-12b-it": 78.0,
  "gemma-3-12b-it": 79.3,
  "gemma-4-26B-a4b-it": 163.8,
  "translategemma-27b-it": 175.5,
  "gemma-3-27b-it": 178.3,
  "gemma-4-31B-it": 199.6
 },
 "fertility": [
  {
   "tokenizer": "IndicTrans2",
   "macro_fertility": 3.8973
  },
  {
   "tokenizer": "Sarvam Translate (Gemma 3)",
   "macro_fertility": 3.2678
  },
  {
   "tokenizer": "Indic Translate (Gemma 4)",
   "macro_fertility": 3.2678
  },
  {
   "tokenizer": "Sarvam-30B",
   "macro_fertility": 2.3754
  },
  {
   "tokenizer": "In-house 128k (ours)",
   "macro_fertility": 1.8148
  }
 ],
 "quality_en_xx_in22gen": [
  {
   "model": "gemma-3-4b-it",
   "en_xx_chrf": 22.9,
   "partial_16_of_22": false,
   "xx_en_chrf": 47.1
  },
  {
   "model": "gemma-3-12b-it",
   "en_xx_chrf": 28.8,
   "partial_16_of_22": false,
   "xx_en_chrf": 53.2
  },
  {
   "model": "translategemma-4b-it",
   "en_xx_chrf": 31.2,
   "partial_16_of_22": true,
   "xx_en_chrf": 49.2
  },
  {
   "model": "gemma-3-27b-it",
   "en_xx_chrf": 31.9,
   "partial_16_of_22": false,
   "xx_en_chrf": 55.5
  },
  {
   "model": "gemma-4-26B-a4b-it",
   "en_xx_chrf": 36.2,
   "partial_16_of_22": false,
   "xx_en_chrf": 56.6
  },
  {
   "model": "translategemma-12b-it",
   "en_xx_chrf": 37.1,
   "partial_16_of_22": true,
   "xx_en_chrf": 55.0
  },
  {
   "model": "gemma-4-31B-it",
   "en_xx_chrf": 39.2,
   "partial_16_of_22": false,
   "xx_en_chrf": 59.6
  },
  {
   "model": "translategemma-27b-it",
   "en_xx_chrf": 39.6,
   "partial_16_of_22": true,
   "xx_en_chrf": 58.5
  },
  {
   "model": "gemini-3.5-flash",
   "en_xx_chrf": 40.3,
   "partial_16_of_22": false,
   "xx_en_chrf": 66.6
  },
  {
   "model": "IndicTrans2",
   "en_xx_chrf": 46.39,
   "partial_16_of_22": false,
   "xx_en_chrf": 62.48,
   "rescored_by_us": true
  },
  {
   "model": "Indic Translate",
   "en_xx_chrf": 47.73,
   "partial_16_of_22": false,
   "xx_en_chrf": 60.33,
   "rescored_by_us": true
  },
  {
   "model": "Sarvam Translate",
   "en_xx_chrf": 48.74,
   "partial_16_of_22": false,
   "xx_en_chrf": 51.82,
   "rescored_by_us": true
  }
 ],
 "quantization_ct2": [
  {
   "variant": "float32",
   "en_xx_chrf": 47.15,
   "xx_en_chrf": 53.87,
   "size_mb": 630.4,
   "p50_ms": 399.2
  },
  {
   "variant": "int16",
   "en_xx_chrf": 47.15,
   "xx_en_chrf": 53.86,
   "size_mb": 323.2,
   "p50_ms": 367.1
  },
  {
   "variant": "int8",
   "en_xx_chrf": 47.16,
   "xx_en_chrf": 53.85,
   "size_mb": 170.6,
   "p50_ms": 374.2
  }
 ],
 "deployment": {
  "checkpoint": "final model (combined → human, step 1,800)",
  "runtime": "CTranslate2 on CPU, inter_threads 1, intra_threads 8, beam 5, IN22-Gen (scripts/deployment_convert_ct2_job.sh)",
  "intra_threads": 8,
  "int8_size_mb": 170.6,
  "int8_p50_ms": 374.2,
  "int8_p95_ms": 687.8,
  "int8_batch32_sent_per_s": 8.4,
  "float32_batch32_sent_per_s": 5.9,
  "serving": "deployment/api (FastAPI) + deployment/frontend, verified running bare-metal; container image NOT built (podman blocked)",
  "pivot": "Indic->Indic routed through English (xx->en then en->xx) by deployment/api/translator.py"
 },
 "significance_test": {
  "method": "sacrebleu paired bootstrap, 1000 resamples, chrF++, float32 baseline, 22 directions pooled per bucket",
  "en_xx": {
   "int16_p": 0.199,
   "int8_p": 0.181,
   "pooled_float32": 47.31,
   "pooled_int16": 47.31,
   "pooled_int8": 47.32,
   "macro_delta_int16": 0.005,
   "macro_delta_int8": 0.009,
   "directions_significant_bonferroni": "0/22 for both"
  },
  "xx_en": {
   "int16_p": 0.212,
   "int8_p": 0.149,
   "pooled_float32": 53.88,
   "pooled_int8": 53.86,
   "macro_delta_int8": -0.017,
   "directions_significant_bonferroni": "0/22 for both"
  }
 },
 "llm_judge": {
  "judge": "Gemma 4 31B-it",
  "checkpoint": "combined → human, no replay, step 1,800",
  "runtime": "CTranslate2 float32 on CPU",
  "test_sets": "IN22-Gen",
  "bucket": "en->xx",
  "n_items": 22528,
  "v2_with_reference": {
   "perfect_pct": 63.8,
   "mean_ge4_pct": 77.0,
   "fluency": 4.62,
   "adequacy": 4.48,
   "overall": 4.33,
   "failed": 0
  },
  "v1_reference_free": {
   "perfect_pct": 66.87,
   "mean_ge4_pct": 78.37,
   "fluency": 4.67,
   "adequacy": 4.5,
   "overall": 4.38,
   "failed": 2
  },
  "human_reference_perfect_pct": 73
 },
 "links": {
  "blog": "https://medium.com/@ujjwal.sharma.cs/multilingual-neural-machine-translation-from-theory-to-implementation-and-deployment-a1e14119e369",
  "report": "http://cse.iitb.ac.in/~ujjwalsharma/blogs/indicMT",
  "code": "https://github.com/ujjwalsharmaIITB/indicMT",
  "weights": "https://huggingface.co/ujjwalsharmaIITB/indicMT",
  "display": {
   "blog": "medium.com/@ujjwal.sharma.cs",
   "report": "cse.iitb.ac.in/~ujjwalsharma/blogs/indicMT",
   "code": "github.com/ujjwalsharmaIITB/indicMT",
   "weights": "huggingface.co/ujjwalsharmaIITB/indicMT"
  }
 },
 "vs_indictrans2": {
  "method": "sacrebleu paired bootstrap, 1000 resamples, chrF++, IN22-Gen, baseline IndicTrans2 1B (re-run by us), 22 directions pooled per bucket",
  "en_xx_macro": [
   46.39,
   47.15
  ],
  "en_xx_pooled": [
   46.53,
   47.31
  ],
  "en_xx_p": 0.001,
  "en_xx_significant": true,
  "en_xx_directions_indicmt_better": 11,
  "en_xx_directions_indictrans2_better": 8,
  "xx_en_macro": [
   62.48,
   53.87
  ],
  "xx_en_p": 0.001
 }
};
