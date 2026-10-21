"""LLM-as-judge pipeline for IndicMT translation quality (spec: PROMPT.md, next to this file).

Scores every hypothesis of one checkpoint (arm + step) on fluency, adequacy and overall (1-5) with a
judge LLM. chrF++/BLEU (evaluation/run_eval.py) stay the headline automatic metrics; the judge adds
what they cannot see (adequacy vs fluency separately, wrong language/script, hallucination, negation).

Inputs (all line-aligned UTF-8 text, one sentence per line, line i = the same multi-parallel IN22 row):
  hypothesis  evaluation/results/<arm>/model_step_<N>/<test-set>.<src>-<tgt>.hyp.txt   (run_eval.py)
  source      128k-spm/data/raw/test/<test-set>/<test-set>.<src>                       (no <xx> tag)
  reference   128k-spm/data/raw/test/<test-set>/<test-set>.<tgt>
  chrF++      evaluation/results/<arm>/model_step_<N>/scores.json                      (optional)

Prompts are read from PROMPT.md at run time (never copied inline) and the SHA-256 of that whole file is
recorded as prompt_sha256 in every record. Variants: V2 = reference-aided (headline), V1 = reference-free
(control). Placeholders are substituted in a single pass without str.format (the system prompt contains
literal braces).

Backends (JudgeBackend.generate_batch): `vllm` (offline batch inference in this process, vLLM imported
lazily), `openai` (chat-completions HTTP endpoint via urllib), `echo` (deterministic offline stub for
testing the whole pipeline without a GPU). Importing this module needs only the standard library.

Output directory (default evaluation/llm_judge/results/<judge-name>/<arm>/model_step_<N>/):
  results.jsonl   one line per item per variant, appended and fsync'd per chunk (a crash loses <= 1 chunk)
  summary.json / summary.md   aggregates (rewritten atomically after every chunk)
  run_meta.json   judge model, backend, seed, prompt_sha256, options, timestamps
  sanity/         with --sanity: identity / mismatch item files, sanity.json, sanity.md

Resume: items already present (same variant, test set, direction, line index; status ok / auto_empty /
failed) are skipped. Resuming into a results.jsonl with a different prompt_sha256, judge model, seed or
few-shot setting is refused (use --force-new to write to a fresh directory). --retry-failed re-attempts
records whose status is "failed". Items whose judge call hit a transport error are never written, so a
resumed run picks them up again. Failed items are never imputed and never enter any mean.

Examples:
  python evaluation/llm_judge/judge.py --backend vllm --model /path/to/gemma-4-31B-it --arm bpcc-human --step 11400
  python evaluation/llm_judge/judge.py --backend echo --arm bpcc-human --step 11400 --limit-per-direction 5 \\
      --out-dir evaluation/llm_judge/results/echo-test
  python evaluation/llm_judge/judge.py --backend echo --arm bpcc-human --step 11400 --sanity
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
from common.languages import LANGUAGE_CODES, LANGUAGES  # noqa: E402

PROMPT_FILE = Path(__file__).resolve().parent / "PROMPT.md"
RAW_TEST_DIR = ROOT / "128k-spm" / "data" / "raw" / "test"
RESULTS_ROOT = ROOT / "evaluation" / "results"
JUDGE_RESULTS_ROOT = Path(__file__).resolve().parent / "results"

EN_SHORT = "en"
INDIC_SHORT_TAGS = [LANGUAGE_CODES[lang[:3]].strip("<>") for lang in LANGUAGES]
DEFAULT_TEST_SETS = ["in22-gen", "in22-conv"]
EXPECTED_LINES = {"in22-gen": 1024, "in22-conv": 1503}
BUCKET_ARGS = {"en-xx": "en->xx", "xx-en": "xx->en"}

# Judge-facing names ("<Name> (<Script> script)"); checked against LANGUAGE_CODES so it cannot drift.
LANG_NAMES = {
    "en": "English (Latin script)",
    "as": "Assamese (Bengali-Assamese script)",
    "bn": "Bengali (Bengali script)",
    "br": "Bodo (Devanagari script)",
    "doi": "Dogri (Devanagari script)",
    "gom": "Konkani (Devanagari script)",
    "gu": "Gujarati (Gujarati script)",
    "hi": "Hindi (Devanagari script)",
    "kn": "Kannada (Kannada script)",
    "ks": "Kashmiri (Perso-Arabic script)",
    "mai": "Maithili (Devanagari script)",
    "ml": "Malayalam (Malayalam script)",
    "mni": "Manipuri / Meiteilon (Meetei Mayek script)",
    "mr": "Marathi (Devanagari script)",
    "ne": "Nepali (Devanagari script)",
    "or": "Odia (Odia script)",
    "pa": "Punjabi (Gurmukhi script)",
    "sa": "Sanskrit (Devanagari script)",
    "sat": "Santali (Ol Chiki script)",
    "sd": "Sindhi (Devanagari script)",
    "ta": "Tamil (Tamil script)",
    "te": "Telugu (Telugu script)",
    "ur": "Urdu (Perso-Arabic script)",
}
# Languages for which the judge must not claim high confidence (PROMPT.md, "Judge reliability").
CONFIDENCE_CAPPED = {"sat", "mni", "ks", "sd", "doi", "br", "gom", "mai"}

CANONICAL_ORDER = ["analysis", "fluency", "adequacy", "overall", "language_ok", "confidence"]
CONFIDENCES = ("high", "medium", "low")
HYP_MAX_CHARS = 1500
TRUNC_MARKER = "[...TRUNCATED BY HARNESS...]"
RAW_RESPONSE_MAX = 4000
STATUSES_DONE = ("ok", "auto_empty", "failed")
PLACEHOLDER_RE = re.compile(r"\{(src_lang_name|tgt_lang_name|source|reference|hypothesis)\}")
PLACEHOLDERS = {"v1": {"src_lang_name", "tgt_lang_name", "source", "hypothesis"},
                "v2": {"src_lang_name", "tgt_lang_name", "source", "reference", "hypothesis"}}


def check_language_table():
    """The name table and the project's tag scheme must cover exactly the same languages."""
    tags = {v.strip("<>") for v in LANGUAGE_CODES.values()}
    assert set(LANG_NAMES) == tags, f"LANG_NAMES drifted from LANGUAGE_CODES: {set(LANG_NAMES) ^ tags}"
    assert set(INDIC_SHORT_TAGS) | {EN_SHORT} == tags


# ---------------------------------------------------------------- prompts

def extract_block(text, heading, nth=0):
    """Content of the first fenced block (fence line exactly ``` or ```text) after the first line that
    starts with `heading`, verbatim minus the trailing newline. `nth` picks a later block (few-shot)."""
    lines = text.split("\n")
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(heading):
            start = i + 1
            break
    if start is None:
        raise ValueError(f"heading not found in prompt file: {heading!r}")
    seen = -1
    i = start
    while i < len(lines):
        if lines[i] in ("```", "```text"):
            seen += 1
            j = i + 1
            while j < len(lines) and lines[j] != "```":
                j += 1
            if j >= len(lines):
                raise ValueError(f"unterminated fenced block after {heading!r}")
            if seen == nth:
                return "\n".join(lines[i + 1:j])
            i = j
        i += 1
    raise ValueError(f"no fenced block #{nth} after {heading!r}")


def load_prompts(path=PROMPT_FILE):
    """Read every prompt block from PROMPT.md and hash the whole file."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    prompts = {
        "system_v2": extract_block(text, "## SYSTEM PROMPT V2"),
        "user_v2": extract_block(text, "## USER TEMPLATE V2"),
        "system_v1": extract_block(text, "## SYSTEM PROMPT V1"),
        "user_v1": extract_block(text, "## USER TEMPLATE V1"),
        "repair": extract_block(text, "## REPAIR MESSAGE"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "few_shot": [(extract_block(text, f"### Example {k}", 0), extract_block(text, f"### Example {k}", 1))
                     for k in "ABC"],
    }
    for v in ("v1", "v2"):
        found = set(PLACEHOLDER_RE.findall(prompts[f"user_{v}"]))
        assert found == PLACEHOLDERS[v], f"user template {v} placeholders {found} != {PLACEHOLDERS[v]}"
    return prompts


def fill_template(template, values):
    """Single-pass placeholder substitution (no str.format: the templates contain literal braces, and a
    sentence that itself contains '{source}' must not be substituted twice)."""
    return PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


def drop_reference(user_text):
    """Few-shot V1 turns: the V2 example user turn without its REFERENCE block."""
    return re.sub(r"REFERENCE TRANSLATION \(one valid human translation\):\n.*?\n\n", "", user_text,
                  count=1, flags=re.S)


def build_messages(prompts, item, hypothesis, few_shot=False, merge_system=False):
    """One fresh conversation per item: system, [few-shot user/assistant pairs], user."""
    v = item["variant"]
    values = {"src_lang_name": LANG_NAMES[item["src_lang"]], "tgt_lang_name": LANG_NAMES[item["tgt_lang"]],
              "source": item["source"], "reference": item["reference"], "hypothesis": hypothesis}
    turns = []
    if few_shot:
        for user_ex, asst_ex in prompts["few_shot"]:
            turns.append({"role": "user", "content": user_ex if v == "v2" else drop_reference(user_ex)})
            turns.append({"role": "assistant", "content": asst_ex})
    turns.append({"role": "user", "content": fill_template(prompts[f"user_{v}"], values)})
    system = prompts[f"system_{v}"]
    if merge_system:
        turns[0] = {"role": "user", "content": system + "\n\n" + turns[0]["content"]}
        return turns
    return [{"role": "system", "content": system}] + turns


def prepare_hypothesis(hyp):
    """Returns (text_for_judge, truncated). Only newline->space and the length cap; nothing else is touched."""
    text = hyp.replace("\r", " ").replace("\n", " ")
    if len(text) > HYP_MAX_CHARS:
        return text[:HYP_MAX_CHARS] + TRUNC_MARKER, True
    return text, False


# ---------------------------------------------------------------- parsing and constraints

def _strip_fence(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _balanced_object(text):
    """Substring from the first '{' to its balanced '}', counting braces outside of strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_reply(raw):
    """Parse and validate one judge reply (PROMPT.md "Parse procedure").

    Returns dict: ok, error, parsed (OrderedDict or None), recovered_by, key_order_ok, analysis_words,
    analysis_too_long."""
    out = {"ok": False, "error": None, "parsed": None, "recovered_by": None, "key_order_ok": None,
           "analysis_words": None, "analysis_too_long": None}
    if raw is None:
        out["error"] = "no reply"
        return out
    text = _strip_fence(raw)
    obj = None
    try:
        obj = json.loads(text, object_pairs_hook=collections.OrderedDict)
    except ValueError:
        cand = _balanced_object(text)
        if cand is not None:
            try:
                obj = json.loads(cand, object_pairs_hook=collections.OrderedDict)
                out["recovered_by"] = "brace_scan"
            except ValueError:
                obj = None
    if not isinstance(obj, dict):
        out["error"] = "not a JSON object"
        return out
    if set(obj) != set(CANONICAL_ORDER):
        out["error"] = f"wrong keys: {sorted(obj)}"
        return out
    for k in ("fluency", "adequacy", "overall"):
        val = obj[k]
        if isinstance(val, bool) or not isinstance(val, int) or not 1 <= val <= 5:
            out["error"] = f"{k} must be an integer 1-5, got {val!r}"
            return out
    if not isinstance(obj["language_ok"], bool):
        out["error"] = f"language_ok must be a bool, got {obj['language_ok']!r}"
        return out
    if obj["confidence"] not in CONFIDENCES:
        out["error"] = f"confidence must be high/medium/low, got {obj['confidence']!r}"
        return out
    if not isinstance(obj["analysis"], str) or not obj["analysis"].strip():
        out["error"] = "analysis must be a non-empty string"
        return out
    order = list(obj)
    out["key_order_ok"] = order == CANONICAL_ORDER
    if order[0] != "analysis":
        out["error"] = "analysis is not the first key"
        return out
    out["analysis_words"] = len(obj["analysis"].split())
    out["analysis_too_long"] = out["analysis_words"] > 60
    out["parsed"] = obj
    out["ok"] = True
    return out


def band_violation(f, a, o, language_ok):
    """True if the band rule is broken: min(F,A) <= O <= min(F,A)+1; A<=2 => O<=A; not language_ok => O<=2 and min(F,A)<=2."""
    low = min(f, a)
    if not (low <= o <= low + 1):
        return True
    if a <= 2 and o > a:
        return True
    if not language_ok and (o > 2 or low > 2):
        return True
    return False


def confidence_cap_violation(src, tgt, confidence):
    return confidence == "high" and (src in CONFIDENCE_CAPPED or tgt in CONFIDENCE_CAPPED)


# ---------------------------------------------------------------- backends

class Reply:
    """One backend answer. text None = transport failure (item is left unwritten)."""

    def __init__(self, text=None, latency_ms=None, error=None):
        self.text, self.latency_ms, self.error = text, latency_ms, error


class JudgeBackend:
    """generate_batch(list of message lists) -> list of Reply, same order. Greedy decoding."""

    name = "abstract"
    merge_system = False

    def generate_batch(self, batch):
        raise NotImplementedError


def _stable_hash(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


class EchoBackend(JudgeBackend):
    """Deterministic offline stub: valid JSON with scores derived from a hash of the hypothesis.

    With inject=True a slice of items misbehave to exercise parsing/repair: prose on the first attempt
    (valid on repair), band-violating JSON on the first attempt (valid on repair), always-invalid output
    (ends as failed), a code-fenced reply and a prose-wrapped reply (both parse without a retry)."""

    name = "echo"

    def __init__(self, inject=False):
        self.inject = inject

    @staticmethod
    def hypothesis_of(messages):
        for m in reversed(messages):
            if m["role"] == "user" and "TRANSLATION TO SCORE:\n" in m["content"]:
                body = m["content"].split("TRANSLATION TO SCORE:\n", 1)[1]
                return body.rsplit("\n\nReturn the JSON object only.", 1)[0]
        return ""

    @staticmethod
    def scores_for(hyp):
        h = _stable_hash(hyp)
        a = 1 + h % 5
        f = 1 + (h >> 3) % 5
        lang_ok = (h >> 8) % 10 != 0
        if not lang_ok:
            a = min(a, 2)  # hard rule 3: wrong language => adequacy <= 2 (keeps the band rule satisfiable)
        o = min(min(f, a) + (h >> 6) % 2, 5)
        if a <= 2:
            o = min(o, a)
        if not lang_ok:
            o = min(o, 2)
        return f, a, o, lang_ok, CONFIDENCES[(h >> 12) % 3], h

    def _one(self, messages):
        is_repair = messages[-1]["role"] == "user" and messages[-1]["content"].startswith("Your reply was not valid.")
        hyp = self.hypothesis_of(messages)
        f, a, o, ok, conf, h = self.scores_for(hyp)
        obj = collections.OrderedDict([("analysis", f"echo stub, no real analysis (h={h % 10000})"),
                                       ("fluency", f), ("adequacy", a), ("overall", o),
                                       ("language_ok", ok), ("confidence", conf)])
        good = json.dumps(obj)
        if self.inject and not is_repair:
            slot = (h >> 16) % 100
            if slot < 4:
                return "I think this translation is decent overall."
            if slot < 7:
                bad = collections.OrderedDict(obj)
                bad["fluency"] = bad["adequacy"] = 2
                bad["overall"] = 4
                return json.dumps(bad)
            if slot < 9:
                return "not json at all"
            if slot < 11:
                return "```json\n" + good + "\n```"
            if slot < 13:
                return "Here is my verdict: " + good + " Hope that helps."
        if self.inject and is_repair and 7 <= (h >> 16) % 100 < 9:
            return "still not json"
        return good

    def generate_batch(self, batch):
        return [Reply(self._one(m), latency_ms=0) for m in batch]


class OpenAIBackend(JudgeBackend):
    """OpenAI-compatible /chat/completions endpoint (vLLM / TGI / llama.cpp server) over urllib."""

    name = "openai"

    def __init__(self, base_url, model, seed, max_tokens, concurrency=8, api_key=None, timeout=300,
                 merge_system=False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.seed, self.max_tokens = model, seed, max_tokens
        self.concurrency, self.timeout = concurrency, timeout
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.merge_system = merge_system

    def _call(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0, "top_p": 1,
                           "max_tokens": self.max_tokens, "seed": self.seed}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        err = None
        for attempt in range(4):  # first try + 3 retries with backoff
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return Reply(data["choices"][0]["message"]["content"] or "", int((time.time() - t0) * 1000))
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code < 500 and e.code != 429:
                    break  # a 4xx is our request's fault; retrying will not help
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
                err = repr(e)
            time.sleep(2 ** attempt)
        return Reply(None, None, err)

    def generate_batch(self, batch):
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._call, batch))


class VllmBackend(JudgeBackend):
    """Offline vLLM batch inference in this process (tensor parallel over the allocated GPUs)."""

    name = "vllm"

    def __init__(self, model, tp, seed, max_tokens, max_model_len=4096, gpu_mem_util=0.90, dtype="bfloat16",
                 system_mode="auto", extra_kwargs=None):
        from vllm import LLM, SamplingParams  # lazy: only this backend needs vLLM
        kwargs = dict(model=model, tensor_parallel_size=tp, max_model_len=max_model_len,
                      gpu_memory_utilization=gpu_mem_util, dtype=dtype, seed=seed)
        kwargs.update(extra_kwargs or {})
        self.llm = LLM(**kwargs)
        self.params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=seed)
        self.merge_system = self._decide_merge(system_mode)

    def _decide_merge(self, mode):
        if mode == "merge":
            return True
        if mode == "native":
            return False
        marker = "SYSTEM-MARKER-7f3a"
        try:
            rendered = self.llm.get_tokenizer().apply_chat_template(
                [{"role": "system", "content": marker}, {"role": "user", "content": "x"}],
                tokenize=False, add_generation_prompt=True)
            return marker not in rendered
        except Exception:  # template rejects the system role
            return True

    def generate_batch(self, batch):
        outs = self.llm.chat(batch, self.params, use_tqdm=True,
                             chat_template_kwargs={"enable_thinking": False})
        return [Reply(o.outputs[0].text, None) for o in outs]


# ---------------------------------------------------------------- data loading

def read_lines(path):
    """splitlines() on the file text; an empty line is a legitimate (empty) hypothesis, never dropped."""
    return Path(path).read_text(encoding="utf-8").splitlines()


def sample_indices(n_lines, test_set, direction, limit, seed):
    """Depends only on (seed, test_set, direction, limit): every arm/step is judged on identical sentences."""
    if limit is None or limit >= n_lines:
        return list(range(n_lines))
    return sorted(random.Random(f"{seed}:{test_set}:{direction}").sample(range(n_lines), limit))


def directions(langs, buckets):
    for short in INDIC_SHORT_TAGS:
        if langs and short not in langs:
            continue
        for src, tgt, bucket in ((EN_SHORT, short, "en->xx"), (short, EN_SHORT, "xx->en")):
            if bucket in buckets:
                yield src, tgt, bucket


def load_direction(hyp_dir, raw_dir, test_set, src, tgt, expected=None):
    """(hyps, srcs, refs) for one direction. Raises ValueError on missing files or a line-count mismatch
    (a partial eval run): the caller aborts that direction."""
    expected = EXPECTED_LINES if expected is None else expected
    hyp_path = Path(hyp_dir) / f"{test_set}.{src}-{tgt}.hyp.txt"
    src_path = Path(raw_dir) / test_set / f"{test_set}.{src}"
    ref_path = Path(raw_dir) / test_set / f"{test_set}.{tgt}"
    for p in (hyp_path, src_path, ref_path):
        if not p.exists():
            raise ValueError(f"missing file {p}")
    hyps, srcs, refs = read_lines(hyp_path), read_lines(src_path), read_lines(ref_path)
    want = expected.get(test_set)
    if not (len(hyps) == len(srcs) == len(refs)) or (want is not None and len(hyps) != want):
        raise ValueError(f"line counts hyp/src/ref = {len(hyps)}/{len(srcs)}/{len(refs)}, expected {want}")
    return hyps, srcs, refs


def build_items(hyp_dir, raw_dir, test_sets, dirs, variants, limit, seed, expected=None, hyp_override=None):
    """All items for the run, in a fixed order. hyp_override(hyps, srcs, refs) -> hyps replaces the
    hypothesis stream (sanity checks). Returns (items, skipped [(test_set, direction, reason)])."""
    items, skipped = [], []
    for test_set in test_sets:
        for src, tgt, bucket in dirs:
            direction = f"{src}-{tgt}"
            try:
                hyps, srcs, refs = load_direction(hyp_dir, raw_dir, test_set, src, tgt, expected)
            except ValueError as e:
                skipped.append((test_set, direction, str(e)))
                continue
            if hyp_override is not None:
                hyps = hyp_override(hyps, srcs, refs)
            idxs = sample_indices(len(hyps), test_set, direction, limit, seed)
            for variant in variants:
                for i in idxs:
                    items.append({"test_set": test_set, "direction": direction, "src_lang": src,
                                  "tgt_lang": tgt, "bucket": bucket, "line_index": i, "variant": variant,
                                  "source": srcs[i], "reference": refs[i], "hypothesis": hyps[i]})
    return items, skipped


# ---------------------------------------------------------------- results file, resume

def item_key(rec):
    return (rec["variant"], rec["test_set"], rec["direction"], rec["line_index"])


def atomic_write(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_results(path):
    """Read a results jsonl. A trailing partial line (killed mid-write) is dropped and the file repaired."""
    path = Path(path)
    if not path.exists():
        return []
    recs, bad = [], 0
    for ln in path.read_text(encoding="utf-8").split("\n"):
        if not ln.strip():
            continue
        try:
            recs.append(json.loads(ln))
        except ValueError:
            bad += 1
    if bad:
        print(f"[resume] dropped {bad} unparseable line(s) from {path}")
        atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    return recs


def check_resume_compat(records, ctx):
    """Refuse to mix results made with a different rubric file, judge model, seed or few-shot setting."""
    problems = []
    for field, want in (("prompt_sha256", ctx["prompt_sha256"]), ("judge_model", ctx["judge_model"]),
                        ("seed", ctx["seed"]), ("few_shot", ctx["few_shot"])):
        have = sorted({str(r.get(field)) for r in records})
        if have and have != [str(want)]:
            problems.append(f"{field}: existing {have} vs this run {want}")
    return problems


def append_records(path, recs):
    """Append + flush + fsync so a crash loses at most the chunk in flight."""
    if not recs:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- judging one chunk

def base_record(item, ctx):
    return {"test_set": item["test_set"], "direction": item["direction"], "src_lang": item["src_lang"],
            "tgt_lang": item["tgt_lang"], "bucket": item["bucket"], "line_index": item["line_index"],
            "arm": ctx["arm"], "step": ctx["step"], "variant": item["variant"], "status": None,
            "auto_scored": False, "source": item["source"], "reference": item["reference"],
            "hypothesis": item["hypothesis"], "analysis": None, "fluency": None, "adequacy": None,
            "overall": None, "language_ok": None, "confidence": None, "analysis_words": None,
            "analysis_too_long": None, "key_order_ok": None, "recovered_by": None,
            "constraint_violation": False, "confidence_cap_violation": False, "hyp_truncated": False,
            "retried": False, "judge_model": ctx["judge_model"], "prompt_sha256": ctx["prompt_sha256"],
            "seed": ctx["seed"], "few_shot": ctx["few_shot"], "system_merged": ctx["system_merged"],
            "latency_ms": None, "raw_response": None, "error": None}


def judge_chunk(items, backend, prompts, ctx):
    """Judge one chunk. Returns (records, n_transport_unwritten). Empty hypotheses are auto-scored
    without a judge call; parse/validation failures and band violations get one repair retry."""
    recs, pending = [], []
    for item in items:
        rec = base_record(item, ctx)
        if not item["hypothesis"].strip():
            rec.update(status="auto_empty", auto_scored=True, analysis="empty output", fluency=1, adequacy=1,
                       overall=1, language_ok=False, confidence="high")
            recs.append(rec)
            continue
        hyp, trunc = prepare_hypothesis(item["hypothesis"])
        rec["hyp_truncated"] = trunc
        msgs = build_messages(prompts, item, hyp, ctx["few_shot"], ctx["system_merged"])
        pending.append((rec, msgs))
    if not pending:
        return recs, 0

    first = backend.generate_batch([m for _, m in pending])
    state = []  # per pending: (rec, msgs, reply, parse, needs_retry)
    for (rec, msgs), rep in zip(pending, first):
        if rep.text is None:
            state.append((rec, msgs, rep, None, False))
            continue
        p = parse_reply(rep.text)
        bad = not p["ok"]
        if p["ok"]:
            o = p["parsed"]
            bad = band_violation(o["fluency"], o["adequacy"], o["overall"], o["language_ok"])
        state.append((rec, msgs, rep, p, bad))

    retry_idx = [i for i, s in enumerate(state) if s[4]]
    retry_replies = {}
    if retry_idx:
        rbatch = [state[i][1] + [{"role": "assistant", "content": state[i][2].text},
                                 {"role": "user", "content": prompts["repair"]}] for i in retry_idx]
        for i, rep in zip(retry_idx, backend.generate_batch(rbatch)):
            retry_replies[i] = rep

    unwritten = 0
    for i, (rec, msgs, rep, p, bad) in enumerate(state):
        if rep.text is None:
            unwritten += 1
            continue
        final_p, final_rep = p, rep
        if i in retry_replies:
            rrep = retry_replies[i]
            rec["retried"] = True
            if rrep.text is None:
                if not p["ok"]:
                    unwritten += 1
                    continue
                # first attempt was a valid (band-violating) answer; keep it, flagged below
            else:
                rp = parse_reply(rrep.text)
                if rp["ok"]:
                    final_p, final_rep = rp, rrep
                elif not p["ok"]:
                    final_p, final_rep = rp, rrep
                # else: retry unusable but the first attempt was a valid answer: keep the first
        rec["latency_ms"] = final_rep.latency_ms
        if final_p["ok"]:
            o = final_p["parsed"]
            rec.update(status="ok", analysis=o["analysis"], fluency=o["fluency"], adequacy=o["adequacy"],
                       overall=o["overall"], language_ok=o["language_ok"], confidence=o["confidence"],
                       analysis_words=final_p["analysis_words"], analysis_too_long=final_p["analysis_too_long"],
                       key_order_ok=final_p["key_order_ok"], recovered_by=final_p["recovered_by"])
            rec["constraint_violation"] = band_violation(o["fluency"], o["adequacy"], o["overall"],
                                                         o["language_ok"])
            rec["confidence_cap_violation"] = confidence_cap_violation(rec["src_lang"], rec["tgt_lang"],
                                                                       o["confidence"])
        else:
            rec.update(status="failed", error=final_p["error"], raw_response=(final_rep.text or "")[:RAW_RESPONSE_MAX])
        recs.append(rec)
    return recs, unwritten


def run_items(items, backend, prompts, ctx, results_path, chunk_size, retry_failed=False, on_chunk=None,
              log=print):
    """Resume-aware driver: skip done items, judge the rest chunk by chunk, append per chunk.
    Returns (all_records_in_file, stats)."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_results(results_path)
    problems = check_resume_compat(existing, ctx)
    if problems:
        raise SystemExit("Refusing to resume into " + str(results_path) + ":\n  " + "\n  ".join(problems)
                         + "\nUse --force-new to write to a fresh directory.")
    if retry_failed and any(r["status"] == "failed" for r in existing):
        existing = [r for r in existing if r["status"] != "failed"]
        atomic_write(results_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in existing))
    done = {item_key(r) for r in existing if r["status"] in STATUSES_DONE}
    todo = [it for it in items if item_key(it) not in done]
    stats = {"n_items": len(items), "n_skipped_done": len(items) - len(todo), "n_judged": 0,
             "n_transport_unwritten": 0}
    all_recs = list(existing)
    log(f"[run] {len(items)} items, {stats['n_skipped_done']} already done, {len(todo)} to do "
        f"({results_path})")
    for c0 in range(0, len(todo), chunk_size):
        chunk = todo[c0:c0 + chunk_size]
        t0 = time.time()
        recs, unwritten = judge_chunk(chunk, backend, prompts, ctx)
        append_records(results_path, recs)
        all_recs.extend(recs)
        stats["n_judged"] += len(recs)
        stats["n_transport_unwritten"] += unwritten
        log(f"[chunk {c0 // chunk_size + 1}/{(len(todo) + chunk_size - 1) // chunk_size}] "
            f"{len(recs)} written, {unwritten} transport failures, {time.time() - t0:.1f}s")
        if on_chunk:
            on_chunk(all_recs)
    return all_recs, stats


# ---------------------------------------------------------------- aggregation

def mean(xs):
    return sum(xs) / len(xs) if xs else None


def agg(recs):
    """Aggregate a list of records. Headline means = scored items with confidence high/medium (auto_empty
    included: they are true 1/1/1); low-confidence mean reported separately; failed never averaged."""
    scored = [r for r in recs if r["status"] in ("ok", "auto_empty")]
    failed = [r for r in recs if r["status"] == "failed"]
    head = [r for r in scored if r["confidence"] != "low"]
    low = [r for r in scored if r["confidence"] == "low"]
    out = {"n": len(recs), "n_scored": len(scored), "n_failed": len(failed),
           "failed_rate": len(failed) / len(recs) if recs else None,
           "unreliable_failed_rate_gt_5pct": bool(recs) and len(failed) / len(recs) > 0.05,
           "n_auto_empty": sum(1 for r in recs if r["status"] == "auto_empty"),
           "n_headline": len(head), "n_low_confidence": len(low),
           "n_language_ok_false": sum(1 for r in scored if r["language_ok"] is False),
           "n_constraint_violation": sum(1 for r in scored if r["constraint_violation"]),
           "constraint_violation_rate": (sum(1 for r in scored if r["constraint_violation"]) / len(scored))
           if scored else None,
           "n_confidence_cap_violation": sum(1 for r in scored if r.get("confidence_cap_violation")),
           "n_retried": sum(1 for r in recs if r["retried"]),
           "n_recovered_brace_scan": sum(1 for r in recs if r.get("recovered_by") == "brace_scan"),
           "n_analysis_too_long": sum(1 for r in scored if r.get("analysis_too_long")),
           "n_key_order_not_canonical": sum(1 for r in scored if r.get("key_order_ok") is False),
           "n_hyp_truncated": sum(1 for r in recs if r["hyp_truncated"])}
    for m in ("fluency", "adequacy", "overall"):
        out[f"mean_{m}"] = mean([r[m] for r in head])
        out[f"mean_{m}_lowconf"] = mean([r[m] for r in low])
        out[f"hist_{m}"] = {str(k): sum(1 for r in scored if r[m] == k) for k in range(1, 6)}
    return out


def language_of(rec):
    return rec["tgt_lang"] if rec["src_lang"] == EN_SHORT else rec["src_lang"]


def rank(xs):
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rho (Pearson on average ranks); None if undefined (n<3 or a constant series)."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sxx * syy) ** 0.5


def load_chrf(scores_path):
    """{(test_set, direction): chrf++} from run_eval.py's scores.json ({} when the file does not exist)."""
    p = Path(scores_path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for ts, dd in data.items():
        for d, v in dd.items():
            if isinstance(v, dict) and "chrf++" in v and not d.startswith("MACRO"):
                out[(ts, d)] = v["chrf++"]
    return out


def group(recs, keyfn):
    g = collections.OrderedDict()
    for r in recs:
        g.setdefault(keyfn(r), []).append(r)
    return g


def summarize(records, chrf, meta):
    """The summary.json structure. One block per variant present in `records`."""
    out = {"meta": meta, "variants": {}}
    for variant, vrecs in group(records, lambda r: r["variant"]).items():
        block = {"total": agg(vrecs)}
        block["by_test_set"] = {k: agg(v) for k, v in group(vrecs, lambda r: r["test_set"]).items()}
        block["by_bucket"] = {k: agg(v) for k, v in group(vrecs, lambda r: r["bucket"]).items()}
        block["by_bucket_test_set"] = {f"{b}|{t}": agg(v) for (b, t), v in
                                       group(vrecs, lambda r: (r["bucket"], r["test_set"])).items()}
        block["by_language"] = {}
        for lang, lrecs in group(vrecs, language_of).items():
            entry = {"all": agg(lrecs)}
            for b, brecs in group(lrecs, lambda r: r["bucket"]).items():
                a = agg(brecs)
                cs = [chrf[(r_ts, r_d)] for r_ts, r_d in {(x["test_set"], x["direction"]) for x in brecs}
                      if (r_ts, r_d) in chrf]
                a["chrf++"] = mean(cs)
                entry[b] = a
            block["by_language"][lang] = entry
        block["by_direction"] = {}
        for (ts, d), drecs in group(vrecs, lambda r: (r["test_set"], r["direction"])).items():
            a = agg(drecs)
            a["chrf++"] = chrf.get((ts, d))
            a["bucket"] = drecs[0]["bucket"]
            a["test_set"] = ts
            block["by_direction"][f"{ts}/{d}"] = a
        corr = {}
        for (ts, b) in {(r["test_set"], r["bucket"]) for r in vrecs}:
            xs, ys = [], []
            for a in block["by_direction"].values():
                if a["test_set"] == ts and a["bucket"] == b and a["chrf++"] is not None \
                        and a["mean_overall"] is not None:
                    xs.append(a["mean_overall"])
                    ys.append(a["chrf++"])
            corr[f"{ts}|{b}"] = {"n_directions": len(xs), "spearman_overall_vs_chrf": spearman(xs, ys)}
        block["chrf_correlation_system_level"] = corr
        out["variants"][variant] = block
    return out


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def summary_markdown(summary):
    meta = summary["meta"]
    L = ["# LLM-as-judge summary", "",
         f"- judge model: `{meta.get('judge_model')}` (backend `{meta.get('backend')}`)",
         f"- arm / step: `{meta.get('arm')}` / {meta.get('step')}",
         f"- prompt_sha256: `{meta.get('prompt_sha256')}`  (few-shot: {meta.get('few_shot')}, "
         f"system merged into user turn: {meta.get('system_merged')})",
         f"- seed: {meta.get('seed')}", "",
         "Headline means are over confidence high+medium (auto-scored empty outputs included); the "
         "low-confidence mean is reported separately, failed items are excluded from every mean.", ""]
    for variant, b in summary["variants"].items():
        name = {"v2": "V2 reference-aided (headline)", "v1": "V1 reference-free (control)"}.get(variant, variant)
        L += [f"## {name}", "", "### Rollup", "",
              "| scope | n | F | A | O | n_lowconf | O_lowconf | lang_ok_false | failed |",
              "|---|---|---|---|---|---|---|---|---|"]

        def row(label, a):
            return (f"| {label} | {a['n']} | {fmt(a['mean_fluency'])} | {fmt(a['mean_adequacy'])} | "
                    f"{fmt(a['mean_overall'])} | {a['n_low_confidence']} | {fmt(a['mean_overall_lowconf'])} | "
                    f"{a['n_language_ok_false']} | {a['n_failed']} |")
        L.append(row("**total**", b["total"]))
        for k, a in b["by_bucket"].items():
            L.append(row(f"bucket {k}", a))
        for k, a in b["by_test_set"].items():
            L.append(row(f"test set {k}", a))
        for k, a in b["by_bucket_test_set"].items():
            L.append(row(f"{k.replace('|', ' / ')}", a))
        L.append("")
        for bucket in ("en->xx", "xx->en"):
            if bucket not in b["by_bucket"]:
                continue
            L += [f"### Per language, {bucket} (test sets pooled)", "",
                  "| language | F | A | O | n | low-conf | lang_ok_false | failed | chrF++ |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for lang, e in sorted(b["by_language"].items()):
                a = e.get(bucket)
                if a:
                    L.append(f"| {lang} | {fmt(a['mean_fluency'])} | {fmt(a['mean_adequacy'])} | "
                             f"{fmt(a['mean_overall'])} | {a['n']} | {a['n_low_confidence']} | "
                             f"{a['n_language_ok_false']} | {a['n_failed']} | {fmt(a.get('chrf++'), 1)} |")
            L.append("")
        t = b["total"]
        L += ["### Counts", "",
              f"- scored {t['n_scored']} / {t['n']}; failed {t['n_failed']} (rate {fmt(t['failed_rate'], 4)}); "
              f"auto-scored empty {t['n_auto_empty']}",
              f"- low-confidence {t['n_low_confidence']}; language_ok=false {t['n_language_ok_false']}",
              f"- constraint violations {t['n_constraint_violation']} (rate {fmt(t['constraint_violation_rate'], 4)}); "
              f"confidence-cap violations {t['n_confidence_cap_violation']}",
              f"- retried {t['n_retried']}; recovered by brace scan {t['n_recovered_brace_scan']}; "
              f"analysis > 60 words {t['n_analysis_too_long']}; non-canonical key order {t['n_key_order_not_canonical']}; "
              f"hypotheses truncated by harness {t['n_hyp_truncated']}", ""]
        unreliable = [k for k, a in b["by_direction"].items() if a["unreliable_failed_rate_gt_5pct"]]
        if unreliable:
            L += ["Directions with failed rate > 5% (unreliable, not averaged away): " + ", ".join(unreliable), ""]
        L += ["### Score distribution (all scored items, counts of 1..5)", "", "| metric | 1 | 2 | 3 | 4 | 5 |",
              "|---|---|---|---|---|---|"]
        for m in ("fluency", "adequacy", "overall"):
            h = t[f"hist_{m}"]
            L.append(f"| {m} | " + " | ".join(str(h[str(k)]) for k in range(1, 6)) + " |")
        L += ["", "### System-level Spearman of mean overall vs corpus chrF++ (across directions)", "",
              "| test set / bucket | directions | rho |", "|---|---|---|"]
        for k, c in sorted(b["chrf_correlation_system_level"].items()):
            L.append(f"| {k.replace('|', ' / ')} | {c['n_directions']} | {fmt(c['spearman_overall_vs_chrf'], 3)} |")
        L.append("")
    return "\n".join(L) + "\n"


def write_summary(records, chrf, meta, out_dir):
    summary = summarize(records, chrf, meta)
    atomic_write(Path(out_dir) / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    atomic_write(Path(out_dir) / "summary.md", summary_markdown(summary))
    return summary


# ---------------------------------------------------------------- sanity checks

def identity_override(hyps, srcs, refs):
    return list(refs)


def mismatch_override(hyps, srcs, refs, offset=37):
    n = len(refs)
    return [refs[(i + offset) % n] for i in range(n)]


def sanity_report(results, chrf_corr):
    """Interpret sanity results against the spec's expected ranges. results: {check: [records]}."""
    rep = {"checks": {}, "chrf_correlation": chrf_corr}
    for name, recs in results.items():
        a = agg(recs)
        scored = [r for r in recs if r["status"] in ("ok", "auto_empty")]
        lang_ok = [r for r in scored if r["language_ok"] is not None]
        pct_ok = (sum(1 for r in lang_ok if r["language_ok"]) / len(lang_ok)) if lang_ok else None
        allmean = {m: mean([r[m] for r in scored]) for m in ("fluency", "adequacy", "overall")}
        if name == "identity":
            passed = allmean["overall"] is not None and allmean["overall"] >= 4.7 and (pct_ok or 0) >= 0.95
            expect = "V1: mean overall 4.7-5.0, language_ok true ~100%"
        else:
            passed = (allmean["adequacy"] is not None and allmean["adequacy"] <= 2.0
                      and allmean["overall"] <= 2.0 and allmean["fluency"] >= 4.0)
            expect = "mean adequacy <= 2, mean overall <= 2, mean fluency >= 4"
        rep["checks"][name] = {"expect": expect, "n_scored": len(scored), "n_failed": a["n_failed"],
                               "mean_all_scored": allmean, "pct_language_ok_true": pct_ok, "pass": bool(passed),
                               "by_direction": {d: {m: mean([r[m] for r in v if r["status"] != "failed"
                                                             and r[m] is not None]) for m in
                                                    ("fluency", "adequacy", "overall")}
                                                for d, v in group(recs, lambda r: r["direction"]).items()}}
    return rep


def sanity_markdown(rep, meta):
    L = ["# LLM-as-judge sanity checks", "", f"- judge model: `{meta.get('judge_model')}`",
         f"- arm / step: `{meta.get('arm')}` / {meta.get('step')}",
         f"- prompt_sha256: `{meta.get('prompt_sha256')}`", "",
         "| check | expected | n | failed | F | A | O | language_ok true | pass |", "|---|---|---|---|---|---|---|---|---|"]
    for name, c in rep["checks"].items():
        m = c["mean_all_scored"]
        L.append(f"| {name} | {c['expect']} | {c['n_scored']} | {c['n_failed']} | {fmt(m['fluency'])} | "
                 f"{fmt(m['adequacy'])} | {fmt(m['overall'])} | {fmt(c['pct_language_ok_true'], 3)} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L += ["", "## System-level Spearman, mean overall vs corpus chrF++ (across directions, from the main run)", ""]
    corr = rep["chrf_correlation"]
    if not corr:
        L.append("No main-run results.jsonl (or no scores.json) found, so no correlation.")
    else:
        L += ["| variant / test set / bucket | directions | rho |", "|---|---|---|"]
        for k, c in sorted(corr.items()):
            L.append(f"| {k.replace('|', ' / ')} | {c['n_directions']} | {fmt(c['spearman_overall_vs_chrf'], 3)} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI

def judge_name_of(model):
    base = Path(str(model).rstrip("/")).name or str(model)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--backend", choices=["vllm", "openai", "echo"], default="echo")
    p.add_argument("--model", default="echo-stub", help="model path (vllm) or served model name (openai)")
    p.add_argument("--judge-name", default=None, help="directory/label name for the judge (default: from --model)")
    p.add_argument("--base-url", default=None, help="openai backend: e.g. http://host:8000/v1")
    p.add_argument("--api-key", default=None)
    p.add_argument("--concurrency", type=int, default=8, help="openai backend: parallel requests")
    p.add_argument("--tp", type=int, default=2, help="vllm tensor_parallel_size")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--vllm-kwargs", default=None, help="JSON dict of extra vllm.LLM(...) arguments")
    p.add_argument("--system-mode", choices=["auto", "native", "merge"], default="auto",
                   help="auto: merge the system prompt into the first user turn only if the chat template "
                        "has no system role (openai backend: auto = native)")
    p.add_argument("--echo-inject", action="store_true", help="echo backend: inject invalid outputs")
    p.add_argument("--arm", default="bpcc-human")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--test-sets", nargs="+", default=DEFAULT_TEST_SETS)
    p.add_argument("--buckets", nargs="+", choices=list(BUCKET_ARGS), default=["en-xx"])
    p.add_argument("--langs", nargs="+", default=None, help="subset of Indic short tags (e.g. hi ta sat)")
    p.add_argument("--variant", choices=["V1", "V2", "both"], default="V2")
    p.add_argument("--limit-per-direction", type=int, default=None, help="sample N sentences per direction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--few-shot", action="store_true")
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--force-new", action="store_true", help="on a prompt/model/seed mismatch write to a fresh dir")
    p.add_argument("--sanity", action="store_true",
                   help="run the sanity checks (identity, mismatch, chrF++ correlation) instead of the main run")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--results-root", default=str(RESULTS_ROOT))
    p.add_argument("--raw-test-dir", default=str(RAW_TEST_DIR))
    p.add_argument("--prompt-file", default=str(PROMPT_FILE))
    return p.parse_args(argv)


def make_backend(args):
    if args.backend == "echo":
        return EchoBackend(inject=args.echo_inject)
    if args.backend == "openai":
        if not args.base_url:
            raise SystemExit("--base-url is required for --backend openai")
        return OpenAIBackend(args.base_url, args.model, args.seed, args.max_tokens, args.concurrency,
                             args.api_key, merge_system=args.system_mode == "merge")
    extra = json.loads(args.vllm_kwargs) if args.vllm_kwargs else None
    return VllmBackend(args.model, args.tp, args.seed, args.max_tokens, args.max_model_len,
                       args.gpu_memory_utilization, args.dtype, args.system_mode, extra)


def main(argv=None):
    args = parse_args(argv)
    check_language_table()
    langs = set(args.langs) if args.langs else None
    if langs and not langs <= set(INDIC_SHORT_TAGS):
        raise SystemExit(f"unknown --langs: {sorted(langs - set(INDIC_SHORT_TAGS))}")
    variants = ["v1", "v2"] if args.variant == "both" else [args.variant.lower()]
    buckets = {BUCKET_ARGS[b] for b in args.buckets}
    judge_name = args.judge_name or judge_name_of(args.model)
    hyp_dir = Path(args.results_root) / args.arm / f"model_step_{args.step}"
    out_dir = Path(args.out_dir) if args.out_dir else JUDGE_RESULTS_ROOT / judge_name / args.arm / f"model_step_{args.step}"
    prompts = load_prompts(args.prompt_file)

    ctx = {"arm": args.arm, "step": args.step, "judge_model": args.model, "prompt_sha256": prompts["sha256"],
           "seed": args.seed, "few_shot": args.few_shot, "system_merged": False}
    results_path = out_dir / "results.jsonl"

    # Refuse-to-mix check up front, before any model is loaded.
    problems = check_resume_compat(load_results(results_path), ctx)
    if problems:
        if not args.force_new:
            raise SystemExit("Refusing to resume into " + str(results_path) + ":\n  " + "\n  ".join(problems)
                             + "\nUse --force-new to write to a fresh directory.")
        out_dir = Path(str(out_dir) + "__" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        results_path = out_dir / "results.jsonl"
        print(f"[force-new] writing to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit_per_direction
    sanity_dirs = None
    if args.sanity:
        limit = limit if limit is not None else 30
        sanity_langs = langs or {"hi", "ta", "or", "as", "sat", "mni"}
        sanity_dirs = list(directions(sanity_langs, {"en->xx", "xx->en"}))
    dirs = sanity_dirs if args.sanity else list(directions(langs, buckets))

    backend = make_backend(args)
    ctx["system_merged"] = bool(backend.merge_system)
    chrf = load_chrf(hyp_dir / "scores.json")
    meta = {"judge_model": args.model, "judge_name": judge_name, "backend": backend.name, "arm": args.arm,
            "step": args.step, "prompt_sha256": prompts["sha256"], "prompt_file": str(args.prompt_file),
            "seed": args.seed, "few_shot": args.few_shot, "system_merged": ctx["system_merged"],
            "variants": variants, "test_sets": args.test_sets, "buckets": sorted(buckets),
            "langs": sorted(langs) if langs else "all", "limit_per_direction": limit,
            "max_tokens": args.max_tokens, "temperature": 0, "chunk_size": args.chunk_size,
            "started": datetime.now().isoformat(timespec="seconds"), "mode": "sanity" if args.sanity else "main"}

    if args.sanity:
        return run_sanity(args, backend, prompts, ctx, meta, out_dir, hyp_dir, dirs, limit, chrf)

    items, skipped = build_items(hyp_dir, args.raw_test_dir, args.test_sets, dirs, variants, limit, args.seed)
    for ts, d, why in skipped:
        print(f"[skip] {ts} {d}: {why}")
    meta["skipped_directions"] = [f"{ts}/{d}: {why}" for ts, d, why in skipped]
    atomic_write(out_dir / "run_meta.json", json.dumps(meta, indent=2))

    def on_chunk(recs):
        write_summary(recs, chrf, meta, out_dir)

    recs, stats = run_items(items, backend, prompts, ctx, results_path, args.chunk_size, args.retry_failed, on_chunk)
    write_summary(recs, chrf, meta, out_dir)
    meta.update(finished=datetime.now().isoformat(timespec="seconds"), run_stats=stats)
    atomic_write(out_dir / "run_meta.json", json.dumps(meta, indent=2))
    print(f"[done] {stats}; summary at {out_dir / 'summary.md'}")
    return 2 if skipped else (3 if stats["n_transport_unwritten"] else 0)


def run_sanity(args, backend, prompts, ctx, meta, out_dir, hyp_dir, dirs, limit, chrf):
    sdir = out_dir / "sanity"
    sdir.mkdir(parents=True, exist_ok=True)
    plan = [("identity", ["v1"], identity_override)]
    plan.append(("mismatch", ["v1", "v2"] if args.variant == "both" else [args.variant.lower()], mismatch_override))
    results = {}
    for name, variants, override in plan:
        items, skipped = build_items(hyp_dir, args.raw_test_dir, args.test_sets, dirs, variants, limit,
                                     args.seed, hyp_override=override)
        for ts, d, why in skipped:
            print(f"[skip] {ts} {d}: {why}")
        recs, _ = run_items(items, backend, prompts, ctx, sdir / f"{name}.jsonl", args.chunk_size,
                            args.retry_failed)
        results[name] = recs
    corr = {}
    main_recs = load_results(out_dir / "results.jsonl")
    if main_recs and chrf:
        for variant, blk in summarize(main_recs, chrf, meta)["variants"].items():
            for k, c in blk["chrf_correlation_system_level"].items():
                corr[f"{variant}|{k}"] = c
    rep = sanity_report(results, corr)
    atomic_write(sdir / "sanity.json", json.dumps(rep, indent=2, ensure_ascii=False))
    atomic_write(sdir / "sanity.md", sanity_markdown(rep, meta))
    print((sdir / "sanity.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
