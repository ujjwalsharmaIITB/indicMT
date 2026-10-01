"""Unit tests for evaluation/llm_judge/judge.py (plain asserts, no pytest needed).

Run: python3 evaluation/llm_judge/test_judge.py   (needs Python 3.8+; standard library only)

Everything runs on the echo backend against a tiny fabricated fixture in a temp directory. The one
exception is test_real_files_line_up, which only reads the real in22-gen en-hi hypothesis/source/
reference files (and is skipped when they do not exist).
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as J  # noqa: E402

PROMPTS = J.load_prompts()
CTX = {"arm": "fix", "step": 1, "judge_model": "echo-stub", "prompt_sha256": PROMPTS["sha256"], "seed": 42,
       "few_shot": False, "system_merged": False}


def make_fixture(tmp, n=6, empty_at=(2,)):
    """hyp dir + raw dir for in22-gen, directions en-hi and hi-en only, n lines each."""
    hyp_dir, raw_dir = Path(tmp) / "hyp", Path(tmp) / "raw"
    (raw_dir / "in22-gen").mkdir(parents=True)
    hyp_dir.mkdir()
    en = [f"English sentence number {i}." for i in range(n)]
    hi = [f"हिन्दी वाक्य संख्या {i}।" for i in range(n)]
    (raw_dir / "in22-gen" / "in22-gen.en").write_text("\n".join(en) + "\n", encoding="utf-8")
    (raw_dir / "in22-gen" / "in22-gen.hi").write_text("\n".join(hi) + "\n", encoding="utf-8")
    hyp_hi = ["" if i in empty_at else f"हिन्दी अनुवाद {i}।" for i in range(n)]
    hyp_en = ["" if i in empty_at else f"English translation {i}." for i in range(n)]
    (hyp_dir / "in22-gen.en-hi.hyp.txt").write_text("\n".join(hyp_hi) + "\n", encoding="utf-8")
    (hyp_dir / "in22-gen.hi-en.hyp.txt").write_text("\n".join(hyp_en) + "\n", encoding="utf-8")
    return hyp_dir, raw_dir


def fixture_items(tmp, variants=("v2",), **kw):
    hyp_dir, raw_dir = make_fixture(tmp, **kw)
    dirs = list(J.directions({"hi"}, {"en->xx", "xx->en"}))
    items, skipped = J.build_items(hyp_dir, raw_dir, ["in22-gen"], dirs, list(variants), None, 42,
                                   expected={"in22-gen": 6})
    assert not skipped, skipped
    return items


GOOD = {"analysis": "fine", "fluency": 4, "adequacy": 4, "overall": 4, "language_ok": True, "confidence": "high"}


def test_prompt_extraction():
    for k in ("system_v2", "user_v2", "system_v1", "user_v1", "repair"):
        assert PROMPTS[k].strip(), k
    assert PROMPTS["system_v2"].startswith("You are a professional translation reviewer")
    assert '{"analysis": string, "fluency": int' in PROMPTS["system_v2"]  # literal braces kept
    assert "THE REFERENCE is ONE valid human translation" in PROMPTS["system_v2"]
    assert "NO REFERENCE is provided" in PROMPTS["system_v1"]
    assert "{reference}" in PROMPTS["user_v2"] and "{reference}" not in PROMPTS["user_v1"]
    assert PROMPTS["repair"].startswith("Your reply was not valid.")
    assert len(PROMPTS["sha256"]) == 64
    assert len(PROMPTS["few_shot"]) == 3
    assert PROMPTS["few_shot"][0][1].startswith('{"analysis"')
    assert "REFERENCE TRANSLATION" not in J.drop_reference(PROMPTS["few_shot"][0][0])
    assert "TRANSLATION TO SCORE:" in J.drop_reference(PROMPTS["few_shot"][0][0])
    J.check_language_table()


def test_substitution_is_not_format():
    item = {"variant": "v2", "src_lang": "en", "tgt_lang": "mr", "source": "brace { } and {reference} inside",
            "reference": "REF", "hypothesis": "HYP"}
    msgs = J.build_messages(PROMPTS, item, "HYP")
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[-1]["content"]
    assert "Source language: English (Latin script)" in user
    assert "Target language: Marathi (Devanagari script)" in user
    assert "brace { } and {reference} inside" in user  # single pass: not re-substituted
    assert user.endswith("Return the JSON object only.")
    assert "{tgt_lang_name}" not in user and "{hypothesis}" not in user
    assert msgs[0]["content"] == PROMPTS["system_v2"]
    v1 = J.build_messages(PROMPTS, dict(item, variant="v1"), "HYP")[-1]["content"]
    assert "REFERENCE" not in v1
    merged = J.build_messages(PROMPTS, item, "HYP", merge_system=True)
    assert [m["role"] for m in merged] == ["user"]
    assert merged[0]["content"].startswith(PROMPTS["system_v2"] + "\n\n")
    fs = J.build_messages(PROMPTS, dict(item, variant="v1"), "HYP", few_shot=True)
    assert [m["role"] for m in fs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert "REFERENCE" not in fs[1]["content"]


def test_hypothesis_preparation():
    assert J.prepare_hypothesis("a\nb") == ("a b", False)
    long, tr = J.prepare_hypothesis("x" * 2000)
    assert tr and long == "x" * 1500 + J.TRUNC_MARKER
    assert J.prepare_hypothesis("x" * 1500)[1] is False


def test_parse_and_repair():
    p = J.parse_reply(json.dumps(GOOD))
    assert p["ok"] and p["key_order_ok"] and p["recovered_by"] is None and p["analysis_words"] == 1
    assert J.parse_reply("```json\n" + json.dumps(GOOD) + "\n```")["ok"]
    p = J.parse_reply('Sure! ' + json.dumps(dict(GOOD, analysis="has {braces} inside")) + " bye")
    assert p["ok"] and p["recovered_by"] == "brace_scan"
    assert not J.parse_reply("no json here")["ok"]
    assert not J.parse_reply(None)["ok"]
    assert not J.parse_reply(json.dumps(dict(GOOD, fluency="4")))["ok"]      # string int
    assert not J.parse_reply(json.dumps(dict(GOOD, fluency=4.0)))["ok"]      # float
    assert not J.parse_reply(json.dumps(dict(GOOD, fluency=6)))["ok"]
    assert not J.parse_reply(json.dumps(dict(GOOD, fluency=True)))["ok"]
    assert not J.parse_reply(json.dumps(dict(GOOD, language_ok="true")))["ok"]
    assert not J.parse_reply(json.dumps(dict(GOOD, confidence="certain")))["ok"]
    assert not J.parse_reply(json.dumps(dict(GOOD, analysis="")))["ok"]
    extra = dict(GOOD, extra=1)
    assert not J.parse_reply(json.dumps(extra))["ok"]
    missing = {k: v for k, v in GOOD.items() if k != "overall"}
    assert not J.parse_reply(json.dumps(missing))["ok"]
    reordered = {k: GOOD[k] for k in ("fluency", "analysis", "adequacy", "overall", "language_ok", "confidence")}
    assert not J.parse_reply(json.dumps(reordered))["ok"]  # analysis not first: failure
    ok_order = {k: GOOD[k] for k in ("analysis", "adequacy", "fluency", "overall", "language_ok", "confidence")}
    p = J.parse_reply(json.dumps(ok_order))
    assert p["ok"] and p["key_order_ok"] is False  # counted, not a failure
    assert J.parse_reply(json.dumps(dict(GOOD, analysis="w " * 61)))["analysis_too_long"] is True


def test_constraints():
    assert not J.band_violation(4, 4, 4, True)
    assert not J.band_violation(4, 4, 5, True)
    assert J.band_violation(4, 4, 3, True)             # below low
    assert J.band_violation(2, 5, 4, True)              # above low+1
    assert not J.band_violation(2, 5, 3, True)
    assert J.band_violation(5, 2, 3, True)              # adequacy <= 2 => overall <= adequacy
    assert not J.band_violation(5, 2, 2, True)
    assert J.band_violation(5, 5, 5, False)             # language_ok false => overall <= 2
    assert J.band_violation(4, 4, 2, False)             # language_ok false => min(F, A) <= 2 too (else no valid overall exists)
    assert not J.band_violation(4, 2, 2, False)         # wrong language: adequacy lowered, overall 2 is consistent
    assert not J.band_violation(2, 4, 2, False)         # wrong script: fluency lowered
    assert not J.band_violation(1, 1, 1, False)
    assert J.confidence_cap_violation("en", "sat", "high") and not J.confidence_cap_violation("en", "sat", "medium")
    assert not J.confidence_cap_violation("en", "hi", "high")


def test_auto_empty_and_echo_no_call():
    class Boom(J.JudgeBackend):
        def generate_batch(self, batch):
            raise AssertionError("judge must not be called for empty hypotheses")

    with tempfile.TemporaryDirectory() as tmp:
        items = [i for i in fixture_items(tmp) if not i["hypothesis"].strip()]
        assert len(items) == 2
        recs, unwritten = J.judge_chunk(items, Boom(), PROMPTS, CTX)
        assert unwritten == 0 and len(recs) == 2
        for r in recs:
            assert r["status"] == "auto_empty" and r["auto_scored"] is True
            assert (r["fluency"], r["adequacy"], r["overall"]) == (1, 1, 1)
            assert r["language_ok"] is False and r["confidence"] == "high"


def test_echo_pipeline_and_retry_paths():
    with tempfile.TemporaryDirectory() as tmp:
        items = fixture_items(tmp, n=6)
        recs, _ = J.judge_chunk(items, J.EchoBackend(), PROMPTS, CTX)
        assert len(recs) == len(items)
        assert all(r["status"] in ("ok", "auto_empty") and not r["constraint_violation"] for r in recs)
        assert recs == J.judge_chunk(items, J.EchoBackend(), PROMPTS, CTX)[0]  # deterministic
    # Injected-invalid outputs: build many items so every injection slot occurs.
    items = []
    for i in range(600):
        items.append({"test_set": "in22-gen", "direction": "en-hi", "src_lang": "en", "tgt_lang": "hi",
                      "bucket": "en->xx", "line_index": i, "variant": "v2", "source": "s", "reference": "r",
                      "hypothesis": f"hypothesis {i}"})
    recs, _ = J.judge_chunk(items, J.EchoBackend(inject=True), PROMPTS, CTX)
    st = {}
    for r in recs:
        st[r["status"]] = st.get(r["status"], 0) + 1
    assert st.get("failed", 0) > 0 and st.get("ok", 0) > 0, st
    for r in recs:
        if r["status"] == "failed":
            assert r["retried"] and r["fluency"] is None and r["overall"] is None and r["raw_response"]
            assert r["error"]
        else:
            assert r["constraint_violation"] is False  # every violating first answer was repaired
    assert any(r["retried"] and r["status"] == "ok" for r in recs)
    assert any(r["recovered_by"] == "brace_scan" for r in recs)


def test_constraint_violation_kept_when_retry_still_violates():
    class Stubborn(J.JudgeBackend):
        def generate_batch(self, batch):
            bad = dict(GOOD, fluency=2, adequacy=2, overall=4)
            return [J.Reply(json.dumps(bad), 0) for _ in batch]

    item = {"test_set": "in22-gen", "direction": "en-hi", "src_lang": "en", "tgt_lang": "hi", "bucket": "en->xx",
            "line_index": 0, "variant": "v2", "source": "s", "reference": "r", "hypothesis": "h"}
    recs, _ = J.judge_chunk([item], Stubborn(), PROMPTS, CTX)
    r = recs[0]
    assert r["status"] == "ok" and r["retried"] and r["constraint_violation"] is True
    assert r["overall"] == 4  # never silently clipped


def test_transport_failure_left_unwritten():
    class Down(J.JudgeBackend):
        def generate_batch(self, batch):
            return [J.Reply(None, None, "timeout") for _ in batch]

    item = {"test_set": "in22-gen", "direction": "en-hi", "src_lang": "en", "tgt_lang": "hi", "bucket": "en->xx",
            "line_index": 0, "variant": "v2", "source": "s", "reference": "r", "hypothesis": "h"}
    recs, unwritten = J.judge_chunk([item], Down(), PROMPTS, CTX)
    assert recs == [] and unwritten == 1


def test_resume_skip_and_refuse():
    with tempfile.TemporaryDirectory() as tmp:
        items = fixture_items(tmp, variants=("v2", "v1"))
        path = Path(tmp) / "out" / "results.jsonl"
        logs = []
        recs, st = J.run_items(items[:7], J.EchoBackend(), PROMPTS, CTX, path, 4, log=logs.append)
        assert st["n_judged"] == 7 and len(J.load_results(path)) == 7
        counting = []

        class Counting(J.EchoBackend):
            def generate_batch(self, batch):
                counting.append(len(batch))
                return super().generate_batch(batch)

        recs, st = J.run_items(items, Counting(), PROMPTS, CTX, path, 100, log=logs.append)
        assert st["n_skipped_done"] == 7 and st["n_judged"] == len(items) - 7
        assert len(J.load_results(path)) == len(items)
        assert len({J.item_key(r) for r in recs}) == len(items)  # no duplicates
        counting.clear()
        recs, st = J.run_items(items, Counting(), PROMPTS, CTX, path, 100, log=logs.append)
        assert st["n_judged"] == 0 and counting == []  # idempotent: zero judge calls
        for field, val in (("prompt_sha256", "deadbeef"), ("judge_model", "other"), ("seed", 7), ("few_shot", True)):
            try:
                J.run_items(items, J.EchoBackend(), PROMPTS, dict(CTX, **{field: val}), path, 100, log=logs.append)
            except SystemExit as e:
                assert field in str(e)
            else:
                raise AssertionError(f"resume with different {field} was not refused")
        # a torn last line (killed mid-write) is repaired, not fatal
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"broken": ')
        assert len(J.load_results(path)) == len(items)
        # retry-failed: failed records are re-attempted
        recs = J.load_results(path)
        recs[0].update(status="failed", fluency=None, adequacy=None, overall=None, error="x")
        J.atomic_write(path, "".join(json.dumps(r) + "\n" for r in recs))
        recs, st = J.run_items(items, J.EchoBackend(), PROMPTS, CTX, path, 100, retry_failed=True, log=logs.append)
        assert st["n_judged"] == 1 and all(r["status"] != "failed" for r in recs)


def test_summary_and_spearman():
    with tempfile.TemporaryDirectory() as tmp:
        items = fixture_items(tmp, variants=("v2", "v1"))
        recs, _ = J.judge_chunk(items, J.EchoBackend(inject=True), PROMPTS, CTX)
        chrf = {("in22-gen", "en-hi"): 40.0, ("in22-gen", "hi-en"): 50.0}
        s = J.summarize(recs, chrf, {"judge_model": "echo-stub"})
        assert set(s["variants"]) == {"v1", "v2"}
        b = s["variants"]["v2"]
        t = b["total"]
        assert t["n"] == 12 and t["n_scored"] + t["n_failed"] == 12
        assert t["n_auto_empty"] == 2  # one empty line x 2 directions
        assert set(b["by_bucket"]) == {"en->xx", "xx->en"} and set(b["by_test_set"]) == {"in22-gen"}
        assert set(b["by_language"]) == {"hi"} and set(b["by_direction"]) == {"in22-gen/en-hi", "in22-gen/hi-en"}
        assert b["by_direction"]["in22-gen/en-hi"]["chrf++"] == 40.0
        for m in ("fluency", "adequacy", "overall"):
            assert sum(t[f"hist_{m}"].values()) == t["n_scored"]
        # headline mean recomputed by hand (high+medium only, failed excluded)
        head = [r["overall"] for r in recs if r["variant"] == "v2" and r["status"] in ("ok", "auto_empty")
                and r["confidence"] != "low"]
        assert abs(t["mean_overall"] - sum(head) / len(head)) < 1e-9
        lows = [r["overall"] for r in recs if r["variant"] == "v2" and r["status"] == "ok" and r["confidence"] == "low"]
        assert t["n_low_confidence"] == len(lows)
        md = J.summary_markdown(s)
        assert "V2 reference-aided" in md and "Score distribution" in md and "Per language, en->xx" in md
    # spearman on known data
    assert abs(J.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(J.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert J.spearman([1, 2], [1, 2]) is None and J.spearman([1, 1, 1], [1, 2, 3]) is None
    assert J.rank([5, 5, 1]) == [2.5, 2.5, 1.0]
    # a failed item never enters means
    fails = [dict(J.base_record({"test_set": "a", "direction": "en-hi", "src_lang": "en", "tgt_lang": "hi",
                                 "bucket": "en->xx", "line_index": 0, "variant": "v2", "source": "", "reference": "",
                                 "hypothesis": "h"}, CTX), status="failed")]
    a = J.agg(fails)
    assert a["n_failed"] == 1 and a["mean_overall"] is None and a["n_scored"] == 0


def test_sampling_is_paired():
    a = J.sample_indices(1024, "in22-gen", "en-hi", 5, 42)
    assert a == J.sample_indices(1024, "in22-gen", "en-hi", 5, 42) and a == sorted(a) and len(a) == 5
    assert a != J.sample_indices(1024, "in22-gen", "en-ta", 5, 42)
    assert J.sample_indices(10, "x", "y", None, 42) == list(range(10))


def test_length_mismatch_aborts_direction():
    with tempfile.TemporaryDirectory() as tmp:
        hyp_dir, raw_dir = make_fixture(tmp)
        (Path(hyp_dir) / "in22-gen.en-hi.hyp.txt").write_text("only one line\n", encoding="utf-8")
        dirs = list(J.directions({"hi"}, {"en->xx"}))
        items, skipped = J.build_items(hyp_dir, raw_dir, ["in22-gen"], dirs, ["v2"], None, 42,
                                       expected={"in22-gen": 6})
        assert items == [] and len(skipped) == 1


def test_end_to_end_cli_and_sanity():
    with tempfile.TemporaryDirectory() as tmp:
        hyp_dir, raw_dir = make_fixture(tmp)
        root = Path(tmp) / "results" / "fix"
        root.mkdir(parents=True)
        (root / "model_step_1").symlink_to(hyp_dir)
        out = Path(tmp) / "out"
        # the fixture has 6 lines, the real expected count is 1024: the CLI must skip and report exit 2
        common = ["--backend", "echo", "--arm", "fix", "--step", "1", "--results-root", str(Path(tmp) / "results"),
                  "--raw-test-dir", str(raw_dir), "--test-sets", "in22-gen", "--langs", "hi",
                  "--buckets", "en-xx", "xx-en", "--out-dir", str(out)]
        assert J.main(common + ["--variant", "both"]) == 2
        J.EXPECTED_LINES["in22-gen"] = 6
        try:
            assert J.main(common + ["--variant", "both", "--echo-inject"]) == 0
            assert (out / "summary.json").exists() and (out / "summary.md").exists() and (out / "run_meta.json").exists()
            n = len(J.load_results(out / "results.jsonl"))
            assert n == 24, n
            assert J.main(common + ["--variant", "both", "--echo-inject"]) == 0
            assert len(J.load_results(out / "results.jsonl")) == n
            assert J.main(common + ["--sanity", "--variant", "V2"]) == 0
            assert (out / "sanity" / "sanity.md").exists()
            assert len(J.load_results(out / "sanity" / "identity.jsonl")) > 0
        finally:
            J.EXPECTED_LINES["in22-gen"] = 1024


def test_real_files_line_up():
    hyp = J.RESULTS_ROOT / "bpcc-human" / "model_step_11400" / "in22-gen.en-hi.hyp.txt"
    if not hyp.exists() or not (J.RAW_TEST_DIR / "in22-gen" / "in22-gen.en").exists():
        print("  (skipped: real evaluation files not present)")
        return
    hyps, srcs, refs = J.load_direction(hyp.parent, J.RAW_TEST_DIR, "in22-gen", "en", "hi")
    assert len(hyps) == len(srcs) == len(refs) == 1024


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
