import csv
import json

from asrbench.cli import main
from asrbench.judge import Judge, llm_errors, parse_verdicts
from asrbench.metrics import score_utterance
from asrbench.report import build
from asrbench.validate import cohen_kappa


def test_judge_caches_and_counts(tmp_path, monkeypatch):
    calls = []

    def fake_call(self, prompt, n):
        calls.append(prompt)
        return [True] + [False] * (n - 1)

    monkeypatch.setattr(Judge, "_call", fake_call)
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    ref, hyp = "मेरा मोबाइल खो गया", "मेरा mobile गया"
    s = score_utterance("u", ref, hyp, "basic")
    pairs, same = j.judge(ref, hyp, s)
    assert len(pairs) == 2 and same == [True, False]
    assert llm_errors(pairs, same) == 1

    # second judge instance reads the cache instead of calling again
    j2 = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    j2.judge(ref, hyp, s)
    assert len(calls) == 1


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_parse_strips_think_and_takes_last_object():
    text = (
        '<think>maybe {"verdicts": [{"i": 1, "same": true}]}</think>\n'
        'draft {"verdicts": [{"i": 1, "same": false}]}\n'
        'final {"verdicts": [{"i": 1, "same": true}, {"i": 3, "same": true}]} done'
    )
    assert parse_verdicts(text, 3) == [True, False, True]


def test_parse_rejects_text_without_verdicts():
    assert parse_verdicts("no json here", 2) is None
    assert parse_verdicts('{"other": 1}', 2) is None
    assert parse_verdicts('<think>{"verdicts": []}</think>', 2) is None


def _score():
    ref, hyp = "मेरा मोबाइल खो गया", "मेरा mobile गया"
    return ref, hyp, score_utterance("u", ref, hyp, "basic")


def test_retry_once_then_succeed(tmp_path, monkeypatch):
    replies = iter([None, [True, False]])
    monkeypatch.setattr(Judge, "_call", lambda self, prompt, n: next(replies))
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    _, same = j.judge(*_score())
    assert same == [True, False]


def test_two_failures_are_unknown_counted_as_errors_and_not_cached(tmp_path, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(Judge, "_call", lambda self, prompt, n: calls.append(1))
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    pairs, same = j.judge(*_score())
    assert same == [None] * len(pairs) and len(calls) == 2
    assert llm_errors(pairs, same) == sum(p['ops'] for p in pairs)
    assert "for u," in caplog.text
    assert not (tmp_path / "c.jsonl").exists()


def test_judge_many_threads_keep_order_and_cache_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(Judge, "_call", lambda self, prompt, n: [True] * n)
    path = tmp_path / "c.jsonl"
    j = Judge("fake", base_url="http://unused", cache=str(path), workers=8)
    items = []
    for k in range(40):
        ref, hyp = f"मेरा मोबाइल खो गया {k}", f"मेरा mobile गया {k}"
        items.append((ref, hyp, score_utterance(f"u{k}", ref, hyp, "basic")))
    out = j.judge_many(items)
    assert len(out) == 40 and all(same == [True] * len(pairs) for pairs, same in out)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40 and len({json.loads(x)["key"] for x in lines}) == 40


def _write_run(tmp_path):
    rows = [{"id": "h1", "lang": "hi", "text": "मेरा मोबाइल खो गया"}, {"id": "t1", "lang": "ta", "text": "வணக்கம் நண்பா"}]
    man = tmp_path / "m.jsonl"
    man.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    hyps = [{"id": "h1", "hyp": "मेरा mobile गया"}, {"id": "t1", "hyp": "வணக்கம்"}]
    (tmp_path / "hyp_sys.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in hyps), encoding="utf-8")
    return man


def test_report_judge_lang_scopes_llm_wer(tmp_path, monkeypatch):
    seen = []

    def fake(self, prompt, n):
        seen.append(prompt)
        return [False] * n

    monkeypatch.setattr(Judge, "_call", fake)
    man = _write_run(tmp_path)
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    text = build(str(man), str(tmp_path), judge=j, n_boot=50, judge_langs={"hi"})
    hi, ta = text.split("## ta")
    assert hi.count("n/a") == 0 and ta.count("| n/a |") == 1
    assert len(seen) == 1 and "மேரா" not in seen[0] and "வணக்கம்" not in seen[0]


def test_label_export_judges_only_sampled_utterances(tmp_path, monkeypatch):
    seen = []

    def fake(self, prompt, n):
        seen.append(prompt)
        return [True] * n

    monkeypatch.setattr(Judge, "_call", fake)
    man = _write_run(tmp_path)
    out = tmp_path / "labels.csv"
    main(["label-export", "--manifest", str(man), "--results", str(tmp_path), "--judge-model", "fake",
          "--judge-base-url", "http://unused", "--n", "1", "--out", str(out)])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["judge_same"] == "1"
    assert len(seen) == 1 and rows[0]["ref"] in seen[0]


def _judge_with_posts(tmp_path, monkeypatch, posts):
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    j.backoff = 0
    calls = []

    def post(prompt):
        calls.append(1)
        r = posts[min(len(calls), len(posts)) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(j, "_post", post)
    return j, calls


def test_timeout_and_transport_errors_are_retried(tmp_path, monkeypatch):
    import httpx

    good = '{"verdicts": [{"i": 1, "same": true}, {"i": 2, "same": false}]}'
    j, calls = _judge_with_posts(tmp_path, monkeypatch, [httpx.ReadTimeout("slow"), httpx.ConnectError("down"), good])
    _, same = j.judge(*_score())
    assert same == [True, False] and len(calls) == 3


def test_persistent_transport_error_gives_unknown_without_crashing(tmp_path, monkeypatch):
    import httpx

    j, calls = _judge_with_posts(tmp_path, monkeypatch, [httpx.ReadTimeout("slow")])
    pairs, same = j.judge(*_score())
    assert same == [None] * len(pairs)
    assert len(calls) == 6


def test_unparsable_reply_is_appended_to_failures_file(tmp_path, monkeypatch):
    j, _ = _judge_with_posts(tmp_path, monkeypatch, ["sorry, cannot do that"])
    j.judge(*_score())
    rows = [json.loads(x) for x in (tmp_path / "judge_failures.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[0]["raw"] == "sorry, cannot do that" and "Reference:" in rows[0]["prompt"]


def test_report_prints_fallback_count(tmp_path, monkeypatch):
    monkeypatch.setattr(Judge, "_call", lambda self, prompt, n: None)
    man = _write_run(tmp_path)
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    text = build(str(man), str(tmp_path), judge=j, n_boot=50, judge_langs={"hi"})
    assert "fell back to unknown (counted as errors) for 1 utterances (sys: 1)" in text


def test_label_export_leaves_judge_same_blank_for_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(Judge, "_call", lambda self, prompt, n: None)
    man = _write_run(tmp_path)
    out = tmp_path / "labels.csv"
    main(["label-export", "--manifest", str(man), "--results", str(tmp_path), "--judge-model", "fake",
          "--judge-base-url", "http://unused", "--n", "10", "--out", str(out)])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows and all(r["judge_same"] == "" for r in rows)


def test_label_score_skips_rows_without_a_judge_verdict(tmp_path):
    from asrbench.validate import score_labels

    path = tmp_path / "l.csv"
    path.write_text("bucket,judge_same,human_same\nlexical,1,1\nlexical,,0\nlexical,0,1\n", encoding="utf-8")
    res = score_labels(str(path))
    assert res["n"] == 2 and res["judge_unknown"] == 1 and res["agreement"] == 0.5


def test_label_export_lang_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(Judge, "_call", lambda self, prompt, n: [True] * n)
    man = _write_run(tmp_path)
    out = tmp_path / "labels.csv"
    main(["label-export", "--manifest", str(man), "--results", str(tmp_path), "--n", "10", "--lang", "hi", "--out", str(out)])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert rows and {r["id"] for r in rows} == {"h1"}


def _fake_http(monkeypatch, statuses, body='{"verdicts": [{"i": 1, "same": true}, {"i": 2, "same": true}]}'):
    import httpx

    calls = []

    def post(url, **kw):
        code = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(code)
        content = {"choices": [{"message": {"content": body}}]} if code == 200 else {"error": "bad key"}
        return httpx.Response(code, json=content, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    return calls


def test_client_error_fails_fast_with_status_and_body(tmp_path, monkeypatch):
    import pytest

    from asrbench.judge import JudgeClientError

    calls = _fake_http(monkeypatch, [401])
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    j.backoff = 0
    with pytest.raises(JudgeClientError, match="401.*bad key"):
        j.judge(*_score())
    assert calls == [401]


def test_429_and_5xx_are_still_retried(tmp_path, monkeypatch):
    calls = _fake_http(monkeypatch, [429, 503, 200])
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    j.backoff = 0
    _, same = j.judge(*_score())
    assert same == [True, True] and calls == [429, 503, 200]
