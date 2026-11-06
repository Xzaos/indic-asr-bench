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


def test_two_failures_count_as_errors_and_are_logged_not_cached(tmp_path, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(Judge, "_call", lambda self, prompt, n: calls.append(1))
    j = Judge("fake", base_url="http://unused", cache=str(tmp_path / "c.jsonl"))
    pairs, same = j.judge(*_score())
    assert same == [False] * len(pairs) and len(calls) == 2
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
