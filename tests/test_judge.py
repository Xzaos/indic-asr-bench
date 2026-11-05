from asrbench.judge import Judge, llm_errors
from asrbench.metrics import score_utterance
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
