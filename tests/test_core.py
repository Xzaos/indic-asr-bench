import numpy as np
import pytest

from asrbench.align import align
from asrbench.failures import classify
from asrbench.metrics import corpus_rate, score_all, score_utterance
from asrbench.normalize import normalize
from asrbench.stats import bootstrap_ci, paired_bootstrap
from asrbench.translit import cross_script_match, skeleton


def test_align_counts():
    al = align("a b c d".split(), "a x c".split())
    assert (al.count("sub"), al.count("del"), al.count("ins")) == (1, 1, 0)
    assert al.errors == 2 and al.n_ref == 4


def test_align_empty_hyp_is_all_deletions():
    al = align("a b c".split(), [])
    assert al.count("del") == 3


def test_normalize_presets():
    s = "नमस्ते, दुनिया।  Hello!"
    assert normalize(s, "raw") == "नमस्ते, दुनिया। Hello!"
    assert normalize(s, "basic") == "नमस्ते दुनिया hello"


def test_indic_folds_nukta_chandrabindu_digits():
    # precomposed क़ (U+0958) and decomposed क + nukta both fold to क
    assert normalize("क़ानून", "indic") == normalize("कानून", "indic")
    assert normalize("क़", "indic") == "क"
    assert normalize("हँसना", "indic") == normalize("हंसना", "indic")
    assert normalize("५००", "indic") == "500"
    assert normalize("कल‍म", "indic") == "कलम"


def test_skeleton_and_cross_script():
    assert skeleton("mobile") == skeleton("मोबाइल")
    assert cross_script_match("phone", "फोन")
    assert cross_script_match("school", "स्कूल")
    assert not cross_script_match("फोन", "फ़ोन")  # same script: not this function's job
    assert not cross_script_match("a", "अ")  # skeleton too short


def test_translit_mode_lowers_wer_on_code_mix():
    ref, hyp = "मेरा मोबाइल फोन खो गया", "मेरा mobile phone खो गया"
    plain = score_utterance("u", ref, hyp, "indic", translit=False)
    tl = score_utterance("u", ref, hyp, "indic", translit=True)
    assert plain.word_errors == 2 and tl.word_errors == 0


def test_missing_hyp_counts_as_deletions():
    sc = score_all({"a": "x y", "b": "z"}, {"a": "x y"}, "basic")
    assert corpus_rate(sc) == pytest.approx(1 / 3)


def test_buckets():
    al = score_utterance("u", "मैं कर के आया 500 रुपये", "मैं करके आया पांच सौ रूपए", "basic").alignment
    items = classify(al)
    buckets = [b for b, *_ in items]
    assert "split_merge" in buckets
    assert "number_format" in buckets
    assert sum(k for *_, k in items) == al.errors


def test_bucket_ops_never_exceed_errors():
    al = score_utterance("u", "a b c", "ab x y z", "basic").alignment
    assert sum(k for *_, k in classify(al)) == al.errors


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    n = rng.integers(5, 20, 200)
    e = rng.binomial(n, 0.2)
    ci = bootstrap_ci(e, n, n_boot=500)
    assert ci.lo <= ci.point <= ci.hi
    assert 0.15 < ci.point < 0.25


def test_paired_bootstrap_detects_real_gap():
    rng = np.random.default_rng(2)
    n = rng.integers(5, 20, 300)
    a = rng.binomial(n, 0.30)
    b = rng.binomial(n, 0.20)
    res = paired_bootstrap(a, b, n, n_boot=1000)
    assert res.diff.point > 0 and res.diff.lo > 0 and res.p_value < 0.01


def test_paired_bootstrap_identical_systems():
    n = np.full(50, 10)
    e = np.full(50, 2)
    assert paired_bootstrap(e, e, n, n_boot=200).p_value == 1.0
