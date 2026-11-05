"""Corpus-level WER and CER under each normalisation preset.

Scores are corpus level (total errors / total reference words), not the mean
of per-utterance WER. Short utterances would otherwise dominate the average.
"""
from __future__ import annotations

from dataclasses import dataclass

from .align import Alignment, align
from .normalize import normalize
from .translit import cross_script_match


@dataclass
class UttScore:
    id: str
    ref_words: int
    word_errors: int
    ref_chars: int
    char_errors: int
    alignment: Alignment


def score_utterance(uid: str, ref: str, hyp: str, preset: str = "basic", translit: bool = False) -> UttScore:
    r, h = normalize(ref, preset), normalize(hyp, preset)
    match = cross_script_match if translit else (lambda a, b: a == b)
    words = align(r.split(), h.split(), match)
    # CER on codepoints with spaces removed, so word-boundary disagreements
    # (a common Hindi compound issue) don't count twice.
    rc, hc = list(r.replace(" ", "")), list(h.replace(" ", ""))
    chars = align(rc, hc)
    return UttScore(uid, words.n_ref, words.errors, len(rc), chars.errors, words)


def corpus_rate(scores: list[UttScore], unit: str = "word") -> float:
    if unit == "word":
        num, den = sum(s.word_errors for s in scores), sum(s.ref_words for s in scores)
    else:
        num, den = sum(s.char_errors for s in scores), sum(s.ref_chars for s in scores)
    return num / den if den else float("nan")


SCORING_MODES = {
    # name: (preset, translit)
    "raw": ("raw", False),
    "basic": ("basic", False),
    "indic": ("indic", False),
    "indic+translit": ("indic", True),
}


def score_all(refs: dict[str, str], hyps: dict[str, str], mode: str) -> list[UttScore]:
    preset, translit = SCORING_MODES[mode]
    missing = set(refs) - set(hyps)
    if missing:
        # An empty hypothesis is scored as all deletions rather than silently
        # dropped. Dropping failed utterances flatters the model.
        hyps = {**hyps, **dict.fromkeys(missing, "")}
    return [score_utterance(uid, refs[uid], hyps[uid], preset, translit) for uid in sorted(refs)]
