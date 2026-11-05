"""Cross-script token matching for code-mixed Hindi.

A reference transcribed in Devanagari ("मोबाइल फोन") and a hypothesis that
writes English loanwords in Latin ("mobile phone") say the same thing, but
plain WER counts two substitutions. This module decides whether a Latin token
and a Devanagari token are plausibly the same word.

The test is deliberately crude: romanise the Devanagari side, reduce both
sides to a consonant skeleton, and compare. मोबाइल -> "mobAila" -> "mbl", and
"mobile" -> "mbl". It will produce some false matches on short words, which is
why skeletons under MIN_SKELETON characters never match, and why the judge
validation step exists.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .normalize import is_latin_token

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit
except ImportError:  # optional dependency
    sanscript = None

MIN_SKELETON = 2

# Collapse consonants that English spellings and Hindi romanisation disagree on.
_FOLDS = [
    (r"ph", "f"),
    (r"bh", "b"),
    (r"kh", "k"),
    (r"gh", "g"),
    (r"th", "t"),
    (r"dh", "d"),
    (r"sh", "s"),
    (r"ch", "c"),
    (r"ck", "k"),
    (r"q", "k"),
    (r"x", "ks"),
    (r"w", "v"),
    (r"z", "j"),
    (r"c(?=[eiy])", "s"),
    (r"c", "k"),
]
_VOWELS = re.compile(r"[aeiouy]")
_NON_ALPHA = re.compile(r"[^a-z]")


def _devanagari_to_latin(tok: str) -> str:
    if sanscript is None:
        raise RuntimeError("pip install indic-transliteration to use translit matching")
    # ITRANS marks long vowels with capitals; lowercase them since the vowels
    # get dropped anyway.
    return _translit(tok, sanscript.DEVANAGARI, sanscript.ITRANS).lower()


@lru_cache(maxsize=100_000)
def skeleton(tok: str) -> str:
    s = tok.lower() if is_latin_token(tok) else _devanagari_to_latin(tok)
    s = _NON_ALPHA.sub("", s)
    for pat, rep in _FOLDS:
        s = re.sub(pat, rep, s)
    s = _VOWELS.sub("", s)
    # Doubled consonants ("summer" vs "समर") shouldn't matter.
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def cross_script_match(a: str, b: str) -> bool:
    """True if a and b are identical, or one is Latin, the other isn't, and
    their consonant skeletons agree."""
    if a == b:
        return True
    if is_latin_token(a) == is_latin_token(b):
        return False
    sa, sb = skeleton(a), skeleton(b)
    return len(sa) >= MIN_SKELETON and sa == sb
