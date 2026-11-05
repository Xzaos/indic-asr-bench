"""Text normalisation for scoring Indic ASR output.

Three presets, applied the same way to references and hypotheses:

    raw    whitespace collapse only
    basic  NFC, strip punctuation, lowercase Latin
    indic  basic + zero-width chars, nukta folding, chandrabindu -> anusvara,
           native digits -> ASCII

The point of keeping all three is to see how much of a model's WER comes
from spelling conventions rather than from mishearing.
"""
from __future__ import annotations

import re
import unicodedata

PRESETS = ("raw", "basic", "indic")

_WS = re.compile(r"\s+")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)

# Devanagari letters with a precomposed nukta form. NFC leaves most of these
# decomposed (base + U+093C), so folding is just dropping the nukta.
_NUKTA = "़"
_CHANDRABINDU = "ँ"
_ANUSVARA = "ं"

# Digit blocks for the scripts we care about. Each block is 10 contiguous
# codepoints starting at zero.
_DIGIT_BLOCKS = {
    "devanagari": 0x0966,
    "bengali": 0x09E6,
    "gurmukhi": 0x0A66,
    "gujarati": 0x0AE6,
    "oriya": 0x0B66,
    "tamil": 0x0BE6,
    "telugu": 0x0C66,
    "kannada": 0x0CE6,
    "malayalam": 0x0D66,
}
_DIGITS = {
    start + i: str(i) for start in _DIGIT_BLOCKS.values() for i in range(10)
}


def _strip_punct(text: str) -> str:
    # Unicode category P* covers danda (U+0964), double danda and the usual
    # ASCII marks. Symbols (S*) like % and ₹ are kept since they carry content.
    return "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)


def _lower_latin(text: str) -> str:
    return "".join(ch.lower() if ch.isascii() else ch for ch in text)


def normalize(text: str, preset: str = "basic") -> str:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, expected one of {PRESETS}")
    if preset == "raw":
        return _WS.sub(" ", text).strip()

    text = unicodedata.normalize("NFC", text)
    text = _strip_punct(text)
    text = _lower_latin(text)

    if preset == "indic":
        text = text.translate(_ZERO_WIDTH)
        # NFD so that precomposed nukta letters (e.g. U+0958 क़) split into
        # base + nukta, then drop the nukta and recompose.
        text = unicodedata.normalize("NFD", text).replace(_NUKTA, "")
        text = unicodedata.normalize("NFC", text)
        text = text.replace(_CHANDRABINDU, _ANUSVARA)
        text = text.translate(_DIGITS)

    return _WS.sub(" ", text).strip()


def is_latin_token(tok: str) -> bool:
    return any(ch.isascii() and ch.isalpha() for ch in tok)


def has_digit(tok: str) -> bool:
    return any(ch.isdigit() for ch in tok)
