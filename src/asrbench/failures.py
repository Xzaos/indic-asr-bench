"""Bucket word errors into failure classes.

Run this on the basic-normalised alignment. The buckets are ordered from
"probably not a real error" to "definitely a real error":

    orthographic   same word once nukta/chandrabindu/digits are folded
    script_mix     same word written in the other script (translit match)
    number_format  one side has digits ("500" vs "पांच सौ")
    split_merge    word boundary disagreement ("कर के" vs "करके")
    near_miss      small character edit distance (spelling, inflection)
    lexical        a different word
    deletion / insertion

Counts per bucket tell you which part of a WER number a better normaliser
would erase and which part needs a better model.
"""
from __future__ import annotations

from collections import Counter

from .align import Alignment, align, edit_distance
from .normalize import has_digit, normalize
from .translit import cross_script_match

NEAR_MISS_RATIO = 0.34


def classify_sub(ref: str, hyp: str) -> str:
    if normalize(ref, "indic") == normalize(hyp, "indic"):
        return "orthographic"
    try:
        if cross_script_match(ref, hyp):
            return "script_mix"
    except RuntimeError:
        pass
    if has_digit(ref) or has_digit(hyp):
        return "number_format"
    d = edit_distance(list(ref), list(hyp))
    if d / max(len(ref), len(hyp), 1) <= NEAR_MISS_RATIO:
        return "near_miss"
    return "lexical"


def _find_merges(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Pairs (i, j) where a[i] + a[i+1] == b[j]."""
    hits, used_a, used_b = [], set(), set()
    for i in range(len(a) - 1):
        if i in used_a or i + 1 in used_a:
            continue
        for j, tok in enumerate(b):
            if j not in used_b and a[i] + a[i + 1] == tok:
                hits.append((i, j))
                used_a.update((i, i + 1))
                used_b.add(j)
                break
    return hits


def _classify_run(run) -> list[tuple[str, str | None, str | None, int]]:
    # Inside a run of consecutive errors the aligner's choice of which token
    # is a deletion and which is a substitution is arbitrary, so look at the
    # run as two token lists and pull out word-boundary splits first.
    R = [op.ref for op in run if op.ref is not None]
    H = [op.hyp for op in run if op.hyp is not None]
    items = []
    for i, j in _find_merges(R, H):  # two ref words written as one
        items.append(("split_merge", f"{R[i]} {R[i + 1]}", H[j], 2))
        R[i] = R[i + 1] = H[j] = None
    R = [t for t in R if t is not None]
    H = [t for t in H if t is not None]
    for j, i in _find_merges(H, R):  # one ref word split in two
        items.append(("split_merge", R[i], f"{H[j]} {H[j + 1]}", 2))
        H[j] = H[j + 1] = R[i] = None
    R = [t for t in R if t is not None]
    H = [t for t in H if t is not None]
    # Re-align what's left: removing a split can expose a word that matches
    # exactly and shouldn't be reported as an error.
    for op in align(R, H).ops:
        if op.kind == "sub":
            items.append((classify_sub(op.ref, op.hyp), op.ref, op.hyp, 1))
        elif op.kind == "del":
            items.append(("deletion", op.ref, None, 1))
        elif op.kind == "ins":
            items.append(("insertion", None, op.hyp, 1))

    # Keep op counts consistent with the alignment so that LLM-WER can never
    # come out above plain WER.
    diff = len(run) - sum(it[3] for it in items)
    if diff and items:
        b, r, h, k = items[-1]
        items[-1] = (b, r, h, max(0, k + diff))
    return items


def classify(al: Alignment) -> list[tuple[str, str | None, str | None, int]]:
    """Returns (bucket, ref, hyp, n_ops) for every error. n_ops sums to
    al.errors."""
    out, run = [], []
    for op in al.ops + [None]:
        if op is not None and op.kind != "eq":
            run.append(op)
            continue
        if run:
            out += _classify_run(run)
            run = []
    return out


def bucket_counts(alignments: list[Alignment]) -> Counter:
    c: Counter = Counter()
    for al in alignments:
        for b, _, _, k in classify(al):
            c[b] += k
    return c
