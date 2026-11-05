"""Levenshtein alignment over tokens with a pluggable equality test.

jiwer would give the same counts for plain WER, but I need the aligned pairs
themselves (to bucket errors) and a custom match function (for the
transliteration-aware score), so the DP lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

Match = Callable[[str, str], bool]


@dataclass
class Op:
    kind: str  # "eq", "sub", "del", "ins"
    ref: str | None
    hyp: str | None


@dataclass
class Alignment:
    ops: list[Op] = field(default_factory=list)

    @property
    def n_ref(self) -> int:
        return sum(op.kind != "ins" for op in self.ops)

    def count(self, kind: str) -> int:
        return sum(op.kind == kind for op in self.ops)

    @property
    def errors(self) -> int:
        return self.count("sub") + self.count("del") + self.count("ins")


def _exact(a: str, b: str) -> bool:
    return a == b


def align(ref: Sequence[str], hyp: Sequence[str], match: Match = _exact) -> Alignment:
    n, m = len(ref), len(hyp)
    # cost[i][j] = edit distance between ref[:i] and hyp[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j

    same = [[match(ref[i], hyp[j]) for j in range(m)] for i in range(n)]
    for i in range(1, n + 1):
        row, prev = cost[i], cost[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (0 if same[i - 1][j - 1] else 1)
            row[j] = min(diag, prev[j] + 1, row[j - 1] + 1)

    # Backtrace. Ties go to diagonal first so that substitutions are preferred
    # over a del+ins pair, which keeps the error buckets readable.
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            step = 0 if same[i - 1][j - 1] else 1
            if cost[i][j] == cost[i - 1][j - 1] + step:
                ops.append(Op("eq" if step == 0 else "sub", ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(Op("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(Op("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return Alignment(ops)


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    return align(a, b).errors
