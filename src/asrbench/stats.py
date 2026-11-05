"""Bootstrap intervals for corpus WER, and a paired test between two systems.

Resampling is over utterances (the unit that was actually sampled), and the
statistic is recomputed as sum(errors) / sum(ref words) on each resample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    lo: float
    hi: float

    def fmt(self, pct: bool = True) -> str:
        k = 100 if pct else 1
        return f"{self.point * k:.2f} [{self.lo * k:.2f}, {self.hi * k:.2f}]"


def _ratio(err: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return err[idx].sum(axis=1) / den[idx].sum(axis=1)


def bootstrap_ci(errors, ref_len, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Interval:
    err, den = np.asarray(errors, float), np.asarray(ref_len, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    boots = _ratio(err, den, idx)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Interval(err.sum() / den.sum(), float(lo), float(hi))


@dataclass
class PairedResult:
    diff: Interval  # WER(a) - WER(b)
    p_value: float


def paired_bootstrap(err_a, err_b, ref_len, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> PairedResult:
    """Same resampled utterances for both systems, so per-utterance difficulty
    cancels out. Two-sided p-value from the share of resamples whose sign
    disagrees with the observed difference."""
    a, b, den = (np.asarray(x, float) for x in (err_a, err_b, ref_len))
    if not (len(a) == len(b) == len(den)):
        raise ValueError("systems must be scored on the same utterances")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = _ratio(a, den, idx) - _ratio(b, den, idx)
    observed = (a.sum() - b.sum()) / den.sum()
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    if observed == 0:
        p = 1.0
    else:
        p = min(1.0, 2 * float(np.mean(np.sign(d) != np.sign(observed))))
    return PairedResult(Interval(float(observed), float(lo), float(hi)), p)
