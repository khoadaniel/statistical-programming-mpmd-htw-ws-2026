"""Drift statistics as pure functions: arrays in, numbers out.

- `ks_statistic`, `ks_test`: two-sample Kolmogorov-Smirnov test for a numeric feature
- `psi`: population stability index over quantile bins of the reference sample
- `psi_from_shares`, `category_shares`, `label_shift`: the same idea for class shares
- `retrain_needed`: a written retraining rule
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

PSI_STABLE, PSI_LARGE = 0.10, 0.25  # common rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 large


def ks_statistic(reference: Sequence[float], current: Sequence[float]) -> float:
    """Largest vertical distance between the two empirical distribution functions.

    >>> ks_statistic([1, 2, 3, 4], [3, 4, 5, 6])
    0.5
    """
    a, b = np.sort(np.asarray(reference, float)), np.sort(np.asarray(current, float))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


@dataclass(frozen=True)
class KSResult:
    statistic: float
    pvalue: float


def ks_test(reference: Sequence[float], current: Sequence[float]) -> KSResult:
    """KS statistic and p-value (scipy). With large samples, tiny differences become significant:
    judge the statistic (an effect size between 0 and 1), not only the p-value."""
    res = stats.ks_2samp(reference, current)
    return KSResult(float(res.statistic), float(res.pvalue))


def psi_from_shares(p: Sequence[float], q: Sequence[float], eps: float = 1e-4) -> float:
    """PSI = sum over bins of (q - p) * ln(q / p); p = reference shares, q = current shares.

    >>> round(psi_from_shares([0.5, 0.5], [0.6, 0.4]), 4)
    0.0405
    """
    p = np.clip(np.asarray(p, float), eps, None)
    q = np.clip(np.asarray(q, float), eps, None)
    if p.shape != q.shape:
        raise ValueError("p and q need the same number of bins")
    return float(np.sum((q - p) * np.log(q / p)))


def quantile_edges(reference: Sequence[float], bins: int = 10) -> np.ndarray:
    """Inner bin edges at the quantiles of the reference: every bin holds about 1/bins of it."""
    return np.unique(np.quantile(np.asarray(reference, float), np.linspace(0, 1, bins + 1))[1:-1])


def bin_shares(values: Sequence[float], edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, np.asarray(values, float), side="right")
    return np.bincount(idx, minlength=len(edges) + 1) / len(values)


def psi(reference: Sequence[float], current: Sequence[float], bins: int = 10) -> float:
    """Population stability index of a numeric feature, over the deciles of the reference sample."""
    edges = quantile_edges(reference, bins)
    return psi_from_shares(bin_shares(reference, edges), bin_shares(current, edges))


def category_shares(labels: Sequence[str], categories: Sequence[str]) -> np.ndarray:
    labels = list(labels)
    if not labels:
        raise ValueError("no labels")
    return np.array([labels.count(c) / len(labels) for c in categories])


@dataclass(frozen=True)
class LabelShift:
    categories: tuple[str, ...]
    reference: tuple[float, ...]
    current: tuple[float, ...]
    psi: float
    chi2_pvalue: float

    def as_dict(self) -> dict:
        return {"categories": list(self.categories), "reference": list(self.reference),
                "current": list(self.current), "psi": self.psi, "chi2_pvalue": self.chi2_pvalue}


def label_shift(reference: Sequence[str], current: Sequence[str],
                categories: Sequence[str] = ("neg", "neu", "pos")) -> LabelShift:
    """Compare class shares (true labels or predicted labels) of two periods."""
    p, q = category_shares(reference, categories), category_shares(current, categories)
    counts = np.array([p * len(reference), q * len(current)]).round()
    pvalue = float(stats.chi2_contingency(counts[:, counts.sum(axis=0) > 0]).pvalue)
    return LabelShift(tuple(categories), tuple(np.round(p, 4)), tuple(np.round(q, 4)),
                      round(psi_from_shares(p, q), 4), pvalue)


def retrain_needed(f1_recent: float | None, f1_reference: float, psi_input: float = 0.0,
                   tolerance: float = 0.03, psi_threshold: float = PSI_LARGE) -> tuple[bool, list[str]]:
    """A written retraining rule: retrain if macro-F1 on recently labelled data drops by more than
    `tolerance`, or if an input feature shifts strongly. Returns the decision and its reasons."""
    reasons = []
    if f1_recent is not None and f1_recent < f1_reference - tolerance:
        reasons.append(f"macro-F1 {f1_recent:.3f} < {f1_reference:.3f} - {tolerance}")
    if psi_input > psi_threshold:
        reasons.append(f"input PSI {psi_input:.3f} > {psi_threshold}")
    return bool(reasons), reasons


def oov_rate(texts: Sequence[str], vocabulary: set[str]) -> float:
    """Share of words in `texts` that are not in the model's vocabulary (out-of-vocabulary rate).

    TODO (exercise 4): lower-case each text, split it into words with re.findall(r"(?u)\\b\\w\\w+\\b", text)
    (the default token pattern of TfidfVectorizer), and return the share of words not in `vocabulary`.
    Return 0.0 when there are no words. Test: tests/test_exercises.py.
    """
    raise NotImplementedError("exercise 4: out-of-vocabulary rate")
