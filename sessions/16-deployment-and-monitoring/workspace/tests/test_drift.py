import math

import numpy as np
import pytest
from scipy import stats

from sentiment_service.drift import (
    bin_shares,
    category_shares,
    ks_statistic,
    ks_test,
    label_shift,
    psi,
    psi_from_shares,
    quantile_edges,
    retrain_needed,
)
from sentiment_service.model import load_model
from sentiment_service.monitor import drift_report

rng = np.random.default_rng(0)


def test_ks_statistic_worked_example_and_scipy_agree():
    assert ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) == 0.5
    a, b = rng.normal(0, 1, 500), rng.normal(0.3, 1, 400)
    assert math.isclose(ks_statistic(a, b), stats.ks_2samp(a, b).statistic)
    assert ks_statistic(a, a) == 0.0


def test_ks_test_detects_a_shift_and_not_noise():
    same = ks_test(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    shifted = ks_test(rng.normal(0, 1, 2000), rng.normal(0.5, 1, 2000))
    assert same.pvalue > 0.01 and shifted.pvalue < 1e-6 and shifted.statistic > same.statistic


def test_psi_from_shares_worked_example():
    # (0.6 - 0.5) ln(0.6/0.5) + (0.4 - 0.5) ln(0.4/0.5) = 0.0182 + 0.0223
    assert round(psi_from_shares([0.5, 0.5], [0.6, 0.4]), 4) == 0.0405
    assert psi_from_shares([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0
    assert psi_from_shares([0.5, 0.5], [1.0, 0.0]) > 1  # an empty bin is clipped, not infinite
    with pytest.raises(ValueError):
        psi_from_shares([0.5, 0.5], [1.0])


def test_quantile_bins_hold_equal_shares_of_the_reference():
    ref = rng.normal(0, 1, 10_000)
    shares = bin_shares(ref, quantile_edges(ref, 10))
    assert len(shares) == 10 and np.allclose(shares, 0.1, atol=0.002)


def test_psi_is_small_for_noise_and_large_for_a_shift():
    ref = rng.lognormal(4.5, 1, 5000)
    assert psi(ref, rng.lognormal(4.5, 1, 5000)) < 0.02
    assert psi(ref, rng.lognormal(5.5, 1, 5000)) > 0.25
    assert psi(ref, ref * 1) == pytest.approx(0.0, abs=1e-9)


def test_label_shift_on_the_case_study_shares():
    ref = ["neg"] * 19 + ["neu"] * 8 + ["pos"] * 73   # training years: 19 % negative
    cur = ["neg"] * 26 + ["neu"] * 8 + ["pos"] * 66   # 2022-2023: 26 % negative
    shift = label_shift(ref * 100, cur * 100)
    assert shift.reference == (0.19, 0.08, 0.73) and shift.current == (0.26, 0.08, 0.66)
    assert 0.02 < shift.psi < 0.04
    assert shift.chi2_pvalue < 1e-6  # 10,000 reviews per period: clearly significant
    assert list(category_shares(["a", "b", "b"], ["a", "b", "c"])) == pytest.approx([1 / 3, 2 / 3, 0])


def test_retraining_rule():
    assert retrain_needed(0.70, 0.69) == (False, [])
    decision, reasons = retrain_needed(0.64, 0.69, psi_input=0.05)
    assert decision and "macro-F1" in reasons[0]
    assert retrain_needed(None, 0.69, psi_input=0.30)[0]       # no labels yet, but large input drift
    assert not retrain_needed(None, 0.69, psi_input=0.15)[0]   # moderate drift: watch, do not retrain


def test_drift_report_with_the_fixture_model(model_dir):
    pipe, meta = load_model(model_dir)
    texts = ["Broke after two days.", "Love it, great value.", "It is okay."] * 10
    report = drift_report(pipe, meta, texts, labels=["neg", "pos", "neu"] * 10)
    assert report["model_version"] == "0.0.0-test" and report["n_current"] == 30
    assert set(report["prediction_shares"]["current_predicted"]) == {"neg", "neu", "pos"}
    assert report["macro_f1_recent"] == 1.0 and report["retrain"]["decision"] in (True, False)
    assert "psi" in report["label_shift"]
