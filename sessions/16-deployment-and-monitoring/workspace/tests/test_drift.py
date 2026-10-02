import math

import numpy as np
import pytest
from scipy import stats

from tariff_service.drift import (
    bin_shares,
    category_shares,
    ks_statistic,
    ks_test,
    label_shift,
    psi,
    psi_categories,
    psi_from_shares,
    quantile_edges,
    retrain_needed,
    unseen_share,
)
from tariff_service.model import load_model
from tariff_service.monitor import drift_report

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
    ref = rng.lognormal(6.3, 0.8, 5000)                 # description lengths, about 550 characters
    assert psi(ref, rng.lognormal(6.3, 0.8, 5000)) < 0.02
    assert psi(ref, rng.lognormal(7.3, 0.8, 5000)) > 0.25
    assert psi(ref, ref * 1) == pytest.approx(0.0, abs=1e-9)


def test_country_shift_after_brexit():
    # training years: GB issued about 4 % of the decisions; test years: none
    reference = {"DE": 0.56, "FR": 0.16, "GB": 0.04, "other": 0.24}
    current = ["DE"] * 59 + ["FR"] * 16 + ["other"] * 25
    value = psi_categories(reference, current)
    assert 0.1 < value < 0.4                            # a vanished category weighs heavily
    assert psi_categories({"DE": 0.5, "FR": 0.5}, ["DE", "FR"] * 50) == pytest.approx(0.0, abs=1e-9)


def test_label_shift_on_chapters():
    ref = ["85"] * 15 + ["39"] * 10 + ["64"] * 5 + ["other"] * 70
    cur = ["85"] * 12 + ["39"] * 10 + ["64"] * 7 + ["other"] * 71
    shift = label_shift(ref * 100, cur * 100)
    assert shift.categories == ("39", "64", "85", "other")
    assert shift.reference == (0.1, 0.05, 0.15, 0.7) and shift.current == (0.1, 0.07, 0.12, 0.71)
    assert 0.005 < shift.psi < 0.02
    assert shift.chi2_pvalue < 1e-3                       # 10,000 decisions per period: significant, but small
    assert list(category_shares(["a", "b", "b"], ["a", "b", "c"])) == pytest.approx([1 / 3, 2 / 3, 0])
    assert set(shift.as_dict(top=2)["largest_changes"]) == {"85", "64"}


def test_unseen_share():
    assert unseen_share({"6403", "6404"}, ["6403", "8524", "6404", "6404"]) == 0.25
    assert unseen_share({"6403"}, []) == 0.0


def test_retraining_rule():
    assert retrain_needed(0.78, 0.77) == (False, [])
    decision, reasons = retrain_needed(0.70, 0.77, psi_input=0.05)
    assert decision and "accuracy" in reasons[0]
    assert retrain_needed(None, 0.77, psi_input=0.30)[0]       # no labels yet, but large input drift
    assert not retrain_needed(None, 0.77, psi_input=0.15)[0]   # moderate drift: watch, do not retrain


def test_drift_report_with_the_fixture_model(model_dir):
    pipe, meta = load_model(model_dir)
    texts = ["Stiefel aus Leder", "Spielzeugauto aus Kunststoff", "Schutzmaske aus Vliesstoff"] * 10
    report = drift_report(pipe, meta, texts, labels=["6403", "9503", "6307"] * 10)
    assert report["model_version"] == "0.0.0-test" and report["n_current"] == 30
    assert report["accuracy_recent"] == 1.0 and report["unseen_headings_share"] == 0.0
    assert "psi" in report["label_shift_chapters"] and report["retrain"]["decision"] in (True, False)
