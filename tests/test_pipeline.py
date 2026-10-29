"""End-to-end checks on the real data (small Monte Carlo for speed)."""
import numpy as np
import pytest

from beefwhatif import analysis
from beefwhatif import nutrition as nu


@pytest.fixture(scope="module")
def R():
    return analysis.run(n_mc=500, n_campus=120)


def test_uk_beef_footprint_in_published_range(R):
    # Between UK farm studies (~30) and Poore & Nemecek's UK-producer mix (~37).
    assert 25 < R["beef_ef"]["uk_attr"] < 45
    assert R["beef_ef"]["uk_conseq"] < R["beef_ef"]["uk_attr"]


def test_equivalent_baskets_match_key_nutrients(R):
    for key in nu.Q2_SCENARIOS:
        bk = R["baskets"][key]
        assert bk.feasible, key
        assert (bk.ratios[list(nu.KEY_TARGETS)] >= 1 - 1e-6).all(), key


def test_whole_food_vegan_cannot_supply_b12(R):
    bk = R["baskets"]["vegan_wholefood"]
    assert bk.shortfall["b12_abs"] == pytest.approx(1.0)


def test_replacement_cuts_emissions(R):
    beef100 = R["per100"].loc["uk_attr", "kg_co2e_per_100g"]
    for key in nu.Q2_SCENARIOS:
        assert R["baskets"][key].ghg < 0.5 * beef100


def test_campus_decomposition_adds_up(R):
    lr = R["campus_lr"]
    parts = lr[["gross", "substitutes_gross", "offcampus", "market_leakage", "substitute_adj"]].sum(axis=1)
    assert np.allclose(parts, lr["net"])
    # market response makes the long-run saving smaller than the accounting claim
    assert (lr["net"] > lr["accounting"]).mean() > 0.9
    # short-run pass-through is lower than long-run
    assert R["campus_sr"]["pass_through_beef"].median() < lr["pass_through_beef"].median()


def test_london_saving_is_plausible(R):
    x = -R["mc_london"][f"london_delta_mt_attr_{analysis.HEADLINE}"].median()
    assert 1.5 < x < 5.0
