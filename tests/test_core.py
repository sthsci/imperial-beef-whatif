"""Unit tests for the analytical building blocks (independent of the data files)."""
import numpy as np
import pandas as pd
import pytest

from beefwhatif import economics as econ
from beefwhatif import emissions as em
from beefwhatif import nutrition as nu
from beefwhatif import welfare as wf


# ------------------------------------------------------------------ economics

def test_pass_through_limits():
    assert econ.pass_through(1.0, -1.0) == pytest.approx(0.5)
    assert econ.pass_through(1e9, -0.5) == pytest.approx(1.0)  # perfectly elastic supply
    assert econ.pass_through(0.0, -0.5) == pytest.approx(0.0)  # fixed supply: full leakage


def test_edm_single_market_matches_pass_through():
    Q = np.array([1000.0])
    eta, eps = np.array([0.8]), np.array([[-0.6]])
    dD = np.array([-10.0])
    dlnP, dQ = econ.edm(Q, eta, eps, dD)
    assert dQ[0] == pytest.approx(-10.0 * 0.8 / (0.8 + 0.6))
    assert dlnP[0] < 0  # price falls when demand falls


def test_edm_cross_price_substitution_offsets():
    # Beef demand falls; with substitutes, other consumers' beef demand rises less
    # once poultry price also changes; check market clearing holds.
    Q = np.array([100.0, 200.0])
    eta = np.array([0.5, 2.0])
    E = np.array([[-0.8, 0.1], [0.05, -0.6]])
    dD = np.array([-1.0, 0.5])
    dlnP, dQ = econ.edm(Q, eta, E, dD)
    demand = Q * (E @ dlnP) + dD
    assert np.allclose(demand, dQ)


# ------------------------------------------------------------------ welfare

def test_nested_logit_cv_reduces_to_mnl():
    alpha, p_b, p_m = 0.5, 0.1, 0.5
    cv_nested = wf.cv_per_meal(alpha, p_b, p_m, 1.0)
    assert cv_nested == pytest.approx(np.log(1 - p_b) / alpha)


def test_nested_logit_cv_vanishes_for_perfect_substitutes():
    assert wf.cv_per_meal(0.5, 0.1, 0.5, 1e-9) == pytest.approx(0.0, abs=1e-6)


def test_alpha_calibration_mnl():
    # MNL: eps = -alpha p (1 - P)
    a = wf.alpha_from_elasticity(-1.5, 6.0, 0.1, 0.5, 1.0)
    assert a == pytest.approx(1.5 / (6.0 * 0.9))


# ------------------------------------------------------------------ emissions

def test_gwp_star_abrupt_removal():
    d = em.gwp_star_delta(1.0, 40, gwp100_ch4=27.0)
    assert d[0] == pytest.approx(-27.0 * 4.53)
    assert d[25] == pytest.approx(-27.0 * (4.53 - 4.25))


def test_gwp_star_cumulative_converges_to_gwp100_long_run():
    d = em.gwp_star_delta(1.0, 2000, gwp100_ch4=27.0)
    gwp100 = -27.0 * np.arange(1, 2001)
    # after many decades the cumulative ratio approaches (4.53-4.25)=0.28 per yr
    ratio = np.cumsum(d)[-1] / gwp100[-1]
    assert 0.25 < ratio < 0.35


def test_beef_flows_chain():
    f = em.beef_flows(10.0, 0.7, 0.1, 0.75)
    assert f.retail == pytest.approx(7.0)
    assert f.eaten == pytest.approx(7.0 * 0.9 * 0.75)


# ------------------------------------------------------------------ nutrition

def _toy_foods():
    cols = dict(unit_mass_g=100.0, satfat_g=1.0, sodium_mg=10.0, fibre_g=0.0, ruminant=0,
                diaas=1.0, haem_frac=0.0, zinc_class="animal", b12_class="none")
    rows = {
        "beef_mince": dict(cols, diet="meat", ruminant=1, energy_kcal=200, a=10.0, b=4.0),
        "x": dict(cols, diet="plant", energy_kcal=100, a=10.0, b=0.0),
        "y": dict(cols, diet="plant", energy_kcal=100, a=0.0, b=8.0),
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["cap_g"] = 300.0
    return df


def test_lp_matches_targets_with_two_foods():
    foods = _toy_foods()
    ghg = pd.Series({"beef_mince": 5.0, "x": 0.1, "y": 0.2})
    bk = nu.optimise_basket(foods, ghg, ["x", "y"], targets={"a": "A", "b": "B"},
                            energy_band=(0.5, 1.5), sodium_allow_mg=1e3, satfat_max_ratio=10)
    assert bk.feasible
    assert bk.units["x"] == pytest.approx(1.0, rel=1e-6)
    assert bk.units["y"] == pytest.approx(0.5, rel=1e-6)
    assert bk.ghg == pytest.approx(0.1 + 0.1)


def test_lp_reports_shortfall_when_infeasible():
    foods = _toy_foods()
    ghg = pd.Series({"beef_mince": 5.0, "x": 0.1, "y": 0.2})
    bk = nu.optimise_basket(foods, ghg, ["x"], targets={"a": "A", "b": "B"},
                            energy_band=(0.5, 1.5), sodium_allow_mg=1e3, satfat_max_ratio=10)
    assert not bk.feasible
    assert bk.shortfall["b"] == pytest.approx(1.0)


def test_single_food_equivalents():
    foods = _toy_foods()
    eq = nu.single_food_equivalents(foods, {"a": "A", "b": "B"})
    assert eq.loc["x", "a"] == pytest.approx(100.0)
    assert np.isinf(eq.loc["x", "b"])
    assert eq.loc["y", "b"] == pytest.approx(50.0)
