"""Greenhouse-gas accounting for beef and its nutritionally equivalent replacements.

Emission factors (EFs) in ``data/emission_factors.csv`` are kg CO2e per kg of
food *as purchased* (raw meat, dry pulses, etc.) on a cradle-to-retail basis.
They are converted to kg CO2e per LP unit *as eaten* with the cooking yield
(kg as eaten per kg as purchased) and the household waste fraction.

Beef is treated separately: its EF is built from the origin x production-system
mix of beef consumed in the UK (``data/beef_supply.csv``), converted from farm
gate per kg carcass weight to retail per kg boneless meat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .params import Params, Z90, load_csv

GWP_SETS = {
    # (CH4, N2O) GWP values for each metric; CO2 = 1 by definition.
    "AR5cc": (34.0, 298.0),  # AR5 GWP100 incl. climate-carbon feedbacks (Poore & Nemecek 2018)
    "AR5": (28.0, 265.0),
    "AR4": (25.0, 298.0),
    "AR6": (27.0, 273.0),  # AR6 GWP100, non-fossil CH4
    "AR6_GWP20": (79.7, 273.0),  # AR6 GWP20, non-fossil CH4
}


def load_efs() -> pd.DataFrame:
    return load_csv("emission_factors.csv", index="food_id")


def load_beef_supply() -> pd.DataFrame:
    return load_csv("beef_supply.csv", index="segment")


def lognormal(rng: np.random.Generator, median, low, high, n: int) -> np.ndarray:
    """Log-normal draws with given median and 90% interval [low, high]."""
    sigma = (np.log(high) - np.log(low)) / (2 * Z90)
    return median * np.exp(sigma * rng.standard_normal(n))


def ef_draws(row: pd.Series, rng: np.random.Generator, n: int) -> np.ndarray:
    """Monte Carlo EF draws for one food: log-normal through its median and 5th/95th percentiles."""
    return lognormal(rng, float(row["ef_median"]), float(row["ef_low"]), float(row["ef_high"]), n)


# --------------------------------------------------------------------------- beef


def beef_segment_ef(supply: pd.DataFrame, retail_yield: float, post_farm: float,
                    consequential: bool = False, column: str = "ef_cwe") -> pd.Series:
    """kg CO2e per kg boneless retail beef for each origin x system segment.

    Farm-gate EFs per kg carcass weight are divided by the carcass-to-retail
    yield and a post-farm-gate adder (processing, packaging, transport, retail)
    is added. With ``consequential=True`` the farm-gate EF is multiplied by the
    segment's ``avoidable_frac`` - the share of emissions that disappears if
    the beef is no longer demanded (for dairy-origin beef the cow's share
    is re-allocated to milk rather than avoided).
    """
    farm = supply[column].astype(float)
    if consequential:
        farm = farm * supply["avoidable_frac"].astype(float)
    return farm / retail_yield + post_farm


def beef_mix_ef(supply: pd.DataFrame, retail_yield: float, post_farm: float,
                consequential: bool = False) -> float:
    seg = beef_segment_ef(supply, retail_yield, post_farm, consequential)
    return float((supply["share"] * seg).sum() / supply["share"].sum())


def beef_mix_gas_shares(supply: pd.DataFrame, retail_yield: float, post_farm: float) -> pd.Series:
    """Emission-weighted CO2/CH4/N2O shares of the UK beef mix at retail (GWP100 basis).

    Post-farm-gate emissions (processing, transport, retail) are counted as CO2.
    """
    s = supply["share"] / supply["share"].sum()
    farm = s * supply["ef_cwe"] / retail_yield
    total = farm.sum() + post_farm
    out = {g: float((farm * supply[f"f_{g}"]).sum() / total) for g in ("co2", "ch4", "n2o")}
    out["co2"] += post_farm / total
    return pd.Series(out)


def beef_mix_samples(supply: pd.DataFrame, retail_yield: np.ndarray, post_farm: np.ndarray,
                     rng: np.random.Generator, consequential: bool = False,
                     share_concentration: float = 200.0) -> np.ndarray:
    """Monte Carlo draws of the UK-mix beef EF (kg CO2e per kg retail).

    Segment farm-gate EFs are log-normal (median ``ef_cwe``, 90% interval
    ``ef_cwe_low``-``ef_cwe_high``); segment shares are Dirichlet around the
    central shares; avoidable fractions are uniform on their stated range.
    """
    n = len(retail_yield)
    shares = rng.dirichlet(share_concentration * supply["share"].to_numpy(float), n)
    total = np.zeros(n)
    for j, (_, seg) in enumerate(supply.iterrows()):
        farm = lognormal(rng, seg["ef_cwe"], seg["ef_cwe_low"], seg["ef_cwe_high"], n)
        if consequential:
            farm = farm * rng.uniform(seg["avoidable_low"], seg["avoidable_high"], n)
        total += shares[:, j] * (farm / retail_yield + post_farm)
    return total


# ------------------------------------------------------------- alternative foods


def ghg_per_unit(foods: pd.DataFrame, efs: pd.DataFrame, waste_frac: float,
                 column: str = "ef_central", beef_ef: float | None = None,
                 ref: str = "beef_mince") -> pd.Series:
    """kg CO2e per LP unit as eaten (100 g of food, or one tablet).

    unit GHG = (unit mass [kg] / cooking yield) / (1 - waste) * EF [kg CO2e/kg purchased]

    Supplements are not cooked or wasted. If ``beef_ef`` is given it overrides
    the beef EF (kg CO2e per kg retail, as purchased).
    """
    ef = efs.loc[foods.index, column].astype(float).copy()
    if beef_ef is not None:
        ef.loc[ref] = beef_ef
    yld = efs.loc[foods.index, "cook_yield"].astype(float)
    waste = np.where(foods["diet"] == "supplement", 0.0, waste_frac)
    return (foods["unit_mass_g"] / 1000.0) / yld / (1.0 - waste) * ef


def gwp_factor(efs: pd.DataFrame, metric: str) -> pd.Series:
    """Multiplier converting each food's EF from its source metric to ``metric``."""
    out = {}
    for fid, row in efs.iterrows():
        ch4_src, n2o_src = GWP_SETS[row["gwp_src"]]
        ch4, n2o = GWP_SETS[metric]
        out[fid] = row["f_co2"] + row["f_ch4"] * ch4 / ch4_src + row["f_n2o"] * n2o / n2o_src
    return pd.Series(out)


def gwp_factor_shares(shares: pd.Series, src: str, metric: str) -> float:
    ch4_src, n2o_src = GWP_SETS[src]
    ch4, n2o = GWP_SETS[metric]
    return float(shares["co2"] + shares["ch4"] * ch4 / ch4_src + shares["n2o"] * n2o / n2o_src)


# ---------------------------------------------------------------- per-capita flows


@dataclass
class BeefFlows:
    """Per-capita beef flows, kg per person per year."""

    cwe: float  # carcass-weight equivalent (supply balance)
    retail: float  # boneless retail weight purchased (incl. out-of-home)
    eaten: float  # cooked, as eaten

    @property
    def eaten_g_day(self) -> float:
        return self.eaten * 1000 / 365.25


def beef_flows(cwe, retail_yield, waste_frac, cook_yield) -> BeefFlows:
    retail = cwe * retail_yield
    eaten = retail * (1 - waste_frac) * cook_yield
    return BeefFlows(cwe=cwe, retail=retail, eaten=eaten)


# --------------------------------------------------------------------------- GWP*


def gwp_star_delta(e0_ch4: float, years: int, gwp100_ch4: float = 27.0,
                   phase_out_years: int = 0, a: float = 4.53, b: float = 4.25,
                   lag: int = 20) -> np.ndarray:
    """Change in warming-equivalent emissions (t CO2-we/yr) when a constant
    methane source ``e0_ch4`` (t CH4/yr) is removed, relative to a baseline in
    which it continues unchanged (Smith et al. 2021 GWP* formulation):

        E*_t = GWP100 * (a * E_t - b * E_{t-lag})

    The source is removed abruptly at t = 0 (``phase_out_years=0``) or linearly
    over ``phase_out_years``. Emissions before t = 0 equal ``e0_ch4``.
    """
    t = np.arange(years)
    if phase_out_years > 0:
        path = e0_ch4 * np.clip(1 - (t + 1) / phase_out_years, 0, 1)
    else:
        path = np.zeros(years)
    lagged = np.where(t - lag >= 0, path[np.clip(t - lag, 0, None)], e0_ch4)
    return gwp100_ch4 * (a * (path - e0_ch4) - b * (lagged - e0_ch4))


def summarise(x: np.ndarray, q=(0.05, 0.5, 0.95)) -> dict[str, float]:
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), **{f"p{int(100 * qq):02d}": float(np.quantile(x, qq)) for qq in q}}


def per_capita_inputs(params: Params, rng: np.random.Generator | None = None, n: int = 1):
    """Central values (rng=None) or Monte Carlo draws of the per-capita inputs."""
    keys = {
        "cwe": "consumption.beef_cwe_kg_per_cap",
        "retail_yield": "conversion.beef_retail_yield",
        "cook_yield": "conversion.beef_cook_yield",
        "waste": "conversion.household_waste_frac",
        "post_farm": "emissions.beef_post_farm_kg_per_kg",
    }
    if rng is None:
        return {k: params.v(p) for k, p in keys.items()}
    return {k: params.sample(p, rng, n) for k, p in keys.items()}
