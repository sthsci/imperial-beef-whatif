"""Market response to a campus beef ban: an equilibrium displacement model (EDM).

A campus that stops selling beef shifts the demand curves of several food
markets: beef demand falls (unless diners buy beef elsewhere) and demand for
the foods in the replacement dishes rises. Each market i has constant-elasticity
supply (elasticity eta_i) and demand with own- and cross-price elasticities
E_ij. For an exogenous demand shift dD (quantity at unchanged prices),

    supply:   dlnQ_i = eta_i dlnP_i
    demand:   dlnQ_i = sum_j E_ij dlnP_j + dD_i / Q_i

so (diag(eta) - E) dlnP = dD / Q and dQ = Q * eta * dlnP. In a single market
the share of a demand shift that reaches production is the pass-through

    phi = eta / (eta - eps)      (eps < 0),

and 1 - phi is "market leakage": the demand freed up by the campus is taken up
by other consumers because the price falls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Params, load_csv


def pass_through(eta, eps):
    """Fraction of a demand shift transmitted to production, eta / (eta - eps)."""
    eta = np.asarray(eta, float)
    eps = np.asarray(eps, float)
    return eta / (eta - eps)


def edm(Q: np.ndarray, eta: np.ndarray, E: np.ndarray, dD: np.ndarray):
    """Solve the linear EDM. Returns (dlnP, dQ) with dQ in the units of Q."""
    A = np.diag(eta) - E
    dlnP = np.linalg.solve(A, dD / Q)
    return dlnP, Q * eta * dlnP


def load_markets() -> pd.DataFrame:
    return load_csv("markets.csv", index="market")


def load_cross() -> pd.DataFrame:
    return load_csv("cross_price.csv", index="market")


def dish_table(params: Params) -> pd.DataFrame:
    """Dish compositions, kg of each food *as purchased* per serving."""
    rows = []
    for dish, comp in params.raw["dishes"].items():
        for food, kg in comp.items():
            if food in ("label", "note"):
                continue
            rows.append({"dish": dish, "food_id": food, "kg": float(kg)})
    return pd.DataFrame(rows).pivot_table(index="dish", columns="food_id", values="kg", fill_value=0.0)


DESTINATIONS = ["other_meat", "fish", "vegetarian", "vegan", "off_campus_beef", "off_campus_other"]
ON_CAMPUS = ["other_meat", "fish", "vegetarian", "vegan"]


def destination_draws(params: Params, rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Where forgone beef meals go. Dirichlet around the central shares."""
    base = np.array([params.v(f"campus.switch.{d}") for d in DESTINATIONS])
    base = base / base.sum()
    if n == 1 and rng is None:
        return pd.DataFrame([base], columns=DESTINATIONS)
    kappa = params.v("campus.switch_concentration")
    return pd.DataFrame(rng.dirichlet(kappa * base, n), columns=DESTINATIONS)


def replacement_purchases(dishes: pd.DataFrame, shares: pd.Series) -> tuple[pd.Series, pd.Series]:
    """kg of each food bought per forgone campus beef meal.

    Returns ``(on_campus, off_campus)``. Diners who leave campus either buy a
    beef meal elsewhere or a non-beef meal with the same average composition as
    the on-campus alternatives.
    """
    on = shares[ON_CAMPUS]
    mix_on = (dishes.loc[ON_CAMPUS].mul(on, axis=0)).sum()
    avg_alt = (dishes.loc[ON_CAMPUS].mul(on / on.sum(), axis=0)).sum()
    off = shares["off_campus_other"] * avg_alt + shares["off_campus_beef"] * dishes.loc["beef"]
    return mix_on, off


def campus_beef(params: Params, rng=None, n: int = 1):
    """Campus beef purchases (kg/yr, FOI-based) and the implied number of beef meals."""
    if rng is None:
        kg = params.v("campus.beef_t_per_year") * 1000
        return kg, kg / params.v("campus.beef_kg_per_meal")
    kg = params.sample("campus.beef_t_per_year", rng, n) * 1000
    return kg, kg / params.sample("campus.beef_kg_per_meal", rng, n)


def campus_scenario(Q: np.ndarray, names: list[str], xp: pd.DataFrame | None, dishes: pd.DataFrame,
                    food_market: pd.Series, shares: pd.Series, beef_kg: float, n_beef: float,
                    beef_ef: float, ef_food: dict[str, float], ef_default: dict[str, float],
                    eta: np.ndarray, eps: np.ndarray) -> dict:
    """One campus scenario: demand shock -> EDM -> emissions decomposition (t CO2e/yr).

    Components (they sum to ``net``):

    * ``gross``             beef removed from campus menus
    * ``substitutes_gross`` replacement food bought (on and off campus), before
                            any market response
    * ``offcampus``         beef re-bought off campus (behavioural leakage)
    * ``market_leakage``    beef re-consumed by others because its price falls
    * ``substitute_adj``    market response of the replacement foods (incl.
                            cross-price effects)

    ``accounting`` is what a procurement inventory would report: beef removed
    plus the on-campus replacements, with no market or off-campus response.
    """
    dishes = dishes.copy()
    dishes.loc["beef", "beef_mince"] = beef_kg / n_beef
    on_meal, off_meal = replacement_purchases(dishes, shares)
    on_tot, off_tot = on_meal * n_beef, off_meal * n_beef
    demand = on_tot.add(off_tot, fill_value=0.0)
    demand.loc["beef_mince"] = demand.get("beef_mince", 0.0) - beef_kg

    dD = np.zeros(len(names))
    ef_m = np.zeros(len(names))
    for i, mkt in enumerate(names):
        foods_m = [f for f in demand.index if food_market.get(f) == mkt]
        dD[i] = demand[foods_m].sum() if foods_m else 0.0
        if mkt == "beef":
            ef_m[i] = beef_ef
        elif foods_m and abs(dD[i]) > 0:
            w = demand[foods_m]
            ef_m[i] = float(sum(w[f] * ef_food[f] for f in foods_m) / w.sum())
        else:
            ef_m[i] = ef_default[mkt]
    E = np.diag(eps)
    if xp is not None:
        for i, mi in enumerate(names):
            for j, mj in enumerate(names):
                if i != j and mi in xp.index and mj in xp.columns:
                    E[i, j] = float(xp.loc[mi, mj])
    dlnP, dQ = edm(Q, eta, E, dD)

    ib = names.index("beef")
    offcampus_beef = shares["off_campus_beef"] * beef_kg
    subs_gross = float(np.delete(dD * ef_m, ib).sum())
    subs_net = float(np.delete(dQ * ef_m, ib).sum())
    on_alt = on_tot.drop(labels=["beef_mince"], errors="ignore")
    on_alt_ghg = float(sum(on_alt[f] * ef_food[f] for f in on_alt.index))
    row = {
        "gross": -beef_kg * beef_ef / 1000,
        "substitutes_gross": subs_gross / 1000,
        "offcampus": offcampus_beef * beef_ef / 1000,
        "market_leakage": (dQ[ib] - dD[ib]) * beef_ef / 1000,
        "substitute_adj": (subs_net - subs_gross) / 1000,
        "accounting": (-beef_kg * beef_ef + on_alt_ghg) / 1000,
        "beef_t": beef_kg / 1000,
        "beef_meals": n_beef,
        "pass_through_beef": dQ[ib] / dD[ib] if dD[ib] != 0 else np.nan,
        "beef_ef": beef_ef,
        "price_beef_pct": 100 * float(dlnP[ib]),
        "dD_beef_kg": dD[ib], "dQ_beef_kg": dQ[ib],
    }
    for i, mkt in enumerate(names):
        row[f"dD_{mkt}"] = dD[i]
        row[f"dQ_{mkt}"] = dQ[i]
        row[f"ef_{mkt}"] = ef_m[i]
    row["net"] = (row["gross"] + row["substitutes_gross"] + row["offcampus"]
                  + row["market_leakage"] + row["substitute_adj"])
    return row


def _setup(params: Params):
    markets = load_markets()
    names = list(markets.index)
    Q = markets["q_kt"].to_numpy(float) * 1e6  # kg/yr
    return markets, names, Q, load_cross(), dish_table(params)


def run_campus(params: Params, foods: pd.DataFrame, efs: pd.DataFrame, beef_ef: np.ndarray,
               rng: np.random.Generator, n: int, horizon: str = "lr",
               cross: bool = True) -> pd.DataFrame:
    """Monte Carlo of the campus beef removal (see :func:`campus_scenario`)."""
    markets, names, Q, xp, dishes = _setup(params)
    dest = destination_draws(params, rng, n)
    beef_kg, n_beef = campus_beef(params, rng, n)
    ef_food = {f: lognormal_draw(rng, efs.loc[f], n) for f in dishes.columns if f != "beef_mince"}
    ef_default = {m: lognormal_draw(rng, efs.loc[markets.loc[m, "default_food"]], n) for m in names}
    cols = (f"eta_{horizon}_low", f"eta_{horizon}", f"eta_{horizon}_high")
    eta = np.column_stack([rng.triangular(*markets.loc[m, list(cols)].astype(float), n) for m in names])
    eps = np.column_stack([rng.triangular(*markets.loc[m, ["eps_low", "eps", "eps_high"]].astype(float), n)
                           for m in names])
    out = []
    for k in range(n):
        out.append(campus_scenario(
            Q, names, xp if cross else None, dishes, foods["market"], dest.iloc[k], beef_kg[k],
            n_beef[k], beef_ef[k], {f: v[k] for f, v in ef_food.items()},
            {m: v[k] for m, v in ef_default.items()}, eta[k], eps[k]))
    return pd.DataFrame(out)


def campus_central(params: Params, foods: pd.DataFrame, efs: pd.DataFrame, beef_ef: float,
                   horizon: str = "lr") -> dict:
    """The same calculation with every input at its central value (worked example)."""
    markets, names, Q, xp, dishes = _setup(params)
    base = pd.Series({d: params.v(f"campus.switch.{d}") for d in DESTINATIONS})
    beef_kg, n_beef = campus_beef(params)
    return campus_scenario(
        Q, names, xp, dishes, foods["market"], base / base.sum(), beef_kg, n_beef, beef_ef,
        {f: float(efs.loc[f, "ef_central"]) for f in dishes.columns if f != "beef_mince"},
        {m: float(efs.loc[markets.loc[m, "default_food"], "ef_central"]) for m in names},
        markets[f"eta_{horizon}"].to_numpy(float), markets["eps"].to_numpy(float))


def lognormal_draw(rng: np.random.Generator, row: pd.Series, n: int) -> np.ndarray:
    from .emissions import ef_draws

    return ef_draws(row, rng, n)
