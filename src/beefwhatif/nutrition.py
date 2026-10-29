"""Nutritional equivalence of beef and alternative foods.

Two complementary analyses are implemented.

* **Single-food equivalents** - the mass of one alternative food that supplies
  the same amount of each nutrient as the reference portion of beef
  (100 g cooked lean beef mince).
* **Optimised baskets** - linear programmes (LPs) that find the combination of
  foods that matches *every* target nutrient of the beef portion at minimum
  greenhouse-gas (GHG) emissions or minimum mass, subject to an energy band,
  saturated-fat and sodium ceilings and per-food portion caps.

Nutrient values in ``data/foods.csv`` are given per *LP unit*: 100 g of food as
eaten (cooked where applicable) or, for supplements, one tablet. Protein, iron,
zinc and vitamin B12 are additionally expressed on a bioavailability-adjusted
basis (``protein_q``, ``iron_abs``, ``zinc_abs``, ``b12_abs``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .params import Params, load_csv

REF = "beef_mince"

#: Nutrients the replacement must supply at least as much of as beef
#: (bioavailability-adjusted where indicated).
TARGETS: dict[str, str] = {
    "protein_q": "Protein (DIAAS-adj.)",
    "iron_abs": "Iron (absorbable)",
    "zinc_abs": "Zinc (absorbable)",
    "b12_abs": "Vitamin B12 (absorbable)",
    "niacin_mg": "Niacin",
    "b6_mg": "Vitamin B6",
    "riboflavin_mg": "Riboflavin",
    "selenium_ug": "Selenium",
    "phosphorus_mg": "Phosphorus",
    "potassium_mg": "Potassium",
}

#: The same targets without any bioavailability adjustment ("label" values).
NAIVE_TARGETS: dict[str, str] = {
    "protein_g": "Protein",
    "iron_mg": "Iron",
    "zinc_mg": "Zinc",
    "b12_ug": "Vitamin B12",
    **{k: v for k, v in TARGETS.items() if not k.endswith(("_q", "_abs"))},
}

#: Key nutrients for which meat is a principal UK source and intakes are marginal
#: in parts of the population (protein quality, iron, zinc, B12, selenium).
KEY_TARGETS: dict[str, str] = {k: TARGETS[k] for k in
                               ("protein_q", "iron_abs", "zinc_abs", "b12_abs", "selenium_ug")}

ALL_FOODS = ["meat", "fish", "egg", "dairy", "plant", "fortified"]

#: Basket scenarios: label, allowed food groups (``diet`` column) and an optional
#: "lead" constraint (groups that must supply at least a share of the protein).
#: Q1 scenarios ask what each diet can do with foods alone; Q2 scenarios must be
#: nutritionally equivalent and may close gaps with supplements.
SCENARIOS: dict[str, tuple[str, list[str], tuple[list[str], float] | None]] = {
    "meat_swap": ("Poultry & pork only", ["meat"], None),
    "omnivore": ("Omnivore, foods only", ALL_FOODS, None),
    "vegetarian": ("Vegetarian, foods only", ["egg", "dairy", "plant", "fortified"], None),
    "vegan_fortified": ("Vegan incl. fortified foods", ["plant", "fortified"], None),
    "vegan_wholefood": ("Vegan, whole foods only", ["plant"], None),
    "eq_meat": ("Poultry/pork-led", ALL_FOODS + ["supplement"], (["meat"], 0.6)),
    "eq_fish": ("Fish-led", ALL_FOODS + ["supplement"], (["fish"], 0.6)),
    "eq_vegetarian": ("Dairy/egg-led", ["egg", "dairy", "plant", "fortified", "supplement"],
                      (["egg", "dairy"], 0.5)),
    "eq_vegan": ("Plant-based", ["plant", "fortified", "supplement"], None),
}
Q1_SCENARIOS = ["meat_swap", "omnivore", "vegetarian", "vegan_fortified", "vegan_wholefood"]
Q2_SCENARIOS = ["eq_meat", "eq_fish", "eq_vegetarian", "eq_vegan"]


def load_foods() -> pd.DataFrame:
    return load_csv("foods.csv", index="food_id")


def load_csv_drv() -> pd.DataFrame:
    """UK dietary reference values for adults 19-50 (``data/drv_uk.csv``)."""
    return load_csv("drv_uk.csv", index="nutrient")


def bio_params(params: Params) -> dict[str, float]:
    """Flatten the ``[bioavailability]`` table into ``{name: central value}``."""
    raw = params.raw["bioavailability"]
    return {k: float(v["value"] if isinstance(v, dict) else v) for k, v in raw.items()}


def with_bioavailability(foods: pd.DataFrame, bio: dict[str, float]) -> pd.DataFrame:
    """Add bioavailability-adjusted protein, iron, zinc and vitamin B12 columns.

    * Protein is weighted by the truncated DIAAS (min(DIAAS, 1)), i.e. protein
      that is not limited by its first-limiting indispensable amino acid.
    * Iron is split into haem and non-haem pools with separate absorption
      fractions.
    * Zinc and vitamin B12 use food-class-specific fractional absorption.
    """
    out = foods.copy()
    out["protein_q"] = out["protein_g"] * out["diaas"].clip(upper=1.0)
    haem = out["iron_mg"] * out["haem_frac"]
    out["iron_abs"] = haem * bio["haem_abs"] + (out["iron_mg"] - haem) * bio["nonhaem_abs"]
    out["zinc_abs"] = out["zinc_mg"] * out["zinc_class"].map(lambda c: bio[f"zinc_abs_{c}"])
    out["b12_abs"] = out["b12_ug"] * out["b12_class"].map(lambda c: bio.get(f"b12_abs_{c}", 0.0))
    return out


def single_food_equivalents(foods: pd.DataFrame, targets: dict[str, str] = TARGETS,
                            ref: str = REF) -> pd.DataFrame:
    """Grams of each single food needed to match each nutrient in the beef portion.

    Returns a frame indexed by food with one column per target nutrient (grams
    as eaten; ``inf`` where the food contains none of the nutrient) plus
    ``binding_g`` (grams needed to match *all* targets) and ``binding``
    (the nutrient that sets that amount).
    """
    beef = foods.loc[ref]
    rows = {}
    for fid, row in foods.drop(index=ref).iterrows():
        if row["diet"] == "supplement":
            continue
        need = {}
        for k in targets:
            a = float(row[k])
            need[k] = np.inf if a <= 0 else row["unit_mass_g"] * float(beef[k]) / a
        rows[fid] = need
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["binding_g"] = out[list(targets)].max(axis=1)
    out["binding"] = out[list(targets)].idxmax(axis=1).map(targets)
    return out


@dataclass
class Basket:
    """Solution of one nutrient-matching linear programme."""

    name: str
    label: str
    objective: str
    units: pd.Series  # LP units of each food per 100 g beef replaced
    grams: pd.Series  # g as eaten (tablets for supplements)
    ghg: float  # kg CO2e per 100 g beef replaced (emission factors used in the LP)
    mass_g: float
    energy_ratio: float
    ratios: pd.Series  # supplied / beef, every target nutrient
    shortfall: pd.Series  # fractional shortfall vs beef (0 = matched)
    satfat_g: float
    sodium_mg: float
    fibre_g: float

    @property
    def feasible(self) -> bool:
        return bool((self.shortfall < 1e-6).all())


def allowed_foods(foods: pd.DataFrame, diets: list[str], ref: str = REF) -> list[str]:
    mask = foods["diet"].isin(diets) & (foods.index != ref) & (foods["ruminant"] == 0)
    return list(foods.index[mask])


def optimise_basket(foods: pd.DataFrame, ghg_unit: pd.Series, allowed: list[str], *,
                    targets: dict[str, str] = TARGETS, ref: str = REF,
                    objective: str = "ghg", energy_band: tuple[float, float] = (0.85, 1.15),
                    satfat_max_ratio: float = 1.0, sodium_allow_mg: float = 150.0,
                    penalty: float = 100.0, name: str = "", label: str = "",
                    lead: tuple[list[str], float] | None = None) -> Basket:
    """Solve the nutrient-matching LP for one food set.

    Decision variables are the LP units x_i >= 0 of each allowed food and one
    slack s_k in [0, 1] per target nutrient (fractional shortfall). The LP is

        min  sum_i c_i x_i + penalty * sum_k s_k
        s.t. sum_i (a_ik / b_k) x_i + s_k >= 1              for each target k
             (1 - d_lo) E_b <= sum_i e_i x_i <= (1 + d_hi) E_b
             sum_i satfat_i x_i <= satfat_max_ratio * satfat_b
             sum_i Na_i x_i <= Na_b + sodium_allow_mg
             0 <= x_i <= cap_i

    where b_k is the beef content of nutrient k and c_i is kg CO2e per unit
    (``objective='ghg'``) or 100 g of mass per unit (``objective='mass'``). The
    slacks keep the LP feasible so that shortfalls can be reported when no
    combination of the allowed foods can match beef. The penalty (100 per unit
    shortfall) dwarfs any basket's emissions (< 1 kg CO2e) so slacks are only
    used when unavoidable.
    """
    F = foods.loc[allowed]
    keys = list(targets)
    n, m = len(F), len(keys)
    b = foods.loc[ref, keys].astype(float).to_numpy()
    A = F[keys].astype(float).to_numpy().T / b[:, None]

    mass_unit = F["unit_mass_g"].to_numpy(float) / 100.0
    ghg = ghg_unit.loc[allowed].to_numpy(float)
    e = F["energy_kcal"].to_numpy(float)
    if objective == "ghg":
        c_x = ghg + 1e-4 * mass_unit  # tiny mass tie-breaker
    elif objective == "mass":
        c_x = mass_unit + 1e-4 * ghg
    elif objective == "energy":
        c_x = e / 1000.0 + 1e-4 * ghg
    else:
        raise ValueError(objective)
    c = np.concatenate([c_x, np.full(m, penalty)])

    rows, rhs = [], []
    rows.append(np.hstack([-A, -np.eye(m)]))
    rhs.append(-np.ones(m))
    e_b = float(foods.loc[ref, "energy_kcal"])
    if np.isfinite(energy_band[1]):
        rows.append(np.hstack([e, np.zeros(m)])[None])
        rhs.append([energy_band[1] * e_b])
    rows.append(np.hstack([-e, np.zeros(m)])[None])
    rhs.append([-energy_band[0] * e_b])
    sf = F["satfat_g"].to_numpy(float)
    rows.append(np.hstack([sf, np.zeros(m)])[None])
    rhs.append([satfat_max_ratio * float(foods.loc[ref, "satfat_g"])])
    na = F["sodium_mg"].to_numpy(float)
    rows.append(np.hstack([na, np.zeros(m)])[None])
    rhs.append([float(foods.loc[ref, "sodium_mg"]) + sodium_allow_mg])

    if lead is not None:  # lead groups supply >= share of DIAAS-adjusted protein
        groups, share = lead
        pq = F["protein_q"].to_numpy(float)
        in_lead = F["diet"].isin(groups).to_numpy()
        coef = np.where(in_lead, -(1 - share) * pq, share * pq)
        rows.append(np.hstack([coef, np.zeros(m)])[None])
        rhs.append([0.0])

    caps = (F["cap_g"].to_numpy(float) / F["unit_mass_g"].to_numpy(float))
    bounds = [(0.0, cap) for cap in caps] + [(0.0, 1.0)] * m
    res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), bounds=bounds,
                  method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP for {name or allowed} failed: {res.message}")

    x = pd.Series(res.x[:n], index=allowed)
    x = x[x > 1e-6]
    s = pd.Series(res.x[n:], index=keys).clip(lower=0.0)
    return summarise_basket(foods, ghg_unit, x, targets=targets, ref=ref, shortfall=s,
                            objective=objective, name=name, label=label)


def summarise_basket(foods: pd.DataFrame, ghg_unit: pd.Series, units: pd.Series, *,
                     targets: dict[str, str] = TARGETS, ref: str = REF,
                     shortfall: pd.Series | None = None, objective: str = "",
                     name: str = "", label: str = "") -> Basket:
    F = foods.loc[units.index]
    cols = list(dict.fromkeys([*targets, *(k for k in TARGETS if k in foods.columns)]))
    supplied = (F[cols].mul(units, axis=0)).sum()
    beef = foods.loc[ref, cols].astype(float)
    ratios = supplied / beef
    if shortfall is None:
        shortfall = (1.0 - ratios[list(targets)]).clip(lower=0.0)
    return Basket(
        name=name, label=label, objective=objective, units=units,
        grams=(units * F["unit_mass_g"]).where(F["diet"] != "supplement", units),
        ghg=float((units * ghg_unit.loc[units.index]).sum()),
        mass_g=float((units * F["unit_mass_g"]).where(F["diet"] != "supplement", 0).sum()),
        energy_ratio=float((units * F["energy_kcal"]).sum() / foods.loc[ref, "energy_kcal"]),
        ratios=ratios, shortfall=shortfall,
        satfat_g=float((units * F["satfat_g"]).sum()),
        sodium_mg=float((units * F["sodium_mg"]).sum()),
        fibre_g=float((units * F["fibre_g"]).sum()),
    )


def solve_scenarios(foods: pd.DataFrame, ghg_unit: pd.Series, settings: dict,
                    targets: dict[str, str] = KEY_TARGETS) -> dict[str, Basket]:
    """Solve every scenario in :data:`SCENARIOS` at minimum emissions."""
    out = {}
    for key, (label, diets, lead) in SCENARIOS.items():
        out[key] = optimise_basket(foods, ghg_unit, allowed_foods(foods, diets), targets=targets,
                                   name=key, label=label, lead=lead, **settings)
    return out


def min_energy(foods: pd.DataFrame, ghg_unit: pd.Series, settings: dict,
               targets: dict[str, str] = KEY_TARGETS) -> pd.DataFrame:
    """Energy cost of full nutritional equivalence.

    For each scenario, the smallest energy (kcal) of a basket that matches all
    targets (no upper energy bound; same fat, sodium and portion limits),
    relative to the 100 g of beef it replaces.
    """
    rows = []
    open_band = dict(settings, energy_band=(0.0, np.inf))
    for key, (label, diets, lead) in SCENARIOS.items():
        bk = optimise_basket(foods, ghg_unit, allowed_foods(foods, diets), targets=targets,
                             objective="energy", name=key, label=label, lead=lead, **open_band)
        rows.append({"scenario": key, "label": label, "feasible": bk.feasible,
                     "energy_ratio": bk.energy_ratio, "max_shortfall": float(bk.shortfall.max()),
                     "worst_nutrient": targets[bk.shortfall.idxmax()] if not bk.feasible else ""})
    return pd.DataFrame(rows).set_index("scenario")


def baskets_table(baskets: dict[str, Basket]) -> pd.DataFrame:
    """Long-format table of basket composition (g or tablets per 100 g beef)."""
    rows = []
    for key, bk in baskets.items():
        for fid, g in bk.grams.items():
            rows.append({"scenario": key, "label": bk.label, "food_id": fid, "amount": g})
    return pd.DataFrame(rows)


def baskets_summary(baskets: dict[str, Basket], targets: dict[str, str] = TARGETS) -> pd.DataFrame:
    """One row per basket. Ratios are reported for all ten nutrients; feasibility
    refers to the targets the basket was optimised for."""
    rows = []
    for key, bk in baskets.items():
        row = {"scenario": key, "label": bk.label, "objective": bk.objective,
               "feasible": bk.feasible, "ghg_kg": bk.ghg, "mass_g": bk.mass_g,
               "energy_ratio": bk.energy_ratio, "satfat_g": bk.satfat_g,
               "sodium_mg": bk.sodium_mg, "fibre_g": bk.fibre_g,
               "max_shortfall": float(bk.shortfall.max()),
               "worst_nutrient": TARGETS[bk.shortfall.idxmax()] if bk.shortfall.max() > 1e-6 else ""}
        row.update({f"ratio_{k}": float(bk.ratios[k]) for k in targets})
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def basket_settings(params: Params) -> dict:
    lp = params.raw["lp"]
    return {
        "energy_band": (1 - float(lp["energy_tol_low"]), 1 + float(lp["energy_tol_high"])),
        "satfat_max_ratio": float(lp["satfat_max_ratio"]),
        "sodium_allow_mg": float(lp["sodium_allow_mg"]),
    }
