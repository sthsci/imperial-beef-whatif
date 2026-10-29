"""Student welfare and nutrient intake when beef leaves the campus menu (prices fixed).

Welfare uses a random-utility (nested) logit model of dish choice. Removing the
beef alternatives changes diners' expected maximum utility; with prices held
constant the compensating variation (CV) per meal occasion is

    MNL:     CV = ln(1 - P_B) / alpha
    nested:  CV = ln(1 - P_M + P_M (1 - P_B / P_M)^lambda) / alpha

(Small & Rosen 1981; Train 2009). P_B is the beef share of meals, P_M the share
of the meat nest that contains beef, lambda in (0, 1] the nest's dissimilarity
parameter (lambda -> 0: other meat dishes are perfect substitutes for beef, no
loss; lambda = 1: plain multinomial logit) and alpha the marginal utility of
money (per GBP). alpha is calibrated from an assumed own-price elasticity of a
beef dish eps_B < 0:

    eps_B = -alpha p_B [1/lambda - (1/lambda - 1) P_B|M - P_B].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .economics import DESTINATIONS, ON_CAMPUS, campus_beef, destination_draws, dish_table
from .params import Params

REPORT_NUTRIENTS = {
    "energy_kcal": "Energy (kcal)",
    "protein_g": "Protein (g)",
    "protein_q": "Protein, DIAAS-adj. (g)",
    "iron_mg": "Iron (mg)",
    "iron_abs": "Iron, absorbable (mg)",
    "zinc_mg": "Zinc (mg)",
    "zinc_abs": "Zinc, absorbable (mg)",
    "b12_ug": "Vitamin B12 (ug)",
    "satfat_g": "Saturated fat (g)",
    "fibre_g": "Fibre (g)",
    "sodium_mg": "Sodium (mg)",
}


def alpha_from_elasticity(eps_b, price, p_b, p_m, lam):
    """Marginal utility of money implied by a beef-dish own-price elasticity."""
    p_b_given_m = p_b / p_m
    bracket = 1 / lam - (1 / lam - 1) * p_b_given_m - p_b
    return -np.asarray(eps_b) / (np.asarray(price) * bracket)


def cv_per_meal(alpha, p_b, p_m, lam):
    """Compensating variation (GBP, negative = loss) per meal occasion."""
    return np.log(1 - p_m + p_m * (1 - p_b / p_m) ** lam) / alpha


def welfare_draws(params: Params, rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Monte Carlo of the welfare loss. All money in GBP per year unless stated.

    The logit loss per forgone beef meal is also reported capped at the cost of
    the outside option (walking off campus to buy a beef meal): no diner can
    lose more than that, because they can always take it.
    """
    meals = params.sample("campus.meals_per_year", rng, n)
    _, n_beef = campus_beef(params, rng, n)
    p_b = np.clip(n_beef / meals, 1e-4, 0.5)
    p_m = np.maximum(params.sample("welfare.meat_nest_share", rng, n), p_b + 0.05)
    lam = params.sample("welfare.nest_lambda", rng, n)
    eps = params.sample("welfare.dish_elasticity", rng, n)
    price = params.sample("welfare.meal_price_gbp", rng, n)
    cap = params.sample("welfare.outside_option_cost_gbp", rng, n)
    alpha = alpha_from_elasticity(eps, price, p_b, p_m, lam)
    cv = cv_per_meal(alpha, p_b, p_m, lam)
    per_beef = -cv / p_b
    return pd.DataFrame({
        "meals": meals, "beef_meals": n_beef, "p_beef": p_b, "p_meat": p_m, "lambda": lam,
        "eps": eps, "price": price, "alpha": alpha, "cv_per_meal": cv,
        "loss_per_beef_meal": per_beef,
        "loss_per_beef_meal_capped": np.minimum(per_beef, cap),
        "loss_total": per_beef * n_beef,
        "loss_capped_total": np.minimum(per_beef, cap) * n_beef,
        "loss_mnl_total": -np.log(1 - p_b) / alpha * meals,
    })


def welfare_central(params: Params) -> dict[str, float]:
    meals = params.v("campus.meals_per_year")
    _, n_beef = campus_beef(params)
    p_b = n_beef / meals
    p_m = params.v("welfare.meat_nest_share")
    lam = params.v("welfare.nest_lambda")
    alpha = float(alpha_from_elasticity(params.v("welfare.dish_elasticity"),
                                        params.v("welfare.meal_price_gbp"), p_b, p_m, lam))
    cv = float(cv_per_meal(alpha, p_b, p_m, lam))
    per_beef = -cv / p_b
    cap = params.v("welfare.outside_option_cost_gbp")
    return {"alpha": alpha, "p_beef": p_b, "cv_per_meal": cv, "loss_per_beef_meal": per_beef,
            "loss_total": per_beef * n_beef, "beef_meals": n_beef,
            "loss_capped_total": min(per_beef, cap) * n_beef}


# ----------------------------------------------------------------- nutrition


def dish_nutrients(dishes: pd.DataFrame, foods: pd.DataFrame, efs: pd.DataFrame,
                   nutrients: list[str] | None = None) -> pd.DataFrame:
    """Nutrients per serving of each dish's protein component (as eaten)."""
    nutrients = nutrients or list(REPORT_NUTRIENTS)
    rows = {}
    for dish, comp in dishes.iterrows():
        tot = pd.Series(0.0, index=nutrients)
        for food, kg in comp.items():
            if kg <= 0:
                continue
            eaten_g = kg * 1000 * float(efs.loc[food, "cook_yield"])
            tot += foods.loc[food, nutrients].astype(float) * eaten_g / float(foods.loc[food, "unit_mass_g"])
        rows[dish] = tot
    return pd.DataFrame(rows).T


def nutrient_change_per_meal(dish_nut: pd.DataFrame, shares: pd.Series) -> pd.Series:
    """Expected change in intake per forgone campus beef meal (as eaten)."""
    beef = dish_nut.loc["beef"]
    on = shares[ON_CAMPUS]
    avg_alt = (dish_nut.loc[ON_CAMPUS].mul(on / on.sum(), axis=0)).sum()
    delta = sum(shares[d] * (dish_nut.loc[d] - beef) for d in ON_CAMPUS)
    delta = delta + shares["off_campus_other"] * (avg_alt - beef)
    return delta  # off-campus beef: no change


def central_shares(params: Params) -> pd.Series:
    base = pd.Series({d: params.v(f"campus.switch.{d}") for d in DESTINATIONS})
    return base / base.sum()


def frequent_eater_change(delta_meal: pd.Series, meals_per_week: float) -> pd.Series:
    """Average daily change in intake for someone who ate beef on campus
    ``meals_per_week`` times a week (during term)."""
    return delta_meal * meals_per_week / 7.0


__all__ = [
    "REPORT_NUTRIENTS", "alpha_from_elasticity", "cv_per_meal", "welfare_draws",
    "welfare_central", "dish_nutrients", "nutrient_change_per_meal", "central_shares",
    "frequent_eater_change", "dish_table", "destination_draws",
]
