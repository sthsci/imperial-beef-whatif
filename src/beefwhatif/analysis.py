"""End-to-end analysis: computes every number reported in the paper.

``run()`` returns a nested dict of results (pandas objects and floats) that the
figure and report modules consume. Nothing in the paper is typed by hand: all
numbers in the LaTeX source come from ``paper/generated/numbers.tex``, which
is written from these results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import economics as econ
from . import emissions as em
from . import nutrition as nu
from . import welfare as wf
from .params import Params

MAIN_BASKETS = ["eq_meat", "eq_fish", "eq_vegetarian", "eq_vegan"]
HEADLINE = "eq_meat"  # basket used for single-number headlines
BEEF_SCENARIOS = {
    "uk_attr": "UK supply mix (attributional)",
    "uk_conseq": "UK supply mix (consequential)",
    "pn_mix": "Global mean, UK herd mix",
    "pn_beef_herd": "Global mean, beef herd",
}
NAIVE_KEY = {k: nu.NAIVE_TARGETS[k] for k in ("protein_g", "iron_mg", "zinc_mg", "b12_ug", "selenium_ug")}
ANIMAL_BIO = {"haem_abs", "zinc_abs_animal", "b12_abs_meat", "b12_abs_fish", "b12_abs_dairy",
              "b12_abs_egg"}


def bio_variant(params: Params, plant_favourable: bool) -> dict[str, float]:
    """Bioavailability set tilted for/against plant foods (for sensitivity)."""
    out = {}
    for k, node in params.raw["bioavailability"].items():
        if not isinstance(node, dict) or "low" not in node:
            out[k] = float(node["value"] if isinstance(node, dict) else node)
            continue
        animal = k in ANIMAL_BIO
        pick_high = (not animal) if plant_favourable else animal
        out[k] = float(node["high"] if pick_high else node["low"])
    return out


def beef_ef_scenarios(params: Params, supply: pd.DataFrame, c: dict) -> dict[str, float]:
    share_dairy = params.v("consumption.dairy_origin_share")
    herd = params.v("emissions.pn_beef_herd")
    dairy = params.v("emissions.pn_beef_dairy")
    return {
        "uk_attr": em.beef_mix_ef(supply, c["retail_yield"], c["post_farm"]),
        "uk_conseq": em.beef_mix_ef(supply, c["retail_yield"], c["post_farm"], consequential=True),
        "pn_mix": share_dairy * dairy + (1 - share_dairy) * herd,
        "pn_beef_herd": herd,
    }


def run(params: Params | None = None, n_mc: int | None = None, n_campus: int | None = None) -> dict:
    params = params or Params()
    meta = params.raw["meta"]
    rng = np.random.default_rng(int(meta["seed"]))
    n_mc = n_mc or int(meta["n_mc"])
    n_campus = n_campus or int(meta["n_campus"])
    R: dict = {"params": params}

    # ------------------------------------------------------------ inputs
    foods_raw = nu.load_foods()
    bio = nu.bio_params(params)
    foods = nu.with_bioavailability(foods_raw, bio)
    efs = em.load_efs()
    supply = em.load_beef_supply()
    c = em.per_capita_inputs(params)
    R.update(foods=foods, efs=efs, supply=supply, central_inputs=c, bio=bio)

    beef_ef = beef_ef_scenarios(params, supply, c)
    R["beef_ef"] = beef_ef
    R["beef_segments"] = pd.DataFrame({
        "share": supply["share"],
        "ef_retail_attr": em.beef_segment_ef(supply, c["retail_yield"], c["post_farm"]),
        "ef_retail_conseq": em.beef_segment_ef(supply, c["retail_yield"], c["post_farm"], True),
    })
    R["beef_gas_shares"] = em.beef_mix_gas_shares(supply, c["retail_yield"], c["post_farm"])

    # ------------------------------------------------------------ nutrition
    waste = c["waste"]
    ghg_unit = em.ghg_per_unit(foods, efs, waste, beef_ef=beef_ef["uk_attr"])
    ghg_unit_pn = em.ghg_per_unit(foods, efs, waste, column="ef_pn", beef_ef=beef_ef["pn_mix"])
    R["ghg_unit"], R["ghg_unit_pn"] = ghg_unit, ghg_unit_pn
    R["equivalents"] = nu.single_food_equivalents(foods, nu.TARGETS)
    R["equivalents_naive"] = nu.single_food_equivalents(foods, nu.NAIVE_TARGETS)
    settings = nu.basket_settings(params)
    baskets = nu.solve_scenarios(foods, ghg_unit, settings)
    R["baskets"] = baskets
    R["baskets_naive"] = nu.solve_scenarios(foods, ghg_unit, settings, targets=NAIVE_KEY)
    R["baskets_full"] = nu.solve_scenarios(foods, ghg_unit, settings, targets=nu.TARGETS)
    R["baskets_summary"] = nu.baskets_summary(baskets)
    R["baskets_table"] = nu.baskets_table(baskets)

    sens = {}
    for tag, fav in (("plant_unfavourable", False), ("plant_favourable", True)):
        f2 = nu.with_bioavailability(foods_raw, bio_variant(params, fav))
        sens[tag] = nu.solve_scenarios(f2, ghg_unit, settings)
    for tag, key in (("energy_strict", "lp.energy_tol_strict"), ("energy_relaxed", "lp.energy_tol_relaxed")):
        band = dict(settings, energy_band=(settings["energy_band"][0], 1 + params.v(key)))
        sens[tag] = nu.solve_scenarios(foods, ghg_unit, band)
    R["min_energy"] = nu.min_energy(foods, ghg_unit, settings)
    sens["naive"] = R["baskets_naive"]
    sens["full10"] = R["baskets_full"]
    R["basket_sensitivity"] = sens

    # ------------------------------------------------- emissions per 100 g eaten
    rows = []
    for key, label in BEEF_SCENARIOS.items():
        g = (0.1 / c["cook_yield"]) / (1 - waste) * beef_ef[key]
        rows.append({"item": key, "label": f"Beef - {label}", "kg_co2e_per_100g": g, "kind": "beef"})
    for key in MAIN_BASKETS + nu.Q1_SCENARIOS:
        bk = baskets[key]
        rows.append({"item": key, "label": bk.label, "kind": "basket", "kg_co2e_per_100g": bk.ghg,
                     "kg_co2e_per_100g_pn": float((bk.units * ghg_unit_pn.loc[bk.units.index]).sum()),
                     "feasible": bk.feasible})
    R["per100"] = pd.DataFrame(rows).set_index("item")

    # ------------------------------------------------- per capita, London, UK
    flows = em.beef_flows(c["cwe"], c["retail_yield"], waste, c["cook_yield"])
    R["flows"] = flows
    pop = {"london": params.v("population.london"), "uk": params.v("population.uk")}
    leggi = params.v("london.leggi_total_mt")
    cbe = params.v("london.consumption_based_mt")
    uk_terr = params.v("uk.territorial_mt")
    diet_fp = params.v("emissions.diet_fp_t_per_cap")
    det = []
    for scen in BEEF_SCENARIOS:
        for key in MAIN_BASKETS:
            e_beef_pc = flows.retail * beef_ef[scen]  # kg CO2e / person / yr
            e_alt_pc = flows.eaten * 10 * baskets[key].ghg
            d_pc = e_alt_pc - e_beef_pc
            det.append({
                "beef_scenario": scen, "basket": key,
                "beef_pc_kg": e_beef_pc, "alt_pc_kg": e_alt_pc, "delta_pc_kg": d_pc,
                "pct_vs_beef": 100 * d_pc / e_beef_pc,
                "pct_diet": 100 * d_pc / (diet_fp * 1000),
                "london_beef_mt": pop["london"] * e_beef_pc / 1e9,
                "london_delta_mt": pop["london"] * d_pc / 1e9,
                "uk_delta_mt": pop["uk"] * d_pc / 1e9,
                "pct_leggi": 100 * pop["london"] * d_pc / 1e9 / leggi,
                "pct_cbe": 100 * pop["london"] * d_pc / 1e9 / cbe,
                "pct_uk_terr": 100 * pop["uk"] * d_pc / 1e9 / uk_terr,
            })
    R["scaling"] = pd.DataFrame(det)

    # Monte Carlo (London) ------------------------------------------------
    d = em.per_capita_inputs(params, rng, n_mc)
    fl = em.beef_flows(d["cwe"], d["retail_yield"], d["waste"], d["cook_yield"])
    ef_beef_attr = em.beef_mix_samples(supply, d["retail_yield"], d["post_farm"], rng)
    ef_beef_cons = em.beef_mix_samples(supply, d["retail_yield"], d["post_farm"], rng, consequential=True)
    alt_ef = {f: em.ef_draws(efs.loc[f], rng, n_mc) for f in foods.index if f != nu.REF}
    mc = {"cwe": d["cwe"], "beef_ef_attr": ef_beef_attr, "beef_ef_conseq": ef_beef_cons,
          "retail_yield": d["retail_yield"], "cook_yield": d["cook_yield"], "waste": d["waste"]}
    for key in MAIN_BASKETS:
        bk = baskets[key]
        g = np.zeros(n_mc)
        for f, u in bk.units.items():
            waste_f = 0.0 if foods.loc[f, "diet"] == "supplement" else d["waste"]
            g += u * (foods.loc[f, "unit_mass_g"] / 1000) / efs.loc[f, "cook_yield"] / (1 - waste_f) * alt_ef[f]
        mc[f"alt_{key}"] = fl.eaten * 10 * g
    mc["beef_attr"] = fl.retail * ef_beef_attr
    mc["beef_conseq"] = fl.retail * ef_beef_cons
    mc = pd.DataFrame(mc)
    for key in MAIN_BASKETS:
        for tag in ("attr", "conseq"):
            mc[f"london_delta_mt_{tag}_{key}"] = pop["london"] * (mc[f"alt_{key}"] - mc[f"beef_{tag}"]) / 1e9
    R["mc_london"] = mc
    leggi_draw = params.sample("london.leggi_total_mt", rng, n_mc)
    R["mc_pct_leggi"] = 100 * mc[f"london_delta_mt_attr_{HEADLINE}"] / leggi_draw

    # Metric sensitivity ---------------------------------------------------
    gas = R["beef_gas_shares"]
    beef_src = str(params.raw["emissions"]["uk_beef_gwp_src"])
    metric_rows = []
    for metric in ("AR6", "AR6_GWP20"):
        fb = em.gwp_factor_shares(gas, beef_src, metric)
        fa = em.gwp_factor(efs.loc[foods.index.drop(nu.REF)], metric)
        for key in MAIN_BASKETS:
            bk = baskets[key]
            alt = flows.eaten * 10 * float((bk.units * ghg_unit.loc[bk.units.index] * fa.loc[bk.units.index]).sum())
            beef = flows.retail * beef_ef["uk_attr"] * fb
            metric_rows.append({"metric": metric, "basket": key,
                                "london_delta_mt": pop["london"] * (alt - beef) / 1e9,
                                "beef_factor": fb})
    R["metrics"] = pd.DataFrame(metric_rows)

    # GWP*: methane from London's beef (UK mix), abrupt and 10-year phase-out
    ch4_src = em.GWP_SETS[beef_src][0]
    e_beef_london_t = pop["london"] * flows.retail * beef_ef["uk_attr"] / 1000  # t CO2e / yr
    e0_ch4 = e_beef_london_t * gas["ch4"] / ch4_src  # t CH4 / yr
    non_ch4_ar6 = e_beef_london_t * (gas["co2"] + gas["n2o"] * 273.0 / em.GWP_SETS[beef_src][1])
    alt_t = pop["london"] * flows.eaten * 10 * baskets[HEADLINE].ghg / 1000
    years = 100
    star_abrupt = em.gwp_star_delta(e0_ch4, years, 27.0)
    star_phase = em.gwp_star_delta(e0_ch4, years, 27.0, phase_out_years=10)
    t = np.arange(years)
    ramp = np.clip((t + 1) / 10, 0, 1)
    gwp100_flow = -(e0_ch4 * 27.0 + non_ch4_ar6) + alt_t
    R["gwpstar"] = pd.DataFrame({
        "year": t + 1,
        "cum_gwp100_mt": np.cumsum(np.full(years, gwp100_flow)) / 1e6,
        "cum_star_abrupt_mt": np.cumsum(star_abrupt - non_ch4_ar6 + alt_t) / 1e6,
        "cum_gwp100_phase_mt": np.cumsum(ramp * gwp100_flow) / 1e6,
        "cum_star_phase_mt": np.cumsum(star_phase + ramp * (-non_ch4_ar6 + alt_t)) / 1e6,
    })
    R["gwpstar_inputs"] = {"e0_ch4_t": e0_ch4, "non_ch4_t": non_ch4_ar6, "alt_t": alt_t,
                           "beef_t": e_beef_london_t}

    # ------------------------------------------------------------ campus
    yields = params.sample("conversion.beef_retail_yield", rng, n_campus)
    post = params.sample("emissions.beef_post_farm_kg_per_kg", rng, n_campus)
    ef_c = em.beef_mix_samples(supply, yields, post, rng, consequential=True)
    R["campus_lr"] = econ.run_campus(params, foods, efs, ef_c, rng, n_campus, "lr")
    R["campus_sr"] = econ.run_campus(params, foods, efs, ef_c, rng, n_campus, "sr")
    R["campus_lr_nocross"] = econ.run_campus(params, foods, efs, ef_c, rng, n_campus, "lr", cross=False)
    markets = econ.load_markets()
    R["markets"] = markets
    R["pass_through_central"] = pd.DataFrame({
        "sr": econ.pass_through(markets["eta_sr"], markets["eps"]),
        "lr": econ.pass_through(markets["eta_lr"], markets["eps"]),
    }, index=markets.index)
    R["campus_beef_t_central"] = params.v("campus.beef_t_per_year")

    # ------------------------------------------------------------ welfare
    W = wf.welfare_draws(params, rng, n_campus)
    cv = params.sample("carbon_value.desnz_central_gbp", rng, n_campus)
    net = R["campus_lr"]["net"].to_numpy()
    W["avoided_t"] = -net
    W["carbon_value"] = cv
    W["benefit_gbp"] = -net * cv
    W["cost_per_t"] = np.where(net < 0, W["loss_total"] / np.maximum(-net, 1e-9), np.inf)
    R["welfare"] = W
    R["welfare_central"] = wf.welfare_central(params)

    dishes = econ.dish_table(params)
    dn = wf.dish_nutrients(dishes, foods, efs)
    shares = wf.central_shares(params)
    delta = wf.nutrient_change_per_meal(dn, shares)
    freq = wf.frequent_eater_change(delta, params.v("campus.frequent_beef_meals_per_week"))
    R.update(dish_nutrients=dn, switch_shares=shares, delta_meal=delta, delta_frequent=freq,
             dishes=dishes)
    drv = nu.load_csv_drv()
    R["drv"] = drv
    rows = []
    for nut, r in drv.iterrows():
        if nut not in freq.index:
            continue
        rows.append({"nutrient": nut, "label": r["label"], "kind": r["kind"],
                     "delta_per_day": freq[nut],
                     "pct_women": 100 * freq[nut] / r["women"], "pct_men": 100 * freq[nut] / r["men"]})
    R["frequent_rni"] = pd.DataFrame(rows).set_index("nutrient")

    # ------------------------------------------------------------ extras
    R["main_baskets"] = MAIN_BASKETS
    R["per100_mc"] = per100_intervals(R, mc, fl, rng, n_mc)
    R["tornado"], R["tornado_base"] = tornado(params, R)
    R["campus_central_lr"] = econ.campus_central(params, foods, efs, beef_ef["uk_conseq"], "lr")
    R["campus_central_sr"] = econ.campus_central(params, foods, efs, beef_ef["uk_conseq"], "sr")
    R["campus_grid"] = campus_grid(params, R)
    R["foi"] = foi_emissions(efs, beef_ef["uk_attr"])
    return R


def per100_intervals(R: dict, mc: pd.DataFrame, fl, rng, n: int) -> pd.DataFrame:
    """kg CO2e per 100 g cooked beef (or its replacement), Monte Carlo 5/50/95%."""
    p = R["params"]
    units = fl.eaten * 10
    rows = [("uk_attr", "Beef, UK supply mix", "beef", mc["beef_attr"] / units),
            ("uk_conseq", "Beef, UK mix (consequential)", "beef", mc["beef_conseq"] / units)]
    herd = em.lognormal(rng, 60.36, 37.57, 269.19, n)
    dairy = em.lognormal(rng, 34.14, 14.93, 56.68, n)
    share = p.v("consumption.dairy_origin_share")
    pn = ((1 - share) * herd + share * dairy) * fl.retail / units
    rows.append(("pn_mix", "Beef, global (Poore & Nemecek)", "beef", pn))
    for key in MAIN_BASKETS:
        rows.append((key, R["baskets"][key].label, "basket", mc[f"alt_{key}"] / units))
    out = pd.DataFrame([{"item": k, "label": lab, "kind": kind, "p05": np.quantile(x, 0.05),
                         "p50": np.median(x), "p95": np.quantile(x, 0.95), "mean": np.mean(x)}
                        for k, lab, kind, x in rows]).set_index("item")
    return out


def tornado(params: Params, R: dict) -> tuple[pd.DataFrame, float]:
    """One-at-a-time sensitivity of London's change (headline basket, UK mix), Mt CO2e/yr."""
    supply, c0, foods, efs = R["supply"], dict(R["central_inputs"]), R["foods"], R["efs"]
    bk = R["baskets"][HEADLINE]
    pop = params.v("population.london")

    def run(c=None, sup=None, col="ef_central"):
        c = c or c0
        sup = supply if sup is None else sup
        fl = em.beef_flows(c["cwe"], c["retail_yield"], c["waste"], c["cook_yield"])
        ef = em.beef_mix_ef(sup, c["retail_yield"], c["post_farm"])
        g = em.ghg_per_unit(foods, efs, c["waste"], column=col)
        basket = float((bk.units * g.loc[bk.units.index]).sum())
        return pop * (fl.eaten * 10 * basket - fl.retail * ef) / 1e9

    base = run()
    rows = []
    names = {"cwe": ("consumption.beef_cwe_kg_per_cap", "Beef consumption per person"),
             "retail_yield": ("conversion.beef_retail_yield", "Carcass-to-retail yield"),
             "cook_yield": ("conversion.beef_cook_yield", "Cooking yield of beef"),
             "waste": ("conversion.household_waste_frac", "Household waste"),
             "post_farm": ("emissions.beef_post_farm_kg_per_kg", "Post-farm-gate emissions")}
    for key, (pname, label) in names.items():
        p = params[pname]
        lo, hi = run(dict(c0, **{key: p.low})), run(dict(c0, **{key: p.high}))
        rows.append({"label": label, "low_value": lo, "high_value": hi})
    for seg, r in supply.iterrows():
        vals = []
        for bound in ("ef_cwe_low", "ef_cwe_high"):
            sup = supply.copy()
            sup.loc[seg, "ef_cwe"] = r[bound]
            vals.append(run(sup=sup))
        rows.append({"label": f"Footprint: {r['label']}", "low_value": vals[0], "high_value": vals[1]})
    rows.append({"label": "Replacement-food EFs (5th/95th pct)", "low_value": run(col="ef_low"),
                 "high_value": run(col="ef_high")})
    t = pd.DataFrame(rows)
    t["span"] = (t["high_value"] - t["low_value"]).abs()
    return t.set_index("label", drop=False), base


def campus_grid(params: Params, R: dict) -> pd.DataFrame:
    """Net campus change (t/yr) over beef pass-through and off-campus beef share."""
    cc = R["campus_central_lr"]
    s0 = R["switch_shares"]["off_campus_beef"]
    gross = -cc["gross"]
    subs0 = cc["substitutes_gross"] + cc["substitute_adj"]
    phis = np.linspace(0, 1, 101)
    shares = np.linspace(0, 0.5, 51)
    grid = np.array([[-(1 - s) * gross * phi + subs0 * (1 - s) / (1 - s0) for phi in phis] for s in shares])
    return pd.DataFrame(grid, index=shares, columns=phis)


def foi_emissions(efs: pd.DataFrame, beef_ef: float) -> pd.DataFrame:
    """Emissions of Taste Imperial meat purchases by year at central EFs (t CO2e)."""
    from .params import load_csv

    foi = load_csv("imperial_foi_meat.csv", index="year")
    ef = {"beef_kg": beef_ef, "venison_kg": efs.loc["venison", "ef_central"],
          "lamb_kg": efs.loc["lamb", "ef_central"], "pork_kg": efs.loc["pork_loin", "ef_central"],
          "chicken_kg": efs.loc["chicken_breast", "ef_central"]}
    for col, e in ef.items():
        foi[col.replace("_kg", "_t")] = foi[col] * e / 1000
    foi["meat_t"] = foi[[c.replace("_kg", "_t") for c in ef]].sum(axis=1)
    return foi
