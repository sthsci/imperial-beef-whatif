"""Figures for the paper (PDF for LaTeX, PNG for README)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch

from . import nutrition as nu
from .economics import pass_through
from .params import FIGURES
from .style import (AXIS, BLUE_RAMP, DIVERGING, DOUBLE, GRID, GROUP_COLORS, INK, INK2, MM,
                    MUTED, SERIES, SINGLE, apply, panel_label)

BLUES = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)


def _hgrid(ax, axis="x"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def short_name(foods: pd.DataFrame, fid: str) -> str:
    return str(foods.loc[fid, "short"]) if "short" in foods.columns else fid


# --------------------------------------------------------------------- Fig. 1


def _mass(g: float) -> str:
    if not np.isfinite(g):
        return "never"
    return f"{g:,.0f} g" if g < 1000 else f"{g / 1000:,.1f} kg"


def fig_nutrition(R: dict) -> None:
    foods, eq = R["foods"], R["equivalents"].copy()
    eq = eq.sort_values("binding_g")
    targets = list(nu.TARGETS)
    fig = plt.figure(figsize=(DOUBLE, 118 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.95, left=0.12, right=0.98,
                          top=0.86, bottom=0.2)

    # a: single-food equivalents heatmap (grams to match each nutrient)
    ax = fig.add_subplot(gs[0, 0])
    vals = eq[targets].to_numpy(float)
    finite = np.where(np.isfinite(vals), np.clip(vals, 10, 5000), np.nan)
    norm = LogNorm(vmin=10, vmax=5000)
    im = ax.imshow(finite, aspect="auto", cmap=BLUES, norm=norm, interpolation="nearest")
    for (i, j), v in np.ndenumerate(vals):
        if not np.isfinite(v):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f0efec",
                                       edgecolor=MUTED, hatch="////", linewidth=0))
    for i in range(len(eq) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.0)
    for j in range(len(targets) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.0)
    ax.set_yticks(range(len(eq)), [short_name(foods, f) for f in eq.index])
    xl = {"protein_q": "Protein*", "iron_abs": "Iron*", "zinc_abs": "Zinc*", "b12_abs": "Vit. B12*",
          "niacin_mg": "Niacin", "b6_mg": "Vit. B6", "riboflavin_mg": "Riboflavin",
          "selenium_ug": "Selenium", "phosphorus_mg": "Phosphorus", "potassium_mg": "Potassium"}
    ax.set_xticks(range(len(targets)), [xl[t] for t in targets], rotation=60, ha="right",
                  rotation_mode="anchor")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, (fid, row) in enumerate(eq.iterrows()):
        ax.text(len(targets) - 0.25, i, _mass(row["binding_g"]), va="center", ha="left",
                fontsize=5.6, color=INK)
    ax.text(len(targets) - 0.25, -0.85, "All ten", va="bottom", ha="left", fontsize=5.6,
            color=INK2, fontweight="bold")
    cax = ax.inset_axes([0.0, 1.035, 0.55, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("g of food (as eaten) matching 100 g cooked beef  (*bioavailability-adjusted)",
                 fontsize=5.6, color=INK2, labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.2, length=2, pad=1)
    panel_label(ax, "a", x=-0.36, y=1.08)

    # b: optimised baskets
    ax = fig.add_subplot(gs[0, 1])
    order = nu.Q1_SCENARIOS + nu.Q2_SCENARIOS
    baskets = R["baskets"]
    xmax = max(baskets[k].mass_g for k in order) * 1.75
    ax.set_xlim(0, xmax)
    pos = ax.get_position()
    axis_pt = pos.width * fig.get_figwidth() * 72
    for y, key in enumerate(order):
        bk = baskets[key]
        left = 0.0
        g = bk.grams
        groups = foods.loc[g.index, "group"]
        for grp, colour in GROUP_COLORS.items():
            members = [f for f in g.index if groups[f] == grp and foods.loc[f, "diet"] != "supplement"]
            for f in sorted(members, key=lambda f: -g[f]):
                ax.barh(y, g[f], left=left, height=0.62, color=colour, edgecolor="white",
                        linewidth=0.8)
                name = short_name(foods, f).split(" (")[0]
                if g[f] / xmax * axis_pt > len(name) * 4.6 * 0.56 + 4:
                    ax.text(left + g[f] / 2, y, name, ha="center", va="center", fontsize=4.6,
                            color="white" if grp not in ("Dairy", "Eggs") else INK)
                left += g[f]
        supp = [f for f in g.index if foods.loc[f, "diet"] == "supplement"]
        note = f"{bk.ghg:.2f} kg CO$_2$e, {bk.energy_ratio:.1f}× kcal"
        if supp:
            note += "\n+ " + " & ".join(short_name(foods, f).split()[0] for f in supp) + " tablet"
        if not bk.feasible:
            worst = bk.shortfall.idxmax()
            note += f"\n{nu.TARGETS[worst].split(' (')[0]} {-100 * bk.shortfall.max():.0f}%"
        ax.text(left + xmax * 0.015, y, note, va="center", ha="left", fontsize=5.0, color=INK)
    ax.axvline(100, color=MUTED, linewidth=0.6)
    ax.text(100, -0.8, "100 g beef", fontsize=5.2, color=INK2, ha="center")
    ax.axhline(len(nu.Q1_SCENARIOS) - 0.5, color=AXIS, linewidth=0.6)
    ax.text(xmax, -0.8, "foods only  ", ha="right", va="center", fontsize=5.0, color=INK2,
            style="italic")
    ax.text(xmax, len(order) - 0.35, "equivalent incl. supplements  ", ha="right", va="center",
            fontsize=5.0, color=INK2, style="italic")
    ax.set_yticks(range(len(order)), [baskets[k].label for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("g as eaten per 100 g cooked beef replaced")
    _hgrid(ax)
    handles = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.12), ncol=2,
              fontsize=5.2, handlelength=1.0, columnspacing=0.8)
    panel_label(ax, "b", x=-0.62, y=1.08)
    _save(fig, "fig1_nutrition")


# --------------------------------------------------------------------- Fig. 2


def _interval(ax, y, lo, mid, hi, colour, marker="o", size=4.5):
    ax.plot([lo, hi], [y, y], color=colour, linewidth=1.4, solid_capstyle="round")
    ax.plot(mid, y, marker=marker, markersize=size, color=colour, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3)


def fig_emissions(R: dict) -> None:
    fig = plt.figure(figsize=(DOUBLE, 72 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], wspace=0.8)
    p100 = R["per100_mc"]

    # a: kg CO2e per 100 g cooked beef replaced
    ax = fig.add_subplot(gs[0, 0])
    rows = p100.index.tolist()
    for y, item in enumerate(rows):
        r = p100.loc[item]
        colour = SERIES[7] if r["kind"] == "beef" else SERIES[0]
        _interval(ax, y, r["p05"], r["p50"], r["p95"], colour)
    ax.set_yticks(range(len(rows)), p100["label"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("kg CO$_2$e per 100 g cooked beef\n(or its nutritional equivalent)")
    _hgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], color=SERIES[7], marker="o", markersize=3, label="Beef"),
                       plt.Line2D([], [], color=SERIES[0], marker="o", markersize=3,
                                  label="Equivalent basket")],
              loc="upper left", bbox_to_anchor=(0.0, 0.62), fontsize=5.2)
    panel_label(ax, "a", x=-0.95, y=1.02)

    # b: London annual change
    ax = fig.add_subplot(gs[0, 1])
    mc = R["mc_london"]
    leggi = R["params"].v("london.leggi_total_mt")
    keys = [k for k in R["main_baskets"]]
    ylab = []
    for y, key in enumerate(keys):
        for dy, tag, colour in ((-0.16, "attr", SERIES[0]), (0.16, "conseq", SERIES[1])):
            x = mc[f"london_delta_mt_{tag}_{key}"]
            _interval(ax, y + dy, x.quantile(0.05), x.median(), x.quantile(0.95), colour, size=4)
        med = mc[f"london_delta_mt_attr_{key}"].median()
        ylab.append(R["baskets"][key].label)
        ax.text(-0.08, y, f"{-100 * med / leggi:.0f}% of\nLEGGI", fontsize=5.0, color=INK2,
                va="center", ha="right")
    ax.set_yticks(range(len(keys)), ylab)
    ax.invert_yaxis()
    ax.set_xlabel("Change in London's emissions\n(Mt CO$_2$e yr$^{-1}$)")
    ax.axvline(0, color=AXIS, linewidth=0.6)
    _hgrid(ax)
    ax.set_xlim(None, 0.05)
    ax.legend(handles=[plt.Line2D([], [], color=SERIES[0], marker="o", markersize=3, label="Attributional"),
                       plt.Line2D([], [], color=SERIES[1], marker="o", markersize=3, label="Consequential")],
              loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=5.2)
    panel_label(ax, "b", x=-0.95, y=1.02)

    # c: GWP100 vs GWP* cumulative
    ax = fig.add_subplot(gs[0, 2])
    g = R["gwpstar"]
    ax.plot(g["year"], g["cum_gwp100_mt"], color=SERIES[0], linewidth=1.4, label="GWP$_{100}$ (AR6)")
    ax.plot(g["year"], g["cum_star_abrupt_mt"], color=SERIES[1], linewidth=1.4,
            label="GWP* (warming-equivalent)")
    ax.plot(g["year"], g["cum_star_phase_mt"], color=SERIES[1], linewidth=0.9, alpha=0.6,
            label="GWP*, 10-yr phase-out")
    ax.axhline(0, color=AXIS, linewidth=0.6)
    ax.set_xlabel("Years after removal")
    ax.set_ylabel("Cumulative change (Mt CO$_2$-we)")
    ax.set_xlim(0, 100)
    _hgrid(ax, "y")
    ax.legend(loc="lower left", fontsize=5.4)
    panel_label(ax, "c", x=-0.32, y=1.02)
    _save(fig, "fig2_emissions")


# --------------------------------------------------------------------- Fig. 3


def fig_campus(R: dict) -> None:
    fig = plt.figure(figsize=(DOUBLE, 70 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.62)

    # a: pass-through surface
    ax = fig.add_subplot(gs[0, 0])
    eta = np.logspace(np.log10(0.05), np.log10(5), 200)
    eps = np.linspace(0.1, 1.6, 200)
    E, P = np.meshgrid(eta, eps)
    phi = pass_through(E, -P)
    cs = ax.contourf(E, P, phi, levels=np.linspace(0, 1, 11), cmap=BLUES)
    cl = ax.contour(E, P, phi, levels=[0.25, 0.5, 0.68, 0.9], colors="white", linewidths=0.6)
    ax.clabel(cl, fmt=lambda v: f"{v:.2f}", fontsize=5, inline=True)
    mk = R["markets"].loc["beef"]
    ax.plot(mk["eta_sr"], -mk["eps"], marker="s", color=INK, markersize=4, markeredgecolor="white")
    ax.annotate("beef,\nshort run", (mk["eta_sr"], -mk["eps"]), xytext=(-5, -4), ha="right", va="top",
                textcoords="offset points", fontsize=5.2, color=INK)
    ax.plot(mk["eta_lr"], -mk["eps"], marker="o", color=INK, markersize=4, markeredgecolor="white")
    ax.annotate("beef,\nlong run", (mk["eta_lr"], -mk["eps"]), xytext=(5, -4), ha="left", va="top",
                textcoords="offset points", fontsize=5.2, color="white")
    ax.set_xscale("log")
    ax.set_xlabel("Supply elasticity $\\eta$")
    ax.set_ylabel("Demand elasticity $|\\varepsilon|$")
    cb = fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.03, location="top")
    cb.set_label("Pass-through $\\phi=\\eta/(\\eta-\\varepsilon)$", fontsize=5.6, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.2, length=2)
    panel_label(ax, "a", x=-0.3, y=1.02)

    # b: decomposition (long run), medians with 90% intervals on the net bars
    ax = fig.add_subplot(gs[0, 1])
    lr, sr = R["campus_lr"], R["campus_sr"]
    steps = [("gross", "Beef removed"), ("substitutes_gross", "Replacement food"),
             ("offcampus", "Beef bought off campus"), ("market_leakage", "Beef market leakage"),
             ("substitute_adj", "Replacement market adj.")]
    level = 0.0
    xs = []
    for i, (col, lab) in enumerate(steps):
        v = lr[col].median()
        colour = DIVERGING[0] if v < 0 else DIVERGING[2]
        ax.bar(i, v, bottom=level, width=0.62, color=colour, edgecolor="white", linewidth=0.8)
        ax.text(i, level + v + (6 if v >= 0 else -6), f"{v:+.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=5.4, color=INK)
        if i < len(steps) - 1:
            ax.plot([i + 0.31, i + 0.69], [level + v] * 2, color=MUTED, linewidth=0.5)
        level += v
        xs.append(lab)
    for j, (frame, lab) in enumerate(((lr, "Net, long run"), (sr, "Net, short run"),
                                      (lr, "Procurement accounting"))):
        col = "accounting" if lab.startswith("Proc") else "net"
        x = frame[col]
        i = len(steps) + j
        ax.bar(i, x.median(), width=0.62, color=MUTED if col == "net" else AXIS,
               edgecolor="white", linewidth=0.8)
        ax.plot([i, i], [x.quantile(0.05), x.quantile(0.95)], color=INK, linewidth=0.8)
        ax.text(i + 0.36, x.median(), f"{x.median():.0f}", ha="left", va="center", fontsize=5.4,
                color=INK)
        xs.append(lab)
    ax.axhline(0, color=AXIS, linewidth=0.6)
    ax.set_xticks(range(len(xs)), xs, rotation=40, ha="right")
    ax.set_ylabel("t CO$_2$e yr$^{-1}$ (median)")
    _hgrid(ax, "y")
    panel_label(ax, "b", x=-0.2, y=1.02)

    # c: distributions of the net effect
    ax = fig.add_subplot(gs[0, 2])
    lo = min(lr["net"].quantile(0.005), sr["net"].quantile(0.005), lr["accounting"].quantile(0.005))
    hi = max(lr["net"].quantile(0.995), sr["net"].quantile(0.995), 0)
    bins = np.linspace(lo, hi, 50)
    for frame, col, colour, lab in ((lr, "accounting", AXIS, "Procurement accounting"),
                                    (lr, "net", SERIES[0], "Net, long run"),
                                    (sr, "net", SERIES[1], "Net, short run")):
        h, edges = np.histogram(frame[col], bins=bins, density=True)
        ax.stairs(h, edges, color=colour, linewidth=1.2, label=lab)
    ax.axvline(0, color=AXIS, linewidth=0.6)
    ax.set_xlabel("Net change (t CO$_2$e yr$^{-1}$)")
    ax.set_ylabel("Probability density")
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=5.2)
    panel_label(ax, "c", x=-0.12, y=1.02)
    ax.set_xlim(lo, 15)
    _save(fig, "fig3_campus")


# --------------------------------------------------------------------- Fig. 4


def fig_students(R: dict) -> None:
    from .welfare import alpha_from_elasticity, cv_per_meal

    P = R["params"]
    fig = plt.figure(figsize=(DOUBLE, 64 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.55)

    # a: per-meal welfare loss vs dish elasticity for three meal prices (lambda drops out)
    ax = fig.add_subplot(gs[0, 0])
    p_b, p_m, lam = (R["welfare_central"]["p_beef"], P.v("welfare.meat_nest_share"),
                     P.v("welfare.nest_lambda"))
    e = np.linspace(0.5, 4.0, 200)
    for price, colour in ((7.5, SERIES[0]), (6.0, SERIES[1]), (4.5, SERIES[2])):
        a = alpha_from_elasticity(-e, price, p_b, p_m, lam)
        loss = -cv_per_meal(a, p_b, p_m, lam) / p_b
        ax.plot(e, loss, color=colour, linewidth=1.4, label=f"meal price £{price:.2f}")
    cap = P["welfare.outside_option_cost_gbp"]
    ax.axhspan(cap.low, cap.high, color=GRID, alpha=0.8, linewidth=0)
    ax.text(3.95, cap.high + 0.15, "cost of buying beef\noff campus", ha="right", va="bottom",
            fontsize=5.2, color=INK2)
    ax.set_xlabel("Own-price elasticity of a beef dish $|\\varepsilon_B|$")
    ax.set_ylabel("Welfare loss per forgone beef meal (£)")
    ax.set_ylim(0, 12)
    _hgrid(ax, "y")
    ax.legend(loc="upper right", fontsize=5.4)
    panel_label(ax, "a", x=-0.25, y=1.02)

    # b: annual welfare loss vs monetised climate benefit
    ax = fig.add_subplot(gs[0, 1])
    W = R["welfare"]
    items = [("benefit_gbp", "Climate benefit,\nlong run", SERIES[0]),
             ("loss_capped_total", "Welfare loss,\ncapped logit", SERIES[1]),
             ("loss_total", "Welfare loss,\nnested logit", SERIES[1])]
    for y, (col, lab, colour) in enumerate(items):
        x = W[col] / 1000
        _interval(ax, y, x.quantile(0.05), x.median(), x.quantile(0.95), colour)
        ax.text(x.quantile(0.95) + 8, y, f"£{x.median():.0f}k", va="center", fontsize=5.4, color=INK)
    ax.set_yticks(range(len(items)), [i[1] for i in items])
    ax.invert_yaxis()
    ax.set_xlim(0, W["loss_total"].quantile(0.95) / 1000 * 1.25)
    ax.set_ylim(len(items) - 0.5, -0.5)
    ax.set_xlabel("£ thousand per year")
    _hgrid(ax)
    panel_label(ax, "b", x=-0.45, y=1.02)

    # c: change in daily intake for a frequent campus beef eater (% of RNI, women 19-50)
    ax = fig.add_subplot(gs[0, 2])
    tab = R["frequent_rni"]
    for y, (nut, r) in enumerate(tab.iterrows()):
        for dy, col, colour in ((-0.17, "pct_women", SERIES[0]), (0.17, "pct_men", SERIES[1])):
            v = r[col]
            ax.barh(y + dy, v, height=0.32, color=colour, edgecolor="white", linewidth=0.6)
        ax.text(min(r["pct_women"], r["pct_men"], 0) - 1.5, y, f"{r['pct_women']:+.0f}%",
                ha="right", va="center", fontsize=5.2, color=INK)
    ax.set_yticks(range(len(tab)), tab["label"])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=0.6)
    ax.set_xlabel("Change in daily intake\n(% of UK reference value)")
    lo = tab[["pct_women", "pct_men"]].min().min()
    ax.set_xlim(lo * 1.6, max(5, tab[["pct_women", "pct_men"]].max().max() * 1.3))
    _hgrid(ax)
    ax.legend(handles=[Patch(color=SERIES[0], label="Women 19–50"),
                       Patch(color=SERIES[1], label="Men 19–50")], loc="lower left", fontsize=5.2)
    panel_label(ax, "c", x=-0.45, y=1.02)
    _save(fig, "fig4_students")


# -------------------------------------------------------------- Extended Data


def ed_emission_factors(R: dict) -> None:
    efs, foods = R["efs"], R["foods"]
    items = [f for f in foods.index if f != nu.REF and foods.loc[f, "diet"] != "supplement"]
    items = sorted(items, key=lambda f: efs.loc[f, "ef_central"])
    fig, ax = plt.subplots(figsize=(SINGLE * 1.25, 80 * MM))
    for y, f in enumerate(items):
        r = efs.loc[f]
        _interval(ax, y, r["ef_low"], r["ef_central"], r["ef_high"], SERIES[0], size=4)
        if np.isfinite(r.get("ef_pn", np.nan)):
            ax.plot(r["ef_pn"], y, marker="D", markersize=3, color=SERIES[1], markeredgecolor="white",
                    markeredgewidth=0.5, zorder=4)
    seg = R["beef_segments"]
    y0 = len(items)
    for k, (name, r) in enumerate(seg.iterrows()):
        ax.plot(r["ef_retail_attr"], y0 + k, marker="o", markersize=4, color=SERIES[7],
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_yticks(range(len(items) + len(seg)),
                  [short_name(foods, f) for f in items]
                  + [f"Beef: {R['supply'].loc[s, 'label']}" for s in seg.index])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("kg CO$_2$e per kg as purchased (beef: per kg boneless retail)")
    _hgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], color=SERIES[0], marker="o", markersize=3,
                                  label="Median and 5th-95th percentile"),
                       plt.Line2D([], [], color=SERIES[1], marker="D", markersize=3, ls="",
                                  label="Poore & Nemecek mean"),
                       plt.Line2D([], [], color=SERIES[7], marker="o", markersize=3, ls="",
                                  label="UK beef supply segments")],
              loc="lower right", fontsize=5.2)
    _save(fig, "ed1_emission_factors")


def ed_tornado(R: dict) -> None:
    t = R["tornado"].sort_values("span")
    base = R["tornado_base"]
    fig, ax = plt.subplots(figsize=(SINGLE * 1.25, 62 * MM))
    for y, (name, r) in enumerate(t.iterrows()):
        ax.barh(y, r["low_value"] - base, left=base, height=0.55, color=SERIES[0], edgecolor="white")
        ax.barh(y, r["high_value"] - base, left=base, height=0.55, color=SERIES[1], edgecolor="white")
    ax.axvline(base, color=INK, linewidth=0.6)
    ax.set_yticks(range(len(t)), t["label"])
    ax.set_xlabel("Change in London's emissions (Mt CO$_2$e yr$^{-1}$), poultry/pork-led basket")
    _hgrid(ax)
    ax.legend(handles=[Patch(color=SERIES[0], label="Input at low bound"),
                       Patch(color=SERIES[1], label="Input at high bound")], loc="lower left", fontsize=5.2)
    _save(fig, "ed2_tornado")


def ed_basket_sensitivity(R: dict) -> None:
    sens = R["basket_sensitivity"]
    base = R["baskets"]
    cases = [("central", "Central (key nutrients)", base)] + [(k, lab, sens[k]) for k, lab in (
        ("naive", "No bioavailability adj."), ("plant_unfavourable", "Plant-unfavourable"),
        ("plant_favourable", "Plant-favourable"), ("energy_strict", "Energy within ±15%"),
        ("full10", "All ten nutrients"))]
    keys = nu.Q1_SCENARIOS + nu.Q2_SCENARIOS
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.8, 60 * MM), sharey=True)
    for c, (tag, lab, res) in enumerate(cases):
        for y, k in enumerate(keys):
            bk = res[k]
            dy = (c - 2.5) * 0.12
            if not bk.feasible:
                axes[0].plot(bk.ghg, y + dy, marker="o", ls="", markersize=3.5, markerfacecolor="white",
                             markeredgecolor=SERIES[c], markeredgewidth=0.8)
                axes[1].plot(bk.mass_g, y + dy, marker="o", ls="", markersize=3.5, markerfacecolor="white",
                             markeredgecolor=SERIES[c], markeredgewidth=0.8)
                continue
            axes[0].plot(bk.ghg, y + dy, marker="o", ls="", color=SERIES[c], markersize=3.5,
                         markeredgecolor="white", markeredgewidth=0.5, label=lab if y == 0 else None)
            axes[1].plot(bk.mass_g, y + dy, marker="o", ls="", color=SERIES[c], markersize=3.5,
                         markeredgecolor="white", markeredgewidth=0.5)
    axes[0].set_yticks(range(len(keys)), [base[k].label for k in keys])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("kg CO$_2$e per 100 g beef replaced")
    axes[1].set_xlabel("Basket mass (g as eaten)")
    for a in axes:
        _hgrid(a)
    axes[1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=SERIES[c], markersize=3.5,
                                       label=lab) for c, (_, lab, _) in enumerate(cases)]
                   + [plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2,
                                 markersize=3.5, label="hollow = shortfall remains")],
                   loc="lower right", fontsize=5)
    panel_label(axes[0], "a", x=-0.7)
    panel_label(axes[1], "b", x=-0.1)
    _save(fig, "ed3_basket_sensitivity")


def ed_campus_grid(R: dict) -> None:
    """Net campus change vs the share of forgone beef meals bought off campus."""
    g = R["campus_grid"]
    fig, ax = plt.subplots(figsize=(SINGLE * 1.15, 60 * MM))
    for phi, colour, lab in ((1.0, SERIES[0], "1.0 (no market leakage)"),
                             (R["campus_lr"]["pass_through_beef"].median(), SERIES[1], "long-run median"),
                             (R["campus_sr"]["pass_through_beef"].median(), SERIES[2], "short-run median"),
                             (0.0, SERIES[7], "0 (production unchanged)")):
        col = g.columns[np.argmin(np.abs(g.columns - phi))]
        ax.plot(g.index, g[col], color=colour, linewidth=1.4, label=f"$\\phi$ = {phi:.2f}: {lab}")
    ax.axhline(0, color=AXIS, linewidth=0.6)
    ax.set_xlabel("Share of forgone beef meals bought as beef off campus")
    ax.set_ylabel("Net change (t CO$_2$e yr$^{-1}$)")
    _hgrid(ax, "y")
    ax.legend(loc="lower right", fontsize=5.2, title="Beef pass-through", title_fontsize=5.4)
    _save(fig, "ed4_campus_grid")


def si_foi(R: dict) -> None:
    foi = R["foi"]
    fig, ax = plt.subplots(figsize=(SINGLE * 1.2, 58 * MM))
    x = np.arange(len(foi))
    for col, lab, colour in (("chicken_kg", "Chicken (lower bound)", SERIES[0]),
                             ("pork_kg", "Pork", SERIES[1]), ("beef_kg", "Beef", SERIES[7]),
                             ("lamb_kg", "Lamb", SERIES[3]), ("venison_kg", "Venison", SERIES[2])):
        y = foi[col] / 1000
        ax.plot(x, y, color=colour, linewidth=1.4, marker="o", markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.6, label=lab)
        ax.text(x[-1] + 0.08, y.iloc[-1], f"{y.iloc[-1]:.1f}", va="center", fontsize=5.2, color=INK)
    ax.axvline(2.6, color=MUTED, linewidth=0.6)
    ax.text(2.55, ax.get_ylim()[1] * 0.62, "beef removed\n(Oct 2024)", fontsize=5.2, color=INK2,
            va="top", ha="right")
    ax.set_xticks(x, foi.index)
    ax.set_xlim(-0.2, len(foi) - 0.55)
    ax.set_ylabel("Purchases (t product per year)")
    _hgrid(ax, "y")
    ax.legend(loc="upper left", fontsize=5.2)
    _save(fig, "si_foi_purchases")


def make_all(R: dict) -> None:
    apply()
    fig_nutrition(R)
    fig_emissions(R)
    fig_campus(R)
    fig_students(R)
    ed_emission_factors(R)
    ed_tornado(R)
    ed_basket_sensitivity(R)
    ed_campus_grid(R)
    si_foi(R)
