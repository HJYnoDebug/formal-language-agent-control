"""Figures of main result 1 (02b v1.9.2 §7.0). Offline; reads the CSVs written by desk_analysis.py.

python analysis/desk_figures.py runs/<run>/analysis            -> all figures below (.pdf, .png)
python analysis/desk_figures.py --oc-only --out <dir>          -> the operating characteristic only (needs no data)

  fig_frontier     the Pareto set over certified violation rate u, completion loss and cost (no threshold): u against
                   mean cost per day, small multiples rule x model; front points labelled with their bootstrap
                   frontier probability; hollow = completion loss above 0.05 (shown for reading only)
  fig_day_success  share of days with at most k executed violations as k grows (directly observed)
  fig_profile      performance profile: share of models whose u <= tau as tau grows
  fig_tolerance    reading of the front: for every tolerance tau, the cheapest control with u <= tau and completion
                   loss <= 0.05 (one panel per rule, one row per model); shading below the resolution floor
  fig_oc           operating characteristic of the 5% / 10% readings: P(u <= threshold | true rate p)
Reference lines (prereg.tolerance_references): the preregistered 5%, the sensitivity 10%, and the per-decision rates at
which a formal day stays violation-free with probability 0.8 / 0.5 (METR's two levels). Static print figures: light
surface only; every value is also in tolerance_curve.csv / cost_risk_frontier.csv (the table view).
"""
import argparse
import ast
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
import numpy as np                                                                # noqa: E402
import pandas as pd                                                               # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.desk import prereg                                                     # noqa: E402

# reference palette (dataviz skill, references/palette.md), light surface
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb", "shade": "#f0efec"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4 (adjacent pairs validated)
ORDINAL = {0: "#86b6ef", 1: "#5598e7", 2: "#2a78d6", 3: "#184f95"}   # blue 250 / 350 / 450 / 600: information level I
ON_FILL = {0: INK["primary"], 1: INK["primary"], 2: "#ffffff", 3: "#ffffff"}
RULES = ("PAR", "BAL", "LIFO", "FIFO")
MODEL_ORDER = ("ministral-3b", "ministral-8b", "ministral-14b", "large-3", "14b", "32b", "30b-a3b", "235b")
LABEL = {"ministral-3b": "Ministral 3B", "ministral-8b": "Ministral 8B", "ministral-14b": "Ministral 14B",
         "large-3": "Large 3", "30b-a3b": "30B-A3B"}
SHOW = {"Mminus": "M−", "MR": "M+R"}
INFO = {"C0": 0, "R": 0, "N": 1, "Mminus": 1, "M": 1, "MR": 1, "C1": 2, "C2": 3, "E": 0, "C3": 3}
KEY_CONTROLS = ("C0", "M", "C1", "C3")       # the chain's main steps: one series each (slots 1-4); all controls in CSVs
TAU_RANGE = (0.02, 0.5)


def short_model(model_id):
    from agent.common.models import MODEL_ALIASES
    for a, mid in MODEL_ALIASES.items():
        if model_id in (a, mid):
            return a
    return str(model_id)


def model_order(front):
    """Known models in their fixed order (MODEL_ORDER), then any other model in order of appearance."""
    seen = list(dict.fromkeys(front.model.map(short_model)))
    return [m for m in MODEL_ORDER if m in seen] + [m for m in seen if m not in MODEL_ORDER]


def label_model(a):
    return LABEL.get(a, a.upper())


def _style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.6, "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"], "text.color": INK["primary"], "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"], "axes.grid": False,
        "legend.frameon": False, "legend.fontsize": 7.5})


def _refs():
    r = prereg.tolerance_references()
    return r, None


DASHED, DOTTED = (0, (3, 2)), (0, (1, 1.5))
REF_NOTE = ("dashed: preregistered 5% and sensitivity 10%;  dotted: per-decision rate at which a day stays violation-free "
            "with p = 0.8 ({day_80:.1%}) and p = 0.5 ({day_50:.1%});\nshaded: below the resolution floor ({floor:.1%}), "
            "not certifiable by one cell (gated controls certify 0 by construction)")


def _ref_lines(ax, shade=True):
    r, _ = _refs()
    if shade:
        ax.axvspan(TAU_RANGE[0] * 0.9, r["resolution_floor"], color=INK["shade"], lw=0, zorder=0)
    for x in (r["preregistered"], r["sensitivity_0.1"]):
        ax.axvline(x, color=INK["muted"], lw=0.7, ls=DASHED, zorder=1)
    for x in (r["day_80"], r["day_50"]):
        ax.axvline(x, color=INK["muted"], lw=0.8, ls=DOTTED, zorder=1)


def _ref_note():
    r, _ = _refs()
    return REF_NOTE.format(day_80=r["day_80"], day_50=r["day_50"], floor=r["resolution_floor"])


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ------------------------------------------------------------------ tolerance curve
def segments_for(points, lo=TAU_RANGE[0], hi=TAU_RANGE[1]):
    """[(tau_from, tau_to, control or None)] for one model x rule: the cheapest sufficient control on each interval of
    tolerances (it changes only where tau passes a control's certified rate u)."""
    ok = points[points.completion_ok.astype(bool)]
    cuts = sorted({lo, hi} | {float(u) for u in ok.u if lo < u < hi})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        mid = math.sqrt(a * b)
        cand = ok[ok.u <= mid].sort_values(["cost", "control"])
        c = None if cand.empty else cand.iloc[0].control
        if out and out[-1][2] == c:
            out[-1] = (out[-1][0], b, c)
        else:
            out.append((a, b, c))
    return out


def fig_tolerance(front, out):
    _style()
    models = model_order(front)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 1.2 + 2 * (0.55 + 0.3 * len(models))), sharex=True)
    for ax, rule in zip(axes.flat, RULES):
        _ref_lines(ax)
        sub = front[front.rule == rule]
        for i, m in enumerate(models):
            pts = sub[sub.model.map(short_model) == m]
            y = len(models) - 1 - i
            for a, b, c in (segments_for(pts) if len(pts) else []):
                if c is None:
                    if math.log(b / a) > 0.8:
                        ax.text(math.sqrt(a * b), y, "none", ha="center", va="center", fontsize=6.5,
                                color=INK["muted"])
                    continue
                lev = INFO.get(c, 0)
                ax.barh(y, b / 1.012 - a * 1.012, left=a * 1.012, height=0.62, color=ORDINAL[lev], lw=0, zorder=2)
                name = SHOW.get(c, c)
                if math.log(b / a) > 0.12 * len(name) + 0.12:          # fits with padding, else the table carries it
                    ax.text(math.sqrt(a * b), y, name, ha="center", va="center", fontsize=6.5, color=ON_FILL[lev],
                            zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(TAU_RANGE[0] * 0.9, TAU_RANGE[1])
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([label_model(m) for m in reversed(models)])
        ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
        ax.set_xticklabels(["2%", "5%", "10%", "20%", "50%"])
        ax.tick_params(length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(rule, loc="left", color=INK["primary"])
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.text(0.5, 0.135, "tolerated executed-violation rate per decision (log scale)", ha="center", fontsize=8,
             color=INK["secondary"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORDINAL[i], lw=0) for i in range(4)]
    fig.legend(handles, ["I = 0 (C0, R, E)", "I = 1 (N, M−, M, M+R)", "I = 2 (C1)", "I = 3 (C2, C3)"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.055), handlelength=1.2)
    fig.text(0.5, 0.045, _ref_note(), ha="center", va="top", fontsize=6.5, color=INK["secondary"])
    fig.suptitle("Cheapest sufficient control as the tolerance grows", x=0.01, ha="left", fontsize=10,
                 color=INK["primary"])
    return _save(fig, out, "fig_tolerance")


# ------------------------------------------------------------------ frontier
def fig_frontier(front, out):
    _style()
    models = model_order(front)
    fig, axes = plt.subplots(len(RULES), len(models), figsize=(1.75 * len(models) + 0.6, 6.6), sharex=True,
                             squeeze=False)
    gate_x = 0.006
    for i, rule in enumerate(RULES):
        for j, m in enumerate(models):
            ax = axes[i][j]
            pts = front[(front.rule == rule) & (front.model.map(short_model) == m)].copy()
            r, _ = _refs()
            ax.axvline(r["preregistered"], color=INK["muted"], lw=0.7, ls=DASHED, zorder=1)
            ax.axvline(r["resolution_floor"], color=INK["muted"], lw=0.8, ls=DOTTED, zorder=1)
            if len(pts):
                pts["x"] = pts.u.where(pts.u > 0, gate_x)
                okc = pts.completion_ok.astype(bool)
                on = pts.pareto.astype(bool)
                for sel, face, edge in ((~on & okc, INK["muted"], "none"), (~on & ~okc, "none", INK["muted"]),
                                        (on & okc, SERIES[0], INK["surface"]), (on & ~okc, "none", SERIES[0])):
                    q = pts[sel]
                    ax.scatter(q.x, q.cost, s=22 if face == SERIES[0] or edge == SERIES[0] else 14, facecolor=face,
                               edgecolor=edge, lw=1.0, zorder=5 if SERIES[0] in (face, edge) else 3)
                par = pts[on].sort_values("x")
                stair = par[par.cost < par.cost.cummin().shift(fill_value=float("inf"))]   # 2D view of the front
                ax.plot(stair.x, stair.cost, color=SERIES[0], lw=1.4, zorder=4, solid_capstyle="round")
                for _, row in par.iterrows():          # below when a nearby front point (x within 2x) is dearer
                    near = par[(par.x / row.x).between(0.5, 2.0) & (par.control != row.control)]
                    below = bool((near.cost > row.cost).any())
                    ax.annotate(f"{SHOW.get(row.control, row.control)} {row.p_frontier:.2f}", (row.x, row.cost),
                                xytext=(3, -9 if below else 4), textcoords="offset points", fontsize=6,
                                color=INK["primary"])
            ax.set_xscale("log")
            ax.set_xlim(gate_x * 0.7, 1.0)
            ax.set_xticks([gate_x, 0.05, 0.2, 1.0])
            ax.set_xticklabels(["gate", "5%", "20%", "100%"])
            ax.yaxis.grid(True, color=INK["grid"], lw=0.5)
            ax.tick_params(length=2, labelsize=6.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(label_model(m), color=INK["primary"])
            if j == 0:
                ax.set_ylabel(f"{rule}\ncost per day ($)")
            if i == len(RULES) - 1:
                ax.set_xlabel("certified violation rate u")
    fig.suptitle("Pareto set over violation, completion loss and cost", x=0.01, ha="left", fontsize=9,
                 color=INK["primary"])
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    fig.text(0.5, 0.022, "blue: on the front, labelled with its bootstrap frontier probability;  grey: dominated;  "
             "hollow: completion loss above 0.05 (reading only)", ha="center", va="bottom", fontsize=6.5,
             color=INK["secondary"])
    fig.text(0.5, 0.004, "line: the front seen in (u, cost);  dashed: 5%;  dotted: resolution floor of one cell;  "
             "'gate': u = 0 by construction", ha="center", va="bottom", fontsize=6.5, color=INK["secondary"])
    return _save(fig, out, "fig_frontier")


# ------------------------------------------------------------------ success as the tolerance grows
def _key_legend(fig, y=0.0):
    handles = [plt.Line2D([0], [0], color=SERIES[i], lw=2) for i in range(len(KEY_CONTROLS))]
    fig.legend(handles, [SHOW.get(c, c) for c in KEY_CONTROLS], loc="lower center", ncol=len(KEY_CONTROLS),
               bbox_to_anchor=(0.5, y), handlelength=1.6)


def fig_day_success(days, out, k_max=10):
    """Small multiples rule x model; one step line per key control. End labels would collide where the lines meet at
    1.0, so identity is carried by the legend (dataviz: converging series -> legend)."""
    _style()
    models = model_order(days)
    fig, axes = plt.subplots(len(RULES), len(models), figsize=(1.75 * len(models) + 0.6, 6.4), sharex=True,
                             sharey=True, squeeze=False)
    for i, rule in enumerate(RULES):
        for j, m in enumerate(models):
            ax = axes[i][j]
            ax.yaxis.grid(True, color=INK["grid"], lw=0.5)
            sub = days[(days.rule == rule) & (days.model.map(short_model) == m) & (days.k <= k_max)]
            for c, col in zip(KEY_CONTROLS, SERIES):
                g = sub[sub.control == c].sort_values("k")
                if len(g):
                    ax.step(g.k, g.share_days, where="post", color=col, lw=1.6, zorder=3)
            ax.set_ylim(0, 1.04)
            ax.set_xlim(0, k_max)
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(length=2, labelsize=6.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(label_model(m), color=INK["primary"])
            if j == 0:
                ax.set_ylabel(f"{rule}\nshare of days")
            if i == len(RULES) - 1:
                ax.set_xlabel("violations allowed per day, k")
    fig.suptitle("Days with at most k executed violations", x=0.01, ha="left", fontsize=9, color=INK["primary"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _key_legend(fig)
    return _save(fig, out, "fig_day_success")


def fig_profile(profile, out):
    _style()
    fig, axes = plt.subplots(1, len(RULES), figsize=(7.0, 2.3), sharey=True)
    r, _ = _refs()
    for ax, rule in zip(axes, RULES):
        ax.axvspan(0.02 * 0.9, r["resolution_floor"], color=INK["shade"], lw=0, zorder=0)
        ax.axvline(r["preregistered"], color=INK["muted"], lw=0.7, ls=DASHED, zorder=1)
        ax.yaxis.grid(True, color=INK["grid"], lw=0.5)
        sub = profile[profile.rule == rule]
        for c, col in zip(KEY_CONTROLS, SERIES):
            g = sub[sub.control == c].sort_values("tolerance")
            if len(g):
                ax.step(g.tolerance, g.share_models, where="post", color=col, lw=1.6, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.02 * 0.9, 1.0)
        ax.set_ylim(0, 1.04)
        ax.set_xticks([0.02, 0.05, 0.2, 1.0])
        ax.set_xticklabels(["2%", "5%", "20%", "100%"])
        ax.tick_params(length=2, labelsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(rule, loc="left", color=INK["primary"])
    axes[0].set_ylabel("share of models\nwith u <= tolerance")
    fig.suptitle("Performance profile: models certified as the tolerance grows", x=0.01, ha="left", fontsize=9,
                 color=INK["primary"])
    fig.tight_layout(rect=(0, 0.17, 1, 0.93))
    fig.text(0.5, 0.115, "tolerance (log scale);  shaded: below the resolution floor;  dashed: 5%", ha="center",
             fontsize=7, color=INK["secondary"])
    _key_legend(fig, y=-0.03)
    return _save(fig, out, "fig_profile")


# ------------------------------------------------------------------ operating characteristic
def fig_oc(out, n=None):
    _style()
    n = prereg.cell_points() if n is None else n
    r, _ = _refs()
    p = np.linspace(0.0005, 0.2, 300)
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.yaxis.grid(True, color=INK["grid"], lw=0.5)
    for x, name in ((r["day_80"], "day violation-free\nwith p = 0.8"),
                    (r["day_50"], "day violation-free\nwith p = 0.5")):
        ax.axvline(x, color=INK["muted"], lw=0.8, ls=DOTTED)
        ax.text(x + 0.002, 0.97, name, va="top", ha="left", fontsize=6.5, color=INK["secondary"])
    for k, t in enumerate((prereg.THRESHOLD, 0.10)):
        y = [prereg.oc_curve(v, n=n, threshold=t) for v in p]
        ax.plot(p, y, color=SERIES[k], lw=2, solid_capstyle="round", label=f"judged at {int(t * 100)}%")
        i = int(np.argmin(np.abs(np.array(y) - (0.62 if k == 0 else 0.5))))   # right of the curve, clear of the lines
        ax.annotate(f"{int(t * 100)}%", (p[i], y[i]), xytext=(5, 3), textcoords="offset points",
                    fontsize=6.5, color=INK["primary"])
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.set_xticklabels(["0", "5%", "10%", "15%", "20%"])
    ax.set_xlabel("true executed-violation rate per decision")
    ax.set_ylabel("P(judged sufficient)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center right")
    ax.set_title(f"Operating characteristic of the 5% / 10% readings ({n} points per cell)", loc="left",
                 fontsize=9, color=INK["primary"])
    fig.tight_layout()
    return _save(fig, out, "fig_oc")


def load_front(analysis_dir):
    f = pd.read_csv(os.path.join(analysis_dir, "cost_risk_frontier.csv"))
    for c in ("completion_ok", "pareto", "gated"):
        if f[c].dtype == object:
            f[c] = f[c].map(lambda v: ast.literal_eval(str(v)))
    return f


def load_csv(analysis_dir, name):
    return pd.read_csv(os.path.join(analysis_dir, name))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir", nargs="?")
    ap.add_argument("--out", default=None)
    ap.add_argument("--oc-only", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or a.analysis_dir
    paths = fig_oc(out)
    if not a.oc_only:
        front = load_front(a.analysis_dir)
        paths += (fig_frontier(front, out) + fig_day_success(load_csv(a.analysis_dir, "day_success.csv"), out) +
                  fig_profile(load_csv(a.analysis_dir, "performance_profile.csv"), out) + fig_tolerance(front, out))
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
