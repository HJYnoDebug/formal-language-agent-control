"""Analysis of the help-desk experiment (02b §7). Offline; reads the episode records of one or more runs.

python analysis/desk_analysis.py runs/<run> [runs/<run2> ...] [--out runs/<run>/analysis]

Main results (02b v1.6 §7.0; v1.9.2: result 1 is the threshold-free Pareto set over certified violation rate,
completion loss and cost, with bootstrap frontier probabilities; day-level success and the performance profile show how
success grows with the tolerance; the tolerance curve and the preregistered 5% map are readings of it; figures:
analysis/desk_figures.py):
  1. minimal-sufficient control map: for every model x rule (pooled over depth), the sufficient controls, I* (lowest
     information level sufficient with the gate off), whether the gate is needed, the minimal elements of the partial
     order and the cheapest sufficient control; depth enters through the stepwise models of H2, H4, H6. A segmented
     map (shallow / deep days; BAL by M_t) is reported descriptively at the 10% threshold;
  2. level signature: along which coordinate each rule's errors grow (H4), and H1-H6 with Holm correction. H1 (v1.7,
     option b) is the direct C0 test regular vs BAL; the I* ordering along the levels is descriptive, BAL vs LIFO /
     FIFO is exploratory (state / decision components) and error rates are also reported on the first point of each
     hold segment and on clean points (robust_measures).
Exploratory (02b §7.1): window floor and memory gain, cost, success and steps, analyses B (effective memory k),
C (violation attribution), E (device consistency), E' (how the notebook N is used), G (two layers), FREE resume
order, self-conditioning; the two-tray probe M2S if it was run (02b §15).

Coordinates (02b v1.6, §14 D2): burial beta_t for LIFO / FIFO and the counter bound M_t for BAL, the coordinates on
which bounded recognisers fail (acceptance item 6); PAR has none (its tests use beta_t as the depth axis). "v1.5"
reruns everything on d_t / c_t as a secondary report. Sufficiency: SUFFICIENCY holds the threshold, the confidence level and how the bound is
computed ("exact": Clopper-Pearson on the segment's points, on all the cell's points when the cell has no violation;
"stepwise": the 02b §7.0 logistic model along the coordinate, which fails under separation and then falls back to
exact). Gated controls execute no violation by construction, so only their completion is judged. Both settings are
fixed before data (pending the design freeze).
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import statistics
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.desk import plan as planmod, prereg                                  # noqa: E402
from agent.desk.runner import run_workflow                                       # noqa: E402
from agent.desk.strategies import WINDOW_POLICIES, make_chooser, simulate        # noqa: E402

COORDINATES = {
    "theory": {"LIFO": "burial", "FIFO": "burial", "BAL": "counter_peak", "LAST": None, "PAR": None},
    "v1.5": {"LIFO": "depth", "FIFO": "depth", "BAL": "excursion", "LAST": None, "PAR": None},
}
COORD = COORDINATES["theory"]
DEPTH_AXIS = {"theory": "burial", "v1.5": "depth"}
SUFFICIENCY = {"threshold": prereg.THRESHOLD, "level": prereg.LEVEL, "completion_margin": prereg.COMPLETION_MARGIN,
               "segment": False, "bound": "exact"}
SEGMENTED_THRESHOLD = 0.10                    # descriptive segmented map (02b v1.6 §7.0)
# H2, H4, H6 contrast "high" = coordinate >= the rule's design median of that coordinate (contrast_cut: prereg.CUTS,
# prereg.CUTS_OTHER for beta_t / M_t); linear per-SD slopes are secondary
INFO = {"C0": 0, "R": 0, "N": 1, "Mminus": 1, "M": 1, "MR": 1, "C1": 2, "C2": 3, "E": 0, "C3": 3, "P": 0, "M2S": 1}
GATE = {"E": 1, "C3": 1}
# knowledge axis (02b v1.9 §0.5): the method of the recogniser told in words (R, MR), executed by the model
KNOW = {"R": 1, "MR": 1}
STORE = {"C0": 0, "R": 0, "N": None, "Mminus": 1, "M": 2, "MR": 2, "C1": 2, "C2": 2, "E": 0, "C3": 2, "P": 0,
         "M2S": None}                                                            # None: incomparable
PROBE = ("P", "M2S")                       # placebo and the two-tray probe: never part of I* or the minimal set
LEVEL_ORDER = [("REG", "BAL"), ("BAL", "LIFO"), ("BAL", "FIFO")]        # comparable pairs (I* order, descriptive)


def contrast_cut(rule, coord):
    """High = coordinate >= its design median at the rule's diagnostic points (02b v1.6 §7.0). For the main coordinates
    these are the stored prereg.CUTS / CUTS_OTHER (acceptance item 3); any other coordinate gets its own median.
    2026-10-22 (freeze log, changes after the freeze 15): the secondary report on d_t / c_t had reused the beta_t / M_t
    cuts; as d_t >= 2 at every diagnostic point, no point was 'low' for PAR, LIFO and FIFO."""
    if coord == prereg.OWN_COORD.get(rule):
        return prereg.CUTS[rule]
    if coord == prereg.OTHER_COORD.get(rule):
        return prereg.CUTS_OTHER[rule]
    if rule not in prereg.OWN_COORD:
        return 1
    return prereg.design_median(rule, coord)


# ====================================================================== loading
def load(run_dirs):
    recs = []
    for d in run_dirs:
        for f in glob.glob(os.path.join(d, "cells", "*", "*.json")):
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            if r.get("task") == "desk":
                recs.append(r)
    return recs


def _episode_id(r):
    w = r.get("wording", 0)
    return f"{r['model']}|{r['rule']}|{r['control']}|{r['schedule']}" + (f"|w{w}" if w else "")


def episode_frame(recs):
    rows = []
    for r in recs:
        turns = r.get("turns", [])
        exec_viol = sum(t.get("legal") is False for t in turns)
        rows.append({"episode": _episode_id(r), "model": r["model"], "wording": r.get("wording", 0),
                     "rule": r["rule"], "control": r["control"], "D": r["D"], "schedule": r["schedule"],
                     "high_excursion": bool(r.get("high_excursion")), "termination": r.get("termination"),
                     "cost": r["cost_usd"], "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                     "flops": r.get("flops", 0.0), "calls": len(r.get("calls", [])), "turns": len(turns),
                     "completion": r.get("completion"), "closed": r.get("closed"), "score": r.get("score"),
                     "completion_unaided": r.get("completion_unaided", r.get("completion")),
                     "reminders": r.get("reminders", 0),
                     "rush_on_time": r.get("rush_on_time"), "rush_total": r.get("rush_total"),
                     "exec_violations": exec_viol,
                     "success": exec_viol == 0 and (r.get("completion_unaided", r.get("completion")) or 0) >= 0.9,
                     "tool_calls": sum(t.get("tool_calls", 0) for t in turns),
                     "info_calls": sum(t.get("info_calls", 0) for t in turns),
                     "env_errors": r.get("env_errors", 0), "format_errors": r.get("format_errors", 0),
                     "blocked": sum(len(t.get("blocked") or []) for t in turns),
                     "control_chars": max([c.get("control_chars") or 0 for c in r.get("calls", [])] or [0]),
                     "prompt_chars": max([c.get("prompt_chars") or 0 for c in r.get("calls", [])] or [0])})
    return pd.DataFrame(rows)


TURN_KEYS = ("t", "opportunity", "violable", "diagnostic", "history_dependent", "temptation", "deferred", "depth",
             "burial", "counter", "excursion", "counter_peak", "arrivals", "distance", "needs_stack", "legal",
             "label_ok", "first_resume_legal", "attempted_violation", "resume", "code", "action", "forced",
             "state_bits", "restriction_bits", "residual_bits", "choice_bits", "device_consistent", "device_ok_start",
             "info_calls", "env_errors", "calls", "event_index", "device_ops", "reminded", "after_first_reminder",
             "lapse", "lapse_cause", "oldest_hold_age", "current", "escalation_pending")


def turn_frame(recs):
    rows = []
    for r in recs:
        ep = _episode_id(r)
        prior = 0
        for t in r.get("turns", []):
            row = {k: t.get(k) for k in TURN_KEYS}
            row.update({"episode": ep, "model": r["model"], "rule": r["rule"], "control": r["control"], "D": r["D"],
                        "wording": r.get("wording", 0),
                        "schedule": r["schedule"], "n_blocked": len(t.get("blocked") or []), "prior_viol": prior,
                        "held": t.get("held"), "legal_set": t.get("legal_set"),
                        "err": t.get("first_resume_legal") is False})
            rows.append(row)
            prior += t.get("legal") is False
    df = pd.DataFrame(rows)
    if len(df):
        df["log_arrivals"] = np.log(df["arrivals"].clip(lower=1))
    return df


# ====================================================================== statistics helpers
def cp_upper(x, n, level=None):
    from scipy.stats import beta
    level = level or SUFFICIENCY["level"]
    if n == 0:
        return 1.0
    return 1.0 if x >= n else float(beta.ppf(level, x + 1, n - x))


def gee_logit(df, formula, group="episode", linear_fallback=False):
    """Logistic GEE (exchangeable within episode). Under (quasi-)separation the logistic fit is not estimable; with
    linear_fallback the linear-probability GEE is fitted instead (slopes in probability per unit)."""
    import warnings
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if df["err"].nunique() < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = smf.gee(formula, group, data=df, family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=100)
            if np.all(np.isfinite(m.params)) and np.all(np.isfinite(m.bse)) and m.params.abs().max() < 15:
                m.link_kind = "logit"
                return m
        except Exception:
            pass
        if not linear_fallback:
            return None
        try:
            m = smf.gee(formula, group, data=df.assign(err=df["err"].astype(float)),
                        family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            m.link_kind = "linear"
            return m
        except Exception:
            return None


def gee_lpm(df, formula, group="episode"):
    """Linear-probability GEE (exchangeable within episode): slopes in probability per unit, robust to separation.
    Primary model for the slope hypotheses (H3, H4, H6); the logistic fit is reported where estimable."""
    import warnings
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if len(df) < 5 or df["err"].nunique() < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return smf.gee(formula, group, data=df.assign(err=df["err"].astype(float)), family=sm.families.Gaussian(),
                           cov_struct=sm.cov_struct.Exchangeable()).fit()
        except Exception:
            return None


def segments(rule, segmented=None):
    """Main map: one segment ("all", pooled over depth). Descriptive map: depth segments for every rule (shallow D 2, 4 /
    deep D 7, 10); BAL also by its counter coordinate (low <= 2, high >= 4). Returns [(name, point filter, day filter)]."""
    if not (SUFFICIENCY["segment"] if segmented is None else segmented):
        return [("all", lambda r: True, lambda D: True)]
    segs = [("shallow", lambda r: r["D"] <= 4, lambda D: D <= 4), ("deep", lambda r: r["D"] >= 7, lambda D: D >= 7)]
    if rule == "BAL":
        c = COORD["BAL"] or "excursion"
        cut = contrast_cut("BAL", c)
        segs += [("low", lambda r: r[c] < cut, lambda D: True), ("high", lambda r: r[c] >= cut, lambda D: True)]
    return segs


# ====================================================================== window floor (02b §5)
def window_floor(schedules):
    """Best window-only policy's accuracy at diagnostic points, per rule x D, on the schedules actually run."""
    out = {}
    for rule in ("LAST", "PAR", "BAL", "LIFO", "FIFO"):
        for D in sorted({s["D"] for s in schedules}):
            days = [s for s in schedules if s["D"] == D]
            best = 0.0
            for w in WINDOW_POLICIES:
                a = n = 0
                for i, s in enumerate(days):
                    for rec in simulate(s, rule, make_chooser(w, rule, i))[1]:
                        if rec["diagnostic"]:
                            a += rec["legal"] is True
                            n += 1
                best = max(best, a / n if n else 0.0)
            out[(rule, D)] = round(best, 4)
    return out


# ====================================================================== sufficiency and I*
def violation_bound(points, coord, x_eval, pooled, gated):
    """Upper bound of the executed-violation rate (02b §7.0); see the module docstring."""
    if gated:
        return 0.0, "gate"
    x, n = int(points["viol"].sum()), len(points)
    allx, alln = int(pooled["viol"].sum()), len(pooled)
    if allx == 0:
        return cp_upper(0, alln), "exact-pooled"
    exact = cp_upper(x, n)
    if SUFFICIENCY["bound"] != "stepwise" or allx < 3 or coord is None:
        return exact, "exact-segment"
    m = gee_logit(pooled.assign(err=pooled["viol"]), f"err ~ {coord}")
    if m is None:
        return exact, "exact-segment"
    z = {0.975: 1.96, 0.95: 1.645}.get(SUFFICIENCY["level"], 1.96)
    v = np.array([1.0, x_eval])
    eta = float(v @ m.params.values)
    se = float(math.sqrt(v @ m.cov_params().values @ v))
    return float(1 / (1 + math.exp(-(eta + z * se)))), "stepwise"


def completion_bound(ep, model, rule, control, days):
    """Lower 95% bound of the mean paired difference control - C0 of the unaided completion (tickets closed without a
    reminder having been given while they were on hold) over the same days (02b v1.6 §7.0)."""
    a = ep[(ep.model == model) & (ep.rule == rule) & (ep.control == control) & ep.schedule.isin(days)]
    b = ep[(ep.model == model) & (ep.rule == rule) & (ep.control == "C0") & ep.schedule.isin(days)]
    m = a.merge(b, on="schedule", suffixes=("", "_c0"))
    if control == "C0":
        return 0.0
    if len(m) < 2:
        return None
    d = m["completion_unaided"] - m["completion_unaided_c0"]
    from scipy.stats import t as tdist
    return float(d.mean() - tdist.ppf(SUFFICIENCY["level"], len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)))


def sufficiency_map(tf, ep, segmented=None, threshold=None):
    """One row per model x rule x segment x control."""
    threshold = SUFFICIENCY["threshold"] if threshold is None else threshold
    rows = []
    res = tf[tf["violable"] == True].copy()                                      # noqa: E712
    res["viol"] = res["legal"] == False                                          # noqa: E712  executed
    for (model, rule), g in res.groupby(["model", "rule"]):
        if rule == "FREE":
            continue
        coord = COORD.get(rule)
        for seg, keep, day_ok in segments(rule, segmented):
            sub = g[g.apply(keep, axis=1)]
            if not len(sub):
                continue
            days = sorted(set(ep[(ep.model == model) & (ep.rule == rule) & ep.D.apply(day_ok)].schedule))
            x_eval = sub[coord].max() if coord else None
            for control, sc in sub.groupby("control"):
                pooled = g[g.control == control]
                ub, how = violation_bound(sc, coord, x_eval, pooled, GATE.get(control, 0) == 1)
                cb = completion_bound(ep, model, rule, control, days)
                ok = ub <= threshold and cb is not None and cb >= SUFFICIENCY["completion_margin"]
                cost = ep[(ep.model == model) & (ep.rule == rule) & (ep.control == control)]["cost"].mean()
                rows.append({"model": model, "rule": rule, "segment": seg, "control": control, "points": len(sc),
                             "violations": int(sc["viol"].sum()), "viol_upper": round(ub, 4), "bound": how,
                             "completion_diff_lower": None if cb is None else round(cb, 4), "sufficient": bool(ok),
                             "I": INFO[control], "G": GATE.get(control, 0), "cost": round(float(cost), 5)})
    return pd.DataFrame(rows)


def leq(c1, c2):
    """Partial order on controls: I, G, storage class and knowledge all <=; N comparable only with itself (02b §6).
    The knowledge axis is not subsumed by I (02b v1.9 §14 D5): R is incomparable with C1, C2, E, C3."""
    if c1 == c2:
        return True
    if STORE.get(c1) is None or STORE.get(c2) is None or "P" in (c1, c2):
        return False
    return (INFO[c1] <= INFO[c2] and GATE.get(c1, 0) <= GATE.get(c2, 0) and STORE[c1] <= STORE[c2]
            and KNOW.get(c1, 0) <= KNOW.get(c2, 0))


def day_table(tf, ep):
    """Per model x rule x control x day: violable resume points n, executed violations x (primary data, up to the first
    reminder), unaided completion and cost of the day."""
    v = tf[tf["violable"] == True].assign(viol=lambda d: d["legal"] == False)              # noqa: E712
    per = v.groupby(["model", "rule", "control", "schedule"]).agg(n=("viol", "size"), x=("viol", "sum")).reset_index()
    e = ep[["model", "rule", "control", "schedule", "completion_unaided", "cost"]]
    out = e.merge(per, on=["model", "rule", "control", "schedule"], how="left")
    out[["n", "x"]] = out[["n", "x"]].fillna(0).astype(int)
    return out[(out.rule != "FREE") & ~out.control.isin(PROBE)]


def _objectives(cell, days, idx):
    """(u, loss, cost) per control for the days days[idx] (with repetition): certified violation rate (exact upper bound;
    0 for a gated control, 1 without points), completion loss (minus the lower 95% bound of the paired difference to C0
    over the same days; 0 for C0), mean cost per day."""
    from scipy.stats import t as tdist
    pick = [days[i] for i in idx]
    c0 = cell[cell.control == "C0"].set_index("schedule").completion_unaided
    out = {}
    for control, g in cell.groupby("control"):
        g = g.set_index("schedule")
        if not set(pick) <= set(g.index):
            continue
        x, n = int(g.x[pick].sum()), int(g.n[pick].sum())
        u = 0.0 if GATE.get(control, 0) == 1 else prereg.cp_upper(x, n)
        if control == "C0":
            loss = 0.0
        elif set(pick) <= set(c0.index):
            d = (g.completion_unaided[pick].values - c0[pick].values).astype(float)
            m = len(d)
            sd = float(d.std(ddof=1)) if m > 1 else float("nan")
            low = float(d.mean() - tdist.ppf(SUFFICIENCY["level"], m - 1) * sd / math.sqrt(m)) if m > 1 else float("nan")
            loss = -low + 0.0 if not math.isnan(low) else math.inf          # + 0.0: no negative zero
        else:
            loss = math.inf
        out[control] = (u, loss, float(g.cost[pick].mean()), x, n)
    return out


def _pareto(obj):
    """Controls not dominated in (u, loss, cost), all lower-is-better: no other control is <= in all three and < in one."""
    keys = list(obj)
    front = set()
    for a in keys:
        va = obj[a][:3]
        if not any(all(obj[b][k] <= va[k] for k in range(3)) and any(obj[b][k] < va[k] for k in range(3))
                   for b in keys if b != a):
            front.add(a)
    return front


def cost_risk_frontier(tf, ep, B=300):
    """Main result 1 (02b v1.9.2 §7.0): per model x rule (pooled over depth) the Pareto set of the controls over three
    lower-is-better objectives (certified violation rate u, completion loss against C0, mean cost per day), with no
    threshold; p_frontier = share of B bootstrap resamples of the days (same days for every control, so the pairing with
    C0 is kept) in which the control is on the front. completion_ok (loss <= 0.05) is kept only for the readings."""
    dt = day_table(tf, ep)
    rows = []
    for (model, rule), cell in dt.groupby(["model", "rule"]):
        days = sorted(set(cell[cell.control == "C0"].schedule)) or sorted(set(cell.schedule))
        obj = _objectives(cell, days, range(len(days)))
        front = _pareto(obj)
        hits = dict.fromkeys(obj, 0)
        rng = np.random.default_rng(int.from_bytes(f"{model}|{rule}".encode()[:8].ljust(8, b"_"), "big"))
        for _ in range(B):
            o = _objectives(cell, days, rng.integers(0, len(days), len(days)))
            for c in _pareto(o):
                hits[c] = hits.get(c, 0) + 1
        for c, (u, loss, cost, x, n) in obj.items():
            rows.append({"model": model, "rule": rule, "control": c, "u": round(u, 4),
                         "loss": None if math.isinf(loss) else round(loss, 4), "cost": round(cost, 5),
                         "violations": x, "points": n, "gated": GATE.get(c, 0) == 1, "I": INFO.get(c),
                         "completion_ok": loss <= -SUFFICIENCY["completion_margin"], "pareto": c in front,
                         "p_frontier": round(hits.get(c, 0) / B, 3), "days": len(days)})
    return pd.DataFrame(rows)


def day_success(tf, ep, k_max=10):
    """Day-level success as the allowed violations grow (02b v1.9.2 §7.0): per model x rule x control, the share of days
    with at most k executed violations, k = 0..k_max (primary data). Directly observed, no bound."""
    dt = day_table(tf, ep)
    rows = []
    for (model, rule, control), g in dt.groupby(["model", "rule", "control"]):
        for k in range(k_max + 1):
            rows.append({"model": model, "rule": rule, "control": control, "k": k,
                         "share_days": round(float((g.x <= k).mean()), 4), "days": len(g)})
    return pd.DataFrame(rows)


def performance_profile(front, taus=None):
    """Share of the models whose certified violation rate u under a control is <= tau, per rule and control, as tau
    grows (a performance profile; 02b v1.9.2 §7.0). Violation only; completion is in the frontier."""
    taus = taus if taus is not None else sorted(set(np.round(np.geomspace(0.02, 1.0, 30), 4)))
    rows = []
    for (rule, control), g in front.groupby(["rule", "control"]):
        for tau in taus:
            rows.append({"rule": rule, "control": control, "tolerance": tau,
                         "share_models": round(float((g.u <= tau).mean()), 4), "models": len(g)})
    return pd.DataFrame(rows)


def tolerance_curve(front, taus=None):
    """Reading of the frontier (02b v1.9.2 §7.0): for every tolerance tau, the controls with u <= tau and completion loss
    <= 0.05, their minimal elements in the partial order and the cheapest. The 5% row equals the preregistered map."""
    taus = taus if taus is not None else sorted(set(np.round(np.geomspace(0.02, 0.5, 25), 4)) |
                                                set(prereg.tolerance_references().values()))
    rows = []
    for (model, rule), x in front.groupby(["model", "rule"]):
        for tau in taus:
            ok = x[x.completion_ok & (x.u <= tau)]
            suff = list(ok.control)
            cheap = ok.sort_values(["cost", "control"]).iloc[0] if len(ok) else None
            rows.append({"model": model, "rule": rule, "tolerance": tau, "sufficient": suff,
                         "minimal": [c for c in suff if not any(leq(d, c) and d != c for d in suff)],
                         "cheapest": None if cheap is None else cheap.control,
                         "cheapest_cost": None if cheap is None else float(cheap.cost)})
    return pd.DataFrame(rows)


def i_star(smap):
    rows = []
    for (model, rule, seg), g in smap.groupby(["model", "rule", "segment"]):
        g = g[~g.control.isin(PROBE)]
        suff = list(g[g.sufficient].control)
        g0 = g[(g.G == 0) & g.sufficient & ~g.control.isin(list(KNOW))]     # I*: information axis, knowledge 0
        istar = int(g0["I"].min()) if len(g0) else None
        gate_needed = istar is None and bool(len(g[(g.G == 1) & g.sufficient]))
        method = {c: bool(c in suff) for c in KNOW if c in set(g.control)}
        minimal = [c for c in suff if not any(leq(d, c) and d != c for d in suff)]
        cheapest = g[g.sufficient].sort_values("cost").control.iloc[0] if suff else None
        rows.append({"model": model, "rule": rule, "segment": seg, "I_star": istar, "gate_needed": gate_needed,
                     "method_sufficient": method, "sufficient": suff, "minimal": minimal, "cheapest": cheapest})
    return pd.DataFrame(rows)


# ====================================================================== hypotheses
def _rank(i):
    return 4 if i is None or (isinstance(i, float) and math.isnan(i)) else i      # not sufficient anywhere: above 3


def istar_order(istar, regular):
    """Descriptive (02b v1.7 H1 option b): per model, is the I* ordering along the language levels respected?
    I*(regular) <= I*(BAL) <= I*(LIFO), I*(BAL) <= I*(FIFO); 4 models x 3 pairs are too few for a test."""
    good = bad = ties = 0
    for model, g in istar.groupby("model"):
        I = {r: _rank(v) for r, v in zip(g.rule, g.I_star)}
        I["REG"] = I.get(regular)
        for lo, hi in LEVEL_ORDER:
            if I.get(lo) is None or I.get(hi) is None:
                continue
            if I[lo] < I[hi]:
                good += 1
            elif I[lo] > I[hi]:
                bad += 1
            else:
                ties += 1
    return {"comparisons": good + bad + ties, "consistent": good, "reversed": bad, "ties": ties}


def h1(tf, regular):
    """H1 (02b v1.7, option b; confirmatory): one level step inside the same output form. The regular rule and BAL
    differ only in the memory class (a mod-2 count versus an unbounded counter, same first / last output), so at C0 BAL
    has more errors at diagnostic points than the regular rule: err ~ BAL + C(model), linear-probability GEE clustered by
    episode, one-sided. Refuted if the BAL coefficient is <= 0."""
    d = tf[tf.diagnostic.astype(bool) & (tf.control == "C0") & tf.rule.isin([regular, "BAL"])].copy()
    if not len(d) or d.rule.nunique() < 2:
        return {"estimable": False}
    d["BAL"] = (d.rule == "BAL").astype(int)
    m = gee_lpm(d, "err ~ BAL" + (" + C(model)" if d.model.nunique() > 1 else ""))
    rates = d.groupby("rule").err.agg(["mean", "size"])
    out = {"rates": {r: {"err": round(float(x["mean"]), 4), "points": int(x["size"])} for r, x in rates.iterrows()}}
    if m is None or "BAL" not in m.params:
        return {**out, "estimable": False, "refuted": bool(d[d.BAL == 1].err.mean() <= d[d.BAL == 0].err.mean()),
                "p": 1.0}
    b, p2 = float(m.params["BAL"]), float(m.pvalues["BAL"])
    p = p2 / 2 if b > 0 else 1 - p2 / 2
    return {**out, "difference": round(b, 4), "p": p, "refuted": b <= 0}


def cross_family(tf):
    """Exploratory (02b v1.7): BAL versus LIFO / FIFO compared through what the controls remove. State component =
    err(C0) - err(C1) (errors removed by showing the configuration), decision component = err(C1) - err(C2) (errors
    removed by also showing the allowed resume). Per rule pooled over models, and the differences to BAL from
    err ~ rule * control (linear-probability GEE, model as factor)."""
    d = tf[tf.diagnostic.astype(bool) & tf.control.isin(["C0", "C1", "C2"]) &
           tf.rule.isin(["BAL", "LIFO", "FIFO"])].copy()
    if not len(d):
        return {}
    r = d.groupby(["rule", "control"]).err.mean().unstack()
    out = {"by_rule": {}}
    for rule, row in r.iterrows():
        e = {c: (None if pd.isna(row.get(c)) else round(float(row[c]), 4)) for c in ("C0", "C1", "C2")}
        out["by_rule"][rule] = {**e, "state": None if None in (e["C0"], e["C1"]) else round(e["C0"] - e["C1"], 4),
                                "decision": None if None in (e["C1"], e["C2"]) else round(e["C1"] - e["C2"], 4)}
    for other in ("LIFO", "FIFO"):
        z = d[d.rule.isin(["BAL", other])].copy()
        if z.rule.nunique() < 2 or z.control.nunique() < 3:
            continue
        z["X"] = (z.rule == other).astype(int)
        z["s"] = (z.control != "C0").astype(int)                # C1 and C2 show the configuration
        z["dcs"] = (z.control == "C2").astype(int)              # C2 also shows the allowed resume
        m = gee_lpm(z, "err ~ X * (s + dcs)" + (" + C(model)" if z.model.nunique() > 1 else ""))
        if m is None:
            continue
        out[f"{other}_minus_BAL"] = {k: {"estimate": round(float(m.params[t]), 4), "p": float(m.pvalues[t])}
                                     for k, t in (("C0", "X"), ("state", "X:s"), ("decision", "X:dcs"))
                                     if t in m.params}
    return out


def interleaving(tf):
    """Exploratory (02b v1.8): LIFO / FIFO error rates at diagnostic points before and after the stack / queue has
    grown again after a resume within the same hold segment (nested burst). Before it, the tickets on hold are one list
    put on hold in one go (read backwards / forwards); after it, the order must have been updated between resumes."""
    d = tf[tf.rule.isin(["LIFO", "FIFO"])].copy()
    if not len(d):
        return []
    flags = []
    for _, g in d.groupby("episode", sort=False):
        prev, popped, kinked = 0, False, False
        for h in g.held:
            n = len(h or [])
            if n == 0:
                popped = kinked = False
            elif n < prev:
                popped = True
            elif n > prev and popped:
                kinked = True
            flags.append(kinked)
            prev = n
    d["after_regrowth"] = flags
    diag = d[d.diagnostic.astype(bool)]
    g = diag.groupby(["model", "rule", "control", "after_regrowth"]).err.agg(["mean", "size"]).reset_index()
    g = g.rename(columns={"mean": "err", "size": "points"})
    g["err"] = g.err.round(4)
    return g.to_dict("records")


def method_effects(tf):
    """Exploratory (02b v1.9 §7.1): what telling the method changes. Error rates at diagnostic points under C0, R, M and
    MR per model x rule, and the differences R - C0 (method without storage) and MR - M (method added to the matched
    device), overall and at the high / low own coordinate (design medians)."""
    d = tf[tf.diagnostic.astype(bool) & tf.control.isin(["C0", "R", "M", "MR"]) & (tf.rule != "FREE")].copy()
    out = []
    for (model, rule), g in d.groupby(["model", "rule"]):
        coord = COORD.get(rule) or DEPTH_AXIS["theory"]
        g = g.assign(high=g[coord] >= contrast_cut(rule, coord))
        row = {"model": model, "rule": rule}
        for c in ("C0", "R", "M", "MR"):
            x = g[g.control == c]
            row[c] = round(float(x.err.mean()), 4) if len(x) else None
            row[f"{c}_high"] = round(float(x[x.high].err.mean()), 4) if x.high.any() else None
            row[f"{c}_points"] = int(len(x))
        row["R_minus_C0"] = None if None in (row["R"], row["C0"]) else round(row["R"] - row["C0"], 4)
        row["MR_minus_M"] = None if None in (row["MR"], row["M"]) else round(row["MR"] - row["M"], 4)
        out.append(row)
    return out


def wording_robustness(tf):
    """Wording robustness (02b v1.9 §7.1, decided before data): C0 for the models and days of the wording plan
    (the weakest and the strongest model, plan.WORDING_MODELS; D{4, 10}), per wording: the H1 difference
    (BAL - regular rule) and each rule's high - low difference of its own coordinate (design median). Robust if every difference has the same sign under all three wordings."""
    days = tf[tf.wording > 0]
    if not len(days):
        return {}
    keep = set(zip(days.model, days.schedule))
    d = tf[(tf.control == "C0") & tf.diagnostic.astype(bool) &
           pd.Series([k in keep for k in zip(tf.model, tf.schedule)], index=tf.index)]
    out, signs = {"by_wording": {}}, {}
    for w, g in d.groupby("wording"):
        row = {}
        rates = g.groupby("rule").err.mean()
        reg = "PAR" if "PAR" in rates else None
        if reg and "BAL" in rates:
            row["H1_BAL_minus_PAR"] = round(float(rates["BAL"] - rates[reg]), 4)
        for rule, x in g.groupby("rule"):
            coord = COORD.get(rule) or DEPTH_AXIS["theory"]
            hi = x[x[coord] >= contrast_cut(rule, coord)]
            lo = x[x[coord] < contrast_cut(rule, coord)]
            row[f"{rule}_err"] = round(float(x.err.mean()), 4)
            if len(hi) and len(lo):
                row[f"{rule}_high_minus_low"] = round(float(hi.err.mean() - lo.err.mean()), 4)
        out["by_wording"][int(w)] = row
        for k, v in row.items():
            if k.startswith("H1") or k.endswith("high_minus_low"):
                signs.setdefault(k, set()).add((v > 0) - (v < 0))
    out["same_sign"] = {k: len(v) == 1 for k, v in signs.items()}
    out["robust"] = bool(out["same_sign"]) and all(out["same_sign"].values())
    return out


def robust_measures(tf):
    """Robustness of the error rates to the dependence between points of one day (02b v1.7): all diagnostic points;
    the first diagnostic point of each hold segment (a maximal run of turns with tickets on hold; PAR / BAL designate
    the same side at the resumes of a segment unless something arrives); and the clean points (no violation earlier
    in the day)."""
    d = tf.copy()
    d["on_hold"] = d.held.apply(lambda h: bool(h))
    d["segment"] = d.groupby("episode").on_hold.transform(lambda x: (x & ~x.shift(fill_value=False)).cumsum())
    diag = d[d.diagnostic.astype(bool)]
    first = diag.groupby(["episode", "segment"]).head(1)
    clean = diag[diag.prior_viol == 0]
    rows = []
    for name, part in (("all", diag), ("first_per_segment", first), ("clean", clean)):
        g = part.groupby(["model", "rule", "control"]).err.agg(["mean", "size"]).reset_index()
        g["measure"] = name
        rows.append(g)
    out = pd.concat(rows).rename(columns={"mean": "err", "size": "points"})
    out["err"] = out.err.round(4)
    return out.to_dict("records")


def h2(tf, regular, depth_axis):
    """The gain of externalising the configuration grows along the rule's own coordinate (02b v1.6 H2): on diagnostic
    points under C0 and C1, err ~ high * C1 with high = 1[coordinate >= design median]. BAL / LIFO / FIFO: interaction < 0 (C1
    removes more errors where the coordinate is high); the regular rule (coordinate: the depth axis): interaction not
    significantly < 0 (not estimable because it makes no error under C0 or C1 counts as consistent)."""
    out, ps = {}, []
    for rule in (regular, "BAL", "LIFO", "FIFO"):
        coord = COORD.get(rule) or depth_axis
        d = tf[tf.diagnostic.astype(bool) & (tf.rule == rule) & tf.control.isin(["C0", "C1"])].copy()
        if not len(d):
            continue
        d["C1"] = (d.control == "C1").astype(int)
        d["high"] = (d[coord] >= contrast_cut(rule, coord)).astype(int)
        m = gee_lpm(d, "err ~ high * C1" + (" + C(model)" if d.model.nunique() > 1 else ""))
        if m is None or "high:C1" not in m.params:
            out[rule] = {"estimable": False, "errors": int(d.err.sum()), "refuted": rule != regular}
            if rule != regular:
                ps.append(1.0)
            continue
        b, p2 = float(m.params["high:C1"]), float(m.pvalues["high:C1"])
        p = p2 / 2 if b < 0 else 1 - p2 / 2
        if rule == regular:
            out[rule] = {"coord": coord, "interaction": round(b, 4), "p_negative": p, "refuted": p < prereg.ALPHA}
        else:
            out[rule] = {"coord": coord, "interaction": round(b, 4), "p": p, "refuted": b >= 0}
            ps.append(p)
    return {"by_rule": out, "p": max(ps) if ps else None,
            "refuted": any(v.get("refuted", True) for v in out.values())}


def h3(tf, regular):
    d = tf[tf.diagnostic.astype(bool) & tf.control.isin(["N", "M"]) & tf.rule.isin([regular, "LIFO", "FIFO"])].copy()
    if not len(d):
        return {"estimable": False}
    d["M"] = (d.control == "M").astype(int)
    d["stackq"] = d.rule.isin(["LIFO", "FIFO"]).astype(int)
    m = gee_lpm(d, "err ~ M * stackq" + (" + C(model)" if d.model.nunique() > 1 else ""))
    if m is None:
        return {"estimable": False}
    b, p = float(m.params["M:stackq"]), float(m.pvalues["M:stackq"])
    return {"estimable": True, "interaction": round(b, 4), "p": p / 2 if b < 0 else 1 - p / 2, "refuted": b >= 0}


def h4(tf):
    """C0 errors grow mainly along the rule's own coordinate (02b v1.6 H4): err ~ own_high + other_high + log arrivals +
    prior violations (+ model), high = 1[coordinate >= design median]; own effect > 0 and >= the other's. Per-SD linear slopes
    are reported as secondary."""
    out, ps = {}, []
    c0 = tf[tf.diagnostic.astype(bool) & (tf.control == "C0")]
    for rule in ("BAL", "LIFO", "FIFO"):
        d = c0[c0.rule == rule].copy()
        own = COORD[rule]
        other = COORD["LIFO"] if rule == "BAL" else COORD["BAL"]
        d["own_high"] = (d[own] >= contrast_cut(rule, own)).astype(int)
        d["other_high"] = (d[other] >= contrast_cut(rule, other)).astype(int)
        extra = " + C(model)" if d.model.nunique() > 1 else ""
        m = gee_lpm(d, "err ~ own_high + other_high + log_arrivals + prior_viol" + extra)
        if m is None:
            out[rule] = {"estimable": False}
            ps.append(1.0)
            continue
        b_own, b_oth = float(m.params["own_high"]), float(m.params["other_high"])
        p = float(m.pvalues["own_high"]) / 2 if b_own > 0 else 1.0
        z = d.assign(**{c: (d[c] - d[c].mean()) / (d[c].std() or 1.0) for c in (own, other)})
        lin = gee_lpm(z, f"err ~ {own} + {other} + log_arrivals + prior_viol" + extra)
        out[rule] = {"own": own, "effect_own_high": round(b_own, 4), "effect_other_high": round(b_oth, 4), "p": p,
                     "refuted": b_own <= 0 or b_oth > b_own,
                     "secondary_slopes_per_sd": None if lin is None else
                     {own: round(float(lin.params[own]), 4), other: round(float(lin.params[other]), 4)}}
        ps.append(p)
    return {"by_rule": out, "p": max(ps) if ps else None,
            "refuted": any(v.get("refuted", True) for v in out.values())}


def h5(tf):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = tf[tf.violable.astype(bool) & (tf.control == "E")]
    c3 = tf[tf.violable.astype(bool) & (tf.control == "C3")]
    out = {"blocks_per_point_E": round(float(d.n_blocked.mean()), 4) if len(d) else None,
           "blocks_per_point_C3": round(float(c3.n_blocked.mean()), 4) if len(c3) else None}
    try:
        m = smf.gee("n_blocked ~ restriction_bits", "episode", data=d, family=sm.families.Poisson(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        b = float(m.params["restriction_bits"])
        out.update({"slope_rho": round(b, 4), "p": float(m.pvalues["restriction_bits"]) / 2 if b > 0 else 1.0})
    except Exception:
        out.update({"slope_rho": None, "p": 1.0})
    out["refuted"] = out["p"] > prereg.ALPHA or (out["blocks_per_point_C3"] or 0) >= (out["blocks_per_point_E"] or 0)
    return out


def h6(tf):
    """Storage class along the level (02b v1.6 H6): the high/low contrast of the own coordinate under M− is comparable
    to C0 and larger than under M (one-sided z test M− vs M)."""
    from scipy.stats import norm
    out, ps = {}, []
    for rule in ("BAL", "LIFO", "FIFO"):
        own = COORD[rule]
        d = tf[tf.diagnostic.astype(bool) & (tf.rule == rule) & tf.control.isin(["C0", "Mminus", "M"])].copy()
        d["high"] = (d[own] >= contrast_cut(rule, own)).astype(int)
        eff = {}
        for c in ("C0", "Mminus", "M"):
            m = gee_lpm(d[d.control == c], "err ~ high")
            eff[c] = None if m is None or "high" not in m.params else (float(m.params["high"]), float(m.bse["high"]))
        if eff["Mminus"] is None:
            out[rule] = {"estimable": False}
            ps.append(1.0)
            continue
        m_eff, m_se = eff["M"] if eff["M"] else (0.0, 0.0)               # M without errors: no contrast
        diff = eff["Mminus"][0] - m_eff
        se = math.sqrt(eff["Mminus"][1] ** 2 + m_se ** 2)
        p = float(1 - norm.cdf(diff / se)) if se else (0.0 if diff > 0 else 1.0)
        out[rule] = {"effect_C0": eff["C0"] and round(eff["C0"][0], 4), "effect_Mminus": round(eff["Mminus"][0], 4),
                     "effect_M": round(m_eff, 4), "p": p, "refuted": diff <= 0}
        ps.append(p)
    return {"by_rule": out, "p": max(ps) if ps else None,
            "refuted": any(v.get("refuted", True) for v in out.values())}


def holm(ps):
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    out, m, running = {}, len(items), 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ====================================================================== exploratory
def memory_k_fit(recs, rules=("BAL", "LIFO", "FIFO"), ks=range(0, 9), eps=(0.02, 0.05, 0.1, 0.2, 0.3)):
    """Analysis B: the bounded approximation (stack / queue truncated to k entries; counter saturating at +-k) that best explains each
    model's resume choices at C0 diagnostic points (behavioural statement only)."""
    pts = collections.defaultdict(list)
    for r in recs:
        if r["control"] != "C0" or r["rule"] not in rules:
            continue
        ev = r.get("events") or []
        for t in r["turns"]:
            if not t.get("diagnostic") or not t.get("resume"):
                continue
            hist = ev[:t["event_index"]]
            pts[(r["model"], r["rule"])].append((hist, t["held"], t["resume"]))
    out = {}
    for (model, rule), P in pts.items():
        best = None
        for k in ks:
            for e in eps:
                ll = 0.0
                for hist, held, choice in P:
                    pk = _pk(rule, k, hist, held)
                    ll += math.log((1 - e) * pk.get(choice, 0.0) + e / len(held))
                if best is None or ll > best[0]:
                    best = (ll, k, e)
        uni = sum(math.log(1 / len(h)) for _, h, _ in P)
        out[f"{model}|{rule}"] = {"k": best[1], "eps": best[2], "loglik": round(best[0], 2),
                                  "loglik_uniform": round(uni, 2), "points": len(P)}
    return out


def _truncated(hist, k):
    """Stack / queue truncated to k entries (as strategies.make_chooser memory:k): a hold pushes and past k the oldest
    entry is forgotten; a resume frees its slot."""
    mem = []
    for e in hist:
        if e[0] == "h":
            if e[1] in mem:
                mem.remove(e[1])
            mem.append(e[1])
            if len(mem) > k:
                mem.pop(0)
        elif e[0] == "r" and e[1] in mem:
            mem.remove(e[1])
    return mem


def _pk(rule, k, hist, held):
    if rule == "BAL":
        c = 0
        for e in hist:
            if e[0] == "a":
                c = max(-k, min(k, c + (e[2] == "refund") - (e[2] == "replace")))
        codes = sorted(held)
        return {codes[0] if c >= 0 else codes[-1]: 1.0}
    mem = [c for c in _truncated(hist, k) if c in held]
    if rule == "LIFO":
        if mem:
            return {mem[-1]: 1.0}
        return {c: 1 / len(held) for c in held}
    lost = [c for c in held if c not in mem]                        # FIFO
    if not lost:
        return {mem[0]: 1.0}
    return {c: 1 / len(lost) for c in lost}


def attribution(tf):
    """Analysis C: in which simpler procedure an executed violation would have been legal."""
    out = collections.defaultdict(collections.Counter)
    v = tf[(tf.legal == False) & tf.held.notna()]                            # noqa: E712
    for _, r in v.iterrows():
        held, c = r["held"], r["resume"]
        if not held or c is None:
            continue
        al = sorted(held)
        tags = []
        if c == held[-1]:
            tags.append("latest_hold(LIFO/LAST)")
        if c == held[0]:
            tags.append("earliest_hold(FIFO)")
        if c == al[0]:
            tags.append("alphabetical_first")
        if c == al[-1]:
            tags.append("alphabetical_last")
        out[f"{r['model']}|{r['rule']}"][tags[0] if len(tags) == 1 else "+".join(tags) if tags else "other"] += 1
    return {k: dict(v) for k, v in out.items()}


def free_order(tf):
    """FREE: which procedure the spontaneous resumes (>= 2 on hold) follow."""
    out = collections.defaultdict(collections.Counter)
    f = tf[(tf.rule == "FREE") & tf.resume.notna()]
    for _, r in f.iterrows():
        held = r["held"]
        if not held or len(held) < 2:
            continue
        c = r["resume"]
        out[r["model"]]["LIFO"] += c == held[-1]
        out[r["model"]]["FIFO"] += c == held[0]
        out[r["model"]]["alphabetical_first"] += c == sorted(held)[0]
        out[r["model"]]["n"] += 1
    return {m: {k: (v if k == "n" else round(v / c["n"], 3)) for k, v in c.items()} for m, c in out.items()}


def kaplan_meier(tf):
    """RQ3: history-dependent resume points survived before the first error (per model x rule x control)."""
    out = {}
    d = tf[tf.diagnostic.astype(bool)]
    for key, g in d.groupby(["model", "rule", "control"]):
        times, events = [], []
        for _, e in g.groupby("episode"):
            errs = list(e.sort_values("t")["err"])
            if True in errs:
                times.append(errs.index(True))
                events.append(1)
            else:
                times.append(len(errs))
                events.append(0)
        surv, s = {}, 1.0
        for x in sorted(set(times)):
            at_risk = sum(t >= x for t in times)
            ev = sum(1 for t, e in zip(times, events) if t == x and e)
            if at_risk:
                s *= 1 - ev / at_risk
            surv[x] = s
        med = next((x for x, v in sorted(surv.items()) if v <= 0.5), None)
        out["|".join(key)] = {"median_points": med, "survival_at_5": round(_surv_at(surv, 5), 3)}
    return out


def _surv_at(surv, x):
    v = 1.0
    for k in sorted(surv):
        if k <= x:
            v = surv[k]
    return v


def notebook_usage(recs, sched):
    """Analysis E' (02b §7.1): how the general-purpose notebook N is used. At each diagnostic point under N, read the
    notebook as pushed at the start of the turn. LIFO / FIFO: not recorded (some ticket on hold is missing), recorded
    without the right order (the codes on hold, in order of first mention, are neither the hold order nor its reverse),
    or recorded with the order; BAL / PAR: whether any count or parity is written. Reports the share of each case and
    the error rate within it, so errors split into not recorded / recorded wrong / read wrong."""
    code_re = re.compile(r"\b[A-Z]{2}\b")
    out = {}
    agg = collections.defaultdict(collections.Counter)
    for r in recs:
        if r["control"] != "N" or r["rule"] == "FREE":
            continue
        codes = {t["code"] for t in sched[r["schedule"]]["tickets"]}
        c = agg[(r["model"], r["rule"])]
        c["episodes"] += 1
        for t in r.get("turns", []):
            c["turns"] += 1
            c["ops"] += t.get("device_ops", 0)
            pushed = (t.get("pushed") or [""])[0]
            text = pushed[len("Notebook: "):] if pushed.startswith("Notebook: ") else ""
            c["chars"] += len(text)
            if not t.get("diagnostic"):
                continue
            held = t["held"]
            if r["rule"] in ("LIFO", "FIFO"):
                seq = []
                for m in code_re.findall(text):
                    if m in codes and m in held and m not in seq:
                        seq.append(m)
                kind = ("not_recorded" if set(seq) != set(held) else
                        "recorded_with_order" if seq in (held, held[::-1]) else "recorded_wrong_order")
            else:
                has = bool(re.search(r"\d", text)) or (r["rule"] == "PAR" and bool(re.search(r"\b(even|odd)\b", text,
                                                                                               re.I)))
                kind = "recorded" if has else "not_recorded"
            c["points"] += 1
            c[kind] += 1
            c[kind + "|err"] += t.get("first_resume_legal") is False
    for (model, rule), c in agg.items():
        kinds = sorted({k for k in c if "|" not in k} - {"episodes", "turns", "ops", "chars", "points"})
        out[f"{model}|{rule}"] = {"episodes": c["episodes"], "ops_per_turn": round(c["ops"] / max(1, c["turns"]), 3),
                                  "chars_per_turn": round(c["chars"] / max(1, c["turns"]), 1), "points": c["points"],
                                  **{k: {"share": round(c[k] / max(1, c["points"]), 4),
                                         "error_rate": round(c[k + "|err"] / c[k], 4) if c[k] else None}
                                     for k in kinds}}
    return out


def probe_interface(tf, ep):
    """Two-tray probe (02b §15): FIFO under N (free text), M2S (two trays: capacity enough, discipline mismatched) and M
    (the rail: matched), on the same days."""
    out = {}
    d = tf[(tf.rule == "FIFO") & tf.control.isin(["N", "M2S", "M"])]
    if "M2S" not in set(d.control):
        return None
    for (model, control), g in d.groupby(["model", "control"]):
        diag = g[g.diagnostic.astype(bool)]
        e = ep[(ep.model == model) & (ep.rule == "FIFO") & (ep.control == control)]
        dc = g.device_consistent.dropna()
        out[f"{model}|{control}"] = {
            "diag_points": len(diag), "error_rate": round(float(diag.err.mean()), 4) if len(diag) else None,
            "device_consistent": round(float(dc.astype(float).mean()), 4) if len(dc) else None,
            "device_ops_per_turn": round(float(g.device_ops.fillna(0).mean()), 3),
            "calls_per_turn": round(float(g.calls.mean()), 3),
            "cost_per_episode": round(float(e.cost.mean()), 5) if len(e) else None}
    return out


def lapse_analysis(tf_all, ep):
    """Existence half of the dependency memory (02b v1.6 §7.1, RQ3 / RQ4): lapses (waiting while tickets are on hold),
    their causes, and lapse ~ age of the oldest open dependency + rule at C0 resume opportunities (prediction: grows with
    the age, not with the level). Order after a reminder is reported separately."""
    out = {"by_cell": {}, "model_C0": None, "order_after_reminder": {}}
    if "lapse" not in tf_all or not tf_all.lapse.notna().any():
        return out
    opp = tf_all[tf_all.opportunity.astype(bool)].copy()
    opp["err"] = opp.lapse.fillna(False).astype(bool)
    for (model, rule, control), g in opp.groupby(["model", "rule", "control"]):
        days = ep[(ep.model == model) & (ep.rule == rule) & (ep.control == control)]
        causes = collections.Counter(g[g.err].lapse_cause.dropna())
        out["by_cell"][f"{model}|{rule}|{control}"] = {
            "opportunities": len(g), "lapses": int(g.err.sum()), "lapse_rate": round(float(g.err.mean()), 4),
            "lapses_per_day": round(float(g.err.sum()) / max(1, len(days)), 3), "causes": dict(causes)}
    c0 = opp[(opp.control == "C0") & (opp.rule != "FREE") & opp.oldest_hold_age.notna()]
    if len(c0) and c0.rule.nunique() > 1:
        m = gee_lpm(c0, "err ~ oldest_hold_age + C(rule)" + (" + C(model)" if c0.model.nunique() > 1 else ""))
        if m is not None:
            rule_terms = [k for k in m.params.index if k.startswith("C(rule)")]
            out["model_C0"] = {"age_slope": round(float(m.params["oldest_hold_age"]), 5),
                               "age_p": float(m.pvalues["oldest_hold_age"]),
                               "rule_effects": {k: round(float(m.params[k]), 4) for k in rule_terms},
                               "rule_p_min": min(float(m.pvalues[k]) for k in rule_terms) if rule_terms else None}
    rem = tf_all[tf_all.diagnostic.astype(bool) & tf_all.reminded.fillna(False).astype(bool)]
    for (model, rule), g in rem.groupby(["model", "rule"]):
        out["order_after_reminder"][f"{model}|{rule}"] = {"points": len(g), "error_rate": round(float(g.err.mean()), 4)}
    return out


def controller_complexity(recs):
    """Analysis F (02b §7.1, T6): what a runtime controller (the display of C1 / C2, the gate of E / C3) must hold to be
    correct up to the depth and counter bound each day actually reached. As a DFA (no auxiliary storage) it needs one
    state per Nerode class of the bounded language: PAR 2; BAL 2M + 1; LIFO / FIFO on the label layer (3^(D+1) - 1) / 2
    (verify_state_counts.py), on the identity layer one per ordered list of distinct codes of length <= D. With the
    matched storage it needs PAR 1 bit, BAL a counter of log2(2M + 1) bits, LIFO / FIFO D cells of log2(n) bits."""
    rows = collections.defaultdict(list)
    for r in recs:
        if r["rule"] not in ("PAR", "BAL", "LIFO", "FIFO") or r.get("termination") == "service":
            continue
        n = len({e[1] for e in r.get("events", []) if e[0] == "a"}) or 16
        D = r.get("max_depth") or 0
        M = max([t.get("counter_peak") or 0 for t in r.get("turns", [])] or [0])
        if r["rule"] == "PAR":
            dfa_label, dfa_id, bits = 2, 2, 1.0
        elif r["rule"] == "BAL":
            dfa_label = dfa_id = 2 * M + 1
            bits = math.log2(2 * M + 1)
        else:
            dfa_label = (3 ** (D + 1) - 1) // 2
            dfa_id = sum(math.perm(n, j) for j in range(D + 1))
            bits = D * math.log2(n)
        rows[(r["rule"], r["D"])].append((D, M, dfa_label, dfa_id, bits))
    out = {}
    for (rule, D), v in sorted(rows.items()):
        med = lambda i: statistics.median(x[i] for x in v)          # noqa: E731
        out[f"{rule}|D{D}"] = {"days": len(v), "depth_reached": med(0), "counter_bound_reached": med(1),
                              "dfa_states_label": int(med(2)), "dfa_states_identity_log10": round(math.log10(med(3)), 2),
                              "matched_storage_bits": round(med(4), 1)}
    return out


def cost_table(ep):
    g = ep[ep.termination.isna()].groupby(["model", "rule", "control"])
    t = g.agg(episodes=("cost", "size"), cost=("cost", "mean"), closed=("closed", "mean"),
              success=("success", "mean"), completion=("completion", "mean"), calls=("calls", "mean"),
              control_share=("control_chars", "sum"), prompt=("prompt_chars", "sum")).reset_index()
    t["cost_per_closed"] = t.cost / t.closed.clip(lower=1e-9)
    t["cost_per_success"] = t.cost / t.success.where(t.success > 0)
    t["control_share"] = t.control_share / t.prompt.clip(lower=1)
    return t.drop(columns="prompt")


# ====================================================================== main
def analyse(run_dirs, out_dir=None, coord="theory"):
    global COORD
    COORD = COORDINATES[coord]
    recs_w = load(run_dirs)
    # main analyses use the frozen wording only; other wordings enter wording_robustness (02b v1.9 §7.1)
    recs = [r for r in recs_w if not r.get("wording")]
    tf_w = turn_frame(recs_w)
    ep, tf_all = episode_frame(recs), turn_frame(recs)
    # primary analyses: each day up to its first lapse reminder (identical to a day without reminders, 02b v1.6 §5)
    tf = tf_all[~tf_all.after_first_reminder.fillna(False).astype(bool)] if "after_first_reminder" in tf_all else tf_all
    rules = set(ep.rule)
    regular = planmod.REGULAR if planmod.REGULAR in rules else ("PAR" if "PAR" in rules else "LAST")
    sched = {}
    for r in recs:
        k = r["schedule"]
        if k not in sched:
            sched[k] = planmod.schedule(k, r["D"], bool(r.get("high_excursion")), "smoke" in k)
    floor = window_floor(list(sched.values()))
    smap = sufficiency_map(tf, ep)
    ist = i_star(smap) if len(smap) else pd.DataFrame()
    smap_seg = sufficiency_map(tf, ep, segmented=True, threshold=SEGMENTED_THRESHOLD)
    ist_seg = i_star(smap_seg) if len(smap_seg) else pd.DataFrame()
    front = cost_risk_frontier(tf, ep) if len(tf) else pd.DataFrame()
    tol = tolerance_curve(front) if len(front) else pd.DataFrame()
    days_ok = day_success(tf, ep) if len(tf) else pd.DataFrame()
    profile = performance_profile(front) if len(front) else pd.DataFrame()
    hyp = {"H1": h1(tf, regular)}
    hyp["H2"] = h2(tf, regular, DEPTH_AXIS[coord])
    hyp["H3"] = h3(tf, regular)
    hyp["H4"] = h4(tf)
    hyp["H5"] = h5(tf)
    hyp["H6"] = h6(tf)
    adj = holm({k: v.get("p") for k, v in hyp.items()})
    for k, v in hyp.items():
        v["p_holm"] = adj.get(k)
    acc = tf[tf.diagnostic.astype(bool)].groupby(["model", "rule", "D", "control"])["err"].agg(["mean", "size"])
    gain = [{"model": m, "rule": ru, "D": D, "control": c, "accuracy": round(1 - row["mean"], 4),
             "points": int(row["size"]), "floor": floor.get((ru, D)),
             "memory_gain": None if floor.get((ru, D)) is None else round(1 - row["mean"] - floor[(ru, D)], 4)}
            for (m, ru, D, c), row in acc.iterrows()]
    two = tf[tf.diagnostic.astype(bool)].groupby(["model", "rule", "control"]).agg(
        identity=("err", "mean"), label=("label_ok", lambda s: float((s == False).mean()))).reset_index()  # noqa
    devm = tf[(tf.control == "M") & tf.device_consistent.notna()].groupby(["model", "rule"]).agg(
        device_consistent=("device_consistent", "mean")).reset_index()
    wf = [run_workflow(s, ru) for s in sched.values() for ru in rules]
    out = {"runs": run_dirs, "coordinates": coord, "sufficiency": SUFFICIENCY, "regular": regular,
           "episodes": len(ep), "turns": len(tf),
           "hypotheses": hyp, "i_star": ist.to_dict("records"),
           "tolerance_references": prereg.tolerance_references(),
           "oc_curve": [{"p": p, **{f"threshold_{t}": round(prereg.oc_curve(p, threshold=t), 4)
                                     for t in (prereg.THRESHOLD,) + prereg.SENSITIVITY["threshold"]}}
                        for p in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2)],
           "cost_risk_frontier": front.to_dict("records"), "tolerance_curve": tol.to_dict("records"),
           "day_success": days_ok.to_dict("records"), "performance_profile": profile.to_dict("records"),
           "i_star_order": istar_order(ist, regular) if len(ist) else None,
           "cross_family": cross_family(tf), "robust_measures": robust_measures(tf),
           "interleaving": interleaving(tf), "method_effects": method_effects(tf),
           "wording_robustness": wording_robustness(tf_w[~tf_w.after_first_reminder.fillna(False).astype(bool)]
                                                    if "after_first_reminder" in tf_w else tf_w),
           "i_star_segmented_10pct": ist_seg.to_dict("records"), "memory_gain": gain,
           "memory_k": memory_k_fit(recs), "attribution": attribution(tf), "free_order": free_order(tf),
           "notebook_usage": notebook_usage(recs, sched), "probe_twostack": probe_interface(tf, ep),
           "lapses": lapse_analysis(tf_all, ep), "controller_complexity": controller_complexity(recs),
           "primary_points": {"turns_primary": len(tf), "turns_all": len(tf_all),
                              "diagnostic_primary": int(tf.diagnostic.astype(bool).sum()),
                              "diagnostic_all": int(tf_all.diagnostic.astype(bool).sum())},
           "kaplan_meier": kaplan_meier(tf), "two_layers": two.round(4).to_dict("records"),
           "device_consistency": devm.round(4).to_dict("records"),
           "workflow": {"completion": float(np.mean([w["completion"] for w in wf])) if wf else None, "cost": 0.0},
           "self_conditioning": {ru: (lambda m: None if m is None else {"prior_viol": round(float(m.params["prior_viol"]), 4),
                                                                        "p": float(m.pvalues["prior_viol"])})(
               gee_lpm(tf[tf.diagnostic.astype(bool) & (tf.rule == ru) & (tf.control == "C0")],
                       "err ~ prior_viol")) for ru in ("BAL", "LIFO")}}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        smap.to_csv(os.path.join(out_dir, "sufficiency_map.csv"), index=False)
        front.to_csv(os.path.join(out_dir, "cost_risk_frontier.csv"), index=False)
        tol.to_csv(os.path.join(out_dir, "tolerance_curve.csv"), index=False)
        days_ok.to_csv(os.path.join(out_dir, "day_success.csv"), index=False)
        profile.to_csv(os.path.join(out_dir, "performance_profile.csv"), index=False)
        smap_seg.to_csv(os.path.join(out_dir, "sufficiency_map_segmented_10pct.csv"), index=False)
        sufficiency_map(tf_all, ep).to_csv(os.path.join(out_dir, "sufficiency_map_all_points.csv"), index=False)
        cost_table(ep).to_csv(os.path.join(out_dir, "cost.csv"), index=False)
        ep.to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
        with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--coordinates", default="theory", choices=sorted(COORDINATES))
    ap.add_argument("--threshold", type=float, default=SUFFICIENCY["threshold"], help="sensitivity: 0.02, 0.10")
    ap.add_argument("--bound", default=SUFFICIENCY["bound"], choices=("exact", "stepwise"))
    a = ap.parse_args(argv)
    SUFFICIENCY.update({"threshold": a.threshold, "bound": a.bound})
    out = analyse(a.runs, a.out or os.path.join(a.runs[0], "analysis"), a.coordinates)
    print(f"{out['episodes']} episodes, {out['turns']} turns; regular rule {out['regular']}; coordinates {a.coordinates}")
    for k, v in out["hypotheses"].items():
        print(f"  {k}: " + json.dumps({kk: vv for kk, vv in v.items() if kk in ("refuted", "p", "p_holm", "consistent",
                                                                                "reversed", "ties")}, default=str))
    for r in out["i_star"][:40]:
        print(f"  I* {r['model']:<36s} {r['rule']:<5s} I*={r['I_star']} gate={r['gate_needed']} "
              f"minimal={r['minimal']} cheapest={r['cheapest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
