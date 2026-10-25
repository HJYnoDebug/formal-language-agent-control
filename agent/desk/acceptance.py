"""Offline acceptance of the help-desk platform (02b §10): must pass before any paid run.

python -m agent.desk.acceptance [--n 24] [--out runs/desk_acceptance.json]

Every threshold is derived (02b §14 A9; agent/desk/prereg.py): from the preregistered 5% / two-sided 95% / -0.05, from
alpha = 0.05 and power 0.80, from theory or definitions, or from the design itself (the formal days). Items marked
REPORT have no pass line.
  1  leak       window floor <= 1 - (error rate an exact test on one cell's diagnostic points detects with power 0.80
                against 5%), so that window level and sufficiency are distinguishable in one cell (about 0.86); no
                window-visible ticket attribute (code order, category, rush, due, just put on hold, kind, steps left)
                picks the legal ticket more or less often than chance among the window-compatible candidates (two-sided,
                cluster-robust by day, Holm over rule x policy, alpha 0.05), on the acceptance days and on the formal
                days, for the rules of the grid; the formal days are balanced as the generator defines (v1.7, 02b §14
                A13)
  2  temptation REPORT (its only effect, fewer diagnostic points, is covered by item 3)
  3  depth      nominal depth reached exactly on every day type (definition) and depth 1 reached; violable points of
                one formal cell >= 72 (zero violations then give an upper bound <= 5%); the stored contrast cuts equal
                the design medians and both contrast groups are populated (minimal detectable effect reported); deep
                days do not use more tokens than shallow ones (slope not significantly > 0)
  3b layers     label-layer violation implies identity-layer violation; on all identity words over 3 tickets up to 6
                events, the label verdict at the first violation equals REVERSE / SAME on the projected word
  4  defer      the never-resume agent has 0 violations and would be judged not sufficient by the preregistered
                completion criterion (lower bound of the paired difference to the compliant agent < -0.05)
  5  window     window legal sets equal a brute-force enumeration of window-compatible configurations; state-based and
                definitional judges agree on every resume of random trajectories
  6  synthetic  capacity-2 recognisers (theory): no error where the own coordinate is <= 1; more errors where it is >= 3
                (one-sided Fisher test); the other coordinate is not a capacity coordinate (errors also occur where it
                is <= 1, unless none are expected there at alpha 0.05)
  7  env        step order, argument dependency, escalation start enforced; such errors are not violations
  7b reminder   definition: an agent that never lapses gets no reminder and the same trajectory with or without
                reminders; a forgetting agent's day is identical up to the first reminder; the reminder names no ticket;
                lapse causes (after a block / device shows nothing / plain) are told apart; unaided completion counts no
                ticket closed after a reminder given while it was on hold
  8  no leak    displays show only what the device's access discipline shows; P shows no ticket code; device and
                control texts carry no instruction on when / how to use them; the method texts (R, MR) follow one
                template, name no ticket, device or place to write; no wording of a procedure uses a scenario word
                (queue, stack, first come, backlog, pile)
  9  tokens     REPORT: fake-agent projections of the formal plan at one and two calls per turn (02b §8 states its
                estimate as this projection; the binding cost checks are the pilot's budget gates)
  10 axes       b_t, r_t, rho_t, chi_t equal hand-computed values on small examples
  11 atomic     M (careful agent) and C1 push the same line (BAL / PAR: except turns whose arrivals move the device);
                C2 = C1 + one allowed line; P has the C1 line's length; C3 pushes = C2 pushes; R = C0 and MR = M in
                everything but the method paragraph of the system prompt (same pushes, same tools)
  12 storage    lower devices stay in their class; a device-only agent never errs with M (definition) and with M−
                errs not less than chance (theory: 1 - 1/|held| for LIFO / FIFO, 1/2 for BAL; one-sided z, Holm over
                the three rules, alpha 0.05)
  12b probe     two-tray probe M2S (FIFO): the trays accept only ticket codes; a careful agent keeps the queue in
                them (consistency 100%) and a device-only agent never errs; budget covers moving a full tray
  14 comprehension  the probe's items (agent/desk/comprehension.py, v1.9): 72 per rule, 18 per depth, PAR / BAL
                half "first" and half "last"; the rendered text alone determines the answer (parsed back and judged by
                the definitional judge) and it is the legal ticket; 24 per rule also under each paraphrase
  13 grammars   analysis/verify_grammars.py (v1.7): PAR has a right-linear (type 3), BAL and LIFO a context-free
                (type 2), FIFO a monotone (type 1) grammar that equals the definition and both platform judges on every
                domain word up to the length bound; the lower-bound witnesses hold (burst shapes, fooling sets); every
                generated day is in the regular input language and the grammars reject the first violation of real
                trajectories exactly where the platform does
"""
import argparse
import itertools
import json
import math
import os
import re
import statistics

from agent.common.models import load_models, resolve_model

from . import devices, plan as planmod, prereg, prompts
from .env import DeskEnv, EnvError
from .generator import make_schedule
from .rules import CATEGORIES, DeskState, RULED, Ticket, judge_events, label_projection_ok, legal_resume, restriction_bits, \
    state_bits, window_legal_sets
from .runner import FakeToolBackend, choice_bits, residual_bits, run_episode
from .strategies import WINDOW_POLICIES, make_chooser, next_step_action, simulate

DEPTHS = planmod.DEPTHS
SPEC_RANGE_USD = (25, 35)            # 02b v1.6 §8 (2026-10-12 prices, smoke: 1.2-1.4 calls per turn); reported only
LABEL = {"refund": "A", "replace": "B", "address": "C"}
CODE_RE = re.compile(r"\b[B-DF-HJ-NP-TV-XZ]{2}\b")


def schedules(n, depths=DEPTHS, split="accept"):
    return {D: [make_schedule(f"desk|{split}|D{D}|s{s}", D, bool(s % 2)) for s in range(n)] for D in depths}


def _acc(recs, field="legal"):
    pts = [r for r in recs if r["diagnostic"]]
    return sum(r[field] is True for r in pts), len(pts)


# ------------------------------------------------------------------ 1, 2, 3
def _z_p(hits, expected, var, alternative="two-sided"):
    from scipy.stats import norm
    if var <= 0:
        return 1.0
    z = (hits - expected) / math.sqrt(var)
    if alternative == "less":
        return float(norm.cdf(z))
    return float(2 * (1 - norm.cdf(abs(z))))


# window-visible attributes of a ticket on hold (02b §14 A13): the score of each candidate; ties share the credit
ATTRIBUTE_POLICIES = {
    "alphabetical first": lambda env, tk: tuple(-ord(ch) for ch in tk.code),
    "alphabetical last": lambda env, tk: tuple(ord(ch) for ch in tk.code),
    "refund first": lambda env, tk: tk.category == "refund",
    "replace first": lambda env, tk: tk.category == "replace",
    "address first": lambda env, tk: tk.category == "address",
    "rush first": lambda env, tk: tk.rush,
    "not rush first": lambda env, tk: not tk.rush,
    "earliest due": lambda env, tk: -tk.due if tk.rush else -math.inf,
    "just put on hold": lambda env, tk: tk.code == env.recent_hold,
    "not just put on hold": lambda env, tk: tk.code != env.recent_hold,
    "escalated first": lambda env, tk: tk.kind == "escalated",
    "fewest steps left": lambda env, tk: -tk.remaining,
}


def attribute_credit(rule, days):
    """Per day: at the diagnostic points of the compliant trajectory, the credit of every attribute policy and its
    chance value. Candidates are the tickets of the window-compatible legal sets (what a window-only agent can know from
    the window: PAR / BAL the alphabetically first or last, LIFO / FIFO any ticket on hold but one the window rules
    out); credit = share of legal ones among the top-scoring candidates; chance = share of legal ones among all
    candidates (1 / number of window legal sets)."""
    out = []
    for s in days:
        at = {}
        base = make_chooser("compliant", rule)

        def chooser(env, ctx, at=at, base=base):
            st = env.st
            if len(st.held) > 1 and not st.current and not env.escalation_pending:
                cands = sorted(set().union(*window_legal_sets(rule, st, env.recent_hold)))
                legal = legal_resume(rule, st)
                row = {"chance": sum(c in legal for c in cands) / len(cands)}
                for name, score in ATTRIBUTE_POLICIES.items():
                    sc = {c: score(env, st.tickets[c]) for c in cands}
                    best = max(sc.values())
                    top = [c for c in cands if sc[c] == best]
                    row[name] = sum(c in legal for c in top) / len(top)
                at[st.t] = row
            return base(env, ctx)
        recs = simulate(s, rule, chooser)[1]
        out.append([at[r["t"]] for r in recs if r["diagnostic"]])
    return out


def attribute_leak(days, rules=RULED):
    """Each attribute policy against chance, per rule: cluster-robust by day (deviation from chance summed per day;
    t with G - 1 degrees of freedom, small-sample factor G / (G - 1)), two-sided, Holm over rule x policy, alpha 0.05."""
    from scipy.stats import t as tdist
    res, pvals = {}, {}
    for rule in rules:
        per_day = attribute_credit(rule, days)
        n = sum(len(d) for d in per_day)
        for name in ATTRIBUTE_POLICIES:
            dev = [sum(p[name] - p["chance"] for p in d) for d in per_day]
            hits = sum(p[name] for d in per_day for p in d)
            exp = sum(p["chance"] for d in per_day for p in d)
            g, ss = len(dev), sum(x * x for x in dev)
            p = 1.0 if ss < 1e-12 else float(2 * tdist.sf(abs(sum(dev)) / math.sqrt(ss * g / (g - 1)), g - 1))
            pvals[(rule, name)] = p
            res[(rule, name)] = {"rate": round(hits / n, 3) if n else None, "chance": round(exp / n, 3) if n else None,
                                 "points": n, "p": round(p, 4)}
    adj = prereg.holm(pvals)
    for k in res:
        res[k]["p_holm"] = round(adj[k], 4)
    worst = min(res, key=lambda k: (res[k]["p_holm"], res[k]["p"]))
    return {"pass": all(v["p_holm"] > prereg.ALPHA for v in res.values()),
            "smallest": {"rule": worst[0], "policy": worst[1], **res[worst]},
            "by_rule": {r: {nm: res[(r, nm)] for nm in ATTRIBUTE_POLICIES} for r in rules}}


def formal_balance():
    """Definition check of the generator's balance on the formal days (generator docstring): per day, PAR / BAL
    designate the alphabetically first and last ticket equally often (+-1), LIFO / FIFO as often as chance (+-1)."""
    bad = []
    for D, seed, s in prereg.formal_days():
        for rule in ("PAR", "BAL", "LIFO", "FIFO"):
            pts = [r for r in simulate(s, rule, make_chooser("compliant", rule))[1] if r["violable"]]
            first = sum(sorted(r["held"])[0] in r["legal_set"] for r in pts)
            last = sum(sorted(r["held"])[-1] in r["legal_set"] for r in pts)
            chance = sum(1 / len(r["held"]) for r in pts)
            if rule in ("PAR", "BAL"):
                ok = abs(first - last) <= 1
            else:
                ok = max(abs(first - chance), abs(last - chance)) <= 1
            if not ok:
                bad.append((D, seed, rule, first, last, round(chance, 1)))
    return {"pass": not bad, "unbalanced": bad}


def check_leak(S, rules=None):
    """Rules of the grid only (LAST is no longer run, 02b §14 D1)."""
    rules = rules or [r for r in planmod.rules_in_grid() if r != "FREE"]
    out, floor_ok = {}, {}
    for rule in rules:
        n_cell = sum(r["diagnostic"] for r in prereg.design_points(rule))
        bound = 1 - prereg.detectable_rate(n_cell)
        best = {}
        for w in WINDOW_POLICIES:
            a = n = 0
            for D, days in S.items():
                for i, s in enumerate(days):
                    x, y = _acc(simulate(s, rule, make_chooser(w, rule, i))[1])
                    a, n = a + x, n + y
            best[w] = a / n if n else float("nan")
        top = max(best.values())
        floor_ok[rule] = top <= bound
        out[rule] = {"best_window": round(top, 3), "best_policy": max(best, key=best.get), "floor_bound": round(bound, 3),
                     "cell_diagnostic_points": n_cell, "pass": floor_ok[rule]}
    attr = {"acceptance_days": attribute_leak([s for days in S.values() for s in days], rules),
            "formal_days": attribute_leak([s for _, _, s in prereg.formal_days()], rules)}
    bal = formal_balance()
    ok = all(floor_ok.values()) and all(a["pass"] for a in attr.values()) and bal["pass"]
    return {"pass": ok, "window_floor": out, "attributes": attr, "formal_balance": bal}


def check_temptation(S, rules=RULED):
    """REPORT: share of resume opportunities with an urgent rush ticket that may not be resumed."""
    out = {}
    for rule in rules:
        per_d = {}
        for D, days in S.items():
            o = t = 0
            for s in days:
                _, recs = simulate(s, rule, make_chooser("compliant", rule))
                o += sum(r["opportunity"] for r in recs)
                t += sum(r["temptation"] for r in recs)
            per_d[D] = round(t / o, 3)
        out[rule] = {"pooled": round(statistics.mean(per_d.values()), 3), "by_D": per_d}
    return {"pass": None, "by_rule": out, "note": "reported; its effect (fewer diagnostic points) is bounded by item 3"}


def check_depth(S, spec, rules_grid, rules=RULED):
    from scipy.stats import linregress
    ref = {D: [simulate(s, "LIFO", make_chooser("reference", "LIFO"))[0].max_depth for s in days]
           for D, days in S.items()}
    depth_ok = all(set(v) == {D} for D, v in ref.items()) and min(min(v) for v in ref.values()) <= 2
    need = prereg.zero_event_n()
    cells, groups = {}, {}
    for rule in [r for r in rules_grid if r != "FREE"]:
        pts = prereg.design_points(rule)
        diag = [r for r in pts if r["diagnostic"]]
        cells[rule] = {"violable": sum(r["violable"] for r in pts), "diagnostic": len(diag)}
        coord, cut = prereg.OWN_COORD[rule], prereg.CUTS[rule]
        hi = sum(r[coord] >= cut for r in diag)
        groups[rule] = {"coord": coord, "cut": cut, "low": len(diag) - hi, "high": hi,
                        "mde": None if not hi or hi == len(diag) else round(prereg.mde_two_groups(len(diag) - hi, hi), 3)}
    own, other = prereg.design_cuts()
    cuts_ok = (all(own[r] == prereg.CUTS[r] for r in prereg.CUTS) and
               all(other[r] == prereg.CUTS_OTHER[r] for r in prereg.CUTS_OTHER))
    points_ok = all(c["violable"] >= need for c in cells.values())
    groups_ok = all(g["low"] > 0 and g["high"] > 0 for g in groups.values())
    xs, ys = [], []
    for D, days in S.items():
        for s in days[:6]:
            xs.append(D)
            ys.append(run_episode(s, "LIFO", "C0", FakeToolBackend(spec, "compliant"), spec)["tokens_in"])
    lr = linregress(xs, ys)
    p_up = lr.pvalue / 2 if lr.slope > 0 else 1 - lr.pvalue / 2
    tokens_ok = p_up > prereg.ALPHA
    diag_day = {rule: {D: round(sum(sum(r["diagnostic"] for r in simulate(s, rule, make_chooser("compliant", rule))[1])
                                    for s in days) / len(days), 2) for D, days in S.items()} for rule in rules}
    return {"pass": depth_ok and points_ok and cuts_ok and groups_ok and tokens_ok,
            "depth_exact": depth_ok, "violable_needed_per_cell": need, "formal_cell_points": cells,
            "contrast_groups": groups, "cuts_match_design": cuts_ok, "design_cuts": {"own": own, "other": other},
            "tokens_slope_per_D": round(lr.slope, 1), "tokens_p_increasing": round(p_up, 4),
            "diag_per_day": diag_day, "max_counter_bound": max(s["excursion_max"] for v in S.values() for s in v)}


# ------------------------------------------------------------------ 3b
def _machine(rule):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "verify_state_counts.py")
    spec = importlib.util.spec_from_file_location("vsc", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_rule({"LIFO": "REVERSE", "FIFO": "SAME"}[rule], 50), mod.SINK


def check_layers(S, max_len=6):
    subset = True
    for rule in RULED:
        for days in S.values():
            for i, s in enumerate(days[:8]):
                _, recs = simulate(s, rule, make_chooser("win_random", rule, i))
                subset &= all(not (r["label_ok"] is False and r["legal"] is not False) for r in recs)
    agree, n_words = True, 0
    for rule in ("LIFO", "FIFO"):
        (init, step, _), sink = _machine(rule)
        for cats in itertools.product(CATEGORIES, repeat=3):
            st0 = DeskState(tickets={f"T{i}": Ticket(f"T{i}", "normal", cats[i], 0, False, 0, "", "")
                                     for i in range(3)})
            todo = [(st0, init, 0)]
            while todo:
                st, m, depth = todo.pop()
                if depth == max_len:
                    continue
                for c in sorted(st.tickets):
                    lab = LABEL[st.tickets[c].category]
                    if c not in st.held:                                   # hold: o(label)
                        st2 = _copy_state(st)
                        st2.put_on_hold(c)
                        todo.append((st2, step(m, ("o", lab)), depth + 1))
                        continue
                    m2 = step(m, ("c", lab))                               # resume: c(label)
                    n_words += 1
                    agree &= label_projection_ok(rule, st, c) == (m2 != sink)
                    if c in legal_resume(rule, st):                        # continue only before the first violation
                        st2 = _copy_state(st)
                        st2.take_off_hold(c)
                        todo.append((st2, m2, depth + 1))
    return {"pass": subset and agree, "label_subset_of_identity": subset, "reverse_same_agree": agree,
            "resume_words_checked": n_words}


def _copy_state(st):
    import copy
    s = copy.copy(st)
    s.held = list(st.held)
    return s


# ------------------------------------------------------------------ 4, 5, 6
def check_defer(S, rules=RULED):
    from scipy.stats import t as tdist
    out, ok = {}, True
    for rule in rules:
        diffs, viol = [], 0
        for days in S.values():
            for s in days:
                e1, _ = simulate(s, rule, make_chooser("compliant", rule))
                e2, r2 = simulate(s, rule, make_chooser("defer", rule))
                diffs.append(e2.finish()["completion"] - e1.finish()["completion"])
                viol += sum(r["legal"] is False for r in r2)
        n = len(diffs)
        sd = statistics.stdev(diffs) if n > 1 else 0.0
        lower = statistics.mean(diffs) - tdist.ppf(prereg.LEVEL, n - 1) * sd / math.sqrt(n)
        p = viol == 0 and lower < prereg.COMPLETION_MARGIN
        ok &= p
        out[rule] = {"mean_completion_difference": round(statistics.mean(diffs), 3), "lower_bound": round(lower, 3),
                     "defer_violations": viol, "pass": p}
    return {"pass": ok, "by_rule": out}


def brute_window_sets(rule, st, recent_hold):
    """Legal sets over all configurations compatible with the window: the hold set and the ticket held by the last
    action are seen; the hold order, whether the latest hold is still on hold, and the counts are not."""
    held = list(st.held)
    sets = set()
    for order in itertools.permutations(held):
        if recent_hold in held and order[-1] != recent_hold:
            continue
        lasts = [recent_hold] if recent_hold in held else [order[-1], None]   # latest hold still held, or gone
        for last in lasts:
            for sign in (0, -1):
                for parity in (0, 1):
                    x = _copy_state(st)
                    x.held = list(order)
                    x.last_held = last if last is not None else "__gone__"
                    x.n_refund = parity + 2
                    x.n_replace = x.n_refund - sign
                    sets.add(frozenset(legal_resume(rule, x)))
    return sets


def check_window(S, rules=RULED):
    enum_ok, judge_ok, n_pts, n_res = True, True, 0, 0
    for rule in rules:
        for days in S.values():
            for i, s in enumerate(days[:6]):
                env = DeskEnv(s, rule)
                ch = make_chooser("win_random", rule, i)
                while not env.done():
                    env.start_turn()
                    if env.st.current is None and len(env.st.held) >= 2 and len(env.st.held) <= 6 \
                            and not env.escalation_pending:
                        mine = {frozenset(x) for x in window_legal_sets(rule, env.st, env.recent_hold)}
                        brute = brute_window_sets(rule, env.st, env.recent_hold)
                        brute = {b for b in brute if b}
                        enum_ok &= mine == brute
                        n_pts += 1
                    env.work(*ch(env, {}))
                flags = [x[2] for x in env.resumes]
                judge_ok &= judge_events(rule, env.events) == flags
                n_res += len(flags)
    return {"pass": enum_ok and judge_ok, "window_enumeration": enum_ok, "points": n_pts,
            "judges_agree": judge_ok, "resumes": n_res}


def _ols(recs, ys, keys):
    """Linear probability model error ~ 1 + keys; returns {key: (slope, t)}."""
    import numpy as np
    X = np.array([[1.0] + [float(r[k]) for k in keys] for r in recs])
    y = np.array(ys, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / max(1, len(y) - len(keys) - 1)
    se = np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(X.T @ X)), 1e-18, None))
    return {k: (float(beta[i + 1]), float(beta[i + 1] / se[i + 1])) for i, k in enumerate(keys)}


def check_synthetic(S):
    """Capacity-2 recognisers (memory:2: the last two hold events; a counter saturating at +-2), theory: exact where
    the own coordinate (burial for LIFO / FIFO, counter bound for BAL) is <= 1, wrong beyond capacity."""
    from scipy.stats import fisher_exact
    out, ok = {}, True
    for rule in RULED:
        recs = []
        for days in S.values():
            for i, s in enumerate(days):
                recs += [r for r in simulate(s, rule, make_chooser("memory:2", rule, i))[1] if r["diagnostic"]]
        errs = [r["legal"] is False for r in recs]
        res = {"errors": sum(errs), "points": len(recs)}
        if rule in ("LAST", "PAR"):
            p = sum(errs) == 0
        else:
            own, other = ("burial", "counter_peak") if rule in ("LIFO", "FIFO") else ("counter_peak", "burial")
            low = [e for r, e in zip(recs, errs) if r[own] <= 1]
            high = [e for r, e in zip(recs, errs) if r[own] >= 3]
            oth = [e for r, e in zip(recs, errs) if r[other] <= 1]
            _, p_step = fisher_exact([[sum(high), len(high) - sum(high)], [sum(low), len(low) - sum(low)]],
                                     alternative="greater")
            rate = sum(errs) / len(errs) if errs else 0.0
            p_none_expected = (1 - rate) ** len(oth)
            other_ok = sum(oth) > 0 or p_none_expected >= prereg.ALPHA
            p = sum(low) == 0 and p_step < prereg.ALPHA and other_ok
            res.update({"own": own, "err_own_le1": f"{sum(low)}/{len(low)}", "err_own_ge3": f"{sum(high)}/{len(high)}",
                        "fisher_p": float(p_step), "err_other_le1": f"{sum(oth)}/{len(oth)}"})
            res["fit"] = {k: [round(v[0], 4), round(v[1], 1)] for k, v in _ols(recs, errs, ("burial", "counter_peak")).items()}
        ok &= bool(p)
        out[rule] = {**res, "pass": bool(p)}
    return {"pass": ok, "by_rule": out}


# ------------------------------------------------------------------ 7, 8
def check_reminder(S, spec):
    s = S[max(S)][1]
    out, ok = {}, True
    a = run_episode(s, "LIFO", "C0", FakeToolBackend(spec, "compliant"), spec, reminders=True)
    b = run_episode(s, "LIFO", "C0", FakeToolBackend(spec, "compliant"), spec, reminders=False)
    same = [(t["action"], t["code"]) for t in a["turns"]] == [(t["action"], t["code"]) for t in b["turns"]]
    out["no_lapse_agent"] = {"reminders": a["reminders"], "identical": same}
    ok &= a["reminders"] == 0 and same
    f_on = run_episode(s, "LIFO", "C0", FakeToolBackend(spec, "forget:compliant"), spec, reminders=True)
    f_off = run_episode(s, "LIFO", "C0", FakeToolBackend(spec, "forget:compliant"), spec, reminders=False)
    k = f_on["first_reminder_t"]
    prefix = ([(t["action"], t["code"]) for t in f_on["turns"][:k]] == [(t["action"], t["code"]) for t in f_off["turns"][:k]])
    texts = [m["content"] for m in f_on["transcript"] if m["role"] == "user" and prompts.REMINDER in (m["content"] or "")]
    no_code = all(not CODE_RE.findall(prompts.REMINDER) for _ in texts)
    aided_ok = f_on["completion_unaided"] <= f_on["completion"]
    out["forgetting_agent"] = {"first_reminder_t": k, "reminders": f_on["reminders"], "prefix_identical": prefix,
                               "reminder_names_no_ticket": no_code, "completion": round(f_on["completion"], 3),
                               "completion_unaided": round(f_on["completion_unaided"], 3),
                               "completion_without_reminders": round(f_off["completion"], 3)}
    ok &= k is not None and prefix and no_code and aided_ok and f_on["reminders"] > 0
    causes = {}
    for ctl, pol in (("E", "forget:stubborn:win_random"), ("M", "forget:compliant"), ("C0", "forget:compliant")):
        r = run_episode(s, "LIFO", ctl, FakeToolBackend(spec, pol), spec)
        causes[ctl] = sorted({t["lapse_cause"] for t in r["turns"] if t["lapse"]})
    out["causes"] = causes
    ok &= "after_block" in causes["E"] and causes["M"] == ["device_empty"] and causes["C0"] == ["plain"]
    return {"pass": ok, **out}



def check_env():
    s = make_schedule("desk|accept_env|D4|s0", 4)
    env = DeskEnv(s, "LIFO")
    env.start_turn()
    first = sorted(env.st.tickets.values(), key=lambda t: (t.arrival, t.code))[0]
    action = first.steps[1]
    errs = []
    for args in ((action, first.code, first.customer_id),             # wrong step (verify comes first)
                 ("lookup", first.code, first.customer_id),           # no such tool (removed in v1.7)
                 ("verify", "ZZ", None),                              # no such ticket
                 (action, first.code, "x")):                          # wrong step and wrong id
        try:
            env.work(*args)
            errs.append(False)
        except EnvError:
            errs.append(True)
    env.work("verify", first.code)
    try:
        env.work(action, first.code, "C-0000")                        # wrong customer id
        errs.append(False)
    except EnvError:
        errs.append(True)
    env2 = DeskEnv(s, "LIFO")                                         # an escalation must be started now
    ch = make_chooser("compliant", "LIFO")
    esc_ok = None
    while not env2.done() and esc_ok is None:
        env2.start_turn()
        if env2.escalation_pending and env2.st.current:
            try:
                env2.work(*next_step_action(env2.st.tickets[env2.st.current]))
                esc_ok = False
            except EnvError:
                esc_ok = True
            break
        env2.work(*ch(env2, {}))
    no_viol = not env.resumes and all(x[2] for x in env2.resumes)
    return {"pass": all(errs) and esc_ok is True and no_viol, "errors_raised": errs, "escalation_enforced": esc_ok,
            "no_violation_recorded": no_viol}


FORBIDDEN = ("when you", "whenever", "each time", "every time you", "should", "remember to", "make sure",
             "put on hold, ", "after you")


SCENARIO_WORDS = ("queue", "stack", "first come", "first-come", "backlog", "pile")
METHOD_FORBIDDEN = ("device", "tool", "write", "note", "board", "clip", "rail", "tray", "marker", "counter")


def check_no_leak(S):
    ok, issues = True, []
    for rule, text in prompts.METHOD.items():
        low = text.lower()
        if not (text.startswith("Keep in mind") and "When you resume a ticket" in text or rule == "LAST"):
            ok, _ = False, issues.append(f"method {rule}: not in the template")
        if CODE_RE.findall(text) or any(w in low for w in METHOD_FORBIDDEN):
            ok, _ = False, issues.append(f"method {rule}: names a ticket, a device or a place to write")
    for rule, variants in prompts.PROCEDURE_VARIANTS.items():
        for i, text in enumerate(variants):
            if any(w in text.lower() for w in SCENARIO_WORDS):
                ok, _ = False, issues.append(f"wording {rule}/{i}: scenario word")
    for kind in devices.DESC:
        txt = devices.device_description(kind).lower()
        for w in FORBIDDEN:
            if w in txt:
                ok, _ = False, issues.append(f"device {kind}: '{w}'")
    for c, txt in prompts.CONTROL_TEXT.items():
        for w in FORBIDDEN:
            if w in txt.lower():
                ok, _ = False, issues.append(f"control {c}: '{w}'")
    for rule in ("LAST", "PAR", "BAL", "LIFO", "FIFO"):
        for days in S.values():
            for s in days[:4]:
                env = DeskEnv(s, rule)
                ch = make_chooser("compliant", rule)
                while not env.done():
                    env.start_turn()
                    from .runner import control_lines
                    c1 = control_lines(rule, "C1", env.st, None)[0]
                    p = control_lines(rule, "P", env.st, None)[0]
                    codes = set(CODE_RE.findall(c1))
                    if rule == "LIFO":
                        allowed = {env.st.held[-1]} if env.st.held else set()
                    elif rule == "FIFO":
                        allowed = {env.st.held[0]} if env.st.held else set()
                    elif rule == "LAST":
                        allowed = {env.st.last_held} if env.st.last_held else set()
                    else:
                        allowed = set()
                    if not codes <= allowed:
                        ok = False
                        issues.append(f"{rule} C1 shows {codes - allowed}")
                    if CODE_RE.findall(p):
                        ok = False
                        issues.append(f"P shows a code: {p}")
                    env.work(*ch(env, {}))
    return {"pass": ok, "issues": issues[:10]}


# ------------------------------------------------------------------ 9, 10
def check_tokens(spec_by_alias, S):
    days = [s for D in S for s in S[D][:4]]
    t1 = [run_episode(s, "LIFO", "C0", FakeToolBackend(spec_by_alias[planmod.STRONG], "compliant"), spec_by_alias[planmod.STRONG])
          for s in days]
    t2 = [run_episode(s, "LIFO", "C0", FakeToolBackend(spec_by_alias[planmod.STRONG], "view:compliant"),
                      spec_by_alias[planmod.STRONG]) for s in days]
    per_model = {}
    for name, recs in (("one_call_per_turn", t1), ("two_calls_per_turn", t2)):
        tin = statistics.mean(r["tokens_in"] for r in recs)
        tout = statistics.mean(r["tokens_out"] for r in recs)
        n = len(planmod.formal_plan()) // len(planmod.MODELS)
        per_model[name] = round(sum(n * (tin * sp.input_price + tout * sp.output_price) / 1e6
                                    for sp in spec_by_alias.values()), 2)
    return {"pass": None, "formal_usd": per_model, "spec_range_usd": list(SPEC_RANGE_USD),
            "note": "02b §8 states its estimate as this projection at 1.5-2 calls per turn; the pilot's budget gates "
                    "are the binding cost checks"}


def check_axes():
    tk = {c: Ticket(c, "escalated", cat, 0, False, 0, "C-1", "O-1") for c, cat in
          (("BB", "refund"), ("CC", "replace"), ("DD", "refund"), ("FF", "address"))}
    st = DeskState(tickets=tk)
    for c in ("BB", "CC", "DD", "FF"):
        st.arrive(c)
    for c in ("BB", "CC", "DD"):
        st.put_on_hold(c)
    tk["FF"].status = "waiting"
    checks = {
        "b_LIFO_d3": (state_bits("LIFO", st), math.log2(6)),
        "b_LAST_d3": (state_bits("LAST", st), math.log2(4)),
        "b_BAL_bound1": (state_bits("BAL", st), math.log2(3)),          # counter path 1, 0, 1: bound M_t = 1
        "b_PAR": (state_bits("PAR", st), 1.0),
        "rho_LIFO": (restriction_bits("LIFO", st), math.log2(3)),
        "rho_LAST_restricted": (restriction_bits("LAST", st), math.log2(3)),
    }
    st2 = _copy_state(st)
    st2.take_off_hold("DD")                                               # the last hold resumed: LAST is free
    checks["rho_LAST_free"] = (restriction_bits("LAST", st2), 0.0)
    env = DeskEnv({"tickets": [], "t_max": 10}, "LIFO")
    env.st = st
    checks["chi_ungated"] = (choice_bits(env, False), math.log2(3 + 1 + 1))   # 3 held + 1 waiting + wait
    checks["chi_gated"] = (choice_bits(env, True), math.log2(1 + 1 + 1))
    snap = {"residual_bits_window": math.log2(3)}
    checks["r_C0"] = (residual_bits("C0", snap, None), math.log2(3))
    checks["r_C1"] = (residual_bits("C1", snap, None), 0.0)
    checks["r_M_ok"] = (residual_bits("M", snap, True), 0.0)
    checks["r_M_wrong"] = (residual_bits("M", snap, False), math.log2(3))
    env.recent_hold = None
    checks["r_window_LIFO_d3"] = (math.log2(len(window_legal_sets("LIFO", st, None))), math.log2(3))
    ok = all(abs(a - b) < 1e-9 for a, b in checks.values())
    return {"pass": ok, "checks": {k: [round(a, 4), round(b, 4)] for k, (a, b) in checks.items()}}


# ------------------------------------------------------------------ 11, 12
def _pushes(s, rule, control, spec, policy="compliant"):
    r = run_episode(s, rule, control, FakeToolBackend(spec, policy), spec)
    return r["turns"]


def check_atomic(S, spec):
    ok, issues, n = True, [], 0
    for rule in ("LAST", "PAR", "BAL", "LIFO", "FIFO"):
        for days in S.values():
            for s in days[:3]:
                tm = _pushes(s, rule, "M", spec, "dev:compliant")
                t1, t2, t3, tp = (_pushes(s, rule, c, spec) for c in ("C1", "C2", "C3", "P"))
                for a, b, c, d, e in zip(tm, t1, t2, t3, tp):
                    n += 1
                    moves = {"BAL": lambda x: any(s_["category"] in ("refund", "replace") for s_ in s["tickets"]
                                                  if s_["code"] in x["arrivals_now"]),
                             "PAR": lambda x: any(s_["category"] == "refund" for s_ in s["tickets"]
                                                  if s_["code"] in x["arrivals_now"])}.get(rule, lambda x: False)
                    if a["pushed"] != b["pushed"] and not moves(b):
                        ok = False
                        issues.append(f"{rule} t={b['t']}: M {a['pushed']} vs C1 {b['pushed']}")
                    if c["pushed"][:1] != b["pushed"] or len(c["pushed"]) != 2:
                        ok = False
                        issues.append(f"{rule} t={b['t']}: C2 {c['pushed']} vs C1 {b['pushed']}")
                    if d["pushed"] != c["pushed"]:
                        ok = False
                        issues.append(f"{rule} t={b['t']}: C3 differs from C2")
                    if len(e["pushed"][0]) != len(b["pushed"][0]):
                        ok = False
                        issues.append(f"{rule} t={b['t']}: P length {len(e['pushed'][0])} vs {len(b['pushed'][0])}")
                if rule == "LAST":
                    continue
                # knowledge axis: R = C0 and MR = M except for the method paragraph (02b v1.9 §6)
                para = "\n\n" + prompts.method_paragraph(rule)
                desc = devices.device_description(devices.DEVICE_FOR[rule])
                if (prompts.system_prompt(rule, "R").replace(para, "") != prompts.system_prompt(rule, "C0") or
                        prompts.system_prompt(rule, "MR", desc).replace(para, "") !=
                        prompts.system_prompt(rule, "M", desc)):
                    ok = False
                    issues.append(f"{rule}: R / MR prompts differ from C0 / M beyond the method paragraph")
                t_r, t_c0 = _pushes(s, rule, "R", spec), _pushes(s, rule, "C0", spec)
                t_mr = _pushes(s, rule, "MR", spec, "dev:compliant")
                if [x["pushed"] for x in t_r] != [x["pushed"] for x in t_c0] or \
                        [x["pushed"] for x in t_mr] != [x["pushed"] for x in tm]:
                    ok = False
                    issues.append(f"{rule}: R / MR pushes differ from C0 / M")
    return {"pass": ok, "turns_compared": n, "issues": issues[:10]}


def check_storage(S, spec):
    reg = devices.Device("register")
    class_ok = True
    for bad in ("hello", "ABC", "12", "BB CC"):
        try:
            reg.apply("board_write", {"ticket": bad}, {"BB", "CC"})
            class_ok = False
        except devices.DeviceError:
            pass
    class_ok &= set(devices.OPS["counter"]) == {"counter_up", "counter_down", "counter_sign"}
    res, ok, p_below = {}, class_ok, {}
    for rule in ("BAL", "LIFO", "FIFO"):
        for control in ("M", "Mminus"):
            err = n = exp = var = 0
            for D, days in S.items():
                for i, s in enumerate(days):
                    for t in run_episode(s, rule, control, FakeToolBackend(spec, "dev:devread", i), spec)["turns"]:
                        if not t["diagnostic"]:
                            continue
                        q = 0.5 if rule == "BAL" else 1 - 1 / len(t["held"])     # chance error of a blind pick
                        err, n = err + (t["first_resume_legal"] is False), n + 1
                        exp, var = exp + q, var + q * (1 - q)
            if control == "M":
                p = err == 0
                res[f"{rule}|M"] = {"errors": err, "points": n, "pass": p}
                ok &= p
            else:
                p_below[rule] = _z_p(err, exp, var, "less")
                res[f"{rule}|Mminus"] = {"errors": err, "points": n, "chance_errors": round(exp, 1),
                                         "p_below_chance": round(p_below[rule], 4)}
    for rule, p_adj in prereg.holm(p_below).items():          # three rules tested: Holm, as in item 1
        res[f"{rule}|Mminus"].update({"p_holm": round(p_adj, 4), "pass": p_adj > prereg.ALPHA})
        ok &= p_adj > prereg.ALPHA
    return {"pass": ok, "device_class": class_ok, "device_only": res}


def check_probe_twostack(S, spec):
    dev = devices.Device("twostack")
    class_ok = True
    try:
        dev.apply("tray1_put", {"ticket": "anything"}, {"BB"})
        class_ok = False
    except devices.DeviceError:
        pass
    viol = cons = n = 0
    for D, days in S.items():
        for s in days[:4]:
            for pol in ("dev:compliant", "dev:devread"):
                r = run_episode(s, "FIFO", "M2S", FakeToolBackend(spec, pol), spec)
                viol += sum(t["legal"] is False for t in r["turns"])
                c = [t["device_consistent"] for t in r["turns"]]
                cons += sum(bool(x) for x in c)
                n += len(c)
    return {"pass": class_ok and viol == 0 and cons == n, "device_class": class_ok, "violations": viol,
            "device_consistency": round(cons / n, 4) if n else None}


# ------------------------------------------------------------------ main
def check_comprehension():
    from collections import Counter
    from . import comprehension as comp
    out, ok = {}, True
    for rule in comp.RULES:
        its = comp.items(rule)
        by_d = Counter(it["D"] for it in its)
        sides = Counter(it["side"] for it in its)
        bad = [it["item"] for it in its if comp.answer_from_text(rule, comp.render(it)) != [it["answer"]]]
        para = sum(it["paraphrased"] for it in its)
        good = (len(its) == prereg.zero_event_n() and set(by_d.values()) == {len(its) // len(DEPTHS)} and not bad
                and para == comp.N_PARA * len(DEPTHS) and len(prompts.PROCEDURE_VARIANTS[rule]) == 3
                and (rule not in ("PAR", "BAL") or sides["first"] == sides["last"]))
        ok &= good
        out[rule] = {"items": len(its), "by_depth": dict(by_d), "answer_side": dict(sides), "paraphrased": para,
                     "not_determined_by_text": bad[:5], "pass": good}
    return {"pass": ok, "by_rule": out, "calls_4_models": len(comp.probe_plan())}


def check_grammars():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "analysis", "verify_grammars.py")
    spec = importlib.util.spec_from_file_location("verify_grammars", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run()


def run_all(n=24, regular=planmod.REGULAR):
    models = load_models()
    specs = {a: resolve_model(a, models) for a in planmod.MODELS}
    spec = specs[planmod.STRONG]
    S = schedules(n)
    grid_rules = planmod.rules_in_grid(regular)
    out = {"regular_rule": regular,
           "1_leak": check_leak(S), "2_temptation": check_temptation(S),
           "3_depth": check_depth(S, spec, grid_rules), "3b_layers": check_layers(S),
           "4_defer": check_defer(S), "5_window": check_window(S), "6_synthetic": check_synthetic(S),
           "7_env": check_env(), "7b_reminder": check_reminder(S, spec), "8_no_leak": check_no_leak(S),
           "9_tokens": check_tokens(specs, S),
           "10_axes": check_axes(), "11_atomic": check_atomic(S, spec), "12_storage": check_storage(S, spec),
           "12b_probe_twostack": check_probe_twostack(S, spec), "13_grammars": check_grammars(),
           "14_comprehension": check_comprehension()}
    out["all_pass"] = all(v["pass"] for k, v in out.items() if isinstance(v, dict) and v.get("pass") is not None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="days per depth")
    ap.add_argument("--regular", default=planmod.REGULAR, choices=("LAST", "PAR"))
    ap.add_argument("--out", default=os.path.join("runs", "desk_acceptance.json"))
    a = ap.parse_args(argv)
    res = run_all(a.n, a.regular)
    for k, v in res.items():
        if isinstance(v, dict):
            mark = "REPORT" if v["pass"] is None else "PASS" if v["pass"] else "FAIL"
            print(f"[{mark}] {k}: " + json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False,
                                                 default=str))
    print("ALL PASS" if res["all_pass"] else "SOME CHECKS FAIL")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
