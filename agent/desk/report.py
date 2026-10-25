"""Post-run report for the help desk (02b §8, §10): checks that need real calls and the pilot gates.

python -m agent.desk.report runs/<run>   -> prints and writes runs/<run>/run_report.json

Gates (pilot):
  Q1 interface   ENGINEERING gate (decides only whether to fix the prompt / tools before the formal run, not a
                 scientific threshold): format errors + text fallbacks < 5% of calls; budget-out turns < 2% of turns
  Q2 provider    served provider = pinned provider; reasoning tokens <= 1 (definition of the model condition)
  Q3 range       REPORT ONLY (internal pilot, 02b §14 A12: it looks at outcome data, so it triggers no design change):
                 the strongest model's C0 error rate on LIFO / FIFO deep days is above the sufficiency threshold (5%;
                 else C0 is already sufficient and the control ladder has nothing to show); the weakest model's C0
                 accuracy on the regular rule is above the window floor (it uses memory). Parts without data: n/a
  Q4 cost        REPORT ONLY since v1.7 (02b §14 A13): mean cost per episode against $0.03 (02b §8 rule 2)
  Q5 depth cost  REPORT ONLY since v1.7: mean cost per episode at D = 10 / at D = 2 against 1.15 (rule 3). Context
                 length no longer depends on D (every day has 16 two-step tickets); what still does is the agent's own
                 information calls, which grow with difficulty (pilot v1.6: 1.68x), i.e. an outcome. Cutting D = 10 on
                 it would change the design on outcome data, which the internal pilot excludes (02b §14 A12); the
                 total budget is guarded by Q6
  Q6 projection  formal plan projected from pilot tokens and the model prices <= $49 (rule 4; v1.9.3: $26 for 2 days
                 per D in v1.5, scaled to 3 days per D and to 40 combinations). Over the cap: budget_action says what
                 rule 4 drops, in this order: MR (exploratory), then the middle dense model Ministral 8B (v1.10; was
                 32B); still over: the author decides (rule 5). The projection counts prompt caching as measured
  Q7 gate        E and C3 execute no illegal resume
  Q8 context     definition: the longest prompt of each model + the output cap fits the pinned endpoint's context
"""
import collections
import glob
import json
import os
import statistics
import sys

from agent.common.models import MODEL_ALIASES, PRICES_PATH, load_models

from . import plan as planmod

from .prereg import MAX_COST_PER_EPISODE, MAX_DEPTH_COST_RATIO, MAX_FORMAL_USD, THRESHOLD    # noqa: E402


def load(run_dir):
    recs = []
    for f in glob.glob(os.path.join(run_dir, "cells", "*", "*.json")):
        with open(f, encoding="utf-8") as fh:
            recs.append(json.load(fh))
    return [r for r in recs if r.get("task") == "desk"]


def _rate(k, n):
    return round(k / n, 4) if n else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 5) if xs else None


def window_floor(rule, recs):
    """Best window-only policy's accuracy at diagnostic points on the days of this run."""
    from .strategies import WINDOW_POLICIES, make_chooser, simulate
    days = {}
    for r in recs:
        if r["rule"] == rule:
            days[r["schedule"]] = planmod.schedule(r["schedule"], r["D"], bool(r.get("high_excursion")),
                                                   "smoke" in r["schedule"])
    best = None
    for w in WINDOW_POLICIES:
        a = n = 0
        for i, s in enumerate(days.values()):
            for x in simulate(s, rule, make_chooser(w, rule, i))[1]:
                if x["diagnostic"]:
                    a, n = a + (x["legal"] is True), n + 1
        if n:
            best = max(best or 0.0, a / n)
    return None if best is None else round(best, 4)


def cell_table(recs):
    """Per model x rule x control: diagnostic points, first-attempt error rate there, deferral, completion, cost."""
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in recs:
        a = agg[(r["model"], r["rule"], r["control"])]
        a["episodes"] += 1
        a["cost"] += r["cost_usd"]
        a["completion"] += r.get("completion") or 0
        for t in r.get("turns", []):
            a["turns"] += 1
            if t.get("diagnostic"):
                a["diag"] += 1
                a["diag_err"] += t.get("first_resume_legal") is False
            if t.get("violable"):
                a["opp"] += 1
                a["deferred"] += bool(t.get("deferred"))
            a["exec_viol"] += t.get("legal") is False
    out = {}
    for (m, ru, c), a in sorted(agg.items()):
        out[f"{m}|{ru}|{c}"] = {"episodes": a["episodes"], "diag_points": a["diag"],
                                "diag_error_rate": _rate(a["diag_err"], a["diag"]),
                                "executed_violations": a["exec_viol"], "deferral_rate": _rate(a["deferred"], a["opp"]),
                                "completion": _rate(a["completion"], a["episodes"]),
                                "cost_per_episode": _rate(a["cost"], a["episodes"])}
    return out


def cached_share(recs, specs):
    """Share of input tokens billed at the cached price, solved from the billed cost of the run's episodes and their
    model's prices (0 for a model without prompt caching); clipped to [0, 1]."""
    tin = bill = 0.0
    for r in recs:
        s = specs.get(r.get("model"))
        if s is None or not s.cache_price or s.cache_price >= s.input_price:
            continue
        tin += r["tokens_in"] * (s.input_price - s.cache_price)
        bill += r["tokens_in"] * s.input_price + r["tokens_out"] * s.output_price - r.get("cost_usd", 0) * 1e6
    return 0.0 if tin <= 0 else max(0.0, min(1.0, bill / tin))


def project_formal(recs, models=None, cells=None):
    """Formal cost from the pilot's mean tokens per episode and each model's prices (episodes per model: the formal plan,
    or `cells`). Input tokens are priced with the cached share measured in the run (cached_share) at each model's own
    cached price, so a model without caching pays the full input price (v1.10: list prices alone overstate a cached run
    several times)."""
    ok = [r for r in recs if r.get("termination") is None and r.get("calls")]
    if not ok:
        return None, {}
    tin = statistics.mean(r["tokens_in"] for r in ok)
    tout = statistics.mean(r["tokens_out"] for r in ok)
    specs = models or load_models()
    f = cached_share(ok, specs)
    per_model = collections.Counter(c["model"] for c in (planmod.formal_plan() if cells is None else cells))
    by = {}
    for alias, n in per_model.items():
        s = specs[MODEL_ALIASES[alias]]
        pin = (1 - f) * s.input_price + f * s.cache_price if s.cache_price else s.input_price
        by[alias] = round(n * (tin * pin + tout * s.output_price) / 1e6, 2)
    return round(sum(by.values()), 2), by


def budget_action(recs, cap=MAX_FORMAL_USD, models=None):
    """02b §8 rule 4 (v1.10): the first of these plans whose projection fits the cap -- the full formal plan, without MR,
    without MR and the middle dense model (Ministral 8B; the dense line keeps its two ends); None without data; a note
    for the author if none fits (rule 5)."""
    full = planmod.formal_plan()
    steps = [("none", full), ("drop MR", [c for c in full if c["control"] != "MR"]),
             ("drop MR and ministral-8b", [c for c in full if c["control"] != "MR" and c["model"] != "ministral-8b"])]
    projections = {}
    for name, cells in steps:
        p, _ = project_formal(recs, models, cells)
        if p is None:
            return None, projections
        projections[name] = p
        if p <= cap:
            return name, projections
    return "over the cap after dropping MR and ministral-8b: the author decides (02b §8 rule 5)", projections


def post_run_report(run_dir):
    recs = load(run_dir)
    calls = [(r, c) for r in recs for c in r.get("calls", [])]
    turns = [(r, t) for r in recs for t in r.get("turns", [])]
    mismatch = collections.Counter((r["model"], c["provider"]) for r, c in calls
                                   if c.get("provider") and c["provider"] != r.get("provider_expected"))
    missing = sum(1 for _, c in calls if not c.get("provider"))
    reasoning = collections.Counter(r["model"] for r, c in calls if (c.get("reasoning_tokens") or 0) > 1)
    trunc = sum(1 for _, c in calls if c.get("completion_tokens", 0) >= 400)
    fmt = sum(r.get("format_errors", 0) for r in recs)
    txt = sum(r.get("text_fallbacks", 0) for r in recs)
    env = sum(r.get("env_errors", 0) for r in recs)
    budget_out = sum(1 for _, t in turns if t.get("forced") and not t.get("blocked"))
    table = cell_table(recs)

    def pooled(model_suffix, rules, control="C0", depths=None):
        a = b = 0
        for r, t in turns:
            if (r["model"] == model_suffix and r["rule"] in rules and r["control"] == control
                    and (depths is None or r["D"] in depths) and t.get("diagnostic")):
                a += t.get("first_resume_legal") is False
                b += 1
        return _rate(a, b), b

    strong, n_s = pooled(MODEL_ALIASES[planmod.STRONG], ("LIFO", "FIFO"), depths=(7, 10))
    weak, n_w = pooled(MODEL_ALIASES[planmod.WEAK], (planmod.REGULAR,))
    floor = window_floor(planmod.REGULAR, recs)
    gate_exec = sum(1 for r, t in turns if r["control"] in ("E", "C3") and t.get("legal") is False)
    ok = [r for r in recs if r.get("termination") is None]
    cost_ep = _mean([r["cost_usd"] for r in ok])
    by_d = {D: _mean([r["cost_usd"] for r in ok if r["D"] == D]) for D in sorted({r["D"] for r in ok})}
    ratio = (round(by_d[10] / by_d[2], 3) if by_d.get(10) and by_d.get(2) else None)
    proj, proj_by = project_formal(recs)
    with open(PRICES_PATH, encoding="utf-8") as f:
        ctx = {m: v.get("context") for m, v in json.load(f).items() if not m.startswith("_")}
    from .runner import MAX_TOKENS
    longest = collections.defaultdict(int)
    for r, c in calls:
        longest[r["model"]] = max(longest[r["model"]], c.get("prompt_tokens") or 0)
    fits = {m: {"longest_prompt": n, "context": ctx.get(m), "fits": ctx.get(m) is None or n + MAX_TOKENS <= ctx[m]}
            for m, n in longest.items()}
    n_calls = len(calls)
    gates = {
        "Q1_interface": {"format_or_text_rate": _rate(fmt + txt, n_calls),
                         "budget_out_turn_rate": _rate(budget_out, len(turns)),
                         "pass": n_calls > 0 and (fmt + txt) / n_calls < 0.05 and budget_out / max(1, len(turns)) < 0.02},
        "Q2_provider": {"mismatch": {f"{m} -> {p}": n for (m, p), n in mismatch.items()}, "no_provider": missing,
                        "reasoning_gt1": dict(reasoning), "pass": not mismatch and not reasoning and missing == 0},
        "Q3_range": {"strongest_C0_LIFO_FIFO_deep_error": strong, "n_points": n_s, "weakest_C0_regular_error": weak,
                     "n_points_weakest": n_w, "regular_window_floor": floor,
                     "strong_above_threshold": None if strong is None else strong > THRESHOLD,
                     "weak_above_floor": None if weak is None or floor is None else 1 - weak > floor,
                     "pass": None},
        "Q4_cost": {"cost_per_episode": cost_ep,
                    "within": None if cost_ep is None else cost_ep <= MAX_COST_PER_EPISODE, "pass": None},
        "Q5_depth_cost": {"cost_by_D": by_d, "D10_over_D2": ratio,
                          "within": None if ratio is None else ratio <= MAX_DEPTH_COST_RATIO, "pass": None},
        "Q6_projection": {"formal_usd": proj, "by_model": proj_by, "cap": MAX_FORMAL_USD,
                          "budget_action": budget_action(recs)[0], "projections": budget_action(recs)[1],
                          "pass": None if proj is None else proj <= MAX_FORMAL_USD},
        "Q7_gate": {"gated_executed_violations": gate_exec, "pass": gate_exec == 0},
        "Q8_context": {"by_model": fits, "pass": None if not fits else all(v["fits"] for v in fits.values())},
    }
    by_model = collections.defaultdict(collections.Counter)
    for r in recs:
        m = by_model[r["model"]]
        m["calls"] += len(r.get("calls", []))
        m["turns"] += len(r.get("turns", []))
        m["format"] += r.get("format_errors", 0)
        m["text"] += r.get("text_fallbacks", 0)
        m["info"] += sum(t.get("info_calls", 0) for t in r.get("turns", []))
        m["env"] += r.get("env_errors", 0)
        m["episodes"] += 1
        m["reminders"] += r.get("reminders", 0)
    rep = {"run": os.path.basename(os.path.normpath(run_dir)), "episodes": len(recs), "calls": n_calls,
           "turns": len(turns), "spent_usd": round(sum(r.get("cost_usd", 0) for r in recs), 4),
           "terminations": dict(collections.Counter(r.get("termination") for r in recs)),
           "errors": {"format_errors": fmt, "text_fallbacks": txt, "env_errors": env, "budget_out_turns": budget_out,
                      "calls_at_output_cap": trunc},
           "by_model": {m: {"calls_per_turn": _rate(c["calls"], c["turns"]), "format_rate": _rate(c["format"], c["calls"]),
                            "text_fallback_rate": _rate(c["text"], c["calls"]),
                            "info_calls_per_turn": _rate(c["info"], c["turns"]),
                            "env_errors_per_turn": _rate(c["env"], c["turns"]),
                            "lapse_reminders_per_episode": _rate(c["reminders"], c["episodes"])}
                        for m, c in by_model.items()},
           "cells": table, "gates": gates}
    with open(os.path.join(run_dir, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return rep


def _mark(p):
    return "n/a " if p is None else ("PASS" if p else "FAIL")


def print_report(rep):
    print(f"\n=== help-desk report: {rep['run']} - {rep['episodes']} episodes, {rep['turns']} turns, "
          f"{rep['calls']} calls, ${rep['spent_usd']} ===")
    print(f"       terminations {rep['terminations']}; errors {rep['errors']}")
    for m, v in rep["by_model"].items():
        print(f"       {m}: {v}")
    for k, v in rep["gates"].items():
        print(f"[{_mark(v['pass'])}] {k}: {({kk: vv for kk, vv in v.items() if kk != 'pass'})}")


if __name__ == "__main__":
    print_report(post_run_report(sys.argv[1]))
