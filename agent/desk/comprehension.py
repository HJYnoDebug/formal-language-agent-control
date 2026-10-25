"""Rule comprehension probe (02b v1.9 §7.1): can the model apply the procedure when everything that has happened today
is written out in one message, so that no memory across turns is needed?

If it can (error rate under the sufficiency threshold), the procedure text is understood and errors in the episodes are
errors of keeping the state across turns, not of the wording. Each item is a diagnostic point of the compliant
trajectory on separate days (split "comprehension"): the event log up to that point, the tickets on hold
(alphabetical, as view_queue shows them) and the question which one may be resumed now; one call, answered with the
tool `answer`. Same system text as C0 (help desk, procedure in the given wording).

Items per model x rule: N_ITEMS = 72 under wording 0 (prereg.zero_event_n: zero errors give an exact upper bound
<= 5%), 18 per depth, at most PER_DAY per day, PAR / BAL half "first" and half "last" answers per depth; the first
N_PARA = 6 per depth of them (24 per rule) under each paraphrase (descriptive). 4 models x 4 rules x (72 + 2 x 24) =
1,920 calls.

python -m agent.desk.comprehension --run d_comprehension --budget 1.0 --backend openrouter
python -m agent.desk.comprehension --summary runs/d_comprehension

Written-reasoning control (2026-10-17, exploratory, freeze log changes after the freeze 6): the same 24 paraphrased
items per rule under wording 0, the weakest and the strongest model, one call with tool_choice "auto" and the
instruction to work the answer out in writing before calling the answer tool; compared item by item with the probe's
answers (forced tool call, no written reasoning, as in the episodes).

python -m agent.desk.comprehension --reasoning --run d_comprehension_reasoning --budget 0.4 --backend openrouter
python -m agent.desk.comprehension --reasoning-summary runs/d_comprehension_reasoning runs/d_comprehension
"""
import argparse
import json
import os
import random
import re

from agent.common import executor
from agent.common.toolcall import MeteredCaller, ToolReply, tool_call_json
from agent.common.ledger import ServiceFailure

from . import plan as planmod, prereg, prompts
from .rules import judge_events
from .strategies import make_chooser, simulate

RULES = ("PAR", "BAL", "LIFO", "FIFO")
SPLIT = "comprehension"
N_ITEMS = prereg.zero_event_n()
N_PARA = 6                       # per depth and paraphrase
PER_DAY = 3
MAX_TOKENS = 100
CONTRACT = "c1.0"
REQUEST = {"refund": "refund", "replace": "replacement", "address": "address change"}
ANSWER_TOOL = [{"type": "function", "function": {
    "name": "answer", "description": "Give the code of the ticket on hold that may be resumed now.",
    "parameters": {"type": "object", "properties": {"ticket": {"type": "string"}}, "required": ["ticket"]}}}]
CODE_RE = re.compile(r"\b[B-DF-HJ-NP-TV-XZ]{2}\b")
ANSWER_RE = re.compile(r"Answer:\s*\**\s*([B-DF-HJ-NP-TV-XZ]{2})\b")
REASONING_MAX_TOKENS = 2000
REASONING_LINE = ("You are asked about a situation at the desk. Before you answer, work it out step by step in writing: "
                  "go through the events in order and keep track of what the procedure needs. Then call the answer "
                  "tool with the ticket.")


def _points(sched, rule):
    """(events so far, tickets on hold, legal ticket) at the diagnostic points of the compliant trajectory."""
    seen = {}
    base = make_chooser("compliant", rule)

    def chooser(env, ctx):
        seen[env.st.t] = len(env.events)
        return base(env, ctx)
    env, recs = simulate(sched, rule, chooser)
    return [(env.events[:seen[r["t"]]], list(r["held"]), r["legal_set"][0]) for r in recs if r["diagnostic"]]


def items(rule):
    """N_ITEMS situations for one rule, deterministic."""
    per_d = N_ITEMS // len(planmod.DEPTHS)
    out = []
    for D in planmod.DEPTHS:
        got, sides = [], {"first": 0, "last": 0}
        seed = 0
        while len(got) < per_d:
            key = planmod.sched_key(SPLIT, D, seed)
            sched = planmod.schedule(key, D, bool(seed % 2))
            pts = _points(sched, rule)
            rng = random.Random(f"{key}|{rule}")
            for i in sorted(rng.sample(range(len(pts)), min(PER_DAY, len(pts)))):
                events, held, legal = pts[i]
                side = "first" if legal == min(held) else "last" if legal == max(held) else "middle"
                if rule in ("PAR", "BAL"):
                    if sides[side] >= per_d // 2:
                        continue
                    sides[side] += 1
                got.append({"rule": rule, "D": D, "schedule": key, "events": [list(e) for e in events],
                            "held": sorted(held), "answer": legal, "side": side})
                if len(got) == per_d:
                    break
            seed += 1
        for k, it in enumerate(got):
            it["paraphrased"] = k < N_PARA
        out += got
    for i, it in enumerate(out):
        it["item"] = i
    return out


def render(item):
    lines = []
    for e in item["events"]:
        if e[0] == "a":
            lines.append(f"Ticket {e[1]} arrived ({REQUEST[e[2]]}).")
        elif e[0] == "h":
            lines.append(f"Ticket {e[1]} was put on hold.")
        elif e[0] == "r":
            lines.append(f"Ticket {e[1]} was resumed.")
        elif e[0] == "c":
            lines.append(f"Ticket {e[1]} was finished.")
    log = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(lines))
    return ("Everything that has happened at the desk so far today, in order:\n" + log + "\n\n"
            f"Tickets on hold now (alphabetical): {', '.join(item['held'])}. No ticket is in progress.\n\n"
            "Question: under the procedure, which ticket on hold may be resumed now? Answer with the answer tool.")


def system_text(rule, wording, reasoning=False):
    return (prompts.system_prompt(rule, "C0", wording=wording).replace("\n\nKeep any reasoning brief.", "") +
            "\n\n" + (REASONING_LINE if reasoning else
                       "You are asked about a situation at the desk. Keep any reasoning brief."))


def parse_log(text):
    """Events back from the rendered text (acceptance item 14: the text alone determines the answer)."""
    cat = {v: k for k, v in REQUEST.items()}
    ev = []
    for m in re.finditer(r"Ticket (\w\w) (arrived \((.+?)\)|was put on hold|was resumed|was finished)\.", text):
        code, what = m.group(1), m.group(2)
        if what.startswith("arrived"):
            ev.append(("a", code, cat[m.group(3)]))
        else:
            ev.append(({"was put on hold": "h", "was resumed": "r", "was finished": "c"}[what], code))
    held = re.search(r"Tickets on hold now \(alphabetical\): ([A-Z, ]+)\.", text).group(1).split(", ")
    return ev, held


def answer_from_text(rule, text):
    """The tickets on hold that the definitional judge accepts as the next resume, from the text alone."""
    ev, held = parse_log(text)
    return [c for c in held if judge_events(rule, ev + [("r", c)])[-1]]


# ------------------------------------------------------------------ cells and execution
def probe_plan(models=planmod.MODELS):
    cells = []
    for m in models:
        for rule in RULES:
            for it in items(rule):
                for w in (0, 1, 2):
                    if w == 0 or it["paraphrased"]:
                        cells.append({"exp": "DC", "task": "desk_comprehension", "model": m, "rule": rule,
                                      "wording": w, "item": it["item"], "D": it["D"]})
    return cells


def cell_key(c):
    return f"exp={c['exp']}|model={c['model']}|rule={c['rule']}|wording={c['wording']}|item={c['item']}"


def reasoning_plan(models=(planmod.WEAK, planmod.STRONG)):
    """The written-reasoning control: the 24 paraphrased items per rule, wording 0 (paired with the probe)."""
    return [{"exp": "DCR", "task": "desk_comprehension", "model": m, "rule": rule, "wording": 0, "item": it["item"],
             "D": it["D"], "mode": "reasoning"} for m in models for rule in RULES for it in items(rule)
            if it["paraphrased"]]


_ITEMS = {}


def get_item(rule, i):
    if rule not in _ITEMS:
        _ITEMS[rule] = items(rule)
    return _ITEMS[rule][i]


def execute(cell, backend, spec, ledger):
    it = get_item(cell["rule"], cell["item"])
    rec = {"cell": cell, "contract_version": CONTRACT, "task": "desk_comprehension", "model": spec.model_id,
           "rule": cell["rule"], "wording": cell["wording"], "item": cell["item"], "D": it["D"], "side": it["side"],
           "answer": it["answer"], "calls": [], "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "flops": 0.0,
           "service_errors": 0, "termination": None}
    reasoning = cell.get("mode") == "reasoning"
    messages = [{"role": "system", "content": system_text(cell["rule"], cell["wording"], reasoning)},
                {"role": "user", "content": render(it)}]
    meta = {"tools": ANSWER_TOOL, "item": it}
    try:
        rep = MeteredCaller(backend, spec, ledger, rec)(messages, REASONING_MAX_TOKENS if reasoning else MAX_TOKENS,
                                                        meta)
    except ServiceFailure as e:
        rec.update({"termination": "service", "error_message": str(e)})
        return rec
    given = None
    for _, name, args in rep.tool_calls:
        if name == "answer" and isinstance(args, dict):
            given = str(args.get("ticket") or "").strip().upper()
            break
    if given is None:                     # text fallback: "Answer: XX" if written, else the last ticket code written
        codes = ANSWER_RE.findall(rep.content or "") or CODE_RE.findall(rep.content or "")
        given = codes[-1] if codes else None
        rec["text_fallback"] = True
    rec.update({"given": given, "correct": given == it["answer"], "reasoning_tokens": rep.reasoning_tokens,
                "provider": rep.provider, "content": (rep.content or "")[:4000 if reasoning else 500],
                "written_chars": len(rep.content or "")})
    if reasoning:
        rec["mode"] = "reasoning"
    return rec


class FakeAnswerBackend:
    """Offline: 'oracle' answers correctly, 'random' picks a ticket on hold, 'first' the alphabetically first."""

    def __init__(self, spec, policy="oracle", seed=0):
        self.spec, self.policy, self.rng = spec, policy, random.Random(seed)

    def call(self, messages, max_tokens, meta=None):
        it = meta["item"]
        pick = {"oracle": lambda: it["answer"], "random": lambda: self.rng.choice(it["held"]),
                "first": lambda: it["held"][0]}[self.policy]()
        p_tok = int(sum(len(m["content"]) for m in messages) / 3.8)
        cost = (p_tok * self.spec.input_price + 10 * self.spec.output_price) / 1e6
        am = {"role": "assistant", "content": None, "tool_calls": [tool_call_json("answer", {"ticket": pick}, "a1")]}
        return ToolReply("", [("a1", "answer", {"ticket": pick})], p_tok, 10, 0, cost, self.spec.provider, 0.0, am)


class FakeReasoningBackend(FakeAnswerBackend):
    """Offline written-reasoning control: writes some working, then answers in text only ('Answer: XX')."""

    def call(self, messages, max_tokens, meta=None):
        rep = super().call(messages, max_tokens, meta)
        pick = rep.tool_calls[0][2]["ticket"]
        text = f"Going through the events: {meta['item']['held'][0]} is on hold, ... Answer: {pick}"
        return ToolReply(text, [], rep.prompt_tokens, 60, 0, rep.cost_usd, rep.provider, 0.0,
                         {"role": "assistant", "content": text})


def make_backend(kind, spec, seed=0):
    if kind.startswith("fake:"):
        return FakeAnswerBackend(spec, kind.split(":", 1)[1], seed)
    from .run import make_backend as desk_backend
    return desk_backend(kind, spec, seed)


def make_reasoning_backend(kind, spec, seed=0):
    """Same backends with tool_choice "auto", so that the model can write before it calls the answer tool."""
    if kind.startswith("fake:"):
        return FakeReasoningBackend(spec, kind.split(":", 1)[1], seed)
    backend = make_backend(kind, spec, seed)
    backend.tool_choice = "auto"
    return backend


def run_probe(cells, run_name, budget, workers=4, backend_kind="fake:oracle", runs_dir=executor.RUNS, quiet=False,
              reasoning=False):
    return executor.run_plan(cells, run_name, budget, execute, cell_key,
                             make_reasoning_backend if reasoning else make_backend, backend_kind, workers=workers,
                             runs_dir=runs_dir, config={"contract_version": CONTRACT, "max_tokens": MAX_TOKENS,
                                                        "mode": "written reasoning" if reasoning else "forced tool call",
                                                        "reasoning_max_tokens": REASONING_MAX_TOKENS},
                             summary_keys=("termination", "cost_usd", "correct"), quiet=quiet)


def reasoning_summary(reasoning_dir, probe_dir):
    """Per model x rule: accuracy with written reasoning and, on the same items, in the probe (wording 0); paired
    counts and the exact McNemar test (two-sided binomial on the discordant items); mean written characters."""
    from scipy.stats import binomtest

    def load(d):
        out = {}
        for root, _, files in os.walk(os.path.join(d, "cells")):
            for f in files:
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    r = json.load(fh)
                if r.get("task") == "desk_comprehension" and r.get("termination") is None and r["wording"] == 0:
                    out[(r["model"], r["rule"], r["item"])] = r
        return out
    rz, base = load(reasoning_dir), load(probe_dir)
    rows = {}
    for (m, rule, i), r in rz.items():
        b = base.get((m, rule, i))
        if b is None:
            continue
        row = rows.setdefault((m, rule), {"model": m, "rule": rule, "items": 0, "correct_reasoning": 0,
                                          "correct_forced": 0, "forced_wrong_reasoning_right": 0,
                                          "forced_right_reasoning_wrong": 0, "written_chars": 0, "text_fallbacks": 0})
        row["items"] += 1
        row["correct_reasoning"] += bool(r["correct"])
        row["correct_forced"] += bool(b["correct"])
        row["forced_wrong_reasoning_right"] += (not b["correct"]) and bool(r["correct"])
        row["forced_right_reasoning_wrong"] += bool(b["correct"]) and not r["correct"]
        row["written_chars"] += r.get("written_chars", 0)
        row["text_fallbacks"] += bool(r.get("text_fallback"))
    out = []
    for row in rows.values():
        k, n = row["forced_wrong_reasoning_right"], row["forced_wrong_reasoning_right"] + row["forced_right_reasoning_wrong"]
        row["mcnemar_p"] = round(binomtest(k, n, 0.5).pvalue, 4) if n else 1.0
        row["written_chars"] = round(row["written_chars"] / row["items"])
        out.append(row)
    return sorted(out, key=lambda x: (x["model"], RULES.index(x["rule"])))


# ------------------------------------------------------------------ summary
def summary(run_dirs):
    """Per model x rule x wording: items, errors, exact two-sided 95% upper bound of the error rate, and whether it is
    <= the sufficiency threshold (understood); PAR / BAL also by answer side."""
    from scipy.stats import beta
    recs = []
    for d in run_dirs:
        for root, _, files in os.walk(os.path.join(d, "cells")):
            for f in files:
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    r = json.load(fh)
                if r.get("task") == "desk_comprehension" and r.get("termination") is None:
                    recs.append(r)
    groups = {}
    for r in recs:
        groups.setdefault((r["model"], r["rule"], r["wording"]), []).append(r)
    out = []
    for (m, rule, w), rs in sorted(groups.items()):
        n, err = len(rs), sum(not r["correct"] for r in rs)
        ub = 1.0 if err == n else float(beta.ppf(prereg.LEVEL, err + 1, n - err))
        row = {"model": m, "rule": rule, "wording": w, "items": n, "errors": err, "error_rate": round(err / n, 4),
               "upper_bound": round(ub, 4), "understood": ub <= prereg.THRESHOLD,
               "text_fallbacks": sum(bool(r.get("text_fallback")) for r in rs)}
        if rule in ("PAR", "BAL"):
            row["errors_by_side"] = {s: sum(not r["correct"] for r in rs if r["side"] == s) for s in ("first", "last")}
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", default="fake:oracle")
    ap.add_argument("--models", default=None)
    ap.add_argument("--summary", nargs="*")
    ap.add_argument("--reasoning", action="store_true", help="the written-reasoning control (exploratory)")
    ap.add_argument("--reasoning-summary", nargs=2, metavar=("REASONING_RUN", "PROBE_RUN"))
    a = ap.parse_args(argv)
    if a.summary:
        for row in summary(a.summary):
            print(json.dumps(row, ensure_ascii=False))
        return 0
    if a.reasoning_summary:
        for row in reasoning_summary(*a.reasoning_summary):
            print(json.dumps(row, ensure_ascii=False))
        return 0
    if a.reasoning:
        cells = reasoning_plan(tuple(a.models.split(",")) if a.models else (planmod.WEAK, planmod.STRONG))
    else:
        cells = probe_plan(tuple(a.models.split(",")) if a.models else planmod.MODELS)
    print(f"{len(cells)} items, backend {a.backend}, budget ${a.budget}")
    stats = run_probe(cells, a.run, a.budget, a.workers, a.backend, reasoning=a.reasoning)
    print(json.dumps(stats, indent=2))
    code = 0 if not (stats["budget_stop"] or stats.get("breaker_stop")) else 2
    from .run import _backup_if_paid
    return _backup_if_paid(a.backend, a.run, "comprehension", executor.RUNS) or code


if __name__ == "__main__":
    raise SystemExit(main())
