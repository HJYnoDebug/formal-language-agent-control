"""Grammars of the four procedure languages and of the input language, verified against the platform (00 §3.4 B, 02b §4;
v1.8). Offline, no cost.  Run:  python analysis/verify_grammars.py [--n 7]

Upper bounds (L_P lies in the class): an explicit grammar of the class generates L_P. Its Chomsky type is checked from
the form of its productions, and on every word of the domain up to length n the grammar's verdict equals the definition
and both platform judges (rules.judge_events on the event history, rules.legal_resume on the state).
Lower bounds (L_P lies in no smaller class): L_P intersected with the regular language of burst shapes (the shapes the
generator produces, checked on its days below) is a language that is known to lie outside the smaller class, and the
classes are closed under intersection with regular languages (00 §3.4 B gives the proofs):
  PAR   two Nerode classes (one bit; no window-only rule decides it)
  BAL   prefixes A A R^i pairwise distinguished by P^j F (legal iff i >= j): unboundedly many classes, not regular
  LIFO  L ∩ h*r* = {h_w r_reverse(w)}: fooling set {h_w : |w| = n} of size k^n, not regular
  FIFO  L ∩ h*r* = {h_w r_w} (the copy language), not context-free (pumping lemma)
Here these intersections are checked exhaustively up to the length bound; the non-membership in the smaller class is a
proof, not a check.

Alphabets and domains
  PAR / BAL (side alphabet): R, P, A an arriving refund / replace / address ticket; F, L resuming the alphabetically
  first / last ticket on hold. Domain (environment): F, L only while at least two tickets are on hold (with one on
  hold every resume is legal and carries no side). In the platform mapping every arriving ticket is put on hold.
  LIFO / FIFO (identity alphabet, one letter per ticket): h_x puts x on hold, r_x resumes x. Domain: r_x only while x
  is on hold, h_x only while it is not. With one ticket per label the label layer equals this layer (identity
  transfer, 00 §3.4 B); acceptance item 3b checks that the label layer is implied by the identity layer.

Grammars (start symbol first; x, y range over the tickets)
  PAR   type 3   E -> R O | P E | A E | F E | ε          O -> R E | P O | A O | L O | ε
  BAL   type 2   S -> Pos | Bp P T                        T -> Neg | Bn R S
                 Bp -> ε | A Bp | F Bp | R Bp P Bp       Bn -> ε | A Bn | L Bn | P Bn R Bn
                 Pos -> ε | A Pos | F Pos | R Bp P Pos | R Pos
                 Neg -> ε | A Neg | L Neg | P Bn R Neg | P Neg
                 (Bp / Bn: back to the starting level without going below / above it; Pos / Neg: never below 0 /
                 never above -1; S at counter 0, T at counter -1; F needs counter >= 0, L counter < 0)
  LIFO  type 2   S -> ε | h_x D r_x S | h_x S             D -> ε | h_x D r_x D
                 (h_x S: x stays on hold to the end; between h_x and r_x everything is resumed)
  FIFO  type 1   S0 -> ε | S     S -> h_x M_x S | h_x M_x | U     U -> h_x U | h_x
                 M_x h_y -> h_y M_x      M_x -> r_x
                 (monotone; each resumed hold carries a marker that moves right past later holds and turns into
                 r_x; markers never pass each other, so resumes follow the hold order; holds that stay on hold, U,
                 come last in the hold order)

Generated days (agent/desk/generator.py): every formal and acceptance day's arrival string (one symbol per time unit:
E escalation, N normal ticket, - nothing; the same for every rule) is accepted by the finite automaton of the input
language (input_automaton: number on hold 0..D, ticket in progress, depth reached, counts of E and N; all bounded, so
the language is regular); on the compliant LIFO / FIFO trajectory of every day the depth D is reached by D consecutive
holds (the prefix h_w, |w| = D, of the fooling set) and holds follow resumes inside a hold segment exactly N_NEST times
(v1.8: the stack / queue is updated between resumes, not only filled and emptied); and on compliant and random-resume
trajectories of the formal days the grammar's first rejected prefix is the platform's first violation (FIFO: by the
queue machine, the grammar being equal to it up to n).
"""
import argparse
import itertools
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.desk import generator, plan, prereg                                    # noqa: E402
from agent.desk.rules import DeskState, Ticket, judge_events, legal_resume       # noqa: E402
from agent.desk.strategies import make_chooser, simulate                         # noqa: E402

EPS = ()
CAT = {"R": "refund", "P": "replace", "A": "address"}
SIDE = ("R", "P", "A", "F", "L")


# ------------------------------------------------------------------ grammars
def g_par():
    return "E", {"E": [("R", "O"), ("P", "E"), ("A", "E"), ("F", "E"), EPS],
                 "O": [("R", "E"), ("P", "O"), ("A", "O"), ("L", "O"), EPS]}


def g_bal():
    return "S", {"S": [("Pos",), ("Bp", "P", "T")], "T": [("Neg",), ("Bn", "R", "S")],
                 "Bp": [EPS, ("A", "Bp"), ("F", "Bp"), ("R", "Bp", "P", "Bp")],
                 "Bn": [EPS, ("A", "Bn"), ("L", "Bn"), ("P", "Bn", "R", "Bn")],
                 "Pos": [EPS, ("A", "Pos"), ("F", "Pos"), ("R", "Bp", "P", "Pos"), ("R", "Pos")],
                 "Neg": [EPS, ("A", "Neg"), ("L", "Neg"), ("P", "Bn", "R", "Neg"), ("P", "Neg")]}


def g_lifo(tickets):
    return "S", {"S": [EPS] + [p for x in tickets for p in ((f"h{x}", "D", f"r{x}", "S"), (f"h{x}", "S"))],
                 "D": [EPS] + [(f"h{x}", "D", f"r{x}", "D") for x in tickets]}


def g_fifo(tickets):
    """Monotone grammar as a list of (left side, right side); nonterminals: S0, S, U, M<x>."""
    rules = [(("S0",), EPS), (("S0",), ("S",)), (("S",), ("U",))]
    for x in tickets:
        rules += [(("S",), (f"h{x}", f"M{x}", "S")), (("S",), (f"h{x}", f"M{x}")),
                  (("U",), (f"h{x}", "U")), (("U",), (f"h{x}",)), ((f"M{x}",), (f"r{x}",))]
        rules += [((f"M{x}", f"h{y}"), (f"h{y}", f"M{x}")) for y in tickets]
    return "S0", rules


def chomsky_type(start, rules, nonterminals):
    """Most restrictive type the productions have: 3 right-linear, 2 context-free, 1 monotone (only S0 -> ε, and S0 on
    no right side), else 0."""
    rules = list(rules)
    if all(len(l) == 1 and l[0] in nonterminals for l, _ in rules):
        if all(r == EPS or (len(r) <= 2 and r[0] not in nonterminals and all(s in nonterminals for s in r[1:]))
               for _, r in rules):
            return 3
        return 2
    eps_ok = all(r != EPS or l == (start,) for l, r in rules) and \
        not any(start in r for l, r in rules if any(r == EPS for l2, r in rules if l2 == (start,)))
    if eps_ok and all(len(r) >= len(l) for l, r in rules if r != EPS) and \
            all(any(s in nonterminals for s in l) for l, _ in rules):
        return 1
    return 0


def cfg_rules(g):
    return [((lhs,), rhs) for lhs, prods in g.items() for rhs in prods]


def earley_prefixes(start, g, word):
    """Earley recogniser for a context-free grammar with ε (Aycock-Horspool); accepted[i] for every prefix word[:i]."""
    nullable, changed = set(), True
    while changed:
        changed = False
        for lhs, prods in g.items():
            if lhs not in nullable and any(all(s in nullable for s in p) for p in prods):
                nullable.add(lhs)
                changed = True
    top = ("<top>", (start,), 0, 0)
    chart = [set() for _ in range(len(word) + 1)]
    chart[0].add(top)
    for i in range(len(word) + 1):
        agenda = list(chart[i])
        while agenda:
            lhs, rhs, dot, org = agenda.pop()
            new = []
            if dot < len(rhs):
                sym = rhs[dot]
                if sym in g:
                    new += [(sym, p, 0, i) for p in g[sym]]
                    if sym in nullable:
                        new.append((lhs, rhs, dot + 1, org))
                elif i < len(word) and sym == word[i]:
                    chart[i + 1].add((lhs, rhs, dot + 1, org))
            else:
                new += [(l2, r2, d2 + 1, o2) for l2, r2, d2, o2 in list(chart[org]) if d2 < len(r2) and r2[d2] == lhs]
            for it in new:
                if it not in chart[i]:
                    chart[i].add(it)
                    agenda.append(it)
    return [("<top>", (start,), 1, 0) in chart[i] for i in range(len(word) + 1)]


def monotone_language(start, rules, n):
    """All terminal words of length <= n (sentential forms of a monotone grammar never shrink, so forms longer than n
    are dropped)."""
    nts = {s for l, _ in rules for s in l if not s.startswith(("h", "r"))} | {start}
    seen, out, todo = {(start,)}, set(), deque([(start,)])
    while todo:
        f = todo.popleft()
        if not any(s in nts for s in f):
            out.add(f)
            continue
        for l, r in rules:
            k = len(l)
            for i in range(len(f) - k + 1):
                if f[i:i + k] == l:
                    g = f[:i] + r + f[i + k:]
                    if len(g) <= n and g not in seen:
                        seen.add(g)
                        todo.append(g)
    return out


# ------------------------------------------------------------------ definitions and platform judges
def side_domain(n):
    """Words over the side alphabet up to length n, F / L only with two or more on hold."""
    out = [()]
    frontier = [((), 0)]
    for _ in range(n):
        nxt = []
        for w, held in frontier:
            for a in SIDE:
                if a in "FL" and held < 2:
                    continue
                v = (w + (a,), held + (1 if a in CAT else -1))
                nxt.append(v)
                out.append(v[0])
        frontier = nxt
    return out


def side_definition(rule, w):
    ref = rep = 0
    for a in w:
        ref += a == "R"
        rep += a == "P"
        if a in "FL":
            first = (ref % 2 == 0) if rule == "PAR" else (ref >= rep)
            if (a == "F") != first:
                return False
    return True


def side_platform(rule, w):
    """Both platform judges on the side word: every arriving ticket is put on hold, F / L resume min / max code."""
    arrivals = [a for a in w if a in CAT]
    codes = [f"{a}{b}" for a, b in itertools.product(generator.LETTERS, repeat=2)][:len(arrivals)]
    tickets = {c: Ticket(c, "normal", CAT[a], 0, False, 0, "", "") for c, a in zip(codes, arrivals)}
    st, events, state_ok, k = DeskState(tickets=tickets), [], True, 0
    for a in w:
        if a in CAT:
            c = codes[k]
            k += 1
            st.arrive(c)
            st.put_on_hold(c)
            events += [("a", c, CAT[a]), ("h", c)]
        else:
            c = min(st.held) if a == "F" else max(st.held)
            state_ok &= c in legal_resume(rule, st)
            st.take_off_hold(c)
            events.append(("r", c))
    return state_ok, all(judge_events(rule, events))


def identity_domain(tickets, n):
    out, frontier = [()], [((), frozenset())]
    for _ in range(n):
        nxt = []
        for w, held in frontier:
            for x in tickets:
                v = (w + (f"r{x}",), held - {x}) if x in held else (w + (f"h{x}",), held | {x})
                nxt.append(v)
                out.append(v[0])
        frontier = nxt
    return out


def identity_definition(rule, w):
    order = []
    for s in w:
        x = s[1:]
        if s[0] == "h":
            order.append(x)
        else:
            if x != (order[-1] if rule == "LIFO" else order[0]):
                return False
            order.remove(x)
    return True


def identity_platform(rule, w):
    xs = sorted({s[1:] for s in w})
    st = DeskState(tickets={x: Ticket(x, "normal", "address", 0, False, 0, "", "") for x in xs})
    events, state_ok, arrived = [], True, set()
    for s in w:
        x = s[1:]
        if s[0] == "h":
            if x not in arrived:
                arrived.add(x)
                st.arrive(x)
                events.append(("a", x, "address"))
            st.put_on_hold(x)
            events.append(("h", x))
        else:
            state_ok &= x in legal_resume(rule, st)
            st.take_off_hold(x)
            events.append(("r", x))
    return state_ok, all(judge_events(rule, events))


# ------------------------------------------------------------------ checks
def check_small(n_side, n_id, tickets=("X", "Y", "Z")):
    """Grammar = definition = both platform judges on every domain word up to the length bound."""
    out = {}
    for rule, (start, g) in (("PAR", g_par()), ("BAL", g_bal())):
        bad = n = acc = 0
        for w in side_domain(n_side):
            gram = earley_prefixes(start, g, list(w))[-1]
            s_ok, e_ok = side_platform(rule, w)
            d = side_definition(rule, w)
            bad += not (gram == d == s_ok == e_ok)
            n += 1
            acc += gram
        typ = chomsky_type(start, cfg_rules(g), set(g))
        out[rule] = {"type": typ, "words": n, "in_language": acc, "disagreements": bad}
    start, g = g_lifo(tickets)
    dom = identity_domain(tickets, n_id)
    bad = acc = 0
    for w in dom:
        gram = earley_prefixes(start, g, list(w))[-1]
        s_ok, e_ok = identity_platform("LIFO", w)
        bad += not (gram == identity_definition("LIFO", w) == s_ok == e_ok)
        acc += gram
    out["LIFO"] = {"type": chomsky_type(start, cfg_rules(g), set(g)), "words": len(dom), "in_language": acc,
                   "disagreements": bad}
    start, rules = g_fifo(tickets)
    lang = monotone_language(start, rules, n_id)
    bad = acc = 0
    for w in dom:
        gram = w in lang
        s_ok, e_ok = identity_platform("FIFO", w)
        bad += not (gram == identity_definition("FIFO", w) == s_ok == e_ok)
        acc += gram
    nts = {s for l, _ in rules for s in l if not s.startswith(("h", "r"))} | {"S0"}
    out["FIFO"] = {"type": chomsky_type(start, rules, nts), "words": len(dom), "in_language": acc,
                   "disagreements": bad, "generated_outside_domain": len(lang - set(dom))}
    expected = {"PAR": 3, "BAL": 2, "LIFO": 2, "FIFO": 1}
    ok = all(v["disagreements"] == 0 and v["type"] == expected[r] for r, v in out.items())
    return {"pass": ok, "by_rule": out}


def check_lower_bounds(k_max=6, tickets=("X", "Y", "Z")):
    """Finite witnesses of the lower bounds (the proofs are in 00 §3.4 B)."""
    out = {}
    start, g = g_par()
    acc = lambda w: earley_prefixes(start, g, list(w))[-1]                       # noqa: E731
    out["PAR_two_classes"] = acc("AAAF") and not acc("RAAF")
    start, g = g_bal()
    rows = [tuple(earley_prefixes(start, g, list("AA" + "R" * i + "P" * j + "F"))[-1] for j in range(k_max + 1))
            for i in range(k_max + 1)]
    out["BAL_distinct_prefixes"] = len(set(rows))
    out["BAL_rows_ok"] = all(rows[i][j] == (i >= j) for i in range(k_max + 1) for j in range(k_max + 1))
    for rule in ("LIFO", "FIFO"):
        good = True
        for n in range(1, len(tickets) + 1):
            for w in itertools.permutations(tickets, n):
                for v in itertools.permutations(w):
                    word = tuple(f"h{x}" for x in w) + tuple(f"r{x}" for x in v)
                    want = v == (tuple(reversed(w)) if rule == "LIFO" else w)
                    s_ok, e_ok = identity_platform(rule, word)
                    good &= s_ok == e_ok == want
        out[f"{rule}_burst_shapes"] = good           # L ∩ h*r* = {h_w r_rev(w)} / {h_w r_w} on distinct tickets
    start, g = g_lifo(("A", "B"))                    # labels: fooling set {h_w : w in {A, B}^n}
    words = ["".join(p) for p in itertools.product("AB", repeat=4)]
    m = [[earley_prefixes(start, g, [f"h{c}" for c in u] + [f"r{c}" for c in reversed(v)])[-1] for v in words]
         for u in words]
    out["LIFO_fooling_set_16"] = all(m[i][j] == (i == j) for i in range(16) for j in range(16))
    ok = out["PAR_two_classes"] and out["BAL_rows_ok"] and out["BAL_distinct_prefixes"] == k_max + 1 and \
        out["LIFO_burst_shapes"] and out["FIFO_burst_shapes"] and out["LIFO_fooling_set_16"]
    return {"pass": ok, **out}


def arrival_string(s):
    ts = {t["arrival"]: ("E" if t["kind"] == "escalated" else "N") for t in s["tickets"]}
    assert len(ts) == len(s["tickets"]), "two arrivals at one time"
    return "".join(ts.get(t, "-") for t in range(s["reference_end"]))


def input_automaton(D, word, n_esc=None, n_norm=None):
    """The input language (generator docstring) as a finite automaton following the reference behaviour. State: h on
    hold (0..D), the ticket in progress (none / escalation / normal), depth D reached, E and N counts (bounded)."""
    n_esc = generator.N_ESC if n_esc is None else n_esc
    n_norm = generator.N_NORM if n_norm is None else n_norm
    h, cur, deep, ne, nn = 0, None, False, 0, 0
    for a in word:
        if a == "E":
            if cur == "N":                              # a normal ticket would be put on hold
                return False
            if cur == "E":
                h += 1
            cur, ne = "E", ne + 1
        elif a == "N":
            if cur is not None or h > 0:                # normal tickets arrive only while the desk is idle
                return False
            cur, nn = "N", nn + 1
        elif cur is not None:                           # "-": close the ticket in progress ...
            cur = None
        elif h > 0:                                     # ... or resume one on hold (it has one step left)
            h -= 1
        if h > D or ne > n_esc or nn > n_norm:
            return False
        deep |= h == D
    return h == 0 and cur is None and deep and ne == n_esc and nn == n_norm


def hold_profile(events):
    """Of the hold / resume projection: the maximal depth, the longest run of consecutive holds, and how many runs of
    holds start after a resume while tickets are still on hold (nested bursts: a push after a pop in one segment)."""
    depth = top = run = best = again = 0
    popped = False
    for e in events:
        if e[0] == "h":
            again += popped and depth > 0
            popped = False
            depth += 1
            top = max(top, depth)
            run += 1
            best = max(best, run)
        elif e[0] == "r":
            depth -= 1
            run = 0
            popped = depth > 0
    return {"max_depth": top, "longest_hold_run": best, "push_after_pop": again}


def side_word(events):
    """Side-alphabet word of a platform trajectory: arrivals R / P / A, resumes with two or more on hold F / L
    (another position: 'X', in no word of the language); resumes with one on hold carry no side and are dropped."""
    order, w, at = [], [], []
    for e in events:
        if e[0] == "a":
            w.append({"refund": "R", "replace": "P", "address": "A"}[e[2]])
        elif e[0] == "h":
            if e[1] in order:
                order.remove(e[1])
            order.append(e[1])
        elif e[0] == "r":
            if len(order) >= 2:
                w.append("F" if e[1] == min(order) else "L" if e[1] == max(order) else "X")
                at.append(len(w))
            else:
                at.append(None)
            order.remove(e[1])
    return w, at


def check_days(n_accept=8, n_random=3):
    out = {"input_language": [], "hold_structure": [], "trajectories": []}
    days = [(D, s, sc, "formal") for D, s, sc in prereg.formal_days()]
    days += [(D, s, generator.make_schedule(plan.sched_key("accept", D, s), D, bool(s % 2)), "accept")
             for D in plan.DEPTHS for s in range(n_accept)]
    for D, seed, sc, split in days:
        a = arrival_string(sc)
        if not input_automaton(D, a):
            out["input_language"].append((split, D, seed, a))
        for rule in ("LIFO", "FIFO"):
            env, _ = simulate(sc, rule, make_chooser("compliant", rule))
            prof = hold_profile(env.events)
            if not (prof["max_depth"] == D and prof["longest_hold_run"] == D and
                    prof["push_after_pop"] == generator.N_NEST):
                out["hold_structure"].append((split, D, seed, rule, prof))
        if split != "formal":
            continue
        for rule in ("PAR", "BAL", "LIFO", "FIFO"):
            choosers = [make_chooser("compliant", rule)] + [make_chooser("win_random", rule, i) for i in range(n_random)]
            for ch in choosers:
                env, recs = simulate(sc, rule, ch)
                verdicts = [r["legal"] for r in recs if r["resume"] is not None]
                first_platform = next((i for i, v in enumerate(verdicts) if v is False), None)
                if rule in ("PAR", "BAL"):
                    start, g = g_par() if rule == "PAR" else g_bal()
                    w, at = side_word(env.events)
                    acc = earley_prefixes(start, g, w)
                    first_gram = next((i for i, pos in enumerate(at) if pos is not None and not acc[pos]), None)
                elif rule == "LIFO":
                    codes = sorted(sc_t["code"] for sc_t in sc["tickets"])
                    start, g = g_lifo(codes)
                    word = [f"{e[0]}{e[1]}" for e in env.events if e[0] in "hr"]
                    acc = earley_prefixes(start, g, word)
                    pos = [i + 1 for i, s in enumerate(word) if s[0] == "r"]
                    first_gram = next((i for i, p in enumerate(pos) if not acc[p]), None)
                else:
                    word = tuple(f"{e[0]}{e[1]}" for e in env.events if e[0] in "hr")
                    pos = [i + 1 for i, s in enumerate(word) if s[0] == "r"]
                    first_gram = next((i for i, p in enumerate(pos) if not identity_definition("FIFO", word[:p])), None)
                if first_gram != first_platform:
                    out["trajectories"].append((D, seed, rule, first_gram, first_platform))
    out["days"] = len(days)
    out["pass"] = not (out["input_language"] or out["hold_structure"] or out["trajectories"])
    return out


def run(n_side=7, n_id=6, n_accept=8):
    res = {"small_words": check_small(n_side, n_id), "lower_bounds": check_lower_bounds(),
           "generated_days": check_days(n_accept)}
    res["pass"] = all(v["pass"] for v in res.values())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-side", type=int, default=7)
    ap.add_argument("--n-id", type=int, default=6)
    a = ap.parse_args()
    r = run(a.n_side, a.n_id)
    for k, v in r.items():
        print(k, v)
    sys.exit(0 if r["pass"] else 1)
