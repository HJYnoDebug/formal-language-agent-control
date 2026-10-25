"""Schedules (02b §8, v1.8): 16 tickets per day, 14 escalated and 2 normal, every ticket two steps (verify -> action),
so that context length and cost do not depend on the depth; the nominal depth D is set only by WHEN escalations arrive.

How arrivals make the hold structure. An escalation must be started in the turn it arrives; if another ticket is in
progress, that one is put on hold. So a burst of m escalations arriving one per time unit while the desk is idle puts
m - 1 of them on hold, and the tickets on hold are then resumed one per turn. Every ticket on hold is an escalation with
one step left (no kind or remaining-work cue, 02b §14 A13); normal tickets arrive in the gaps and the reference
behaviour finishes them before the next burst, so it never puts a normal ticket on hold.
  - bursts: one of D + 1 escalations (the hold depth D), the rest of 2..D + 1 (a last single escalation if 1 is left);
  - N_NEST nested bursts of NEST_SIZE escalations (v1.8, 02b §14 A14): each arrives at a resume turn of a burst after
    at least one resume, with tickets still on hold, so tickets are put on hold again after others have been resumed.
    Without them every burst is h^m r^m and LIFO / FIFO only ask to read one list backwards / forwards; with them the
    stack / queue must be updated between resumes. N_NEST = 1 and NEST_SIZE = 2 are the same for every D: a D = 10 day
    has 14 - 11 = 3 escalations besides its deep burst, room for exactly one nested burst of 2, and at D = 2 at most
    one ticket is on hold after a resume, so a nested burst of more than 2 would exceed the depth;
  - N_INS single escalations inserted before a violable resume (started at once, closed in the next turn, never on
    hold). v1.7 needed 1 so that PAR / BAL could change side while a burst unwinds; the nested burst's arrivals do this
    now, and N_INS = 0 is the smallest number for which every formal and acceptance day can be balanced (with 0 the
    violable points per cell are 86 >= 72, prereg.zero_event_n).

Input language (02b §4, 00 §3.4 B; analysis/verify_grammars.py). One symbol per time unit (E an escalation arrives,
N a normal ticket arrives, - nothing arrives). A day is a word accepted by a finite automaton that follows the
reference behaviour: it keeps the number on hold h (0..D), whether a ticket is in progress and of which kind, whether
depth D has been reached and how many E / N have arrived; an E with an escalation in progress increases h, an E with a
normal ticket in progress or an N with tickets on hold or in progress is rejected (no normal ticket is ever put on
hold), - closes the ticket in progress or else resumes one (h - 1). Accepted: h never above D, D reached, 14 E and 2 N,
nothing left at the end. All bounded, so the language is regular; it is the same for every rule (the rules differ only
in which resumes are legal, never in the input).

Balance of the stimulus set (02b §14 A13; all models run on the same days, so chance correlations in these days
would be shared by every model):
  - categories are redrawn until PAR and BAL designate the alphabetically first and the last ticket about equally
    often at the day's diagnostic points (|first - last| <= 1);
  - codes are redrawn until, for LIFO and FIFO, the designated ticket is alphabetically first, and last, about as
    often as chance at the day's diagnostic points (|observed - expected| <= 1).
Every ticket on hold has one step left, so the resume turns and the tickets on hold at each of them depend only on the
arrival times: they are simulated once per skeleton (_structure) and the redraws are checked by counting.
High-excursion days (odd seeds) also need a counter bound of at least 4. Rush tickets (4 of 16):
due = arrival + 2 + slack, slack uniform 8..30, independent of the hold order.
"""
import hashlib
import random

from .rules import CATEGORIES

LETTERS = "BCDFGHJKLMNPQRSTVWXZ"
N_ESC, N_NORM, N_RUSH, N_INS, N_NEST, NEST_SIZE = 14, 2, 4, 0, 1, 2
STEPS = 2                   # every ticket: verify -> action
RUSH_SLACK = (8, 30)        # due = arrival + steps + slack (calibrated offline: acceptance items 2, 3)
T_SLACK = 10                # the day ends this many time units after the reference behaviour has closed every ticket
MAX_ATTEMPTS = 400
MAX_REDRAWS = 600


class GenerationError(Exception):
    pass


def rng_for(key, stream):
    return random.Random(int.from_bytes(hashlib.sha256(f"{key}|{stream}".encode()).digest()[:8], "big"))


def _excursion_max(categories):
    c = best = 0
    for cat in categories:
        c += (cat == "refund") - (cat == "replace")
        best = max(best, abs(c))
    return best


def _burst_sizes(D, rng, n_esc):
    """One burst of D + 1 escalations; the rest in bursts of 2..D + 1 (a last single escalation if 1 is left)."""
    sizes, rest = [D + 1], n_esc - (D + 1)
    while rest > 0:
        m = rng.randint(min(2, rest), min(D + 1, rest))
        sizes.append(m)
        rest -= m
    rng.shuffle(sizes)
    return sizes


def _arrivals(sizes, n_norm, n_ins, n_nest, rng, nest_size=2):
    """(time, kind) list. Bursts start while the desk is idle; normal tickets arrive in the gaps (done before the next
    burst). Two kinds of escalations arrive while a burst unwinds, at a resume turn (no current ticket):
      - n_nest nested bursts of nest_size escalations, each after at least one resume of its burst with tickets still
        on hold: the first is started at once, each later one puts the previous one on hold, so the stack grows again
        after it has shrunk (holds interleave with resumes; without them every burst is h^m r^m);
      - n_ins singles, each before a violable resume (two or more on hold): started at once, closed in the next turn,
        never on hold; they only add to the counts PAR / BAL depend on.
    Returns None if the bursts leave no room for them."""
    nest_slots = [(g, j) for g, m in enumerate(sizes) for j in range(2, m)]      # before resume j: m - j >= 1 on hold
    if len(nest_slots) < n_nest:
        return None
    nests = set(rng.sample(nest_slots, n_nest))
    plan = []                                           # per burst: "nest" or the number on hold before a resume
    for g, m in enumerate(sizes):
        seq, h, j = [], m - 1, 0
        while h > 0:
            j += 1
            if (g, j) in nests:
                seq.append("nest")
                h += nest_size - 1
            seq.append(h)
            h -= 1
        plan.append(seq)
    single_slots = [(g, i) for g, seq in enumerate(plan) for i, h in enumerate(seq) if h != "nest" and h >= 2]
    if len(single_slots) < n_ins:
        return None
    singles = set(rng.sample(single_slots, n_ins))
    per_gap = [0] * (len(sizes) + 1)
    for _ in range(n_norm):
        per_gap[rng.randrange(len(per_gap))] += 1
    arrivals, t = [], 0
    for g in range(len(sizes) + 1):
        for _ in range(per_gap[g]):
            arrivals.append((t, "normal"))
            t += STEPS
        t += rng.randint(1, 2)
        if g < len(sizes):
            m = sizes[g]
            arrivals += [(t + i, "escalated") for i in range(m)]
            t += m + 1                                  # m verify steps, then the last escalation is closed
            for i, ev in enumerate(plan[g]):
                if ev == "nest":                        # nested burst: verify each, then close the last
                    arrivals += [(t + k, "escalated") for k in range(nest_size)]
                    t += nest_size + 1
                    continue
                if (g, i) in singles:                   # inserted single: verify, close
                    arrivals.append((t, "escalated"))
                    t += STEPS
                t += 1                                  # resume turn
    return arrivals


def _structure(sched):
    """Decision structure of a day, independent of codes and categories: every ticket on hold has one step left, so
    the resume opportunities (times, tickets on hold) depend only on the arrival times. Simulated once per skeleton:
    LIFO / FIFO reference runs give the tickets on hold (as indices) and the designated one at every violable point."""
    from .strategies import make_chooser, simulate          # local import: strategies imports env
    idx = {t["code"]: i for i, t in enumerate(sched["tickets"])}
    out = {}
    for rule in ("LIFO", "FIFO"):
        recs = simulate(sched, rule, make_chooser("compliant", rule))[1]
        pts = [r for r in recs if r["violable"]]
        out[rule] = [([idx[c] for c in r["held"]], idx[r["legal_set"][0]]) for r in pts]
        out["times"] = [r["t"] for r in pts]
    return out


def _side_balanced(arrivals, cats, times):
    """PAR and BAL designate the alphabetically first and the last ticket about equally often (|first - last| <= 1)."""
    for rule in ("PAR", "BAL"):
        first = 0
        for t in times:
            ref = sum(a <= t and c == "refund" for (a, _), c in zip(arrivals, cats))
            rep = sum(a <= t and c == "replace" for (a, _), c in zip(arrivals, cats))
            first += (ref % 2 == 0) if rule == "PAR" else (ref >= rep)
        if abs(2 * first - len(times)) > 1:
            return False
    return True


def _alpha_balanced(codes, struct):
    """LIFO / FIFO: the designated ticket is alphabetically first, and last, about as often as chance (+-1)."""
    for rule in ("LIFO", "FIFO"):
        pts = struct[rule]
        expected = sum(1 / len(h) for h, _ in pts)
        for pick in (min, max):
            observed = sum(pick(h, key=lambda i: codes[i]) == leg for h, leg in pts)
            if abs(observed - expected) > 1:
                return False
    return True


def make_schedule(key, D, high_excursion=False, n_esc=N_ESC, n_norm=N_NORM, n_rush=N_RUSH, n_ins=N_INS,
                  n_nest=N_NEST, balance=True):
    """Formal days use the defaults (16 tickets, balanced); the smoke plan uses a short day without balancing."""
    from .strategies import make_chooser, simulate
    if not 1 <= D <= n_esc - n_ins - n_nest * NEST_SIZE - 1:
        raise GenerationError("D + 1 escalations must fit into the day")
    rs, rc, rk, rr, ri = (rng_for(key, s) for s in ("skeleton", "categories", "codes", "rush", "ids"))
    for _ in range(MAX_ATTEMPTS):
        arrivals = _arrivals(_burst_sizes(D, rs, n_esc - n_ins - n_nest * NEST_SIZE), n_norm, n_ins, n_nest, rs,
                             NEST_SIZE)
        if arrivals is None:
            continue
        n = len(arrivals)
        rush_idx = set(rr.sample(range(n), n_rush))
        dues = [a + STEPS + rr.randint(*RUSH_SLACK) for a, _ in arrivals]
        ids = [(f"C-{ri.randint(1000, 9999)}", f"O-{ri.randint(1000, 9999)}") for _ in arrivals]
        pool = [a + b for a in LETTERS for b in LETTERS]

        def build(codes, cats):
            return {"key": key, "D": D, "high_excursion": high_excursion, "t_max": 10 ** 6,
                    "excursion_max": _excursion_max(cats),
                    "tickets": [{"code": c, "kind": k, "category": cat, "arrival": a, "rush": i in rush_idx,
                                 "due": dues[i], "customer_id": ids[i][0], "order_id": ids[i][1]}
                                for i, ((a, k), c, cat) in enumerate(zip(arrivals, codes, cats))]}

        codes = rk.sample(pool, n)
        cats = [rc.choice(CATEGORIES) for _ in arrivals]
        sched = build(codes, cats)
        env, _ = simulate(sched, "LIFO", make_chooser("reference", "LIFO"))
        held_kinds = {env.st.tickets[e[1]].kind for e in env.events if e[0] == "h"}
        if env.max_depth != D or not held_kinds <= {"escalated"}:
            continue
        struct = _structure(sched)
        ok = False
        for _ in range(MAX_REDRAWS):                         # categories: counter bound and PAR / BAL sides
            cats = [rc.choice(CATEGORIES) for _ in arrivals]
            if high_excursion and _excursion_max(cats) < min(4, n // 3):
                continue
            if not balance or _side_balanced(arrivals, cats, struct["times"]):
                ok = True
                break
        if not ok:
            continue
        ok = False
        for _ in range(MAX_REDRAWS):                         # codes: alphabetical rank for LIFO / FIFO
            codes = rk.sample(pool, n)
            if not balance or _alpha_balanced(codes, struct):
                ok = True
                break
        if not ok:
            continue
        sched = build(codes, cats)
        env, _ = simulate(sched, "LIFO", make_chooser("reference", "LIFO"))
        sched["reference_end"] = env.st.t
        sched["t_max"] = env.st.t + T_SLACK
        sched["balanced"] = balance
        return sched
    raise GenerationError(f"no schedule for D={D} ({key})")
