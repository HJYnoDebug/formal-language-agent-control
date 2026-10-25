"""Verify the minimal-DFA state counts in 00_thesis_spec.md §3.4 (v5.5, alignment review 2026-10-04).

What is minimised (the object of the count, stated explicitly):
  * the rule's own prefix-safe language L_r (rules.policy_check): acceptance = no violation so far; the common
    environment (no more outputs than records) is NOT part of FIRST / MOD3;
  * restricted to the depth-bounded domain: an arrival that would exceed D unanswered records is outside the domain
    and goes to the reject sink (for FIRST / MOD3, which keep no record list, D plays no role);
  * the count excludes the reject sink.
Each language is built as a machine, all reachable states are explored and Moore-minimised, and the result is
compared with the formula:

    FIRST    4                      (start + one state per label)
    MOD3     3                      (sum mod 3)
    COUNT    D + 1                  (unanswered count 0..D)
    REVERSE  (k^(D+1) - 1)/(k - 1)  (stack contents of length 0..D)
    SAME     (k^(D+1) - 1)/(k - 1)  (queue contents of length 0..D)

These are minimal state counts of exactly these languages; they are not claims about a checker that may assume
its input is environment-legal (a promise problem), which could be smaller.

Stdlib only. Run:  python analysis/verify_state_counts.py
"""
from collections import deque

LABELS = "ABC"                      # k = 3
VALUE = {"A": 0, "B": 1, "C": 2}
SINK = "SINK"


def alphabet(rule):
    opens = [("o", a) for a in LABELS]
    if rule == "COUNT":
        return opens + [("c", "Y"), ("c", "N"), ("x", None)]
    return opens + [("c", a) for a in LABELS] + [("x", None)]


def make_rule(rule, D):
    """Return (initial_state, step, accepting) for the depth-bounded monitor."""
    if rule == "FIRST":
        def step(s, sym):
            kind, a = sym
            if kind == "o":
                return a if s is None else s
            if kind == "c":
                return s if (s is not None and a == s) else SINK
            return s
        return None, step, lambda s: True

    if rule == "MOD3":
        def step(s, sym):
            kind, a = sym
            if kind == "o":
                return (s + VALUE[a]) % 3
            if kind == "c":
                return s if a == LABELS[s] else SINK
            return s
        return 0, step, lambda s: True

    if rule == "COUNT":
        def step(m, sym):
            kind, a = sym
            if kind == "o":
                return m + 1 if m < D else SINK
            if kind == "c":
                if a == "Y":
                    return m - 1 if m >= 2 else SINK
                return 0 if m == 1 else SINK
            return m
        return 0, step, lambda m: True          # prefix acceptance

    if rule in ("REVERSE", "SAME"):
        def step(s, sym):
            kind, a = sym
            if kind == "o":
                return s + (a,) if len(s) < D else SINK
            if kind == "c":
                if not s:
                    return SINK
                if rule == "REVERSE":
                    return s[:-1] if s[-1] == a else SINK
                return s[1:] if s[0] == a else SINK
            return s
        return (), step, lambda s: True         # prefix acceptance

    raise ValueError(rule)


def minimal_states(rule, D):
    init, step, accepting = make_rule(rule, D)
    sigma = alphabet(rule)
    # reachable states
    states, queue = {init}, deque([init])
    while queue:
        s = queue.popleft()
        for sym in sigma:
            t = step(s, sym)
            if t != SINK and t not in states:
                states.add(t)
                queue.append(t)
    all_states = list(states) + [SINK]
    delta = {s: [SINK if s == SINK else step(s, sym) for sym in sigma] for s in all_states}
    # Moore refinement
    cls = {s: (s != SINK and accepting(s)) for s in all_states}
    while True:
        sig = {s: (cls[s], tuple(cls[t] for t in delta[s])) for s in all_states}
        ids, new = {}, {}
        for s in all_states:
            new[s] = ids.setdefault(sig[s], len(ids))
        if len(ids) == len(set(cls.values())):
            break
        cls = new
    n_classes = len(set(cls.values()))
    dead = {cls[SINK]}
    return n_classes - len(dead), len(states)


def formula(rule, D, k=len(LABELS)):
    return {"FIRST": 4, "MOD3": 3, "COUNT": D + 1,
            "REVERSE": (k ** (D + 1) - 1) // (k - 1),
            "SAME": (k ** (D + 1) - 1) // (k - 1)}[rule]


if __name__ == "__main__":
    ok = True
    print(f"{'rule':8s} {'D':>2s} {'minimal':>8s} {'formula':>8s}")
    for rule in ["FIRST", "MOD3", "COUNT", "REVERSE", "SAME"]:
        for D in range(1, 7):
            n_min, _ = minimal_states(rule, D)
            f = formula(rule, D)
            ok &= (n_min == f)
            print(f"{rule:8s} {D:2d} {n_min:8d} {f:8d}{'' if n_min == f else '   MISMATCH'}")
    print("\nformula at the spec's depths (k = 3):")
    for rule in ["FIRST", "MOD3", "COUNT", "REVERSE", "SAME"]:
        print(f"  {rule:8s} " + "  ".join(f"D={D}: {formula(rule, D):,}" for D in (4, 8, 12, 16)))
    print("\nAll checks passed" if ok else "\nMISMATCH FOUND")
    raise SystemExit(0 if ok else 1)
