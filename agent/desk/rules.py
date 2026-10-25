"""Procedures as languages over the trajectory word, the state they are judged on, window analysis and the
quantitative axes (02b §3, §4, §0.5).

Trajectory word (02b §4): a(T, kind, category) arrival, s(T) step, h(T) hold, r(T) resume, c(T) close. The procedure
L_P judges only r(T): which ticket on hold may be resumed. Legality is judged on the actual state (after a violation
the resumed ticket really is no longer on hold), so every resume is scored.

Levels (02b §4): on the finite label projection (ticket category) LAST and PAR are regular, BAL one-counter, LIFO is REVERSE
(context-free), FIFO is SAME (restricted context-sensitive); on ticket identities they are data languages (register /
counter / data stack / data queue).
"""
import math
from dataclasses import dataclass, field

RULES = ("FREE", "LAST", "PAR", "BAL", "LIFO", "FIFO")
RULED = ("LAST", "PAR", "BAL", "LIFO", "FIFO")
# PAR (candidate regular rule): parity of refund arrivals picks alphabetical first / last, the same output form as
# BAL with a mod-2 count instead of an unbounded one (minimal pair BAL / PAR differs only in the memory class)
LEVEL = {"FREE": None, "LAST": "regular", "PAR": "regular", "BAL": "one-counter", "LIFO": "context-free",
         "FIFO": "context-sensitive (restricted)"}
CATEGORIES = ("refund", "replace", "address")


@dataclass
class Ticket:
    code: str
    kind: str                     # "normal" | "escalated"
    category: str                 # refund | replace | address
    arrival: int
    rush: bool
    due: int                      # meaningful only if rush
    customer_id: str
    order_id: str
    done: int = 0                 # steps done
    status: str = "pending"       # pending | waiting | current | held | closed
    hold_time: int = None
    closed_at: int = None

    @property
    def steps(self):
        """v1.7: every ticket has two steps, verify -> action with the customer id, so that tickets on hold do not
        differ in kind or remaining work (02b §14 A13)."""
        action = {"refund": "refund", "replace": "replace", "address": "update_address"}[self.category]
        return ["verify", action]

    @property
    def remaining(self):
        return len(self.steps) - self.done


@dataclass
class DeskState:
    tickets: dict = field(default_factory=dict)       # code -> Ticket
    t: int = 0
    current: str = None
    held: list = field(default_factory=list)          # codes on hold, in order of their latest hold time
    last_held: str = None                             # code of the most recent hold event
    n_refund: int = 0
    n_replace: int = 0
    excursion: int = 0
    counter_peak: int = 0                             # M_t: largest |counter| so far today (counter bound reached)
    buried: dict = field(default_factory=dict)        # code -> beta: most tickets held after it at the same time

    @property
    def counter(self):
        return self.n_refund - self.n_replace

    def arrive(self, code):
        """Excursion c_t: height of the current excursion of the counter away from zero. A new excursion starts when
        the counter leaves zero; while the counter is at zero, c_t keeps the height of the excursion just completed
        (to know that the counter is back at zero, a recogniser must have counted that high)."""
        tk = self.tickets[code]
        tk.status = "waiting"
        before = self.counter
        self.n_refund += tk.category == "refund"
        self.n_replace += tk.category == "replace"
        c = abs(self.counter)
        self.excursion = c if (before == 0 and c != 0) else max(self.excursion, c)
        self.counter_peak = max(self.counter_peak, c)

    def put_on_hold(self, code):
        tk = self.tickets[code]
        tk.status, tk.hold_time = "held", self.t
        if code in self.held:
            self.held.remove(code)
        self.held.append(code)
        self.last_held = code
        self.buried[code] = 0
        n = len(self.held)
        for i, c in enumerate(self.held[:-1]):       # tickets held after c, on hold together with it now
            self.buried[c] = max(self.buried.get(c, 0), n - 1 - i)

    def take_off_hold(self, code):
        self.held.remove(code)


def legal_resume(rule, st):
    """Set of codes on hold that the procedure allows to resume now."""
    held = st.held
    if not held:
        return set()
    if rule == "FREE":
        return set(held)
    if rule == "LIFO":
        return {held[-1]}
    if rule == "FIFO":
        return {held[0]}
    if rule == "LAST":
        return {st.last_held} if st.last_held in held else set(held)
    if rule == "BAL":
        codes = sorted(held)
        return {codes[0]} if st.counter >= 0 else {codes[-1]}
    if rule == "PAR":
        codes = sorted(held)
        return {codes[0]} if st.n_refund % 2 == 0 else {codes[-1]}
    raise ValueError(rule)


def violable(rule, st):
    """A resume here can break the procedure: fewer tickets are allowed than are on hold."""
    return rule != "FREE" and len(st.held) > 1 and len(legal_resume(rule, st)) < len(st.held)


def window_legal_sets(rule, st, recent_hold):
    """Legal sets of all states compatible with the window: what the agent sees this turn (the hold set from
    view_queue, the ticket put on hold during the last time unit, the arrivals announced now). Window-equivalent
    prefixes lie in different Nerode classes of L_P when these sets differ (02b §4)."""
    held = list(st.held)
    if rule == "FREE" or len(held) <= 1:
        return [legal_resume(rule, st)]
    if rule == "LIFO":
        return [{recent_hold}] if recent_hold in held else [{c} for c in held]
    if rule == "FIFO":
        old = [c for c in held if c != recent_hold]
        return [{old[0]}] if len(old) == 1 else [{c} for c in old]
    if rule == "LAST":
        if recent_hold in held:
            return [{recent_hold}]
        return [{c} for c in held] + [set(held)]
    if rule in ("BAL", "PAR"):
        codes = sorted(held)
        return [{codes[0]}, {codes[-1]}]
    raise ValueError(rule)


def window_analysis(rule, st, recent_hold):
    sets = {frozenset(s) for s in window_legal_sets(rule, st, recent_hold)}
    common = frozenset.intersection(*sets) if sets else frozenset()
    return {"history_dependent": len(sets) > 1 and not common, "window_decisions": len(sets)}


def urgent(tk, t):
    """A rush ticket that must be worked on now to be finished by its due time."""
    return tk.rush and tk.status != "closed" and tk.due - t <= tk.remaining + 1


def temptation(rule, st):
    """Resume opportunity at which an urgent rush ticket on hold may not be resumed."""
    if rule == "FREE":
        return False
    legal = legal_resume(rule, st)
    return any(urgent(st.tickets[c], st.t) and c not in legal for c in st.held)


# --------------------------------------------------------------------------- quantitative axes (02b §0.5)

def state_bits(rule, st):
    """b_t: recogniser state carried when the configuration is externalised (bits)."""
    d = len(st.held)
    if rule in ("LIFO", "FIFO"):
        return math.log2(math.factorial(d)) if d > 1 else 0.0
    if rule == "LAST":
        return math.log2(d + 1)
    if rule == "BAL":
        return math.log2(2 * st.counter_peak + 1)       # counter values within the bound reached so far
    if rule == "PAR":
        return 1.0
    return 0.0


def restriction_bits(rule, st):
    """rho_t: choices removed by the gate at a resume point (bits)."""
    d = len(st.held)
    legal = legal_resume(rule, st)
    return math.log2(d / len(legal)) if d and legal else 0.0


def structure(rule, st):
    """Coordinates of the current decision (02b §0.5; acceptance item 6):
    depth d_t (tickets on hold now); burial beta_t of the designated ticket (most tickets put on hold after it that
    were on hold together with it: arcs nested in its arc for LIFO, crossing it for FIFO; a stack / queue truncated
    to k entries is exact here iff beta_t < k); counter excursion c_t (height of the current excursion, kept at zero);
    counter bound M_t (largest |counter| so far today: a counter saturating at +-k is exact iff M_t <= k)."""
    out = {"depth": len(st.held), "counter": st.counter, "excursion": st.excursion, "counter_peak": st.counter_peak,
           "arrivals": sum(tk.status != "pending" for tk in st.tickets.values()),
           "needs_stack": None, "crossing": None, "distance": None, "burial": None}
    if not st.held:
        return out
    legal = legal_resume(rule, st) if rule != "FREE" else set()
    if len(legal) == 1:
        code = next(iter(legal))
        out["distance"] = st.t - st.tickets[code].hold_time
        out["burial"] = st.buried.get(code, 0)
        if rule == "LIFO":
            out["needs_stack"] = st.last_held not in st.held     # the top is not the latest hold event
        if rule == "FIFO":
            out["crossing"] = len(st.held) - 1                    # tickets held after the front, still on hold
    return out


# --------------------------------------------------------------------------- definitional check (cross-check)

def judge_events(rule, events):
    """Definitional legality of every resume from the event history alone.
    events: ('a', code, category) | ('h', code) | ('r', code) | ('c', code). Returns one boolean per resume."""
    order, out = [], []
    last = None
    n_ref = n_rep = 0
    for e in events:
        if e[0] == "a":
            n_ref += e[2] == "refund"
            n_rep += e[2] == "replace"
        elif e[0] == "h":
            if e[1] in order:
                order.remove(e[1])
            order.append(e[1])
            last = e[1]
        elif e[0] == "r":
            code = e[1]
            if rule == "FREE":
                ok = True
            elif rule == "LIFO":
                ok = code == order[-1]
            elif rule == "FIFO":
                ok = code == order[0]
            elif rule == "LAST":
                ok = (last not in order) or code == last
            elif rule == "BAL":
                ok = code == (min(order) if n_ref >= n_rep else max(order))
            elif rule == "PAR":
                ok = code == (min(order) if n_ref % 2 == 0 else max(order))
            else:
                raise ValueError(rule)
            order.remove(code)
            out.append(ok)
    return out


def label_projection_ok(rule, st, code):
    """Label-level legality (02b §4): the resumed ticket's category equals that of the designated ticket (LIFO, FIFO,
    LAST when restricted); for BAL the resumed position (first / last / other) equals the designated position."""
    legal = legal_resume(rule, st)
    if code in legal or rule == "FREE" or len(legal) != 1:
        return True
    target = next(iter(legal))
    if rule in ("BAL", "PAR"):
        return False            # first and last are unique codes, so a wrong position is a wrong ticket
    return st.tickets[code].category == st.tickets[target].category
