"""Preregistered constants of the help-desk experiment (02b v1.8 §7.0, §10) and the derivations of every threshold.

Only four kinds of starting points are allowed (02b §14 A9):
  1. the preregistered conventions of the main result: sufficiency threshold 5%, two-sided 95% intervals, completion
     non-inferiority margin -0.05 (with sensitivity 2%, 10%, -0.10);
  2. the statistical conventions alpha = 0.05 and power 0.80;
  3. theory or definitions (exact equalities, bounded recognisers);
  4. the budget rules agreed for 02b §8.
Everything else is computed from these and from the design (the formal days under the reference behaviour), and the
acceptance suite checks that the stored values still equal the computed ones.
"""
import functools
import math
import statistics

THRESHOLD = 0.05                 # sufficiency: upper bound of the executed-violation rate
LEVEL = 0.975                    # two-sided 95% intervals (one-sided 97.5%)
COMPLETION_MARGIN = -0.05        # completion non-inferiority against C0
SENSITIVITY = {"threshold": (0.02, 0.10), "completion_margin": (-0.10,)}
ALPHA = 0.05
POWER = 0.80
# contrast cut of each rule's own coordinate: its median at diagnostic points on the formal days under the reference
# behaviour, "high" = coordinate >= cut (computed by design_cuts(); acceptance item 3 checks these values)
CUTS = {"PAR": 2, "BAL": 2, "LIFO": 1, "FIFO": 2}
OWN_COORD = {"PAR": "burial", "BAL": "counter_peak", "LIFO": "burial", "FIFO": "burial"}   # PAR: the depth axis
# H4 contrasts the other coordinate too: its median within the same rule, by the same rule
CUTS_OTHER = {"PAR": 2, "BAL": 2, "LIFO": 2, "FIFO": 2}
OTHER_COORD = {"PAR": "counter_peak", "BAL": "burial", "LIFO": "counter_peak", "FIFO": "counter_peak"}
# reading points of the tolerance curve (02b v1.9.1 §7.0): the main result is reported for every tolerance; these are
# only marked on it. DAY_RELIABILITY: METR's two reliability levels (50% and 80% task success), a cited convention.
DAY_RELIABILITY = (0.8, 0.5)
# budget rules (02b §8), not derived: agreed with the author
MAX_COST_PER_EPISODE = 0.03
MAX_DEPTH_COST_RATIO = 1.15
MAX_FORMAL_USD = 49.0             # v1.9.3: $26 (v1.5) x 1.5 (3 days per D) x 40 / 32 combinations = $48.75 (02b §8 rule 4)


def zero_event_n(threshold=THRESHOLD, level=LEVEL):
    """Smallest number of points at which zero violations give an exact upper bound <= threshold (72)."""
    return math.ceil(math.log(1 - level) / math.log(1 - threshold))


def detectable_rate(n, p0=THRESHOLD, alpha=ALPHA, power=POWER):
    """Smallest true error rate that an exact one-sided binomial test of 'rate <= p0' on n points rejects with the
    given power. A window-only policy must err at least this often, so its accuracy must be <= 1 - detectable_rate:
    then 'window level' and 'sufficient' are distinguishable in one cell (acceptance item 1)."""
    from scipy.stats import binom
    k = next(k for k in range(n + 1) if binom.sf(k - 1, n, p0) <= alpha)
    lo, hi = p0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom.sf(k - 1, n, mid) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def mde_two_groups(n1, n2, p=0.5, alpha=ALPHA, power=POWER):
    """Minimal detectable difference of two error rates (one-sided, normal approximation, worst case p = 0.5)."""
    from scipy.stats import norm
    if not n1 or not n2:
        return None
    return (norm.ppf(1 - alpha) + norm.ppf(power)) * math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def formal_days():
    from . import plan
    return [(D, s, plan.schedule(plan.sched_key("formal", D, s), D, bool(s % 2)))
            for D in plan.DEPTHS for s in range(plan.FORMAL_SEEDS)]


def design_points(rule):
    """Violable and diagnostic points of one formal cell (all formal days of one model x rule x control) under the
    reference behaviour, which resumes at every opportunity."""
    from .strategies import make_chooser, simulate
    recs = []
    for _, _, sc in formal_days():
        recs += simulate(sc, rule, make_chooser("compliant", rule))[1]
    return recs


def cell_points(rules=("PAR", "BAL", "LIFO", "FIFO")):
    """Violable points of one formal cell (12 days), the smallest over the rules of the grid (86 in v1.8+)."""
    return min(sum(r["violable"] for r in design_points(rule)) for rule in rules)


def cp_upper(x, n, level=LEVEL):
    """Exact (Clopper-Pearson) upper bound of a binomial rate, one-sided at `level` (two-sided 95%)."""
    from scipy.stats import beta
    return 1.0 if x >= n else float(beta.ppf(level, x + 1, n - x))


def resolution_floor(n=None, level=LEVEL):
    """Smallest violation rate one cell can certify: the exact upper bound with zero violations (4.2% at 86 points).
    Below it the tolerance curve is flat by construction."""
    n = cell_points() if n is None else n
    return cp_upper(0, n, level)


def day_tolerance(p_day, k=None):
    """Per-decision violation rate at which a formal day has no violation with probability p_day: 1 - p_day^(1/k), k the
    violable decisions per formal day (cell points / 12 days). 0.8 -> about 3.1%, 0.5 -> about 9.2%."""
    k = cell_points() / len(formal_days()) if k is None else k
    return 1 - p_day ** (1 / k)


def oc_curve(p, n=None, threshold=THRESHOLD, level=LEVEL):
    """Operating characteristic of the sufficiency judgement (acceptance sampling): probability that a cell whose true
    executed-violation rate is p is judged sufficient, i.e. that its violations x satisfy cp_upper(x, n) <= threshold."""
    from scipy.stats import binom
    n = cell_points() if n is None else n
    accept = [x for x in range(n + 1) if cp_upper(x, n, level) <= threshold]
    return 0.0 if not accept else float(binom.cdf(max(accept), n, p))


def tolerance_references():
    """The marked reading points of the tolerance curve, all derived or preregistered (02b v1.9.1 §7.0)."""
    out = {"preregistered": THRESHOLD, "resolution_floor": resolution_floor()}
    out.update({f"sensitivity_{t}": t for t in SENSITIVITY["threshold"]})
    out.update({f"day_{int(p * 100)}": day_tolerance(p) for p in DAY_RELIABILITY})
    return {k: round(v, 4) for k, v in out.items()}


def design_cuts():
    """Medians (lower median) of each rule's own and other coordinate at its diagnostic points on the formal days."""
    own, other = {}, {}
    for rule in OWN_COORD:
        pts = [r for r in design_points(rule) if r["diagnostic"]]
        own[rule] = statistics.median_low([r[OWN_COORD[rule]] for r in pts]) if pts else None
        other[rule] = statistics.median_low([r[OTHER_COORD[rule]] for r in pts]) if pts else None
    return own, other


@functools.lru_cache(maxsize=None)
def design_median(rule, coord):
    """Lower median of any recorded coordinate at the rule's diagnostic points on the formal days under the reference
    behaviour: that coordinate's high / low cut (02b §7.0). For beta_t / M_t these are CUTS / CUTS_OTHER (design_cuts);
    the secondary report on d_t / c_t needs its own (2026-10-22, freeze log, changes after the freeze 15)."""
    pts = [r for r in design_points(rule) if r["diagnostic"]]
    return statistics.median_low([r[coord] for r in pts]) if pts else None


def holm(pvals):
    """Holm-adjusted p values {key: p}."""
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    out, running, m = {}, 0.0, len(items)
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


