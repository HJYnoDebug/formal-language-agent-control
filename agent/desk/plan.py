"""Help-desk plans (02b §8). A cell is one episode (one day). Schedule keys are shared across models, rules and
controls (pairing; the same day under every condition) and carry the split, so smoke / pilot / formal never overlap.
Odd seeds draw high-excursion days (the counter coordinate of BAL has range)."""
from functools import lru_cache

from .generator import make_schedule
from .rules import RULED

MODELS = ("ministral-3b", "ministral-8b", "ministral-14b", "large-3")    # 02b §14 D10: Mistral 3 family
WEAK, STRONG = "ministral-3b", "large-3"
DEPTHS = (2, 4, 7, 10)
FORMAL_SEEDS = 3          # 12 days per model x rule x control: ~81 violable resume points (02b §8, §14 D3)
PLACEBO_SEEDS = 2
MAIN_RUN = "d_main"       # one run directory for the internal pilot and the formal run (02b §8, §14 A12)
PILOT_MODELS = (STRONG,)  # internal pilot: the strongest model (does the strongest model reach a ceiling?)
# the regular representative (02b v1.6, §14 D1): PAR, restrictive at every resume point and a minimal pair with BAL;
# LAST stays implemented (exploratory only)
REGULAR = "PAR"
FULL_CONTROLS = ("C0", "R", "N", "Mminus", "M", "MR", "C1", "C2", "E", "C3")
WORDING_MODELS = (WEAK, STRONG)           # wording robustness (02b v1.9 §7.1): C0 under the two paraphrases
SMOKE = {"n_esc": 4, "n_norm": 2, "n_rush": 1, "n_ins": 0, "n_nest": 0, "balance": False}   # short day: depth 3 from one burst of 4


def controls_for(rule):
    if rule == "FREE":
        return ("C0",)
    return tuple(c for c in FULL_CONTROLS if not (c == "Mminus" and rule in ("LAST", "PAR")))


def rules_in_grid(regular=REGULAR):
    return tuple(r for r in RULED if r not in ("LAST", "PAR") or r == regular)


def sched_key(split, D, seed):
    return f"desk|{split}|D{D}|s{seed}"


@lru_cache(maxsize=None)
def schedule(key, D, high_excursion, short=False):
    return make_schedule(key, D, high_excursion, **(SMOKE if short else {}))


def get_schedule(cell):
    return schedule(cell["schedule"], cell["D"], cell["high_excursion"], cell.get("short", False))


def cell_key(c):
    fields = ["exp", "task", "model", "rule", "control", "wording", "D", "seed"]
    return "|".join(f"{k}={c[k]}" for k in fields if c.get(k) is not None)


def _cell(exp, model, rule, control, D, seed, split, short=False, wording=None):
    c = {"exp": exp, "task": "desk", "model": model, "rule": rule, "control": control, "D": D, "seed": seed,
         "schedule": sched_key(split, D, seed), "high_excursion": bool(seed % 2)}
    if short:
        c["short"] = True
    if wording:
        c["wording"] = wording
    return c


def grid(split, models, depths, seeds, regular=REGULAR):
    cells = []
    for m in models:
        for D in depths:
            for s in seeds:
                cells.append(_cell("D0", m, "FREE", "C0", D, s, split))           # no procedure: own tendencies
                for r in rules_in_grid(regular):
                    for c in controls_for(r):
                        cells.append(_cell("D1", m, r, c, D, s, split))
    return cells


def smoke_plan():
    """Tool-calling smoke: every model, LIFO under C0 / N / M− / M / C1 / E / C3, one short day (6 tickets)."""
    return [_cell("DS", m, "LIFO", c, 3, 0, "smoke", short=True) for m in MODELS
            for c in ("C0", "N", "Mminus", "M", "C1", "E", "C3")]


def pilot_plan(regular=REGULAR):
    """Internal pilot (02b §8, §14 A12): the first slice of the formal plan, same days and cell keys, run under
    MAIN_RUN so that the formal run skips it: the strongest model x 40 combinations x D{2, 10} x formal day 0 =
    80 episodes."""
    return grid("formal", PILOT_MODELS, (2, 10), (0,), regular)


def formal_plan(regular=REGULAR):
    """4 models x 40 combinations x D{2, 4, 7, 10} x 3 days = 1,920 episodes (40 = PAR 9 + BAL / LIFO / FIFO 10 each
    + FREE; v1.9 added R and MR)."""
    return grid("formal", MODELS, DEPTHS, range(FORMAL_SEEDS), regular)


def think_plan():
    """T (exploratory, freeze log changes after the freeze 14): C0 plus a think tool and the instruction to reason
    before each step; the weakest and the strongest model x PAR / BAL / LIFO / FIFO x D{4, 10} x the 3 formal days of
    each depth (the same days as the formal C0 cells, for the paired comparison) = 48 episodes."""
    return [_cell("DT", m, r, "T", D, s, "formal") for m in (WEAK, STRONG) for r in ("PAR", "BAL", "LIFO", "FIFO")
            for D in (4, 10) for s in range(FORMAL_SEEDS)]


def placebo_plan():
    """P: the weakest and the strongest model x LIFO / FIFO x D{4, 10} x 2 days (formal days, so P pairs with C0 / C1) = 16 episodes."""
    return [_cell("DP", m, r, "P", D, s, "formal") for m in (WEAK, STRONG) for r in ("LIFO", "FIFO")
            for D in (4, 10) for s in range(PLACEBO_SEEDS)]


def probe_twostack_plan():
    """Exploratory probe M2S (02b §15; decided after the formal experiment): FIFO with two trays instead of the rail,
    4 models x D{2, 4, 7, 10} x the 3 formal days = 48 episodes, paired with FIFO under N and M."""
    return [_cell("DX", m, "FIFO", "M2S", D, s, "formal") for m in MODELS for D in DEPTHS for s in range(FORMAL_SEEDS)]


def wording_plan():
    """Wording robustness (02b v1.9 §7.1): C0 under paraphrases 1 and 2, the weakest and the strongest model x
    PAR / BAL / LIFO / FIFO x
    D{4, 10} x the 3 formal days = 96 episodes, paired with the formal C0 cells (wording 0) on the same days."""
    return [_cell("DW", m, r, "C0", D, s, "formal", wording=w) for m in WORDING_MODELS
            for r in ("PAR", "BAL", "LIFO", "FIFO") for w in (1, 2) for D in (4, 10) for s in range(FORMAL_SEEDS)]


PLANS = {"smoke": smoke_plan, "pilot": pilot_plan, "formal": formal_plan, "placebo": placebo_plan,
         "probe_twostack": probe_twostack_plan, "wording": wording_plan, "think": think_plan}
