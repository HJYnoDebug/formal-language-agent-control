"""Fixed agents and the offline simulator (02b §5, §10): reference / compliant, window-only policies (the memory
floor), finite-state approximations of the recogniser (memory:k), synthetic agents with planted effects, and the
device-only agents used to check the storage-class axis.

A chooser maps (env, ctx) to one work action (tool, code, ref). It may read the true state (these are offline
agents); window-only policies read only what a window shows.
"""
import random

from .env import DeskEnv
from .rules import legal_resume, urgent

NO_BOARD = ("C0", "R", "N", "Mminus", "E", "P")  # synthetic agents: no matched state to read


def next_step_action(tk):
    step = tk.steps[tk.done]
    if step == "verify":
        return ("verify", tk.code, None)
    return (step, tk.code, tk.customer_id)


def _base(env, resume_pick, defer=False):
    st = env.st
    if env.escalation_pending:
        return ("verify", env.escalation_pending, None)
    if st.current:
        return next_step_action(st.tickets[st.current])
    if st.held and not defer:
        code = resume_pick(env)
        return next_step_action(st.tickets[code])
    waiting = sorted((tk for tk in st.tickets.values() if tk.status == "waiting"), key=lambda x: (x.arrival, x.code))
    if waiting:
        return next_step_action(waiting[0])
    return ("wait", None, None)


def make_chooser(name, rule, seed=0):
    rng = random.Random(seed)

    def legal_pick(env):
        legal = legal_resume(rule, env.st)
        return env.st.held[-1] if rule == "FREE" else sorted(legal)[0] if len(legal) > 1 else next(iter(legal))

    def window_pick(kind):
        def pick(env):
            st = env.st
            held = sorted(st.held)
            if kind == "win_random":
                return rng.choice(held)
            if kind == "win_alpha_first":
                return held[0]
            if kind == "win_alpha_last":
                return held[-1]
            if kind == "win_recent":
                return env.recent_hold if env.recent_hold in st.held else rng.choice(held)
            if kind == "win_not_recent":
                rest = [c for c in held if c != env.recent_hold]
                return rng.choice(rest or held)
            if kind == "win_due":
                rush = sorted((st.tickets[c].due, c) for c in held if st.tickets[c].rush)
                return rush[0][1] if rush else rng.choice(held)
            raise ValueError(kind)
        return pick

    def truncated(env, k):
        """A stack / queue truncated to k entries (the bounded recognizer of 00 §3.4, acceptance item 6): a hold pushes,
        and past k the oldest entry is forgotten; a resume removes its entry and frees the slot. Kept in hold order, it
        is the most recent part of the actual stack / queue. (v1.7 remembered the last k hold events instead, which is
        the same only while every burst is h^m r^m.)"""
        mem = []
        for e in env.events:
            if e[0] == "h":
                if e[1] in mem:
                    mem.remove(e[1])
                mem.append(e[1])
                if len(mem) > k:
                    mem.pop(0)
            elif e[0] == "r" and e[1] in mem:
                mem.remove(e[1])
        return mem

    def memory_pick(k):
        def pick(env):
            st = env.st
            mem = truncated(env, k)
            if rule == "PAR":
                return legal_pick(env)              # one bit: within any k >= 1
            if rule == "BAL":
                c = 0
                for e in env.events:
                    if e[0] == "a":
                        c = max(-k, min(k, c + (e[2] == "refund") - (e[2] == "replace")))
                codes = sorted(st.held)
                return codes[0] if c >= 0 else codes[-1]
            if rule == "LAST":
                return legal_pick(env)              # needs only the last hold event: within any k >= 1
            if rule == "LIFO":
                if mem:                             # the remembered part is the top of the stack
                    return mem[-1]
                return rng.choice(st.held)          # every ticket on hold has been forgotten
            if rule == "FIFO":
                lost = [c for c in st.held if c not in mem]
                if not lost:                         # every ticket on hold is remembered: the order is known
                    return mem[0]
                return rng.choice(lost)              # the front is among the forgotten (oldest) ones
            return legal_pick(env)
        return pick

    def device_only(kind):
        def pick(env):
            st = env.st
            if kind == "m_only":
                return legal_pick(env)               # the matched device, correctly kept, shows the answer
            # M−: a counter (LIFO / FIFO) knows how many are held but not their order; a register (BAL) cannot count
            if rule == "BAL":
                codes = sorted(st.held)
                return rng.choice([codes[0], codes[-1]])
            return rng.choice(st.held)
        return pick

    if name in ("compliant", "reference"):
        return lambda env, ctx: _base(env, legal_pick)
    if name == "defer":
        return lambda env, ctx: _base(env, legal_pick, defer=True)
    if name.startswith("win_"):
        p = window_pick(name)
        return lambda env, ctx: _base(env, p)
    if name.startswith("memory:"):
        p = memory_pick(int(name.split(":")[1]))
        return lambda env, ctx: _base(env, p)
    if name in ("m_only", "mminus_only"):
        p = device_only(name)
        return lambda env, ctx: _base(env, p)
    if name == "greedy_deadline":
        p = window_pick("win_due")
        return lambda env, ctx: _base(env, p)
    if name.startswith("synthetic:"):
        _, k, prob = name.split(":")
        mem = memory_pick(int(k))
        prob = float(prob)

        def ch(env, ctx):
            st = env.st
            if not env.escalation_pending and st.current is None and st.held and rule != "FREE":
                legal = legal_resume(rule, st)
                tempting = [c for c in st.held if urgent(st.tickets[c], st.t) and c not in legal]
                if tempting and rng.random() < prob:
                    return next_step_action(st.tickets[tempting[0]])
            if ctx.get("control", "C0") in NO_BOARD:
                return _base(env, mem)
            return _base(env, legal_pick)
        return ch
    raise ValueError(name)


WINDOW_POLICIES = ("win_random", "win_alpha_first", "win_alpha_last", "win_recent", "win_not_recent", "win_due")


def simulate(schedule, rule, chooser, gate=False, control="C0"):
    """Run a chooser offline; returns (env, records) with one record per turn (snapshot + action outcome)."""
    env = DeskEnv(schedule, rule, gate=gate)
    records = []
    while not env.done():
        env.start_turn()
        snap = env.snapshot()
        blocked = 0
        while True:
            tool, code, ref = chooser(env, {"control": control})
            status, text, info = env.work(tool, code, ref)
            if status == "blocked":
                blocked += 1
                if blocked >= 2:
                    status, text, info = env.forced_wait()
                    info["forced"] = True
                    break
                # an offline chooser would repeat itself; fall back to the legal choice after one block
                legal = sorted(legal_resume(rule, env.st))
                tool, code, ref = next_step_action(env.st.tickets[legal[0]])
                status, text, info = env.work(tool, code, ref)
                break
            break
        rec = dict(snap)
        rec.update({"action": info.get("action"), "code": info.get("code"), "resume": info.get("resume"),
                    "legal": info.get("legal"), "label_ok": info.get("label_ok"), "blocked": blocked,
                    "forced": bool(info.get("forced"))})
        rec["deferred"] = snap["opportunity"] and info.get("resume") is None and not snap["escalation_pending"]
        rec["diagnostic"] = snap["diagnostic_candidate"] and info.get("resume") is not None
        records.append(rec)
    return env, records
