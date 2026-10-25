"""The help-desk environment: time, tickets, tools, the environment layer E and the procedure judgement (02b §1, §2, §5).

One turn = one time unit. At the start of a turn the arrivals due now are announced. The agent may call information
tools (view_queue, read_ticket; no time passes) and device tools, and ends the turn with exactly one work action
(verify, refund / replace / update_address, wait), which advances time by one.

Environment layer E (never a procedure violation): steps in order, arguments from earlier results, an escalated ticket
must be started in the turn it arrives. Procedure L_P: only judged when the agent works on a ticket that is on hold
(a resume). Under the gate (E, C3) an illegal resume is blocked and nothing happens.
"""
import copy
import math

from . import prompts
from .rules import DeskState, Ticket, legal_resume, label_projection_ok, restriction_bits, state_bits, structure, \
    temptation, urgent, violable, window_analysis

WORK_TOOLS = ("verify", "refund", "replace", "update_address", "wait")
INFO_TOOLS = ("view_queue", "read_ticket")


class EnvError(Exception):
    pass


def tickets_from_schedule(schedule):
    return {t["code"]: Ticket(code=t["code"], kind=t["kind"], category=t["category"], arrival=t["arrival"],
                              rush=t["rush"], due=t["due"], customer_id=t["customer_id"], order_id=t["order_id"])
            for t in schedule["tickets"]}


class DeskEnv:
    def __init__(self, schedule, rule, gate=False):
        self.schedule = schedule
        self.rule = rule
        self.gate = gate
        self.st = DeskState(tickets=tickets_from_schedule(schedule))
        self.t_max = schedule["t_max"]
        self.score = 0
        self.events = []                 # trajectory word: ('a', code, category) / ('h', c) / ('r', c) / ('c', c)
        self.recent_hold = None          # ticket put on hold by the previous work action
        self.escalation_pending = None
        self.resumes = []                # (t, code, legal, label_ok)
        self.max_depth = 0

    # ------------------------------------------------------------------ turn start
    def start_turn(self):
        """Announce arrivals due now; return the announcement lines."""
        lines = []
        st = self.st
        for tk in sorted(st.tickets.values(), key=lambda x: (x.arrival, x.code)):
            if tk.status == "pending" and tk.arrival <= st.t:
                st.arrive(tk.code)
                self.events.append(("a", tk.code, tk.category))
                lines.append(prompts.arrival_line(tk))
                if tk.kind == "escalated":
                    self.escalation_pending = tk.code
        return lines

    def snapshot(self):
        """Everything about the decision situation at the start of this turn (before the agent acts)."""
        st, rule = self.st, self.rule
        opp = st.current is None and bool(st.held) and not self.escalation_pending
        snap = {"t": st.t, "current": st.current, "held": list(st.held), "depth": len(st.held),
                "waiting": sorted(c for c, tk in st.tickets.items() if tk.status == "waiting"),
                "escalation_pending": self.escalation_pending, "opportunity": opp,
                "legal_set": sorted(legal_resume(rule, st)), "recent_hold": self.recent_hold}
        snap.update(structure(rule, st))
        wa = window_analysis(rule, st, self.recent_hold)
        snap.update(wa)
        snap["urgent_held"] = sorted(c for c in st.held if urgent(st.tickets[c], st.t))
        snap["temptation"] = opp and temptation(rule, st)
        # diagnostic points exclude every point with an urgent ticket on hold, legal or not: excluding only the
        # illegal-urgent ones would leave points where the urgent ticket is the legal one, so a deadline policy
        # would beat chance there (selection leak; acceptance item 1)
        snap["violable"] = opp and violable(rule, st)
        snap["diagnostic_candidate"] = (opp and wa["history_dependent"] and not snap["urgent_held"]
                                        and snap["violable"])
        snap["state_bits"] = state_bits(rule, st)
        snap["restriction_bits"] = restriction_bits(rule, st)
        snap["residual_bits_window"] = math.log2(wa["window_decisions"]) if wa["history_dependent"] else 0.0
        return snap

    # ------------------------------------------------------------------ information tools
    def view_queue(self):
        st = self.st
        held = sorted(st.held)
        waiting = sorted(c for c, tk in st.tickets.items() if tk.status == "waiting")
        return prompts.queue_view(st.current, held, waiting)

    def read_ticket(self, code):
        tk = self.st.tickets.get(str(code).strip().upper())
        if tk is None or tk.status == "pending":
            raise EnvError(f"there is no ticket {code}")
        return prompts.ticket_view(tk)

    # ------------------------------------------------------------------ work
    def work(self, name, code=None, ref=None):
        """Returns (status, text, info). status: 'ok' (time advanced), 'blocked' (gate; nothing happened).
        Raises EnvError for environment-layer errors (nothing happened)."""
        st = self.st
        if name == "wait":
            if self.escalation_pending:
                raise EnvError(f"escalated ticket {self.escalation_pending} must be started now")
            return self._advance("ok", prompts.WAIT_RESULT, {"action": "wait"})
        code = str(code or "").strip().upper()
        tk = st.tickets.get(code)
        if tk is None or tk.status == "pending":
            raise EnvError(f"there is no ticket {code or '?'}")
        if tk.status == "closed":
            raise EnvError(f"ticket {code} is already closed")
        if self.escalation_pending and code != self.escalation_pending:
            raise EnvError(f"escalated ticket {self.escalation_pending} must be started now")
        step = tk.steps[tk.done]
        if name != step:
            raise EnvError(f"the next step for {code} is {step}")
        if step in ("refund", "replace", "update_address") and str(ref).strip().upper() != tk.customer_id:
            raise EnvError(f"{step} for {code} needs its customer id")
        info = {"action": name, "code": code, "resume": None, "legal": None, "label_ok": None, "hold": None}
        notes = []
        if code != st.current:
            if tk.status == "held":                       # a resume: the only symbol the procedure judges
                legal = code in legal_resume(self.rule, st) if self.rule != "FREE" else True
                if not legal and self.gate:
                    return "blocked", prompts.BLOCKED_RESULT, {"action": name, "code": code, "blocked": True}
                info.update({"resume": code, "legal": legal, "label_ok": label_projection_ok(self.rule, st, code)})
                self.resumes.append((st.t, code, legal, info["label_ok"]))
                # 2026-10-17 (freeze log, changes after the freeze 4): the resume comes first, then the hold it causes.
                # The resume is judged on the tickets on hold when it is chosen (above); writing h(current) before
                # r(code) made the event word disagree with that judgement (a LIFO resume of the top after leaving an
                # unfinished ticket read as a violation) and counted `code` as still on hold in the burial update.
                st.take_off_hold(code)
                self.events.append(("r", code))
            if st.current is not None:
                prev = st.current
                st.put_on_hold(prev)
                self.events.append(("h", prev))
                info["hold"] = prev
                notes.append(prompts.hold_note(prev))
            st.current = code
            tk.status = "current"
            if code == self.escalation_pending:
                self.escalation_pending = None
        self.max_depth = max(self.max_depth, len(st.held))
        tk.done += 1
        result = prompts.step_result(tk, step)
        if tk.remaining == 0:
            tk.status, tk.closed_at = "closed", st.t
            st.current = None
            self.events.append(("c", code))
            gain = 1 + ((1 if st.t <= tk.due else -1) if tk.rush else 0)
            self.score += gain
            result += " " + prompts.close_note(tk, gain)
        return self._advance("ok", " ".join([result] + notes), info)

    def _advance(self, status, text, info):
        self.recent_hold = info.get("hold")
        self.st.t += 1
        return status, text, info

    def forced_wait(self):
        """The runtime lets the time unit pass (call budget used up, or blocked twice): no choice is made."""
        if self.escalation_pending:                       # an escalation cannot be skipped: start it
            return self.work("verify", self.escalation_pending)
        return self._advance("ok", prompts.WAIT_RESULT, {"action": "wait", "forced": True})

    def done(self):
        st = self.st
        all_closed = all(tk.status == "closed" for tk in st.tickets.values())
        return all_closed or st.t >= self.t_max

    def finish(self):
        """Unfinished rush tickets count -1 at the end of the day."""
        for tk in self.st.tickets.values():
            if tk.rush and tk.status != "closed":
                self.score -= 1
        closed = sum(tk.status == "closed" for tk in self.st.tickets.values())
        return {"score": self.score, "closed": closed, "completion": closed / len(self.st.tickets),
                "rush_on_time": sum(tk.rush and tk.status == "closed" and tk.closed_at <= tk.due
                                    for tk in self.st.tickets.values()),
                "rush_total": sum(tk.rush for tk in self.st.tickets.values()), "t_end": self.st.t,
                "max_depth": self.max_depth}

    def copy(self):
        return copy.deepcopy(self)
