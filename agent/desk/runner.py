"""One help-desk day with native tool calling (02b §1, §2, §6).

One turn = one time unit. The turn message announces the arrivals and carries the control's pushed lines; the agent
may call information tools (view_queue, read_ticket) and device tools, and ends the turn with exactly one work action
(verify, refund / replace / update_address, wait). Several tool calls in one reply are executed in order until
the work action; later calls get an 'ignored' result (every tool_call id receives a tool message, as the API requires).

Controls (02b §6): pushed lines at the start of every turn
  N / M− / M  the agent's own device, read through its access discipline (devices.read_line)
  C1          the matched device kept by the runtime, same read line as M
  C2 / C3     C1 + "Allowed to resume now: ..."
  P           an unrelated line with the same number of characters as the C1 line
  E / C3      the gate: an illegal resume is blocked (nothing happens); after two blocks in a turn the time unit
              passes (the runtime never picks a ticket)
Environment-layer errors (wrong step, wrong id, escalation not started) are reported and nothing happens; they are
not procedure violations. If the call budget of a turn runs out without a work action, the time unit passes.

Lapse reminder (02b §2, §5): if the agent chooses to wait while nothing is current, tickets are on hold and no
escalation is pending (some ticket on hold is always legal to resume, so the wait helps nothing: a lapse), the next
turn message says only "Reminder: some tickets are still on hold." Every lapse is recorded with its cause (after a
blocked attempt / the device shows nothing on hold / plain). Up to the first reminder the trajectory is identical to a
day without reminders, so the primary analyses use that part; tickets closed after a reminder was given while they
were on hold do not count toward the unaided completion.

Backends: agent.common.toolcall.OpenRouterToolBackend (real calls) or FakeToolBackend (offline agents).
"""
import copy
import json
import math
import random
import threading

from agent.common.ledger import ServiceFailure
from agent.common.toolcall import MeteredCaller, ToolReply, chars_of, text_calls, tool_call_json

from . import devices, prompts
from .devices import canonical, consistent, matches, read_line
from .env import DeskEnv, EnvError, WORK_TOOLS
from .rules import legal_resume
from .strategies import make_chooser, next_step_action, simulate

CONTRACT_VERSION = "d1.6"                   # d1.6: resume written before the hold it causes (freeze log changes 4); d1.5: Mistral 3 models (02b §14 D10); d1.4: R / MR, wordings (D4-D6)
CONTROLS = ("C0", "R", "N", "Mminus", "M", "MR", "C1", "C2", "E", "C3", "P", "M2S", "T")
THINK_CONTROLS = ("T",)                   # exploratory, freeze log changes after the freeze 14: C0 + a think tool
THINK_EXTRA_CALLS = 2                     # thinking takes calls: + 2 per turn, so it does not force waits
THINK_MAX_TOKENS = 1200                   # room for the written reasoning in one call
PROBE_CONTROLS = ("M2S",)                 # exploratory probe, not in the formal plan (02b §15)
GATED = ("E", "C3")
DISPLAY = ("C1", "C2", "C3")              # runtime-kept matched device
ALLOWED_LINE = ("C2", "C3")
DEVICE_CONTROLS = ("N", "Mminus", "M", "MR", "M2S")
MAX_CALLS_PER_TURN = 4
EXTRA_CALLS_DEVICE = 3                    # N / M− / M: + 3 + one per arrival this turn
MAX_BLOCKS_PER_TURN = 2
MAX_FORMAT_ERRORS_PER_EPISODE = 15
MAX_TOKENS = 400

POSITIONAL = {"verify": ("ticket",), "read_ticket": ("ticket",),
              "refund": ("ticket", "ref"), "replace": ("ticket", "ref"), "update_address": ("ticket", "ref"),
              "board_write": ("ticket",), "tray_put": ("ticket",), "rail_add": ("ticket",), "notes_write": ("text",),
              "think": ("thought",),
              "tray1_put": ("ticket",), "tray2_put": ("ticket",)}


def check_condition(rule, control):
    if control not in CONTROLS:
        raise ValueError(f"unknown control {control}")
    if rule == "FREE" and control != "C0":
        raise ValueError("FREE runs only under C0")
    if control == "Mminus" and rule not in devices.LOWER_FOR:
        raise ValueError(f"{rule} has no lower storage class (M−)")
    if control == "M2S" and rule != "FIFO":
        raise ValueError("the two-tray probe M2S runs only with FIFO")


def device_kind_for(rule, control):
    return devices.device_kind(rule, control) if control in DEVICE_CONTROLS else None


def display_line(rule, st):
    """C1–C3: the matched device in its canonical content, read through its access discipline (same text as M)."""
    kind, content = canonical(rule, st)
    return read_line(kind, content)


def allowed_line(rule, st):
    if not st.held:
        return "Allowed to resume now: none (nothing on hold)."
    legal = legal_resume(rule, st)
    if len(legal) == len(st.held) and len(st.held) > 1:
        return "Allowed to resume now: any ticket on hold."
    return "Allowed to resume now: " + ", ".join(sorted(legal)) + "."


def control_lines(rule, control, st, device):
    """Lines pushed at the start of a turn (after the arrivals)."""
    if rule == "FREE" or control in ("C0", "R", "E"):
        return []
    if control in DEVICE_CONTROLS:
        return [read_line(device.kind, device.content)]
    if control == "P":
        return [prompts.placebo_line(len(display_line(rule, st)), st.t)]
    out = [display_line(rule, st)]
    if control in ALLOWED_LINE:
        out.append(allowed_line(rule, st))
    return out


def choice_bits(env, gated):
    """chi_t: log2 of the work actions the agent can actually execute now (distinct tickets it may work on + wait)."""
    st = env.st
    if env.escalation_pending:
        return 0.0
    held = legal_resume(env.rule, st) if gated and env.rule != "FREE" else set(st.held)
    n = len(held) + (st.current is not None) + sum(tk.status == "waiting" for tk in st.tickets.values()) + 1
    return math.log2(n)


def residual_bits(control, snap, device_ok):
    """r_t: decisions the window plus the control's display still leaves open (02b §0.5)."""
    if control in DISPLAY:
        return 0.0
    if control in ("M", "MR", "M2S") and device_ok:
        return 0.0
    return snap["residual_bits_window"]


def _args(name, args):
    """Normalise arguments (named or positional from the text fallback) to (ticket, ref, text)."""
    args = dict(args or {})
    pos = args.pop("_positional", None)
    if pos:
        for k, v in zip(POSITIONAL.get(name, ()), pos):
            args.setdefault(k, v)
    ref = args.get("ref") or args.get("customer_id")
    return args.get("ticket"), ref, args


def device_shows_nothing(device, held):
    """The agent's own device shows nothing on hold although something is (lapse cause 'device_empty')."""
    if device is None or not held:
        return False
    if device.kind in ("stack", "queue"):
        return not device.content
    if device.kind == "twostack":
        return not device.content[0] and not device.content[1]
    if device.kind == "notes":
        return not any(c in device.content for c in held)
    return False


def _user(text, spec):
    return {"role": "user", "content": text + ("\n/no_think" if spec.no_think else "")}


# ====================================================================== offline agents

def device_target(kind, rule, st):
    """What a careful agent keeps in its device (after the action of this turn)."""
    if kind == "notes":
        return (f"On hold, oldest hold first: {' '.join(st.held) or 'none'}. Last put on hold: {st.last_held or '-'}. "
                f"Refunds minus replacements: {st.counter:+d}.")
    if kind == "stack" or kind == "queue":
        return list(st.held)
    if kind == "counter":
        return st.counter if rule == "BAL" else len(st.held)       # M− for LIFO / FIFO: counts what is on hold
    if kind == "marker":
        return st.n_refund % 2
    if kind == "register":
        return st.last_held                                         # LAST (M) and BAL (M−): the latest hold
    raise ValueError(kind)


def device_ops(device, target):
    """Device operations that turn the device's content into target, within its access discipline."""
    kind, cur = device.kind, device.snapshot()
    if kind == "notes":
        return [] if cur == target else [("notes_write", {"text": target})]
    if kind == "register":
        if cur == target:
            return []
        return [("board_write", {"ticket": target})] if target else [("board_erase", {})]
    if kind == "counter":
        n = target - cur
        return [("counter_up", {})] * max(0, n) + [("counter_down", {})] * max(0, -n)
    if kind == "marker":
        return [] if cur == target else [("marker_flip", {})]
    put, take = ("tray_put", "tray_take") if kind == "stack" else ("rail_add", "rail_take")
    if kind == "stack":
        k = 0
        while k < min(len(cur), len(target)) and cur[k] == target[k]:
            k += 1
        return [(take, {})] * (len(cur) - k) + [(put, {"ticket": c}) for c in target[k:]]
    for j in range(len(cur) + 1):                                # queue: take j from the front, add the rest
        rest = cur[j:]
        if target[:len(rest)] == rest:
            return [(take, {})] * j + [(put, {"ticket": c}) for c in target[len(rest):]]
    raise AssertionError("unreachable")


def twostack_ops(device, env, act):
    """A careful agent with two trays under FIFO (probe M2S): holds go on tray 1; to resume the earliest hold it takes
    from tray 2, first moving all of tray 1 onto tray 2 when tray 2 is empty (the classic queue from two stacks)."""
    st, ops = env.st, []
    one, two = list(device.content[0]), list(device.content[1])
    code = act[1]
    tk = st.tickets.get(code) if code else None
    if tk is not None and tk.status == "held":
        if not two:
            while one:
                ops.append(("move_1_to_2", {}))
                two.append(one.pop())
        if two and two[-1] == code:
            ops.append(("tray2_take", {}))
    if code and st.current and code != st.current:
        ops.append(("tray1_put", {"ticket": st.current}))
    return ops


def device_read_pick(device, env, rng):
    """A device-only agent: decides the resume from its own device (and the visible hold set) alone (02b §10.12)."""
    st, kind, c = env.st, device.kind, device.content
    held = sorted(st.held)
    if kind == "twostack":                # the earliest hold: top of tray 2, or the bottom of tray 1 once moved over
        one, two = c
        if two and two[-1] in st.held:
            return two[-1]
        if not two and one and one[0] in st.held:
            return one[0]
    if kind == "stack" and c and c[-1] in st.held:
        return c[-1]
    if kind == "queue" and c and c[0] in st.held:
        return c[0]
    if kind == "counter" and env.rule == "BAL":
        return held[0] if c >= 0 else held[-1]
    if kind == "marker" and env.rule == "PAR":
        return held[0] if c == 0 else held[-1]
    if kind == "register" and env.rule == "LAST":
        return c if c in st.held else held[0]
    if env.rule in ("BAL", "PAR"):     # the device does not determine the side: guess first or last
        return rng.choice([held[0], held[-1]])
    return rng.choice(held)            # the device does not determine the ticket


class FakeToolBackend:
    """Offline agent. policy = [view:][text:][dev:][stubborn:]<strategy> | malformed | flaky:<p>:<policy>.
    view: calls view_queue before acting each turn; forget: waits at every resume opportunity unless reminded this
    turn; text: writes the action as text (no tool call); dev: keeps its
    device (N / M− / M / M2S) in the careful encoding, in the same reply as the action; stubborn: repeats a blocked
    resume.
    Strategies: strategies.make_chooser names, or devread (decides resumes from the device only)."""

    def __init__(self, spec, policy="compliant", seed=0):
        self.spec, self.policy = spec, policy
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.n = 0
        self.choosers = {}

    def _chooser(self, strat, rule, seed):
        key = (strat, rule, seed)
        if key not in self.choosers:
            self.choosers[key] = make_chooser(strat, rule, seed)
        return self.choosers[key]

    def call(self, messages, max_tokens, meta=None):
        meta = meta or {}
        pol = self.policy
        with self.lock:
            r = self.rng.random()
            self.n += 1
            cid = f"fake_{self.n}"
        if pol.startswith("flaky:"):
            _, p, pol = pol.split(":", 2)
            if r < float(p):
                raise RuntimeError("LLM API call failed: HTTP 503: simulated outage")
        content, calls = "", []
        if pol == "malformed":
            content = "Let me think about which ticket to take."
        else:
            mods = set()
            while ":" in pol and pol.split(":", 1)[0] in ("view", "text", "dev", "stubborn", "forget", "think"):
                m, pol = pol.split(":", 1)
                mods.add(m)
            env, device = meta["env"], meta.get("device")
            if "think" in mods and not meta.get("thought"):
                calls = [(cid, "think", {"thought": "Going through what is on hold before I act."})]
            elif "view" in mods and not meta.get("viewed"):
                calls = [(cid, "view_queue", {})]
            else:
                act = self._act(pol, mods, env, device, meta)
                if "text" in mods:
                    content = "Action: " + (f"{act[0]}({', '.join(a for a in act[1:] if a)})")
                else:
                    if "dev" in mods and device is not None and not meta.get("device_done"):
                        after = env.copy()
                        try:
                            after.work(*act)
                        except EnvError:
                            pass
                        ops = (twostack_ops(device, env, act) if device.kind == "twostack" else
                               device_ops(device, device_target(device.kind, env.rule, after.st)))
                        calls = [(f"{cid}_d{i}", n, a) for i, (n, a) in enumerate(ops)]
                    calls.append((cid, act[0], {k: v for k, v in (("ticket", act[1]), ("ref", act[2])) if v}))
        am = {"role": "assistant", "content": content or None}
        if calls:
            am["tool_calls"] = [tool_call_json(n, a, i) for i, n, a in calls]
        p_tok = int(chars_of(messages) / 3.8) + 6 * len(messages) + int(len(json.dumps(meta.get("tools") or [])) / 3.8)
        c_tok = max(5, int((len(content) + 40 * len(calls)) / 3.8))
        cost = (p_tok * self.spec.input_price + c_tok * self.spec.output_price) / 1e6
        return ToolReply(content, calls, p_tok, c_tok, 0, cost, self.spec.provider, 0.0, am)

    def _act(self, strat, mods, env, device, meta):
        st = env.st
        rule = env.rule
        if ("forget" in mods and st.current is None and st.held and not env.escalation_pending
                and not meta.get("reminded")):
            return ("wait", None, None)            # forgets what is on hold until reminded (tests of the reminder)
        if strat == "devread":
            if env.escalation_pending or st.current or not st.held:
                act = self._chooser("compliant", rule, 0)(env, {})
            else:
                view = device
                if "dev" in mods and view.kind != "twostack":   # record this turn's arrivals first, then read
                    view = copy.deepcopy(device)
                    valid = {c for c, tk in st.tickets.items() if tk.status != "pending"}
                    for name, args in device_ops(view, device_target(view.kind, rule, st)):
                        view.apply(name, args, valid)
                act = next_step_action(st.tickets[device_read_pick(view, env, self.rng)])
        else:
            act = self._chooser(strat, rule, meta.get("seed", 0))(env, {"control": meta.get("control", "C0")})
        blocked = meta.get("blocked_codes") or []
        if act[1] in blocked and "stubborn" not in mods:
            rest = sorted(c for c in st.held if c not in blocked)
            if rest:
                act = next_step_action(st.tickets[self.rng.choice(rest)])
            else:
                waiting = sorted(c for c, tk in st.tickets.items() if tk.status == "waiting")
                act = next_step_action(st.tickets[waiting[0]]) if waiting else ("wait", None, None)
        return act


# ====================================================================== one episode

def run_episode(schedule, rule, control, backend, spec, ledger=None, cell=None, seed=0, reminders=True):
    check_condition(rule, control)
    wording = (cell or {}).get("wording", 0)
    gated = control in GATED
    env = DeskEnv(schedule, rule, gate=gated)
    kind = device_kind_for(rule, control)
    device = devices.Device(kind) if kind else None
    device_tools = set(devices.OPS[kind]) if kind else set()
    device_desc = devices.device_description(kind) if kind else None
    system = prompts.system_prompt(rule, control, device_desc, wording)
    tools = prompts.BASE_TOOLS + (devices.tool_schemas(kind) if kind else []) + \
        ([prompts.THINK_TOOL] if control in THINK_CONTROLS else [])
    max_tokens = THINK_MAX_TOKENS if control in THINK_CONTROLS else MAX_TOKENS
    tool_names = tuple(t["function"]["name"] for t in tools)
    rec = {"cell": cell or {}, "contract_version": CONTRACT_VERSION, "task": "desk", "model": spec.model_id,
           "provider_expected": spec.provider, "rule": rule, "control": control, "wording": wording, "device": kind,
           "schedule": schedule["key"], "D": schedule["D"], "high_excursion": schedule.get("high_excursion"),
           "calls": [], "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "flops": 0.0, "service_errors": 0,
           "format_errors": 0, "text_fallbacks": 0, "env_errors": 0, "device_ops": 0, "device_errors": 0,
           "thinks": 0, "think_chars": 0, "termination": None, "turns": []}
    call = MeteredCaller(backend, spec, ledger, rec)
    messages = [{"role": "system", "content": system}]
    # control text in the context (for the cost split): the control paragraph of the system prompt, the device's
    # tool schemas, pushed lines, device results and gate messages
    ctrl = len(system) - len(prompts.system_prompt(rule, "C0", wording=wording))
    if kind:
        ctrl += len(json.dumps(devices.tool_schemas(kind)))

    rs = {"remind_next": False, "n": 0, "first": None, "aided": set(), "prev_blocked": False}

    def finish(termination=None):
        rec["termination"] = termination
        rec["completed"] = termination is None
        rec.update(env.finish())
        tks = env.st.tickets
        rec["reminders"] = rs["n"]
        rec["first_reminder_t"] = rs["first"]
        rec["aided"] = sorted(rs["aided"])
        rec["completion_unaided"] = sum(tk.status == "closed" and c not in rs["aided"] for c, tk in tks.items()) / len(tks)
        rec["events"] = [list(e) for e in env.events]
        rec["transcript"] = messages
        return rec

    try:
        while not env.done():
            n_events = len(env.events)
            lines = env.start_turn()
            arrivals_now = [e[1] for e in env.events[n_events:] if e[0] == "a"]
            snap = env.snapshot()
            chi = choice_bits(env, gated)
            st_before = copy.copy(env.st)
            st_before.held = list(env.st.held)
            device_ok_start = matches(rule, device, env.st)
            device_empty = device_shows_nothing(device, env.st.held)
            oldest_age = (env.st.t - min(env.st.tickets[c].hold_time for c in env.st.held)) if env.st.held else None
            reminded = reminders and rs["remind_next"] and bool(env.st.held)
            if reminded:
                rs["n"] += 1
                rs["first"] = env.st.t if rs["first"] is None else rs["first"]
                rs["aided"] |= set(env.st.held)
            pushed = control_lines(rule, control, env.st, device)
            push_chars = sum(len(x) + 1 for x in pushed)
            ctrl += push_chars
            messages.append(_user(prompts.turn_message(env.st.t, lines, pushed,
                                                       prompts.REMINDER if reminded else None), spec))
            budget = MAX_CALLS_PER_TURN + (EXTRA_CALLS_DEVICE + len(lines) if device else 0)
            if kind == "twostack":                # moving one tray onto the other takes one call per code
                budget += len(env.st.held)
            if control in THINK_CONTROLS:
                budget += THINK_EXTRA_CALLS
            tm = {"calls": 0, "tool_calls": 0, "info_calls": 0, "viewed": False, "env_errors": 0, "format_errors": 0,
                  "text_fallback": 0, "blocked": [], "forced": False, "device_ops": 0, "device_errors": 0,
                  "push_chars": push_chars, "pushed": pushed, "thinks": 0}
            info = None
            while tm["calls"] < budget and info is None:
                meta = {"env": env, "device": device, "tools": tools, "viewed": tm["viewed"], "control": control,
                        "blocked_codes": [b["code"] for b in tm["blocked"]], "seed": seed,
                        "device_done": tm["device_ops"] > 0, "prompt_chars": chars_of(messages),
                        "control_chars": ctrl, "reminded": reminded, "thought": tm["thinks"] > 0}
                rep = call(messages, max_tokens, meta)
                tm["calls"] += 1
                calls = rep.tool_calls
                am = dict(rep.assistant_message)
                messages.append(am)
                if not calls:
                    calls = text_calls(rep.content, tool_names)
                    if calls:
                        tm["text_fallback"] += 1
                        rec["text_fallbacks"] += 1
                    else:
                        tm["format_errors"] += 1
                        rec["format_errors"] += 1
                        messages.append(_user(prompts.NO_TOOL_HINT, spec))
                        if rec["format_errors"] > MAX_FORMAT_ERRORS_PER_EPISODE:
                            return finish("format")
                        continue
                tm["tool_calls"] += len(calls)
                results = []
                for cid, name, args in calls:
                    if info is not None:
                        results.append((cid, prompts.IGNORED_RESULT))
                        continue
                    ticket, ref, a = _args(name, args)
                    if name == "view_queue":
                        tm["viewed"] = True
                        tm["info_calls"] += 1
                        results.append((cid, env.view_queue()))
                    elif name == "read_ticket":
                        tm["info_calls"] += 1
                        try:
                            results.append((cid, env.read_ticket(ticket)))
                        except EnvError as err:
                            tm["env_errors"] += 1
                            rec["env_errors"] += 1
                            results.append((cid, f"Not possible: {err}."))
                    elif name == "think" and control in THINK_CONTROLS:
                        tm["thinks"] += 1
                        rec["thinks"] += 1
                        rec["think_chars"] += len(str(a.get("thought") or ""))
                        results.append((cid, prompts.THINK_RESULT))
                    elif name in device_tools:
                        valid = {c for c, tk in env.st.tickets.items() if tk.status != "pending"}
                        try:
                            out = device.apply(name, a, valid)
                            tm["device_ops"] += 1
                            rec["device_ops"] += 1
                        except devices.DeviceError as err:
                            out = f"Not possible: {err}."
                            tm["device_errors"] += 1
                            rec["device_errors"] += 1
                        ctrl += len(out)
                        results.append((cid, out))
                    elif name in WORK_TOOLS:
                        try:
                            status, text, res = env.work(name, ticket, ref)
                        except EnvError as err:
                            tm["env_errors"] += 1
                            rec["env_errors"] += 1
                            results.append((cid, f"Not done: {err}. No time passed."))
                            continue
                        if status == "blocked":
                            tm["blocked"].append({"code": res["code"], "t": env.st.t})
                            if len(tm["blocked"]) >= MAX_BLOCKS_PER_TURN:
                                status, _, res = env.forced_wait()
                                res["forced"] = True
                                tm["forced"] = True
                                info = res
                                text = prompts.BLOCKED_FINAL
                            ctrl += len(text)
                            results.append((cid, text))
                        else:
                            info = res
                            results.append((cid, text))
                    else:
                        tm["format_errors"] += 1
                        rec["format_errors"] += 1
                        results.append((cid, f"Unknown tool {name!r}."))
                if "tool_calls" in am:
                    for cid, r in results:
                        messages.append({"role": "tool", "tool_call_id": cid, "content": r})
                elif results:
                    messages.append(_user("\n".join(r for _, r in results), spec))
            if info is None:                     # call budget used up without a work action
                tm["forced"] = True
                _, text, info = env.forced_wait()
                info["forced"] = True
                messages.append(_user("No step was taken this turn. " + text, spec))
            trec = {k: v for k, v in snap.items()}
            trec.update({"event_index": n_events + len(arrivals_now),          # history the agent has seen
                         "arrivals_now": arrivals_now, "action": info.get("action"), "code": info.get("code"),
                         "resume": info.get("resume"), "legal": info.get("legal"), "label_ok": info.get("label_ok"),
                         "hold": info.get("hold")})
            trec.update({k: v for k, v in tm.items() if k != "viewed"})
            trec["deferred"] = snap["opportunity"] and info.get("resume") is None
            trec["diagnostic"] = snap["diagnostic_candidate"] and (info.get("resume") is not None or bool(tm["blocked"]))
            # first attempt at this resume opportunity: a blocked attempt counts as an attempted violation
            trec["first_resume_legal"] = (False if tm["blocked"] else info.get("legal"))
            trec["attempted_violation"] = bool(tm["blocked"]) or info.get("legal") is False
            trec["device_content"] = device.snapshot() if device else None
            trec["device_ok_start"] = device_ok_start if device is not None else None
            trec["device_consistent"] = consistent(rule, device, st_before, env.st) if device is not None else None
            trec["choice_bits"] = chi
            trec["residual_bits"] = residual_bits(control, snap, device_ok_start)
            lapse = (info.get("action") == "wait" and not tm["forced"] and snap["current"] is None
                     and bool(snap["held"]) and not snap["escalation_pending"])
            cause = None
            if lapse:
                cause = ("after_block" if tm["blocked"] or rs["prev_blocked"] else
                         "device_empty" if device_empty else "plain")
            trec.update({"reminded": reminded, "after_first_reminder": rs["first"] is not None, "lapse": lapse,
                         "lapse_cause": cause, "oldest_hold_age": oldest_age})
            rs["remind_next"] = lapse
            rs["prev_blocked"] = bool(tm["blocked"])
            rec["turns"].append(trec)
        return finish(None)
    except ServiceFailure as e:
        rec["service_error_message"] = str(e)
        return finish("service")


def run_workflow(schedule, rule):
    """W: the program schedules by the procedure itself (no model, chi = 0); the reference for completion and cost."""
    env, recs = simulate(schedule, rule, make_chooser("reference", rule))
    out = {"task": "desk", "model": "workflow", "rule": rule, "control": "W", "schedule": schedule["key"],
           "D": schedule["D"], "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "termination": None,
           "completed": True, "turns": recs}
    out.update(env.finish())
    return out
