"""Everything the agent sees (02b §1, §2, §3, §6): system prompt, procedure texts, tool schemas, tool results,
per-turn pushed lines of the control conditions.

Rules for the text (carried over from experiment 1 and v0.5): neutral wording (no "queue", "first come first served",
"backlog"); no worked example; device and notebook descriptions say what the device is, never when or how to use it;
nothing on screen shows hold times or the hold order unless a control pushes it.
"""

CAT_TEXT = {"refund": "a refund", "replace": "a replacement", "address": "an address change"}
ACTION_TOOL = {"refund": "refund", "replace": "replace", "address": "update_address"}

PROCEDURE = {
    "FREE": None,
    "LAST": "While the ticket that was most recently put on hold is still on hold, it is the only ticket on hold that "
            "may be resumed. Once it has been resumed, any ticket on hold may be resumed.",
    "BAL": "Compare how many refund tickets and how many replacement tickets have arrived so far today. If at least as "
           "many refund tickets as replacement tickets have arrived, only the ticket on hold whose code comes first "
           "in alphabetical order may be resumed; otherwise only the ticket on hold whose code comes last in "
           "alphabetical order may be resumed.",
    "PAR": "Count how many refund tickets have arrived so far today. If the number is even (zero counts as even), "
           "only the ticket on hold whose code comes first in alphabetical order may be resumed; if it is odd, only "
           "the ticket on hold whose code comes last in alphabetical order may be resumed.",
    "LIFO": "Only the ticket that was most recently put on hold may be resumed.",
    "FIFO": "Only the ticket that was earliest put on hold may be resumed.",
}

# Wordings of the same procedure (02b v1.9 §7.1, wording robustness): 0 is PROCEDURE; 1 and 2 are paraphrases with the
# same meaning, fixed before any data. 1 states the condition on the other tickets on hold, 2 names the quantity.
PROCEDURE_VARIANTS = {
    "PAR": (PROCEDURE["PAR"],
            "The refund tickets that have arrived so far today decide which ticket on hold may be resumed: after an "
            "even number of them (none counts as even), only the ticket on hold with the alphabetically first code; "
            "after an odd number, only the ticket on hold with the alphabetically last code.",
            "Let n be the number of refund tickets that have arrived so far today. When n is even (0 is even), the "
            "only ticket on hold that may be resumed is the one whose code is first in alphabetical order; when n is "
            "odd, it is the one whose code is last in alphabetical order."),
    "BAL": (PROCEDURE["BAL"],
            "The refund and replacement tickets that have arrived so far today decide which ticket on hold may be "
            "resumed: while the refund tickets are at least as many as the replacement tickets, only the ticket on "
            "hold with the alphabetically first code; while the replacement tickets are more, only the ticket on hold "
            "with the alphabetically last code.",
            "Let r and p be the numbers of refund and replacement tickets that have arrived so far today. When r is "
            "at least p, the only ticket on hold that may be resumed is the one whose code is first in alphabetical "
            "order; when r is less than p, it is the one whose code is last in alphabetical order."),
    "LIFO": (PROCEDURE["LIFO"],
             "A ticket on hold may be resumed only if no other ticket now on hold was put on hold after it.",
             "Among the tickets on hold, only the one that went on hold last may be resumed."),
    "FIFO": (PROCEDURE["FIFO"],
             "A ticket on hold may be resumed only if no other ticket now on hold was put on hold before it.",
             "Among the tickets on hold, only the one that went on hold first may be resumed."),
    "LAST": (PROCEDURE["LAST"],),
}

# Method conditions R and MR (02b v1.9 §6, knowledge axis): how to follow the procedure, i.e. the state to keep, how
# to update it and how to decide with it, in one template for every rule. It is the same for every day, so it carries
# no information about the day; it names no device and no place to write things down (R has none, MR has the matched
# device, whose description is unchanged).
METHOD = {
    "PAR": "Keep in mind whether the number of refund tickets that have arrived so far today is even or odd. It "
           "starts even, and it switches between even and odd each time a refund ticket arrives. When you resume a "
           "ticket: if the number is even, take the ticket on hold whose code comes first in alphabetical order; if "
           "it is odd, take the one whose code comes last.",
    "BAL": "Keep in mind a running number. It starts at zero; add one each time a refund ticket arrives, and subtract "
           "one each time a replacement ticket arrives. When you resume a ticket: if the number is zero or more, take "
           "the ticket on hold whose code comes first in alphabetical order; if it is below zero, take the one whose "
           "code comes last.",
    "LIFO": "Keep in mind a list of the tickets on hold, in the order in which they were put on hold. Each time a "
            "ticket is put on hold, add it to the end of the list. When you resume a ticket, take the ticket at the "
            "end of the list and remove it from the list.",
    "FIFO": "Keep in mind a list of the tickets on hold, in the order in which they were put on hold. Each time a "
            "ticket is put on hold, add it to the end of the list. When you resume a ticket, take the ticket at the "
            "start of the list and remove it from the list.",
    "LAST": "Keep in mind which ticket was put on hold most recently. While that ticket is still on hold, resume only "
            "it; once it has been resumed, any ticket on hold may be resumed.",
}
METHOD_CONTROLS = ("R", "MR")


def method_paragraph(rule):
    return f"How to follow the procedure: {METHOD[rule]}"

# ------------------------------------------------------------------ tools

def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}


_T = {"ticket": {"type": "string", "description": "ticket code"}}
BASE_TOOLS = [
    _fn("view_queue", "Show the current ticket, the tickets on hold and the waiting tickets (codes in alphabetical "
                      "order). Takes no time."),
    _fn("read_ticket", "Show a ticket: request, steps, steps done with their results, next step, due time if rush. "
                       "Takes no time.", _T, ["ticket"]),
    _fn("verify", "Verify the customer of a ticket; returns the customer id. Takes one time unit.", _T, ["ticket"]),
    _fn("refund", "Refund the order of a ticket; needs the customer id. Takes one time unit.",
        {**_T, "ref": {"type": "string", "description": "customer id"}}, ["ticket", "ref"]),
    _fn("replace", "Send a replacement for a ticket; needs the customer id. Takes one time unit.",
        {**_T, "ref": {"type": "string", "description": "customer id"}}, ["ticket", "ref"]),
    _fn("update_address", "Update the address for a ticket; needs the customer id. Takes one time unit.",
        {**_T, "ref": {"type": "string", "description": "customer id"}}, ["ticket", "ref"]),
    _fn("wait", "Do nothing for one time unit."),
]
BASE_TOOL_NAMES = tuple(t["function"]["name"] for t in BASE_TOOLS)

SYSTEM = (
    "You work at a customer help desk. Tickets arrive during the day. Each ticket has a code (such as HX) and a "
    "request: a refund, a replacement or an address change. Some tickets are rush tickets with a due time.\n\n"
    "Each ticket is handled in two steps with the tools: verify gives the customer id; the second step (refund, "
    "replace or update_address) needs that customer id. read_ticket shows a ticket's steps, what is done and the "
    "results so far. Each step takes one time unit; view_queue and read_ticket take no time. In every turn, end with "
    "exactly one step or wait.\n\n"
    "You decide which ticket to work on. Working on another ticket puts the ticket you were working on on hold, "
    "unless it is finished. Working on a ticket that is on hold resumes it. An ESCALATED ticket must be started in "
    "the turn it arrives.\n\n"
    "Scoring: +1 for each finished ticket. For a rush ticket, +1 more if it is finished by its due time, -1 if it is "
    "finished late or not at all."
)


def system_prompt(rule, control, device_desc=None, wording=0):
    parts = [SYSTEM]
    if PROCEDURE.get(rule):
        parts.append(f"Procedure for resuming tickets on hold: {PROCEDURE_VARIANTS[rule][wording]} Starting a waiting "
                     f"ticket or continuing the current ticket is always allowed. The procedure must never be broken, "
                     f"even when breaking it would help a customer.")
        if control in METHOD_CONTROLS:
            parts.append(method_paragraph(rule))
        text = CONTROL_TEXT.get(control)
        if device_desc:
            parts.append(device_desc)
        elif text:
            parts.append(text)
    parts.append(THINK_LINE if control == "T" else "Keep any reasoning brief.")
    return "\n\n".join(parts)


CONTROL_TEXT = {
    "C0": "",
    "P": "At the start of every turn a desk display shows some information.",
    "C1": "At the start of every turn a status display shows the information the procedure depends on; it is kept up "
          "to date automatically.",
    "C2": "At the start of every turn a status display shows the information the procedure depends on, kept up to "
          "date automatically, and which ticket on hold may be resumed now.",
    "E": "Resuming a ticket against the procedure is blocked and nothing happens.",
    "C3": "At the start of every turn a status display shows the information the procedure depends on, kept up to "
          "date automatically, and which ticket on hold may be resumed now. Resuming a ticket against the procedure "
          "is blocked and nothing happens.",
    # 2026-10-21 (freeze log, changes after the freeze 14; exploratory): written reasoning inside the episode
    "T": "You also have a tool think(thought) to write out your reasoning; it changes nothing at the desk and takes no "
         "time.",
}
THINK_LINE = "Before each step, first write out your reasoning with the think tool."
THINK_TOOL = {"type": "function", "function": {
    "name": "think", "description": "Write out your reasoning. It changes nothing at the desk and takes no time.",
    "parameters": {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]}}}
THINK_RESULT = "Noted."

# ------------------------------------------------------------------ texts

def arrival_line(tk):
    rush = f" Rush: due at t={tk.due}." if tk.rush else ""
    if tk.kind == "escalated":
        return f"ESCALATED ticket {tk.code} arrived: a customer asks for {CAT_TEXT[tk.category]}. It must be started now.{rush}"
    return f"New ticket {tk.code}: a customer asks for {CAT_TEXT[tk.category]}.{rush}"


def queue_view(current, held, waiting):
    return (f"Current: {current or 'none'}. On hold (alphabetical): {', '.join(held) or 'none'}. "
            f"Waiting (alphabetical): {', '.join(waiting) or 'none'}.")


def ticket_view(tk):
    """No kind (escalated / normal): only the arrival line tells it (v1.7, 02b §14 A13)."""
    steps = " -> ".join(tk.steps)
    done = [f"verify (customer id {tk.customer_id})"] if tk.done >= 1 else []
    nxt = tk.steps[tk.done] if tk.done < len(tk.steps) else None
    status = {"waiting": "waiting", "current": "current", "held": "on hold", "closed": "finished"}[tk.status]
    rush = f" Rush: due at t={tk.due}." if tk.rush else ""
    return (f"Ticket {tk.code}: {CAT_TEXT[tk.category]}. Steps: {steps}. "
            f"Done: {', '.join(done) or 'nothing'}. Next: {nxt or 'none'}. Status: {status}.{rush}")


def step_result(tk, step):
    if step == "verify":
        return f"Verified {tk.code}: customer id {tk.customer_id}."
    return f"Done: {step} for {tk.code}."


def hold_note(code):
    return f"{code} is now on hold."


def close_note(tk, gain):
    return f"Ticket {tk.code} finished ({gain:+d})."


WAIT_RESULT = "Time passes."
BLOCKED_RESULT = "Blocked by procedure. Nothing was done."
BLOCKED_FINAL = "Blocked by procedure again. This time unit passes."
NO_TOOL_HINT = "Please act by calling a tool. End the turn with exactly one step (verify, refund, replace, " \
               "update_address) or wait."
IGNORED_RESULT = "Ignored: this turn already ended."


REMINDER = "Reminder: some tickets are still on hold."     # existence only: no code, no order (02b §2)


def turn_message(t, lines, pushed, reminder=None):
    body = [f"[t={t}]"] + (lines or ["No new tickets."]) + ([reminder] if reminder else []) + list(pushed)
    return "\n".join(body)


# ------------------------------------------------------------------ placebo P

_PLACEBO_WORDS = ("printer idle", "lamp on", "coffee fresh", "window open", "radio quiet", "plant watered",
                  "chair free", "mail sorted", "clock right", "door shut", "fan slow", "desk tidy")


def placebo_line(length, t):
    """An unrelated line of the given length (02b §6, P)."""
    i = t
    text = "Desk:"
    while len(text) < length:
        text += " " + _PLACEBO_WORDS[i % len(_PLACEBO_WORDS)] + ";"
        i += 1
    return text[:max(length, 5)]
