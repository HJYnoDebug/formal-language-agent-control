"""Storage devices for the controls N (unrestricted notebook), M (device of the procedure's class) and M− (one class
lower), and the runtime-maintained display of C1 / C2 / C3 (02b §6).

Restricted access as in the automaton: a stack shows only its top, a queue only its front, a counter only its sign, a
register its one code. The read of a device is pushed at the start of every turn (M−, M: the agent's device; C1–C3:
the same device kept up to date by the runtime), so M -> C1 changes only who performs the update delta.

Argument checks keep each device in its class: the register and the stack / queue accept only codes of tickets that
exist in this episode (a register that accepted any string could hold an unbounded number or list); the counter only
moves by one.

Probe M2S (02b §15, exploratory): FIFO with two trays (two stacks) instead of the rail. Two stacks can hold a queue
(the classic queue from two stacks), so the capacity is enough, but the access discipline does not match the
procedure: the earliest hold is reached only after moving one tray onto the other.
"""
DEVICE_FOR = {"LAST": "register", "PAR": "marker", "BAL": "counter", "LIFO": "stack", "FIFO": "queue"}
LOWER_FOR = {"BAL": "register", "LIFO": "counter", "FIFO": "counter"}   # one class lower (M−); LAST / PAR have none
NOTES_CAP = 600

OPS = {
    "register": {"board_write": ("write", "ticket"), "board_erase": ("erase", None), "board_read": ("read", None)},
    "marker": {"marker_flip": ("flip", None), "marker_look": ("look", None)},
    "counter": {"counter_up": ("up", None), "counter_down": ("down", None), "counter_sign": ("sign", None)},
    "stack": {"tray_put": ("put", "ticket"), "tray_take": ("take", None), "tray_look": ("look", None)},
    "queue": {"rail_add": ("put", "ticket"), "rail_take": ("take", None), "rail_look": ("look", None)},
    "notes": {"notes_write": ("write", "text"), "notes_read": ("read", None)},
    "twostack": {"tray1_put": ("put1", "ticket"), "tray1_take": ("take1", None), "tray2_put": ("put2", "ticket"),
                 "tray2_take": ("take2", None), "move_1_to_2": ("move12", None), "move_2_to_1": ("move21", None),
                 "trays_look": ("look", None)},
}
DESC = {
    "register": "a note board (board_write(ticket), board_erase(), board_read()) that holds at most one ticket code. "
                "Its content is shown to you at the start of every turn.",
    "marker": "an even/odd marker (marker_flip(), marker_look()) that shows either even or odd and starts at even. "
              "It is shown to you at the start of every turn.",
    "counter": "a balance counter (counter_up(), counter_down(), counter_sign()) that moves up or down by one and "
               "shows only whether it is above, at or below zero. Its sign is shown to you at the start of every turn.",
    "stack": "a ticket tray (tray_put(ticket), tray_take(), tray_look()): codes can only be put on top or taken from "
             "the top, and only the top code is visible. Its top is shown to you at the start of every turn.",
    "queue": "a ticket rail (rail_add(ticket), rail_take(), rail_look()): codes are added at the back and taken from "
             "the front, and only the front code is visible. Its front is shown to you at the start of every turn.",
    "notes": f"a notebook (notes_write(text); up to {NOTES_CAP} characters; writing replaces what was there). Its "
             f"content is shown to you at the start of every turn.",
    "twostack": "two ticket trays, tray 1 and tray 2 (tray1_put(ticket), tray1_take(), tray2_put(ticket), "
                "tray2_take(), move_1_to_2(), move_2_to_1(), trays_look()): codes can only be put on top of a tray or "
                "taken from its top, move_1_to_2() takes the top code of tray 1 and puts it on top of tray 2 (and "
                "move_2_to_1() the reverse), and only the top code of each tray is visible. Their tops are shown to "
                "you at the start of every turn.",
}
TOOL_DESC = {
    "board_write": "Write a ticket code on the note board (replaces what was there).",
    "board_erase": "Wipe the note board.", "board_read": "Read the note board.",
    "marker_flip": "Switch the marker between even and odd.", "marker_look": "Look at the marker.",
    "counter_up": "Move the balance counter up by one.", "counter_down": "Move the balance counter down by one.",
    "counter_sign": "See whether the balance counter is above, at or below zero.",
    "tray_put": "Put a ticket code on top of the tray.", "tray_take": "Take the top code off the tray.",
    "tray_look": "Look at the top code of the tray.",
    "rail_add": "Add a ticket code at the back of the rail.", "rail_take": "Take the front code off the rail.",
    "rail_look": "Look at the front code of the rail.",
    "notes_write": "Write text in the notebook (replaces what was there).", "notes_read": "Read the notebook.",
    "tray1_put": "Put a ticket code on top of tray 1.", "tray1_take": "Take the top code off tray 1.",
    "tray2_put": "Put a ticket code on top of tray 2.", "tray2_take": "Take the top code off tray 2.",
    "move_1_to_2": "Take the top code of tray 1 and put it on top of tray 2.",
    "move_2_to_1": "Take the top code of tray 2 and put it on top of tray 1.",
    "trays_look": "Look at the top codes of both trays.",
}


class DeviceError(Exception):
    pass


def device_kind(rule, control):
    if control == "N":
        return "notes"
    if control in ("M", "MR"):
        return DEVICE_FOR[rule]
    if control == "Mminus":
        return LOWER_FOR.get(rule)
    if control == "M2S":
        return "twostack"
    return None


def device_description(kind):
    return (f"You also have {DESC[kind]} It holds only what you put in it, and nothing changes it except your own "
            f"actions. You may use it to keep track of whatever you need.")


def tool_schemas(kind):
    out = []
    for name, (_, arg) in OPS[kind].items():
        props, req = {}, []
        if arg:
            props = {arg: {"type": "string", "description": "ticket code" if arg == "ticket" else "the text"}}
            req = [arg]
        out.append({"type": "function", "function": {"name": name, "description": TOOL_DESC[name],
                                                      "parameters": {"type": "object", "properties": props,
                                                                     "required": req}}})
    return out


class Device:
    def __init__(self, kind):
        self.kind = kind
        self.content = {"register": None, "marker": 0, "counter": 0, "stack": [], "queue": [], "notes": "",
                        "twostack": [[], []]}[kind]
        self.ops = 0

    def snapshot(self):
        if self.kind == "twostack":
            return [list(self.content[0]), list(self.content[1])]
        return list(self.content) if isinstance(self.content, list) else self.content

    def apply(self, tool, args, valid_codes):
        op, arg = OPS[self.kind][tool]
        val = (args or {}).get(arg) if arg else None
        if arg == "ticket":
            val = str(val or "").strip().upper()
            if val not in valid_codes:
                raise DeviceError(f"{val or '?'} is not a ticket code")
        if self.kind == "notes":
            if op == "write":
                text = str(val or "")
                if len(text) > NOTES_CAP:
                    raise DeviceError(f"the notebook holds at most {NOTES_CAP} characters")
                self.content = text
                out = "Written."
            else:
                out = read_line("notes", self.content)
        elif self.kind == "register":
            if op == "write":
                self.content = val
            elif op == "erase":
                self.content = None
            out = read_line("register", self.content)
        elif self.kind == "marker":
            if op == "flip":
                self.content = 1 - self.content
            out = read_line("marker", self.content)
        elif self.kind == "counter":
            self.content += {"up": 1, "down": -1, "sign": 0}[op]
            out = read_line("counter", self.content)
        elif self.kind == "twostack":
            one, two = self.content
            if op in ("put1", "put2"):
                (one if op == "put1" else two).append(val)
                out = f"{val} added to tray {op[-1]}."
            elif op == "look":
                out = read_line("twostack", self.content)
            else:
                src = one if op in ("take1", "move12") else two
                if not src:
                    raise DeviceError("that tray is empty")
                c = src.pop()
                if op == "move12":
                    two.append(c)
                    out = f"Moved {c} to tray 2."
                elif op == "move21":
                    one.append(c)
                    out = f"Moved {c} to tray 1."
                else:
                    out = f"Took {c}."
        else:
            if op == "put":
                self.content.append(val)
                out = f"{val} added."
            elif not self.content:
                raise DeviceError("it is empty")
            elif op == "take":
                c = self.content.pop(-1 if self.kind == "stack" else 0)
                out = f"Took {c}."
            else:
                out = read_line(self.kind, self.content)
        self.ops += 1
        return out


def read_line(kind, content):
    """What the device shows (its access discipline); pushed at the start of every turn."""
    if kind == "register":
        return f"Board: {content}." if content else "Board: blank."
    if kind == "marker":
        return "Marker: odd." if content else "Marker: even."
    if kind == "counter":
        s = "above zero" if content > 0 else "at zero" if content == 0 else "below zero"
        return f"Balance counter: {s}."
    if kind == "stack":
        return f"Tray (top): {content[-1]}." if content else "Tray: empty."
    if kind == "queue":
        return f"Rail (front): {content[0]}." if content else "Rail: empty."
    if kind == "notes":
        return f"Notebook: {content}" if content else "Notebook: empty."
    if kind == "twostack":
        return " ".join(f"Tray {i} (top): {t[-1]}." if t else f"Tray {i}: empty." for i, t in ((1, content[0]),
                                                                                              (2, content[1])))
    raise ValueError(kind)


def canonical(rule, st):
    """The procedure's state in the canonical encoding of its matched device (runtime display of C1–C3, and the
    reference for the mechanism check of M)."""
    kind = DEVICE_FOR[rule]
    if kind in ("stack", "queue"):
        return kind, list(st.held)
    if kind == "counter":
        return kind, st.counter
    if kind == "marker":
        return kind, st.n_refund % 2
    return kind, st.last_held


def logical(rule, device):
    """The content the procedure's state is compared with: the device itself for the matched device; for the two trays
    under FIFO, the queue they hold (tray 2 from the top down, then tray 1 from the bottom up). None if not comparable."""
    if device is None:
        return None
    if device.kind == DEVICE_FOR.get(rule):
        return device.snapshot()
    if device.kind == "twostack" and rule == "FIFO":
        one, two = device.snapshot()
        return list(reversed(two)) + one
    return None


def matches(rule, device, st):
    """Device holds the procedure's state now (r_t is 0 then); None for devices that cannot (N, M−)."""
    got = logical(rule, device)
    return None if got is None else got == canonical(rule, st)[1]


def consistent(rule, device, st_before, st_after):
    """M (and the M2S probe): device content equals the canonical state before or after the action (as F05)."""
    got = logical(rule, device)
    if got is None:
        return None
    return got == canonical(rule, st_before)[1] or got == canonical(rule, st_after)[1]
