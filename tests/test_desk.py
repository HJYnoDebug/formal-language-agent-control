"""Tests for the help-desk platform (agent/desk). Offline only; the full acceptance suite (02b §10) runs here on a
reduced sample, and in full with python -m agent.desk.acceptance."""
import json

import os

import pytest

from agent.common.models import load_models, resolve_model
from agent.desk import acceptance, devices, plan
from agent.desk.env import DeskEnv
from agent.desk.generator import make_schedule
from agent.desk.rules import RULED, DeskState, Ticket, judge_events, legal_resume, violable
from agent.desk.runner import FakeToolBackend, allowed_line, control_lines, run_episode, run_workflow
from agent.desk.strategies import make_chooser, simulate

SPEC = resolve_model("ministral-8b", load_models())


def _state(cats, held):
    tk = {c: Ticket(c, "escalated", cat, 0, False, 0, "C-1", "O-1") for c, cat in cats.items()}
    st = DeskState(tickets=tk)
    for c in cats:
        st.arrive(c)
    for c in held:
        st.put_on_hold(c)
    return st


# ---------------------------------------------------------------- procedures
def test_legal_sets_by_rule():
    st = _state({"BB": "refund", "CC": "replace", "DD": "refund", "FF": "address"}, ["DD", "BB", "CC"])
    assert legal_resume("LIFO", st) == {"CC"}
    assert legal_resume("FIFO", st) == {"DD"}
    assert legal_resume("LAST", st) == {"CC"}
    assert legal_resume("BAL", st) == {"BB"}               # 2 refunds >= 1 replacement: alphabetical first
    assert legal_resume("PAR", st) == {"BB"}               # 2 refunds: even
    st.take_off_hold("CC")
    assert legal_resume("LAST", st) == {"BB", "DD"}        # the latest hold resumed: free
    assert not violable("LAST", st) and violable("LIFO", st)


def test_bal_and_par_flip():
    st = _state({"BB": "replace", "CC": "replace", "DD": "refund"}, ["BB", "CC", "DD"])
    assert legal_resume("BAL", st) == {"DD"}               # 1 refund < 2 replacements: last
    assert legal_resume("PAR", st) == {"DD"}               # 1 refund: odd -> last


def test_judges_agree_on_random_trajectories():
    for rule in RULED:
        for seed in range(5):
            s = make_schedule(f"t|D7|s{seed}", 7, bool(seed % 2))
            env, _ = simulate(s, rule, make_chooser("win_random", rule, seed))
            assert judge_events(rule, env.events) == [x[2] for x in env.resumes]


def test_burial_and_counter_peak():
    st = _state({"BB": "refund", "CC": "refund", "DD": "refund", "FF": "replace"}, ["BB", "CC", "DD"])
    assert st.buried == {"BB": 2, "CC": 1, "DD": 0}
    st.take_off_hold("DD")
    st.put_on_hold("FF")
    assert st.buried["BB"] == 2 and st.buried["CC"] == 1
    assert st.counter_peak == 3 and st.counter == 2        # FF arrived before the holds (replace)


def test_excursion_kept_at_zero():
    st = DeskState(tickets={c: Ticket(c, "normal", cat, 0, False, 0, "", "") for c, cat in
                            (("BB", "refund"), ("CC", "refund"), ("DD", "replace"), ("FF", "replace"),
                             ("GG", "replace"))})
    seen = []
    for c in ("BB", "CC", "DD", "FF", "GG"):
        st.arrive(c)
        seen.append((st.counter, st.excursion))
    assert seen == [(1, 1), (2, 2), (1, 2), (0, 2), (-1, 1)]


# ---------------------------------------------------------------- generator
@pytest.mark.parametrize("D", plan.DEPTHS)
def test_schedule_depth_exact(D):
    for seed in range(4):
        s = make_schedule(f"t|D{D}|s{seed}", D, bool(seed % 2))
        assert len(s["tickets"]) == 16
        assert sum(t["kind"] == "escalated" for t in s["tickets"]) == 14
        assert sum(t["rush"] for t in s["tickets"]) == 4
        for rule in ("PAR", "BAL", "LIFO", "FIFO"):
            left = set()
            base = make_chooser("compliant", rule)

            def chooser(env, ctx, base=base, left=left):     # steps left of the tickets on hold, at every turn
                left.update(env.st.tickets[c].remaining for c in env.st.held)
                return base(env, ctx)
            env, recs = simulate(s, rule, chooser)
            assert env.max_depth == D and env.finish()["completion"] == 1.0
            assert {env.st.tickets[e[1]].kind for e in env.events if e[0] == "h"} == {"escalated"}
            assert left == {1}
            pts = [r for r in recs if r["violable"]]
            first = sum(sorted(r["held"])[0] in r["legal_set"] for r in pts)
            last = sum(sorted(r["held"])[-1] in r["legal_set"] for r in pts)
            chance = sum(1 / len(r["held"]) for r in pts)
            if rule in ("PAR", "BAL"):                      # balance of the stimulus set (generator docstring)
                assert abs(first - last) <= 1
            else:
                assert abs(first - chance) <= 1 and abs(last - chance) <= 1


def test_schedule_deterministic():
    assert make_schedule("t|D4|s1", 4, True) == make_schedule("t|D4|s1", 4, True)


# ---------------------------------------------------------------- environment layer
def test_env_errors_are_not_violations():
    assert acceptance.check_env()["pass"]


def test_gate_blocks_and_forces_wait():
    s = make_schedule("t|D7|s0", 7)
    r = run_episode(s, "LIFO", "E", FakeToolBackend(SPEC, "stubborn:win_random"), SPEC)
    assert sum(t["legal"] is False for t in r["turns"]) == 0
    blocked = [t for t in r["turns"] if t["blocked"]]
    assert blocked and all(t["forced"] for t in blocked if len(t["blocked"]) >= 2)
    assert all(t["action"] == "wait" for t in blocked if t["forced"])


# ---------------------------------------------------------------- controls and devices
def test_device_class_checks():
    reg = devices.Device("register")
    with pytest.raises(devices.DeviceError):
        reg.apply("board_write", {"ticket": "anything"}, {"BB"})
    st = devices.Device("stack")
    st.apply("tray_put", {"ticket": "BB"}, {"BB", "CC"})
    st.apply("tray_put", {"ticket": "CC"}, {"BB", "CC"})
    assert devices.read_line("stack", st.content) == "Tray (top): CC."
    q = devices.Device("queue")
    q.apply("rail_add", {"ticket": "BB"}, {"BB", "CC"})
    q.apply("rail_add", {"ticket": "CC"}, {"BB", "CC"})
    assert devices.read_line("queue", q.content) == "Rail (front): BB."
    m = devices.Device("marker")
    m.apply("marker_flip", {}, set())
    assert devices.read_line("marker", m.content) == "Marker: odd."


def test_allowed_line():
    st = _state({"BB": "refund", "CC": "replace"}, ["BB", "CC"])
    assert allowed_line("LIFO", st) == "Allowed to resume now: CC."
    st.take_off_hold("CC")
    st.put_on_hold("CC")
    st.take_off_hold("CC")
    assert allowed_line("LAST", st) == "Allowed to resume now: BB."


def test_placebo_length_matches_c1():
    s = make_schedule("t|D4|s0", 4)
    env = DeskEnv(s, "FIFO")
    ch = make_chooser("compliant", "FIFO")
    while not env.done():
        env.start_turn()
        c1 = control_lines("FIFO", "C1", env.st, None)
        p = control_lines("FIFO", "P", env.st, None)
        assert len(c1[0]) == len(p[0])
        env.work(*ch(env, {}))


def test_prompts_have_no_operational_hints():
    assert acceptance.check_no_leak(acceptance.schedules(2, (4,)))["pass"]


@pytest.mark.parametrize("rule", ["LAST", "PAR", "BAL", "LIFO", "FIFO"])
def test_all_controls_run(rule):
    s = make_schedule("t|D4|s1", 4, True)
    for control in plan.controls_for(rule) + ("P",):
        r = run_episode(s, rule, control, FakeToolBackend(SPEC, "dev:compliant"), SPEC)
        assert r["termination"] is None and r["completion"] == 1.0
        assert sum(t["legal"] is False for t in r["turns"]) == 0
        json.dumps(r)                                      # records are serialisable


def test_free_only_c0_and_lower_device_rules():
    s = make_schedule("t|D2|s0", 2)
    with pytest.raises(ValueError):
        run_episode(s, "FREE", "C1", FakeToolBackend(SPEC), SPEC)
    with pytest.raises(ValueError):
        run_episode(s, "LAST", "Mminus", FakeToolBackend(SPEC), SPEC)


def test_text_fallback_and_format_errors():
    s = make_schedule("t|D2|s0", 2)
    r = run_episode(s, "LIFO", "C0", FakeToolBackend(SPEC, "text:compliant"), SPEC)
    assert r["termination"] is None and r["text_fallbacks"] == len(r["turns"])
    r = run_episode(s, "LIFO", "C0", FakeToolBackend(SPEC, "malformed"), SPEC)
    assert r["termination"] == "format"


def test_service_failure_terminates():
    s = make_schedule("t|D2|s0", 2)
    r = run_episode(s, "LIFO", "C0", FakeToolBackend(SPEC, "flaky:1.0:compliant"), SPEC)
    assert r["termination"] == "service"


def test_workflow_reference():
    s = make_schedule("t|D7|s0", 7)
    w = run_workflow(s, "FIFO")
    assert w["completion"] == 1.0 and w["cost_usd"] == 0.0


# ---------------------------------------------------------------- plans
def test_plan_sizes():
    assert len(plan.smoke_plan()) == 28
    assert len(plan.pilot_plan()) == 80
    formal = {plan.cell_key(c) for c in plan.formal_plan()}
    assert all(plan.cell_key(c) in formal for c in plan.pilot_plan())       # internal pilot: a slice of the formal plan
    assert len(plan.formal_plan()) == 1920
    assert len(plan.placebo_plan()) == 16
    keys = [plan.cell_key(c) for c in plan.formal_plan()]
    assert len(set(keys)) == len(keys)
    assert {c["rule"] for c in plan.formal_plan()} == {"FREE", "PAR", "BAL", "LIFO", "FIFO"}


def test_executor_resume(tmp_path):
    from agent.desk import run
    cells = plan.smoke_plan()[:6]
    st = run.run_plan(cells, "t", 1.0, 2, "fake:compliant", runs_dir=str(tmp_path), quiet=True)
    assert st["done"] == 6
    st = run.run_plan(cells, "t", 1.0, 2, "fake:compliant", runs_dir=str(tmp_path), quiet=True)
    assert st["done"] == 0 and st["remaining"] == 0


# ---------------------------------------------------------------- acceptance (reduced sample)
@pytest.fixture(scope="module")
def S():
    return acceptance.schedules(24)          # the statistical checks need the full acceptance sample


@pytest.mark.parametrize("name", ["check_leak", "check_defer", "check_window", "check_synthetic", "check_layers"])
def test_acceptance_offline(S, name):
    res = getattr(acceptance, name)(S)
    assert res["pass"], json.dumps(res)[:2000]


def test_temptation_is_reported_only(S):
    res = acceptance.check_temptation(S)
    assert res["pass"] is None and all(0 <= v["pooled"] <= 1 for v in res["by_rule"].values())


# ---------------------------------------------------------------- derived thresholds (prereg)
def test_prereg_derivations():
    from agent.desk import prereg
    assert prereg.zero_event_n() == 72                                  # 1 - 0.025 ** (1 / 72) <= 0.05
    assert 1 - 0.025 ** (1 / 71) > 0.05
    r = prereg.detectable_rate(66)
    assert 0.10 < r < 0.20 and prereg.detectable_rate(200) < r          # more points, finer resolution
    assert prereg.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    own, other = prereg.design_cuts()
    assert own == prereg.CUTS and other == prereg.CUTS_OTHER


def test_acceptance_axes():
    assert acceptance.check_axes()["pass"]


def test_acceptance_atomic(S):
    res = acceptance.check_atomic({D: v[:2] for D, v in S.items()}, SPEC)
    assert res["pass"], res["issues"]


def test_acceptance_storage(S):
    res = acceptance.check_storage({D: v[:8] for D, v in S.items()}, SPEC)
    assert res["pass"], res


def test_acceptance_depth(S):
    res = acceptance.check_depth(S, SPEC, plan.rules_in_grid())
    assert res["pass"], res


# ---------------------------------------------------------------- analysis (end to end on a small fake run)
def test_analysis_runs(tmp_path):
    import importlib.util
    import os
    from agent.desk import run
    cells = plan.pilot_plan()
    run.run_plan(cells, "a", 5.0, 4, "fake:synthetic:2:0.3", runs_dir=str(tmp_path), quiet=True)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "desk_analysis.py")
    spec = importlib.util.spec_from_file_location("desk_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.analyse([str(tmp_path / "a")], str(tmp_path / "a" / "analysis"))
    assert out["episodes"] == len(cells) and out["regular"] == "PAR"
    assert set(out["hypotheses"]) >= {"H1", "H2", "H3", "H4", "H5", "H6"}
    assert all(h.get("p_holm") is None or 0 <= h["p_holm"] <= 1 for h in out["hypotheses"].values())
    assert os.path.exists(tmp_path / "a" / "analysis" / "sufficiency_map.csv")
    assert "difference" in out["hypotheses"]["H1"] or out["hypotheses"]["H1"].get("estimable") is False
    measures = {r["measure"] for r in out["robust_measures"]}
    assert measures == {"all", "first_per_segment", "clean"} and "by_rule" in out["cross_family"]
    assert {r["after_regrowth"] for r in out["interleaving"]} == {False, True}
    assert all("method_sufficient" in r for r in out["i_star"]) and out["method_effects"]


# ---------------------------------------------------------------- two-tray probe M2S (02b §15) and analysis E'
def test_twostack_device():
    d = devices.Device("twostack")
    valid = {"BB", "CC", "DD"}
    for c in ("BB", "CC", "DD"):
        d.apply("tray1_put", {"ticket": c}, valid)
    assert devices.read_line("twostack", d.content) == "Tray 1 (top): DD. Tray 2: empty."
    for _ in range(3):
        d.apply("move_1_to_2", {}, valid)
    assert devices.read_line("twostack", d.content) == "Tray 1: empty. Tray 2 (top): BB."
    assert d.apply("tray2_take", {}, valid) == "Took BB."
    with pytest.raises(devices.DeviceError):
        d.apply("tray1_take", {}, valid)
    with pytest.raises(devices.DeviceError):
        d.apply("tray2_put", {"ticket": "XX"}, valid)


def test_probe_runs_and_stays_out_of_formal():
    assert len(plan.probe_twostack_plan()) == 48
    assert all(c["control"] != "M2S" for c in plan.formal_plan())
    with pytest.raises(ValueError):
        run_episode(make_schedule("t|D4|s0", 4), "LIFO", "M2S", FakeToolBackend(SPEC), SPEC)
    s = make_schedule("t|D10|s1", 10, True)
    for pol in ("dev:compliant", "dev:devread"):
        r = run_episode(s, "FIFO", "M2S", FakeToolBackend(SPEC, pol), SPEC)
        assert r["termination"] is None and sum(t["legal"] is False for t in r["turns"]) == 0
        assert all(t["device_consistent"] for t in r["turns"])
    r = run_episode(s, "FIFO", "M2S", FakeToolBackend(SPEC, "devread"), SPEC)   # no tray kept: errs
    assert sum(t["legal"] is False for t in r["turns"]) > 0


def test_notebook_usage_analysis():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "desk_analysis.py")
    spec = importlib.util.spec_from_file_location("desk_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = make_schedule("t|D7|s1", 7, True)
    recs = [run_episode(s, rule, "N", FakeToolBackend(SPEC, pol), SPEC) for rule in ("LIFO", "BAL")
            for pol in ("dev:compliant", "compliant")]
    out = mod.notebook_usage(recs, {s["key"]: s})
    lifo = out[f"{SPEC.model_id}|LIFO"]
    assert lifo["points"] > 0
    assert lifo["recorded_with_order"]["share"] == 0.5 and lifo["not_recorded"]["share"] == 0.5
    assert out[f"{SPEC.model_id}|BAL"]["recorded"]["share"] == 0.5


# ---------------------------------------------------------------- lapse reminder (02b §2, §5)
def test_lapse_reminder():
    S = {10: [make_schedule("t|D10|s0", 10), make_schedule("t|D10|s1", 10, True)]}
    res = acceptance.check_reminder(S, SPEC)
    assert res["pass"], res


def test_raise_cap_is_explicit_and_logged(tmp_path):
    import json as _json
    from agent.desk import run
    cells = plan.smoke_plan()[:2]
    run.run_plan(cells, "cap", 0.5, 1, "fake:compliant", runs_dir=str(tmp_path), quiet=True)
    run.run_plan(cells, "cap", 5.0, 1, "fake:compliant", runs_dir=str(tmp_path), quiet=True)
    m = _json.load(open(tmp_path / "cap" / "manifest.json", encoding="utf-8"))
    assert m["budget_cap_usd"] == 0.5                                       # resuming does not change the cap
    run.run_plan(cells, "cap", 5.0, 1, "fake:compliant", runs_dir=str(tmp_path), quiet=True, raise_cap=True)
    m = _json.load(open(tmp_path / "cap" / "manifest.json", encoding="utf-8"))
    assert m["budget_cap_usd"] == 5.0 and (tmp_path / "cap" / "budget_cap_changes.jsonl").exists()


# ---------------------------------------------------------------- infrastructure: network failures (freeze amendment 2)
def test_transport_retries_transient_network_errors(monkeypatch):
    import http.client
    import io
    import json as _json
    import urllib.request
    from agent.harness.llm_client import HTTPTransport
    calls = {"n": 0}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return Resp(_json.dumps({"ok": True}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    tr = HTTPTransport(base_delay=0.0, max_delay=0.0)
    assert tr.post_chat_completion("http://x", "k", {}) == {"ok": True} and calls["n"] == 3


def test_executor_isolates_cells_and_stops_on_outage(tmp_path):
    from agent.common import executor
    from agent.desk import run

    def broken(cell, backend, spec, ledger):
        raise ConnectionResetError("simulated outage")

    cells = plan.smoke_plan()[:12]
    st = executor.run_plan(cells, "net", 1.0, broken, plan.cell_key, run.make_backend,
                           "fake:compliant", workers=1, runs_dir=str(tmp_path), quiet=True)
    assert st["failed"] == executor.MAX_CONSECUTIVE_SERVICE_FAILURES and st["breaker_stop"]
    st = run.run_plan(cells, "net", 1.0, 1, "fake:compliant", runs_dir=str(tmp_path), quiet=True)
    assert st["done"] == 12 and st["remaining"] == 0                     # failed cells are rerun on resume


def test_executor_refuses_other_contract(tmp_path):
    import json
    from agent.desk import run
    cells = plan.smoke_plan()[:1]
    run.run_plan(cells, "c", 1.0, 1, "fake:compliant", runs_dir=str(tmp_path), quiet=True)
    snap = tmp_path / "c" / "config_snapshot.json"
    cfg = json.loads(snap.read_text(encoding="utf-8"))
    cfg["contract_version"] = "d0.0"
    snap.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(SystemExit):
        run.run_plan(cells, "c", 1.0, 1, "fake:compliant", runs_dir=str(tmp_path), quiet=True)


def _verify_grammars():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "verify_grammars.py")
    spec = importlib.util.spec_from_file_location("verify_grammars", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_input_automaton_rejects():
    vg = _verify_grammars()
    ok = "EEE---"                                        # depth 2, then resume twice
    assert vg.input_automaton(2, ok, n_esc=3, n_norm=0)
    assert not vg.input_automaton(1, ok, n_esc=3, n_norm=0)          # depth above D
    assert not vg.input_automaton(3, ok, n_esc=3, n_norm=0)          # depth D never reached
    assert not vg.input_automaton(2, "NE-", n_esc=1, n_norm=1)       # a normal ticket would be put on hold
    assert not vg.input_automaton(2, "EEN---", n_esc=2, n_norm=1)    # a normal ticket while tickets are on hold
    assert not vg.input_automaton(2, "EEE--", n_esc=3, n_norm=0)     # a ticket left on hold


def test_nested_burst_every_day():
    """v1.8: on every day the stack / queue grows again after a resume exactly once; without the nested burst it
    never does (every burst is h^m r^m)."""
    vg = _verify_grammars()
    for D in plan.DEPTHS:
        for seed in range(3):
            s = make_schedule(f"t|D{D}|s{seed}", D, bool(seed % 2))
            pure = make_schedule(f"t|D{D}|s{seed}", D, bool(seed % 2), n_nest=0, n_ins=1)
            for rule in ("LIFO", "FIFO"):
                prof = vg.hold_profile(simulate(s, rule, make_chooser("compliant", rule))[0].events)
                assert prof == {"max_depth": D, "longest_hold_run": D, "push_after_pop": 1}
                assert vg.hold_profile(simulate(pure, rule, make_chooser("compliant", rule))[0].events)[
                    "push_after_pop"] == 0


def test_analysis_b_recovers_k(tmp_path):
    """Analysis B finds the capacity of a synthetic agent that keeps stack / queue truncated to 2 entries and a counter
    saturating at +-2 (the bounded recognizers of acceptance item 6)."""
    import importlib.util
    import os
    from agent.desk import run
    cells = [c for c in plan.formal_plan() if c["model"] == plan.STRONG and c["control"] == "C0" and
             c["rule"] in ("BAL", "LIFO", "FIFO") and c["D"] in (7, 10)]
    run.run_plan(cells, "k", 5.0, 4, "fake:view:memory:2", runs_dir=str(tmp_path), quiet=True)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "desk_analysis.py")
    spec = importlib.util.spec_from_file_location("desk_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fit = mod.memory_k_fit(mod.load([str(tmp_path / "k")]))
    assert {v["k"] for v in fit.values()} == {2} and len(fit) == 3


# ---------------------------------------------------------------- v1.9: method conditions, wordings, comprehension
def test_plans_v19():
    assert len(plan.formal_plan()) == 1920 and len(plan.pilot_plan()) == 80 and len(plan.wording_plan()) == 96
    keys = [plan.cell_key(c) for c in plan.formal_plan() + plan.wording_plan()]
    assert len(set(keys)) == len(keys) and all("wording" not in k for k in keys[:1920])


def test_method_conditions_differ_only_by_the_method():
    from agent.desk import prompts
    for rule in ("PAR", "BAL", "LIFO", "FIFO"):
        para = "\n\n" + prompts.method_paragraph(rule)
        assert prompts.system_prompt(rule, "R").replace(para, "") == prompts.system_prompt(rule, "C0")
        assert prompts.PROCEDURE_VARIANTS[rule][0] == prompts.PROCEDURE[rule]
    s = make_schedule("t|D4|s0", 4)
    r = run_episode(s, "LIFO", "MR", FakeToolBackend(SPEC, "dev:compliant"), SPEC)
    m = run_episode(s, "LIFO", "M", FakeToolBackend(SPEC, "dev:compliant"), SPEC)
    assert r["device"] == m["device"] == "stack" and [t["pushed"] for t in r["turns"]] == [t["pushed"] for t in m["turns"]]
    w = run_episode(s, "LIFO", "C0", FakeToolBackend(SPEC, "compliant"), SPEC, cell={"wording": 2})
    assert w["wording"] == 2


def test_comprehension_probe(tmp_path):
    from agent.desk import comprehension as comp
    assert acceptance.check_comprehension()["pass"]
    cells = [c for c in comp.probe_plan((plan.STRONG,)) if c["rule"] == "FIFO"]
    assert len(cells) == 72 + 2 * 24
    comp.run_probe(cells, "c", 1.0, 4, "fake:oracle", runs_dir=str(tmp_path), quiet=True)
    rows = comp.summary([str(tmp_path / "c")])
    assert {(r["wording"], r["items"], r["errors"]) for r in rows} == {(0, 72, 0), (1, 24, 0), (2, 24, 0)}
    assert next(r for r in rows if r["wording"] == 0)["understood"]          # 72 items, 0 errors: bound <= 5%


# ---------------------------------------------------------------- v1.9.1: tolerance curve, frontier, OC curve
def _load_analysis(name):
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prereg_tolerance_references():
    from agent.desk import prereg
    r = prereg.tolerance_references()
    assert prereg.cell_points() == 86 and r["resolution_floor"] == 0.042
    assert abs(r["day_80"] - 0.0307) < 1e-3 and abs(r["day_50"] - 0.0922) < 1e-3
    assert abs(prereg.oc_curve(0.01) - 0.421) < 1e-3                     # 0 violations needed at 5%, 86 points
    assert prereg.oc_curve(0.01, threshold=0.10) > 0.98


def test_contrast_cut_per_coordinate():
    """2026-10-22 (freeze log, changes after the freeze 15): a coordinate's high / low cut is its own design median. The
    main coordinates keep the stored cuts; the secondary d_t gets its own, so both groups have points (with the beta_t
    cuts every diagnostic point had d_t >= cut, so no point was 'low')."""
    from agent.desk import prereg
    da = _load_analysis("desk_analysis.py")
    for rule in ("PAR", "BAL", "LIFO", "FIFO"):
        assert da.contrast_cut(rule, prereg.OWN_COORD[rule]) == prereg.CUTS[rule]
        assert da.contrast_cut(rule, prereg.OTHER_COORD[rule]) == prereg.CUTS_OTHER[rule]
    for rule in ("PAR", "LIFO", "FIFO"):
        pts = [r for r in prereg.design_points(rule) if r["diagnostic"]]
        cut = da.contrast_cut(rule, "depth")
        assert any(r["depth"] < cut for r in pts) and any(r["depth"] >= cut for r in pts)
        assert all(r["depth"] >= prereg.CUTS[rule] for r in pts)            # the old cut left no 'low' point


def test_pareto_set_day_success_profile_figures(tmp_path):
    """v1.9.2: three-objective Pareto set (u, completion loss, cost) with bootstrap frontier probabilities, the tolerance
    reading, day-level success and the performance profile, on a synthetic cell; all five figures render."""
    import pandas as pd
    from agent.desk import prereg
    da = _load_analysis("desk_analysis.py")
    #            control: (violations per day, completion, cost)
    spec = {"C0": (3, 1.0, 0.010), "R": (2, 1.0, 0.011), "M": (0, 1.0, 0.013), "C1": (0, 1.0, 0.012),
            "C2": (0, 1.0, 0.014), "E": (0, 0.8, 0.016), "C3": (0, 1.0, 0.018)}
    days = [f"d{i}" for i in range(6)]
    tf, ep = [], []
    for c, (x, comp, cost) in spec.items():
        for i, d in enumerate(days):
            xd = 1 if (c == "M" and i == 0) else x
            tf += [{"model": "m", "rule": "LIFO", "control": c, "schedule": d, "violable": True, "legal": k >= xd}
                   for k in range(10)]
            ep.append({"model": "m", "rule": "LIFO", "control": c, "schedule": d, "completion_unaided": comp,
                       "cost": cost})
    tf, ep = pd.DataFrame(tf), pd.DataFrame(ep)
    front = da.cost_risk_frontier(tf, ep, B=50)
    assert set(front[front.pareto].control) == {"C0", "R", "C1", "E", "C3"}   # E: cheaper than C3, worse completion
    f = front.set_index("control")
    assert f.loc["C1", "u"] == round(prereg.cp_upper(0, 60), 4) and f.loc["E", "u"] == 0.0
    assert not f.loc["E", "completion_ok"] and f.loc["C3", "loss"] == 0.0
    assert all(0 <= v <= 1 for v in front.p_frontier) and f.loc["C2", "p_frontier"] == 0.0
    tol = da.tolerance_curve(front, taus=[0.03, 0.07, 0.35, 0.5])
    assert list(tol.sort_values("tolerance").cheapest) == ["C3", "C1", "R", "C0"]
    days_ok = da.day_success(tf, ep)
    c0 = days_ok[days_ok.control == "C0"].set_index("k").share_days
    assert c0[2] == 0.0 and c0[3] == 1.0
    prof = da.performance_profile(front, taus=[0.03, 0.5])
    assert set(prof[(prof.control == "C3")].share_models) == {1.0}
    figs = _load_analysis("desk_figures.py")
    paths = (figs.fig_frontier(front, str(tmp_path)) + figs.fig_tolerance(front, str(tmp_path)) +
             figs.fig_day_success(days_ok, str(tmp_path)) + figs.fig_profile(prof, str(tmp_path)) +
             figs.fig_oc(str(tmp_path)))
    assert len(paths) == 10 and all(os.path.getsize(p) > 0 for p in paths)


def test_budget_action_order():
    """02b §8 rule 4 (v1.9.3): over the cap, drop MR first, then 32B; otherwise the author decides."""
    from agent.desk import prereg
    from agent.desk.report import budget_action, project_formal
    recs = [{"termination": None, "calls": [{}], "tokens_in": 600000, "tokens_out": 6000}]
    full, _ = project_formal(recs)
    no_mr, _ = project_formal(recs, cells=[c for c in plan.formal_plan() if c["control"] != "MR"])
    assert no_mr < full and prereg.MAX_FORMAL_USD == 49.0
    assert budget_action(recs, cap=full)[0] == "none"
    assert budget_action(recs, cap=(full + no_mr) / 2)[0] == "drop MR"
    assert budget_action(recs, cap=0.01)[0].startswith("over the cap")
    assert budget_action([])[0] is None


def test_backup_roundtrip(tmp_path):
    """agent/common/backup.py: zip + SHA-256 logged twice; verify detects a changed archive."""
    from agent.common import backup
    run = tmp_path / "runs" / "r1"
    (run / "cells").mkdir(parents=True)
    (run / "cells" / "a.json").write_text('{"x": 1}', encoding="utf-8")
    (run / "done.jsonl").write_text("{}\n", encoding="utf-8")
    dest, log = tmp_path / "backups", tmp_path / "freeze" / "backups.tsv"
    info = backup.backup_run(str(run), dest=str(dest), note="t", log=str(log))
    assert info["files"] == 2 and os.path.exists(info["zip"]) and (dest / "MANIFEST.tsv").exists()
    assert backup.verify(dest=str(dest), log=str(log)) == [(os.path.basename(info["zip"]), "ok")]
    with open(info["zip"], "ab") as f:
        f.write(b"tamper")
    assert backup.verify(dest=str(dest), log=str(log))[0][1] == "hash differs"


def test_projection_counts_caching():
    """v1.10: the cached share is solved from the billed cost and applied at each model's cached price."""
    from agent.common.models import MODEL_ALIASES, load_models
    from agent.desk.report import cached_share, project_formal
    specs = load_models()
    s = specs[MODEL_ALIASES[plan.STRONG]]
    tin, tout, f = 200000, 1500, 0.9
    billed = (tin * ((1 - f) * s.input_price + f * s.cache_price) + tout * s.output_price) / 1e6
    recs = [{"model": s.model_id, "termination": None, "calls": [{}], "tokens_in": tin, "tokens_out": tout,
             "cost_usd": billed}]
    assert abs(cached_share(recs, specs) - f) < 1e-9
    total, by = project_formal(recs, specs)
    n = sum(c["model"] == plan.STRONG for c in plan.formal_plan())
    assert abs(by[plan.STRONG] - round(n * billed, 2)) < 0.011


def test_route_and_request_extra_reach_the_payload():
    import dataclasses
    from agent.common.models import load_models, resolve_model
    from agent.common.toolcall import OpenRouterToolBackend
    seen = []

    class Capture:
        def post_chat_completion(self, base_url, api_key, payload, timeout=120):
            seen.append(payload)
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "view_queue", "arguments": "{}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0}, "provider": "Mistral"}

    models = load_models()
    large = resolve_model("large-3", models)
    small = resolve_model("small-4", models)
    meta = {"tools": []}
    OpenRouterToolBackend(dataclasses.replace(large, route=None), "k", transport=Capture()).call([], 10, meta)
    OpenRouterToolBackend(large, "k", transport=Capture()).call([], 10, meta)          # config: route mistral/eu
    OpenRouterToolBackend(small, "k", transport=Capture()).call([], 10, meta)
    assert seen[0]["provider"]["order"] == ["Mistral"] and "reasoning" not in seen[0]
    assert seen[1]["provider"]["order"] == ["mistral/eu"] and seen[1]["provider"]["allow_fallbacks"] is False
    assert seen[2]["reasoning"] == {"effort": "none"} and seen[2]["provider"]["order"] == ["Mistral"]


def test_interleave_and_per_model_cap(tmp_path):
    import threading
    import time as _time
    from agent.common import executor
    assert [c["id"] for c in executor.interleave([{"model": "a", "id": 1}, {"model": "a", "id": 2},
                                                  {"model": "a", "id": 3}, {"model": "b", "id": 4}])] == [1, 4, 2, 3]
    cells = [c for c in plan.smoke_plan() if c["model"] in ("large-3", "ministral-3b")]
    running, peak, lock = {}, {}, threading.Lock()

    def execute(cell, backend, spec, ledger):
        with lock:
            running[cell["model"]] = running.get(cell["model"], 0) + 1
            peak[cell["model"]] = max(peak.get(cell["model"], 0), running[cell["model"]])
        _time.sleep(0.05)
        with lock:
            running[cell["model"]] -= 1
        return {"termination": None, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}

    stats = executor.run_plan(cells, "cap", 1.0, execute, plan.cell_key, lambda kind, spec, seed=0: None, "fake",
                              workers=6, runs_dir=str(tmp_path), quiet=True)
    assert stats["done"] == len(cells) == 14
    assert peak["large-3"] <= 2                       # config: Large 3 max_concurrency 2
    assert peak["ministral-3b"] >= 1


def test_rate_limit_window_and_retry_after(monkeypatch):
    import io
    import urllib.error
    from agent.harness import llm_client
    slept, calls = [], {"n": 0}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def urlopen(req, timeout=None, fails=12, code=429):
        calls["n"] += 1
        if calls["n"] <= fails:
            raise urllib.error.HTTPError("u", code, "x", {"Retry-After": "7"}, io.BytesIO(b'{"error": "rate-limited upstream"}'))
        return Resp()

    monkeypatch.setattr(llm_client.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", urlopen)
    t = llm_client.HTTPTransport()
    assert t.post_chat_completion("http://x", "k", {}) == {"ok": True}     # 12 x 429 > max_retries 8: still waits
    assert len(slept) == 12 and min(slept) >= 7.0 and max(slept) <= 600.0
    calls["n"], slept[:] = 0, []
    monkeypatch.setattr(llm_client.urllib.request, "urlopen", lambda req, timeout=None: urlopen(req, timeout, 99, 503))
    with pytest.raises(RuntimeError):
        t.post_chat_completion("http://x", "k", {})                       # a 5xx keeps the old 8-retry window
    assert len(slept) == 8


def _spending_execute(calls=4, cost=0.001, pause=0.0, stop_dir=None):
    """A fake episode that bills `calls` calls through the ledger, as MeteredCaller does."""
    import time as _time
    from agent.common import executor

    def execute(cell, backend, spec, ledger):
        if stop_dir is not None:
            executor.request_stop(stop_dir)
        for _ in range(calls):
            ledger.reserve(cost)
            _time.sleep(pause)
            ledger.settle(cost, cost)
        return {"termination": None, "cost_usd": calls * cost, "tokens_in": 0, "tokens_out": 0}
    return execute


def _cells(n, model="ministral-3b"):
    return [{"exp": "T", "model": model, "id": i} for i in range(n)]


def _key(c):
    return f"T|{c['model']}|{c['id']}"


def test_journal_counts_unfinished_and_cut_lines(tmp_path):
    import json as _json
    import os as _os
    from agent.common import executor
    run = lambda cells, cap=1.0: executor.run_plan(cells, "j", cap, _spending_execute(), _key,                 # noqa: E731
                                                   lambda kind, spec, seed=0: None, "fake", workers=2,
                                                   runs_dir=str(tmp_path), quiet=True)
    run(_cells(4))
    rd = str(tmp_path / "j")
    assert abs(executor.spend_of(rd)["spent"] - 0.016) < 1e-9
    with open(_os.path.join(rd, "done.jsonl"), "a", encoding="utf-8") as f:       # a process killed while writing
        f.write('{"key": "T|ministral-3b|4", "sta')
    with open(_os.path.join(rd, "calls.jsonl"), "a", encoding="utf-8") as f:      # an episode killed half-way
        f.write(_json.dumps({"key": "T|ministral-3b|4", "attempt": "dead", "cost_usd": 0.002, "t": 1.0}) + "\n")
    sp = executor.spend_of(rd)
    assert sp["cut_lines"] == 1 and sp["unfinished_episodes"] == 1 and abs(sp["spent"] - 0.018) < 1e-9
    stats = run(_cells(6))
    recs, bad = executor.read_jsonl(_os.path.join(rd, "done.jsonl"))
    assert bad == 1 and len(recs) == 6 and stats["remaining"] == 0             # the record after the cut line survives
    assert abs(executor.spend_of(rd)["spent"] - (6 * 0.004 + 0.002)) < 1e-9


def test_budget_stop_keeps_the_money_of_the_cut_episode(tmp_path):
    from agent.common import executor
    stats = executor.run_plan(_cells(4), "b", 0.0105, _spending_execute(), _key, lambda kind, spec, seed=0: None,
                              "fake", workers=1, runs_dir=str(tmp_path), quiet=True)
    sp = executor.spend_of(str(tmp_path / "b"))
    assert stats["budget_stop"] and sp["unfinished_episodes"] == 1
    assert abs(sp["spent"] - 0.010) < 1e-9 and sp["spent"] > 2 * 0.004                # 2 episodes + the cut one


def test_run_lock_and_stop_file(tmp_path):
    import json as _json
    import os as _os
    import socket as _socket
    import subprocess as _sp
    import sys as _sys
    from agent.common import executor
    rd = tmp_path / "l"
    rd.mkdir()
    other = _sp.Popen([_sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (rd / "run.lock").write_text(_json.dumps({"pid": other.pid, "host": _socket.gethostname(), "started": "x"}))
        assert executor.status(str(rd))["state"] == "running"
        with pytest.raises(SystemExit):
            executor.run_plan(_cells(2), "l", 1.0, _spending_execute(), _key, lambda kind, spec, seed=0: None,
                              "fake", runs_dir=str(tmp_path), quiet=True)
    finally:
        other.kill()
        other.wait()
    assert executor.status(str(rd))["state"].startswith("interrupted")
    stats = executor.run_plan(_cells(6), "l", 1.0, _spending_execute(stop_dir=str(rd)), _key,
                              lambda kind, spec, seed=0: None, "fake", workers=1, runs_dir=str(tmp_path), quiet=True)
    assert stats["stopped"].startswith("stop requested") and 0 < stats["remaining"] < 6
    assert not (rd / "run.lock").exists() and not (rd / "STOP").exists()
    stats = executor.run_plan(_cells(6), "l", 1.0, _spending_execute(), _key, lambda kind, spec, seed=0: None,
                              "fake", workers=2, runs_dir=str(tmp_path), quiet=True)
    assert stats["remaining"] == 0 and executor.status(str(rd))["state"] == "not running"


def test_killed_process_resumes(tmp_path):
    """A real hard kill (TerminateProcess on Windows) in the middle of episodes, then a resume."""
    import os as _os
    import subprocess as _sp
    import sys as _sys
    import time as _time
    from agent.common import executor
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    driver = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
              "import test_desk as t\n"
              "from agent.common import executor\n"
              "executor.run_plan(t._cells(12), 'k', 1.0, t._spending_execute(5, 0.001, 0.05), t._key,\n"
              "                  lambda kind, spec, seed=0: None, 'fake', workers=3, runs_dir=%r, quiet=True)\n"
              % (root, _os.path.join(root, "tests"), str(tmp_path)))
    proc = _sp.Popen([_sys.executable, "-c", driver])
    done, calls = tmp_path / "k" / "done.jsonl", tmp_path / "k" / "calls.jsonl"
    lines = lambda p: len(p.read_text().splitlines()) if p.exists() else 0                          # noqa: E731
    t0 = _time.time()
    while _time.time() - t0 < 60 and not (lines(done) >= 3 and lines(calls) > 5 * lines(done)):   # mid-episode
        _time.sleep(0.01)
    proc.kill()
    proc.wait()
    st = executor.status(str(tmp_path / "k"))
    assert st["state"].startswith("interrupted") and 3 <= st["done"] < 12
    before = executor.spend_of(str(tmp_path / "k"))
    assert before["unfinished_episodes"] >= 1 and before["unfinished_usd"] > 0
    stats = executor.run_plan(_cells(12), "k", 1.0, _spending_execute(5, 0.001), _key,
                              lambda kind, spec, seed=0: None, "fake", workers=3, runs_dir=str(tmp_path), quiet=True)
    after = executor.spend_of(str(tmp_path / "k"))
    recs, _ = executor.read_jsonl(str(done))
    assert stats["remaining"] == 0 and len({r["key"] for r in recs if r["status"] == "done"}) == 12
    assert len(recs) == 12                                       # no cell recorded twice
    assert abs(after["spent"] - (12 * 0.005 + before["unfinished_usd"])) < 1e-9
    assert executor.status(str(tmp_path / "k"))["state"] == "not running"


def test_ctrl_c_lets_running_episodes_finish(tmp_path, monkeypatch):
    from agent.common import executor
    real_wait, n = executor.cf.wait, {"calls": 0}

    def wait(*a, **k):
        n["calls"] += 1
        if n["calls"] == 2:
            raise KeyboardInterrupt
        return real_wait(*a, **k)

    monkeypatch.setattr(executor.cf, "wait", wait)
    stats = executor.run_plan(_cells(10), "c", 1.0, _spending_execute(3, 0.001, 0.2), _key,
                              lambda kind, spec, seed=0: None, "fake", workers=2, runs_dir=str(tmp_path), quiet=True)
    sp = executor.spend_of(str(tmp_path / "c"))
    assert stats["stopped"] == "Ctrl+C" and 0 < stats["remaining"] < 10
    assert sp["unfinished_episodes"] == 0                        # the running episodes finished and were recorded
    assert not (tmp_path / "c" / "run.lock").exists()


def test_switch_resume_writes_resume_before_hold():
    """Leaving an unfinished current ticket to resume a held one (seen in the 2026-10-17 pilot): the resume is judged
    on the tickets on hold when it is chosen, and the event word must agree (r before the h it causes)."""
    from agent.desk.strategies import next_step_action
    n_cases = 0
    for rule in ("LIFO", "FIFO", "BAL", "PAR"):
        for seed in range(3):
            for pick_legal in (True, False):
                env = DeskEnv(plan.schedule(plan.sched_key("formal", 10, seed), 10, bool(seed % 2)), rule)
                ch = make_chooser("compliant", rule, seed)
                switched = False
                while not env.done():
                    env.start_turn()
                    st = env.st
                    cur = st.tickets.get(st.current) if st.current else None
                    if not switched and cur is not None and cur.done >= 1 and len(st.held) >= 2 \
                            and not env.escalation_pending:
                        legal = legal_resume(rule, st)
                        pool = sorted(legal) if pick_legal else sorted(set(st.held) - legal)
                        if pool:
                            code, prev = pool[0], st.current
                            env.work(*next_step_action(st.tickets[code]))
                            assert env.events[-2:] == [("r", code), ("h", prev)] or \
                                env.events[-3:-1] == [("r", code), ("h", prev)]          # (a close may follow)
                            assert env.resumes[-1][2] == pick_legal
                            switched = True
                            n_cases += 1
                            continue
                    env.work(*ch(env, {}))
                assert judge_events(rule, env.events) == [x[2] for x in env.resumes]
    assert n_cases >= 12


def test_written_reasoning_control(tmp_path):
    from agent.desk import comprehension as C
    cells = C.reasoning_plan()
    assert len(cells) == 192 and {c["model"] for c in cells} == {plan.WEAK, plan.STRONG}
    assert all(C.get_item(c["rule"], c["item"])["paraphrased"] and c["wording"] == 0 for c in cells)
    probe = [c for c in C.probe_plan((plan.WEAK, plan.STRONG)) if c["wording"] == 0]
    C.run_probe(probe, "p", 10.0, backend_kind="fake:random", runs_dir=str(tmp_path), quiet=True)
    stats = C.run_probe(cells, "r", 10.0, backend_kind="fake:oracle", runs_dir=str(tmp_path), quiet=True,
                        reasoning=True)
    assert stats["done"] == 192 and stats["remaining"] == 0
    rows = C.reasoning_summary(str(tmp_path / "r"), str(tmp_path / "p"))
    assert len(rows) == 8 and all(r["items"] == 24 and r["correct_reasoning"] == 24 for r in rows)
    assert all(r["text_fallbacks"] == 24 and r["written_chars"] > 0 for r in rows)      # answered in text: parsed
    assert all(r["forced_right_reasoning_wrong"] == 0 and 0 <= r["mcnemar_p"] <= 1 for r in rows)
    assert sum(r["correct_forced"] for r in rows) < 192                                  # the random probe errs


def test_capped_model_does_not_hold_back_the_others(tmp_path):
    """Large 3 (max_concurrency 2) is slow; the Ministral cells must not wait for its slots (freeze log changes 7)."""
    import threading
    import time as _time
    from agent.common import executor
    cells = [c for c in plan.smoke_plan() if c["model"] in ("large-3", "ministral-3b")]
    cells = [dict(c, seed=i) for i, c in enumerate(cells * 2)]                # 14 Large 3 + 14 Ministral 3B
    end, peak, now, lock = {"large-3": [], "ministral-3b": []}, {}, {}, threading.Lock()
    t0 = _time.time()

    def execute(cell, backend, spec, ledger):
        m = cell["model"]
        with lock:
            now[m] = now.get(m, 0) + 1
            peak[m] = max(peak.get(m, 0), now[m])
        _time.sleep(0.25 if m == "large-3" else 0.02)
        with lock:
            now[m] -= 1
            end[m].append(_time.time() - t0)
        return {"termination": None, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0}

    stats = executor.run_plan(cells, "free", 1.0, execute, lambda c: f"{c['model']}|{c['control']}|{c['seed']}",
                              lambda kind, spec, seed=0: None, "fake", workers=4, runs_dir=str(tmp_path), quiet=True)
    assert stats["remaining"] == 0 and peak["large-3"] <= 2 and peak["ministral-3b"] >= 2
    assert max(end["ministral-3b"]) < sorted(end["large-3"])[3]       # all Ministral cells done early


def test_finish_reason_is_recorded_per_call():
    from agent.common.models import load_models, resolve_model
    from agent.common.toolcall import MeteredCaller, OpenRouterToolBackend

    class Capture:
        def post_chat_completion(self, base_url, api_key, payload, timeout=120):
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "wait", "arguments": "{}"}}]}, "finish_reason": "tool_calls",
                                 "native_finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.0}, "provider": "Mistral"}

    spec = resolve_model("ministral-3b", load_models())
    rec = {"calls": [], "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "flops": 0.0, "service_errors": 0}
    rep = MeteredCaller(OpenRouterToolBackend(spec, "k", transport=Capture()), spec, None, rec)([], 10, {"tools": []})
    assert rep.finish_reason == "tool_calls/tool_calls" and rec["calls"][0]["finish_reason"] == "tool_calls/tool_calls"


def test_status_counts_the_plan_of_the_last_start(tmp_path):
    """A filtered start in a directory that already holds other done cells (formal stage A after the pilot)."""
    from agent.common import executor
    run = lambda cells: executor.run_plan(cells, "s", 1.0, _spending_execute(1, 0.0), _key,              # noqa: E731
                                          lambda kind, spec, seed=0: None, "fake", runs_dir=str(tmp_path), quiet=True)
    run(_cells(5))
    run(_cells(8)[5:])
    st = executor.status(str(tmp_path / "s"))
    assert st["done"] == 8 and st["cells_in_plan"] == 3 and st["done_in_plan"] == 3 and st["not_done"] == 0


def test_empty_provider_error_reply_is_retried_not_shown(monkeypatch):
    """An empty zero-token reply with finish_reason "error": retried silently (freeze log changes 11); if the identical
    request fails every time, it reaches the runner as a reply without a tool call, marked (changes 12)."""
    from agent.common import toolcall
    from agent.common.models import load_models, resolve_model
    monkeypatch.setattr(toolcall.time, "sleep", lambda s: None)
    empty = {"choices": [{"message": {"content": None}, "finish_reason": "error", "native_finish_reason": "error"}],
             "usage": {"prompt_tokens": 900, "completion_tokens": 0, "cost": 0.0}, "provider": "Mistral"}
    good = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "wait", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 5, "cost": 0.0001}, "provider": "Mistral"}

    class Seq:
        def __init__(self, replies):
            self.replies, self.n = list(replies), 0

        def post_chat_completion(self, base_url, api_key, payload, timeout=120):
            self.n += 1
            return self.replies.pop(0)

    spec = resolve_model("ministral-3b", load_models())
    t = Seq([empty, empty, good])
    rep = toolcall.OpenRouterToolBackend(spec, "k", transport=t).call([], 10, {"tools": []})
    assert t.n == 3 and rep.tool_calls[0][1] == "wait" and rep.completion_tokens == 5
    assert toolcall.provider_error_reply(empty) and not toolcall.provider_error_reply(good)
    stuck = Seq([empty] * 10)                    # persistent: after every retry it reaches the runner as a reply
    rep = toolcall.OpenRouterToolBackend(spec, "k", transport=stuck).call([], 10, {"tools": []})
    assert stuck.n == toolcall.OpenRouterToolBackend.EMPTY_RETRIES + 1 and rep.provider_error and not rep.tool_calls
    assert rep.finish_reason == "error/error"


def test_void_cells_reruns_only_them(tmp_path):
    import json as _json
    from agent.common import executor
    run = lambda: executor.run_plan(_cells(6), "v", 1.0, _spending_execute(2, 0.001), _key,                  # noqa: E731
                                    lambda kind, spec, seed=0: None, "fake", runs_dir=str(tmp_path), quiet=True)
    run()
    rd = str(tmp_path / "v")
    keys = {_key(c) for c in _cells(6)[:2]}
    assert executor.void_cells(rd, keys, "test", str(tmp_path / "arch")) == sorted(keys)
    assert len(os.listdir(tmp_path / "arch")) == 2
    st = executor.status(rd)
    assert st["done"] == 4 and st["failed_to_rerun"] == 2 and abs(st["spent_usd"] - 0.012) < 1e-9
    stats = run()
    assert stats["done"] == 2 and stats["remaining"] == 0                     # only the two voided cells ran again
    assert abs(executor.spend_of(rd)["spent"] - 0.016) < 1e-9                 # their first run's money still counts


def test_think_control(tmp_path):
    """T (freeze log changes 14): a think tool that changes nothing and takes no time; offered only under T."""
    from agent.desk import prompts, run
    cells = plan.think_plan()
    assert len(cells) == 48 and {c["control"] for c in cells} == {"T"}
    stats = run.run_plan(cells[:6], "t", 1e6, workers=2, backend_kind="fake:think:view:compliant",
                         runs_dir=str(tmp_path), quiet=True)
    assert stats["done"] == 6 and stats["remaining"] == 0
    from agent.desk import report as R
    for r in R.load(str(tmp_path / "t")):
        assert r["thinks"] > 0 and r["think_chars"] > 0 and r["termination"] is None
        sysmsg = r["transcript"][0]["content"]
        assert prompts.THINK_LINE in sysmsg and "Keep any reasoning brief." not in sysmsg
        assert all(t.get("thinks", 0) <= 1 for t in r["turns"])           # the fake thinks once per turn
        assert sum(1 for t in r["turns"] if t.get("forced")) == 0           # thinking does not use up the turn
        compliant = [t for t in r["turns"] if t.get("violable")]
        assert all(t.get("legal") is not False for t in compliant)
    # same day, same policy without the tool: identical judgements, no think calls offered
    c0 = dict(cells[0], control="C0", exp="X")
    rec = run_episode(plan.get_schedule(c0), c0["rule"], "C0", FakeToolBackend(SPEC, "view:compliant"),
                      SPEC, None, c0)
    assert rec["thinks"] == 0 and prompts.THINK_LINE not in rec["transcript"][0]["content"]
