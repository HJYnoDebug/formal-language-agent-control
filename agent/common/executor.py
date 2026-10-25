"""Resumable, budget-capped, concurrent executor for a list of cells (one cell = one episode).

done.jsonl (append-only, one line per finished cell) is the source of truth for status; spend is calls.jsonl (one line
per billed call, 2026-10-17) plus done.jsonl costs of runs from before that journal; manifest.json is a periodic snapshot. Cells already 'done' are skipped on restart; a cell that failed on the service is retried on the
next start. Stops cleanly at the budget cap. Prints progress / ETA every 15 s. Interruptions (2026-10-17): see
run_plan; `python -m agent.common.executor runs/<run>` prints the status of a run, `--stop` asks it to stop.

The experiment package supplies: execute(cell, backend, spec, ledger) -> record, key_fn(cell) -> str,
backend_factory(kind, spec, seed) -> backend, config (dict snapshot) and report(run_dir) (optional).
"""
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import socket
import sys
import threading
import time
import uuid

from agent.harness.run_manifest import BudgetExceeded, Manifest

from .ledger import Ledger
from .models import PRICES_PATH, ROOT, load_models, resolve_model

RUNS = os.path.join(ROOT, "runs")
MAX_CONSECUTIVE_SERVICE_FAILURES = 8     # the network or the provider is down: stop instead of failing every cell


def summary_of(rec, keys):
    return {k: rec.get(k) for k in keys if k in rec}


def interleave(cells, key="model"):
    """Round-robin over models, keeping each model's own order, so that a model capped by max_concurrency runs
    alongside the others instead of all of its cells waiting at the end (2026-10-17, freeze log changes 2)."""
    groups = {}
    for c in cells:
        groups.setdefault(c[key], []).append(c)
    out, queues = [], [list(g) for g in groups.values()]
    while queues:
        for q in queues:
            out.append(q.pop(0))
        queues = [q for q in queues if q]
    return out


LOCK_FILE, STOP_FILE, JOURNAL_FILE, DONE_FILE = "run.lock", "STOP", "calls.jsonl", "done.jsonl"


def read_jsonl(path):
    """Records of an append-only log. A line cut off by a killed process is skipped and counted, not fatal."""
    out, bad = [], 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        bad += 1
    return out, bad


def open_append(path):
    """Append handle that starts on a fresh line, so a cut-off last line never swallows the next record."""
    if os.path.exists(path) and os.path.getsize(path):
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            cut = f.read(1) != b"\n"
        if cut:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
    return open(path, "a", encoding="utf-8")


def spend_of(run_dir):
    """Money spent in a run: every billed call in calls.jsonl (2026-10-17), plus done.jsonl records without any
    journaled call (runs before the journal). Attempts in the journal with no done.jsonl line are episodes that never
    finished (killed process, Ctrl+C, budget stop): their cost counts, and the cell is run again."""
    done, bad_done = read_jsonl(os.path.join(run_dir, DONE_FILE))
    calls, bad_calls = read_jsonl(os.path.join(run_dir, JOURNAL_FILE))
    by_attempt = collections.defaultdict(float)
    for c in calls:
        by_attempt[c.get("attempt")] += c.get("cost_usd", 0.0)
    finished = {d.get("attempt") for d in done if d.get("attempt")}
    unfinished = {a: v for a, v in by_attempt.items() if a not in finished}
    legacy = sum(d["cost_usd"] for d in done if d.get("attempt") not in by_attempt)
    return {"spent": sum(by_attempt.values()) + legacy, "done": done, "unfinished_episodes": len(unfinished),
            "unfinished_usd": sum(unfinished.values()), "cut_lines": bad_done + bad_calls,
            "last_activity": max([c.get("t", 0) for c in calls] + [d.get("t", 0) for d in done] + [0])}


def pid_alive(pid):
    """Is `pid` a running Python process on this machine (Windows: without signalling it)."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.restype = wintypes.HANDLE
        h = k.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            if not k.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:     # STILL_ACTIVE
                return False
            buf, n = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
            if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
                return "python" in buf.value.lower()          # a reused pid of another program is not our run
            return True
        finally:
            k.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(run_dir):
    path = os.path.join(run_dir, LOCK_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        info = {}
    info["alive"] = info.get("host") == socket.gethostname() and pid_alive(info.get("pid"))
    return info


def acquire_lock(run_dir):
    """One process per run directory (2026-10-17: two processes once ran the same cell). A lock whose process is gone
    means the previous run was interrupted; it is returned so that the caller can say so."""
    path = os.path.join(run_dir, LOCK_FILE)
    prev = read_lock(run_dir)
    if prev is not None:
        if prev["alive"] or prev.get("host") not in (None, socket.gethostname()):
            raise SystemExit(f"{run_dir} is in use by process {prev.get('pid')} on {prev.get('host')} (started "
                             f"{prev.get('started')}). If no run is going on, delete {path}.")
        os.remove(path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{run_dir} was just taken by another process ({path})") from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "argv": sys.argv}, f)
    return prev


def release_lock(run_dir):
    info = read_lock(run_dir)
    if info is not None and info.get("pid") == os.getpid():
        os.remove(os.path.join(run_dir, LOCK_FILE))


def void_cells(run_dir, keys, reason, archive_dir):
    """Mark finished cells to be run again (2026-10-19, freeze log changes after the freeze 11): their cell records are
    copied to archive_dir first, then a 'void' line per key is appended to done.jsonl, so the next start runs them
    again. Their money stays counted (calls.jsonl). Refused while a process runs the directory."""
    acquire_lock(run_dir)
    try:
        os.makedirs(archive_dir, exist_ok=True)
        cells_dir = os.path.join(run_dir, "cells")
        found = set()
        for root, _, files in os.walk(cells_dir):
            for f in files:
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    rec = json.load(fh)
                if rec.get("cell_key") in keys:
                    found.add(rec["cell_key"])
                    with open(os.path.join(archive_dir, f), "w", encoding="utf-8") as out:
                        json.dump(rec, out, ensure_ascii=False)
        missing = set(keys) - found
        if missing:
            raise SystemExit(f"no cell record for {len(missing)} key(s), nothing voided: {sorted(missing)[:3]}")
        with open_append(os.path.join(run_dir, DONE_FILE)) as f:
            for k in sorted(keys):
                f.write(json.dumps({"key": k, "status": "void", "cost_usd": 0.0, "t": time.time(),
                                    "reason": reason}, ensure_ascii=False) + "\n")
        return sorted(keys)
    finally:
        release_lock(run_dir)


def request_stop(run_dir):
    """Ask a running executor to stop: no new episodes start, running ones finish and are recorded."""
    with open(os.path.join(run_dir, STOP_FILE), "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def status(run_dir):
    """Progress of a run without running it (python -m agent.common.executor --status runs/<run>)."""
    sp = spend_of(run_dir)
    last = {}
    for d in sp["done"]:
        last[d["key"]] = d["status"]
    plan_path, n_cells, keys = os.path.join(run_dir, "plan.json"), None, None
    if os.path.exists(plan_path):
        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)
        n_cells, keys = plan.get("n_cells"), plan.get("keys")
    cap = None
    man = os.path.join(run_dir, "manifest.json")
    if os.path.exists(man):
        with open(man, encoding="utf-8") as f:
            cap = json.load(f).get("budget_cap_usd")
    lk = read_lock(run_dir)
    n_done = sum(1 for s in last.values() if s == "done")
    # the plan of the last start may be a slice of what the directory holds (2026-10-18: a filtered formal run in
    # d_main, where the pilot's 80 episodes are also done): count the plan's own cells by key
    plan_done = None if keys is None else sum(1 for k in keys if last.get(k) == "done")
    return {"run_dir": run_dir, "cells_in_plan": n_cells, "done_in_plan": plan_done, "done": n_done,
            "failed_to_rerun": sum(1 for s in last.values() if s != "done"),
            "not_done": None if plan_done is None else len(keys) - plan_done, "spent_usd": round(sp["spent"], 4),
            "cap_usd": cap, "unfinished_episodes": sp["unfinished_episodes"],
            "unfinished_usd": round(sp["unfinished_usd"], 4), "cut_lines": sp["cut_lines"],
            "state": "running" if lk and lk["alive"] else "interrupted (lock left by a process that is gone)" if lk
            else "not running", "pid": lk.get("pid") if lk else None, "started": lk.get("started") if lk else None,
            "stop_requested": os.path.exists(os.path.join(run_dir, STOP_FILE)),
            "last_activity": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sp["last_activity"]))
            if sp["last_activity"] else None}


def run_plan(cells, run_name, budget, execute, key_fn, backend_factory, backend_kind, workers=4, runs_dir=RUNS,
             config=None, report=None, summary_keys=("termination", "cost_usd", "tokens_in", "tokens_out"),
             quiet=False, raise_cap=False):
    """raise_cap: a resumed run keeps the cap stored in its manifest; only an explicit raise_cap lifts it to `budget`,
    and every change is appended to budget_cap_changes.jsonl.

    Interruptions (2026-10-17): one process per run directory (run.lock); every billed call is journaled, so a killed
    run resumes with its true spend; a STOP file in the run directory (request_stop) or one Ctrl+C stops starting new
    episodes and lets the running ones finish; a second Ctrl+C exits at once (the journal keeps the money spent)."""
    run_dir = os.path.join(runs_dir, run_name)
    snap_path = os.path.join(run_dir, "config_snapshot.json")
    if config and os.path.exists(snap_path):        # never resume a run recorded under another protocol
        with open(snap_path, encoding="utf-8") as f:
            old = json.load(f).get("contract_version")
        if old is not None and old != config.get("contract_version"):
            raise SystemExit(f"{run_dir} holds contract {old}, this code is {config.get('contract_version')}: "
                             "move that run aside or choose another run name")
    os.makedirs(os.path.join(run_dir, "cells"), exist_ok=True)
    prev_lock = acquire_lock(run_dir)
    try:
        return _run_locked(cells, run_name, run_dir, snap_path, budget, execute, key_fn, backend_factory,
                           backend_kind, workers, config, report, summary_keys, quiet, raise_cap, prev_lock)
    finally:
        release_lock(run_dir)


def _run_locked(cells, run_name, run_dir, snap_path, budget, execute, key_fn, backend_factory, backend_kind, workers,
                config, report, summary_keys, quiet, raise_cap, prev_lock):
    say = (lambda msg: None) if quiet else (lambda msg: print(f"[{run_name}] {msg}", flush=True))
    stop_path = os.path.join(run_dir, STOP_FILE)
    if os.path.exists(stop_path):
        os.remove(stop_path)
        say("removed the stop request left by the previous run")
    manifest = Manifest.load_or_create(os.path.join(run_dir, "manifest.json"), budget)
    if raise_cap and budget > manifest.budget_cap_usd:
        with open(os.path.join(run_dir, "budget_cap_changes.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "from": manifest.budget_cap_usd,
                                "to": budget}) + "\n")
        manifest.budget_cap_usd = budget
        manifest._save()
    sp = spend_of(run_dir)
    for d in sp["done"]:
        manifest.cells[d["key"]] = {"status": d["status"], "result": d.get("summary"), "updated_at": d["t"]}
    spent = sp["spent"]
    manifest.spent_usd = spent
    if prev_lock is not None:
        say(f"the previous run (pid {prev_lock.get('pid')}, started {prev_lock.get('started')}) was interrupted")
    if sp["unfinished_episodes"]:
        say(f"{sp['unfinished_episodes']} episode(s) never finished (interrupted or budget stop): "
            f"${sp['unfinished_usd']:.4f} spent on them is counted; those cells run again")
    if sp["cut_lines"]:
        say(f"skipped {sp['cut_lines']} cut-off log line(s) (a killed process); those cells run again")
    ledger = Ledger(manifest.budget_cap_usd, spent, os.path.join(run_dir, JOURNAL_FILE))
    say(f"budget cap ${manifest.budget_cap_usd:.2f}, already spent ${spent:.4f}")
    done_log = open_append(os.path.join(run_dir, DONE_FILE))
    last_save = [time.time(), 0]
    with open(os.path.join(run_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "backend": backend_kind, "n_cells": len(cells),
                   "keys": [key_fn(c) for c in cells], "cells": cells}, f, ensure_ascii=False)
    with open(PRICES_PATH, encoding="utf-8") as f:
        prices = json.load(f)
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump({"model_prices": prices, **(config or {})}, f, ensure_ascii=False, indent=2)
    models = load_models()
    backends, lock = {}, threading.Lock()

    def backend_for(alias):
        with lock:
            if alias not in backends:
                spec = resolve_model(alias, models)
                backends[alias] = (backend_factory(backend_kind, spec, seed=len(backends)), spec)
            return backends[alias]

    todo = interleave([c for c in cells if manifest.status(key_fn(c)) != "done"])
    caps = {}                                        # ModelSpec.max_concurrency: at most n episodes of a model at once
    for alias in {c["model"] for c in todo}:
        try:
            n = resolve_model(alias, models).max_concurrency
        except KeyError:                             # unknown alias: the cell fails in backend_for, as before
            n = None
        if n:
            caps[alias] = n
    stop = threading.Event()
    progress = {"t0": time.time(), "last": 0.0}
    stats = {"done": 0, "failed": 0, "budget_stop": False, "breaker_stop": False, "stopped": False}
    streak = [0]

    def save_manifest():
        try:
            manifest._save()
        except OSError as e:                         # e.g. the file held open by a virus scanner: done.jsonl is the truth
            say(f"manifest snapshot not saved ({type(e).__name__}: {e}); done.jsonl is unaffected")

    def work(cell):
        if stop.is_set():
            return
        key = key_fn(cell)
        attempt = uuid.uuid4().hex[:12]
        backend, spec = backend_for(cell["model"])
        try:
            ledger.begin(key, attempt)
            rec = execute(cell, backend, spec, ledger)
        except BudgetExceeded as e:
            stop.set()
            stats["budget_stop"] = str(e)
            return
        except Exception as e:                       # one broken cell must not abort the run: record it as failed
            import traceback
            rec = {"termination": "error", "cost_usd": ledger.attempt_cost(attempt), "tokens_in": 0, "tokens_out": 0,
                   "error_message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
            say(f"cell {key} raised {type(e).__name__}: {e}")
        name = hashlib.sha1(key.encode()).hexdigest()[:16]
        sub = os.path.join(run_dir, "cells", cell["exp"])
        os.makedirs(sub, exist_ok=True)
        rec["cell_key"] = key
        rec["attempt"] = attempt
        with open(os.path.join(sub, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        failed = rec.get("termination") in ("service", "error")
        ledger.persist(rec["cost_usd"])
        status_ = "failed" if failed else "done"
        with lock:
            now = time.time()
            done_log.write(json.dumps({"key": key, "status": status_, "cost_usd": rec["cost_usd"], "t": now,
                                       "attempt": attempt, "summary": summary_of(rec, summary_keys)},
                                      ensure_ascii=False) + "\n")
            done_log.flush()
            manifest.cells[key] = {"status": status_, "result": summary_of(rec, summary_keys), "updated_at": now}
            manifest.spent_usd = ledger.spent
            last_save[1] += 1
            if now - last_save[0] > 30 or last_save[1] >= 200:
                save_manifest()
                last_save[:] = [now, 0]
            stats["failed" if failed else "done"] += 1
            streak[0] = streak[0] + 1 if failed else 0
            if streak[0] >= MAX_CONSECUTIVE_SERVICE_FAILURES and not stop.is_set():
                stop.set()
                stats["breaker_stop"] = f"{streak[0]} consecutive failed cells: stopped (network or provider down?)"
                say(stats["breaker_stop"])
            n = stats["done"] + stats["failed"]
            if now - progress["last"] > 15 or n == len(todo):
                progress["last"] = now
                el = now - progress["t0"]
                eta = el / n * (len(todo) - n)
                say(f"{n}/{len(todo)} cells ({n / len(todo):.0%}), failed {stats['failed']}, "
                    f"spent ${ledger.spent:.4f} of ${ledger.cap:.2f}, elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min")

    queue, running, load = list(todo), {}, collections.Counter()

    def dispatch():
        """Start cells on free workers in queue order; a model at its cap is skipped, not waited for (2026-10-18,
        freeze log changes 7: waiting workers had held the other models back to the capped model's pace)."""
        i = 0
        while len(running) < workers and i < len(queue) and not stop.is_set():
            m = queue[i]["model"]
            if m in caps and load[m] >= caps[m]:
                i += 1
                continue
            cell = queue.pop(i)
            load[m] += 1
            running[ex.submit(work, cell)] = m

    def halt(reason):
        if not stats["stopped"]:
            stats["stopped"] = reason
            stop.set()
            say(f"{reason}: no new episodes start; waiting for {len(running)} running episode(s) to finish"
                + (" (Ctrl+C again to quit at once; the money spent so far stays in the journal)"
                   if reason == "Ctrl+C" else ""))

    ex = cf.ThreadPoolExecutor(max_workers=workers)
    try:
        dispatch()
        while running:
            try:                                     # short waits keep the main thread responsive to Ctrl+C
                finished, _ = cf.wait(list(running), timeout=1.0, return_when=cf.FIRST_COMPLETED)
                for fut in finished:
                    load[running.pop(fut)] -= 1
                    fut.result()
                if os.path.exists(stop_path):
                    halt("stop requested (STOP file)")
                dispatch()
            except KeyboardInterrupt:
                if stats["stopped"] == "Ctrl+C":
                    say("second Ctrl+C: quitting now; unfinished episodes run again on the next start")
                    ledger.close()
                    done_log.close()
                    release_lock(run_dir)
                    os._exit(130)
                halt("Ctrl+C")
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        ledger.close()
        done_log.close()
    if os.path.exists(stop_path):
        os.remove(stop_path)
    manifest.spent_usd = ledger.spent
    save_manifest()
    stats["spent_usd"] = spend_of(run_dir)["spent"]
    stats["remaining"] = len([c for c in cells if manifest.status(key_fn(c)) != "done"])
    if not quiet and report is not None:
        report(run_dir)
    return stats


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="status of a run, or ask a running one to stop")
    ap.add_argument("run_dir")
    ap.add_argument("--stop", action="store_true", help="running episodes finish, no new ones start")
    a = ap.parse_args(argv)
    if a.stop:
        request_stop(a.run_dir)
        print(f"stop requested for {a.run_dir}")
    print(json.dumps(status(a.run_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
