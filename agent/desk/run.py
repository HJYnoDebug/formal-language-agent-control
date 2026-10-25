"""Run a help-desk plan: python -m agent.desk.run --plan pilot --run d_pilot_2026-10-13 --budget 4.0

--backend openrouter            real calls (needs API_KEY_PATH; the key is read from that file and never printed)
--backend fake:<policy>         offline (see runner.FakeToolBackend), e.g. fake:view:compliant, fake:dev:memory:2

Uses the shared executor (agent/common/executor.py): resumable via done.jsonl, budget cap, progress / ETA, report.
--status / --stop with --run: progress of a run, or ask the running process to stop after its running episodes.
"""
import argparse
import json
import os
import sys

from agent.common import executor
from agent.common.toolcall import OpenRouterToolBackend

from . import plan as planmod, runner
from .report import post_run_report, print_report


def execute_cell(cell, backend, spec, ledger):
    return runner.run_episode(planmod.get_schedule(cell), cell["rule"], cell["control"], backend, spec, ledger, cell,
                              seed=cell.get("seed", 0))


def make_backend(kind, spec, seed=0):
    if kind == "openrouter":
        path = os.environ.get("API_KEY_PATH")
        if not path or not os.path.exists(path):
            raise SystemExit("set API_KEY_PATH to the file holding the OpenRouter key")
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
        return OpenRouterToolBackend(spec, key)
    if kind.startswith("fake:"):
        return runner.FakeToolBackend(spec, kind.split(":", 1)[1], seed)
    raise SystemExit(f"unknown backend {kind}")


def report(run_dir):
    print_report(post_run_report(run_dir))


def config():
    return {"contract_version": runner.CONTRACT_VERSION, "max_tokens": runner.MAX_TOKENS,
            "max_calls_per_turn": runner.MAX_CALLS_PER_TURN, "extra_calls_device": runner.EXTRA_CALLS_DEVICE,
            "max_blocks_per_turn": runner.MAX_BLOCKS_PER_TURN,
            "max_format_errors_per_episode": runner.MAX_FORMAT_ERRORS_PER_EPISODE, "regular_rule": planmod.REGULAR}


def run_plan(cells, run_name, budget, workers=4, backend_kind="fake:compliant", runs_dir=executor.RUNS, quiet=False,
             raise_cap=False):
    return executor.run_plan(cells, run_name, budget, execute_cell, planmod.cell_key, make_backend, backend_kind,
                             workers=workers, runs_dir=runs_dir, config=config(), report=None if quiet else report,
                             summary_keys=("termination", "cost_usd", "tokens_in", "tokens_out", "completion",
                                           "score"), quiet=quiet, raise_cap=raise_cap)


def _backup_if_paid(backend, run_name, plan_name, runs_dir):
    """Paid data is backed up at once (agent/common/backup.py); a failed backup is loud and changes the exit code."""
    if backend != "openrouter":
        return 0
    from agent.common.backup import backup_run
    try:
        info = backup_run(os.path.join(runs_dir, run_name), note=f"plan={plan_name}, after the run")
    except Exception as e:                                   # noqa: BLE001
        print(f"BACKUP FAILED for {run_name}: {type(e).__name__}: {e}")
        return 3
    print(f"backup {info['zip']}  files={info['files']}  sha256={info['sha256']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(planmod.PLANS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=float)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backend", default="fake:compliant")
    ap.add_argument("--models", default=None)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--controls", default=None, help="only these controls, e.g. C0,M (a first slice of a plan)")
    ap.add_argument("--depths", default=None, help="only these depths D, e.g. 10")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--raise-cap", action="store_true", help="lift the stored cap of a resumed run to --budget (logged)")
    ap.add_argument("--status", action="store_true", help="print the progress of --run and exit (no calls)")
    ap.add_argument("--stop", action="store_true", help="ask the process running --run to stop after its running episodes")
    a = ap.parse_args(argv)
    if a.status or a.stop:
        return executor.main([os.path.join(executor.RUNS, a.run)] + (["--stop"] if a.stop else [])) or 0
    if a.plan is None or a.budget is None:
        ap.error("--plan and --budget are required to run")
    cells = planmod.PLANS[a.plan]()
    if a.models:
        cells = [c for c in cells if c["model"] in set(a.models.split(","))]
    if a.rules:
        cells = [c for c in cells if c["rule"] in set(a.rules.split(","))]
    if a.controls:
        cells = [c for c in cells if c["control"] in set(a.controls.split(","))]
    if a.depths:
        cells = [c for c in cells if c["D"] in {int(d) for d in a.depths.split(",")}]
    if a.limit:
        cells = cells[: a.limit]
    print(f"{len(cells)} episodes, backend {a.backend}, budget ${a.budget}")
    stats = run_plan(cells, a.run, a.budget, a.workers, a.backend, raise_cap=a.raise_cap)
    print(json.dumps(stats, indent=2))
    code = 0 if not (stats["budget_stop"] or stats.get("breaker_stop") or stats.get("stopped")) else 2
    return _backup_if_paid(a.backend, a.run, a.plan, executor.RUNS) or code


if __name__ == "__main__":
    sys.exit(main())
