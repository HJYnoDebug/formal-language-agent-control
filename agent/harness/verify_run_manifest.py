"""Scripted verification of run_manifest.py -- pure stdlib file I/O in a
scratch directory, no LLM/API/environment needed.
"""
import shutil
import sys
import tempfile
import os

from run_manifest import BudgetExceeded, Manifest, cell_key

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


tmp_dir = tempfile.mkdtemp(prefix="verify_run_manifest_")
manifest_path = os.path.join(tmp_dir, "run1", "manifest.json")

# ---------------------------------------------------------------------------
# 1. cell_key is a stable, deterministic string.
# ---------------------------------------------------------------------------
k1 = cell_key("alfworld", "A-enforce", "high", "task_007", 3)
k2 = cell_key("alfworld", "A-enforce", "high", "task_007", 3)
check("cell_key: same inputs -> identical key", k1 == k2)
check("cell_key: different seed -> different key",
      k1 != cell_key("alfworld", "A-enforce", "high", "task_007", 4))

# ---------------------------------------------------------------------------
# 2. Fresh manifest: everything starts pending.
# ---------------------------------------------------------------------------
m = Manifest.load_or_create(manifest_path, budget_cap_usd=1.0)
check("fresh manifest: unknown cell status is 'pending'", m.status(k1) == "pending")
check("fresh manifest: file was created on disk", os.path.exists(manifest_path))

all_keys = [k1, k2 if k2 != k1 else cell_key("alfworld", "A-enforce", "high", "task_008", 3)]
check("pending_cells: a fresh manifest returns every key as pending",
      set(m.pending_cells(all_keys)) == set(all_keys))

# ---------------------------------------------------------------------------
# 3. mark_in_progress / mark_done / mark_failed, and persistence across reload.
# ---------------------------------------------------------------------------
m.mark_in_progress(k1)
check("mark_in_progress: status updates in memory", m.status(k1) == "in_progress")

m.mark_done(k1, {"n_steps": 12, "won": True})
check("mark_done: status becomes 'done'", m.status(k1) == "done")

m2 = Manifest.load_or_create(manifest_path, budget_cap_usd=999.0)
check("reload: budget_cap_usd is read from the FILE, not overridden by the reload call's argument",
      m2.budget_cap_usd == 1.0)
check("reload: a done cell's status survives a reload", m2.status(k1) == "done")
check("reload: pending_cells excludes the done cell",
      k1 not in m2.pending_cells(all_keys))

other_key = cell_key("scienceworld", "P", "mid", "task_x", 1)
m2.mark_failed(other_key, RuntimeError("environment crashed"))
check("mark_failed: status becomes 'failed'", m2.status(other_key) == "failed")
check("pending_cells: a FAILED cell counts as still-pending for a resumed run "
      "(re-run from scratch, not resumed mid-episode)",
      other_key in m2.pending_cells([other_key]))

# ---------------------------------------------------------------------------
# 4. Budget tracking.
# ---------------------------------------------------------------------------
m3 = Manifest.load_or_create(os.path.join(tmp_dir, "run2", "manifest.json"), budget_cap_usd=1.0)
m3.check_budget(0.5)  # should not raise -- well within cap
m3.add_cost(0.5)
check("budget: remaining_budget reflects cost added so far",
      abs(m3.remaining_budget() - 0.5) < 1e-9)

m3.check_budget(0.5)  # exactly at cap -- should not raise
try:
    m3.check_budget(0.51)
    check("budget: check_budget raises BudgetExceeded when spent+estimate > cap", False)
except BudgetExceeded as e:
    check("budget: check_budget raises BudgetExceeded when spent+estimate > cap", True)
    check("budget: BudgetExceeded carries spent/cap/estimated_next_call for logging",
          abs(e.spent - 0.5) < 1e-9 and abs(e.cap - 1.0) < 1e-9 and abs(e.estimated_next_call - 0.51) < 1e-9)

m3.add_cost(0.5)  # now fully spent
try:
    m3.check_budget(0.0001)
    check("budget: even a tiny next-call estimate is refused once the cap is fully spent", False)
except BudgetExceeded:
    check("budget: even a tiny next-call estimate is refused once the cap is fully spent", True)

m4 = Manifest.load_or_create(os.path.join(tmp_dir, "run2", "manifest.json"), budget_cap_usd=1.0)
check("budget: cumulative spend survives a reload", abs(m4.spent_usd - 1.0) < 1e-9)

shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
else:
    print("All checks passed.")
