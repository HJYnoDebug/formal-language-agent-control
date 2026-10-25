"""Cell-level checkpoint/resume + run-level budget cap (harness_design.md
sec 2.3's third and fourth bullets).

A "cell" is one (env, arm, model_tier, task_id, seed) -- one row of spec
sec 5's control-mode table crossed with one task and one model tier. The
manifest tracks each cell's status (pending/done/failed) plus the run's
cumulative cost, and is the single source of truth a resumed run reads to
decide what is left to do.

Deliberately NOT step-level resume (harness_design.md sec 2.3): a failed or
interrupted cell is re-run FROM SCRATCH, never resumed mid-episode. This is
recorded here again (not just in harness_design.md) because it changes this
module's shape directly -- there is no per-step checkpoint API at all, only
mark_done()/mark_failed() for a whole cell.
"""
import json
import os
import time


def cell_key(env, arm, model_tier, task_id, seed):
    """A cell's canonical string key. JSON object keys must be strings, so
    this (not a tuple) is what Manifest actually stores cells under -- kept
    as a single function so every caller builds the same key the same way."""
    return "|".join(str(x) for x in (env, arm, model_tier, task_id, seed))


class BudgetExceeded(Exception):
    """Raised by Manifest.check_budget() -- callers (episode_runner_v2.py)
    catch this to stop a run CLEANLY (record it in the manifest) rather than
    let it propagate as an uncaught crash, per harness_design.md sec 2.3:
    "超出则整个运行干净停止（记录到 manifest,不是异常崩溃）"."""

    def __init__(self, spent, cap, estimated_next_call):
        self.spent = spent
        self.cap = cap
        self.estimated_next_call = estimated_next_call
        super().__init__(
            f"budget exceeded: spent ${spent:.4f} + estimated next call "
            f"${estimated_next_call:.4f} > cap ${cap:.4f}"
        )


class Manifest:
    """One Manifest per RUN (i.e. per budget batch -- spec sec 10's batches
    each get their own budget cap and their own manifest file, so batches
    can be run independently without one's spend counting against
    another's cap)."""

    def __init__(self, path, budget_cap_usd):
        self.path = path
        self.budget_cap_usd = budget_cap_usd
        self.cells = {}  # cell_key -> {"status": ..., "result": ..., "updated_at": ...}
        self.spent_usd = 0.0

    @classmethod
    def load_or_create(cls, path, budget_cap_usd):
        """Resumes from `path` if it already exists (budget_cap_usd is then
        read from the file, NOT overwritten by this call's argument, so a
        resumed run cannot silently change its own cap mid-flight); creates
        a fresh manifest at `path` otherwise."""
        if os.path.exists(path):
            m = cls(path, budget_cap_usd)
            m._load()
            return m
        m = cls(path, budget_cap_usd)
        m._save()
        return m

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.budget_cap_usd = data["budget_cap_usd"]
        self.spent_usd = data["spent_usd"]
        self.cells = data["cells"]

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "budget_cap_usd": self.budget_cap_usd,
            "spent_usd": self.spent_usd,
            "cells": self.cells,
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)  # atomic on POSIX; avoids a half-written
        # manifest if the process dies mid-save, which matters here more than
        # for a plain trace file since resume logic reads this file back.

    def status(self, key):
        cell = self.cells.get(key)
        return cell["status"] if cell else "pending"

    def pending_cells(self, all_keys):
        """all_keys the caller wants covered this run -> the subset still
        pending or failed (i.e. NOT done) -- this is what a resumed run
        should iterate over instead of all_keys."""
        return [k for k in all_keys if self.status(k) != "done"]

    def mark_in_progress(self, key):
        self.cells[key] = {"status": "in_progress", "result": None, "updated_at": time.time()}
        self._save()

    def mark_done(self, key, result):
        self.cells[key] = {"status": "done", "result": result, "updated_at": time.time()}
        self._save()

    def mark_failed(self, key, error):
        self.cells[key] = {
            "status": "failed",
            "result": {"error": str(error)},
            "updated_at": time.time(),
        }
        self._save()

    def add_cost(self, amount_usd):
        self.spent_usd += amount_usd
        self._save()

    def remaining_budget(self):
        return self.budget_cap_usd - self.spent_usd

    def check_budget(self, estimated_next_call_usd):
        """Raises BudgetExceeded if spending this much more would exceed the
        cap; a no-op (returns None) otherwise. Called BEFORE every LLM call,
        not after -- the point is to never place a call that would blow the
        budget, not to notice afterward (harness_design.md sec 2.3: "每次
        LLM 调用前检查")."""
        if self.spent_usd + estimated_next_call_usd > self.budget_cap_usd:
            raise BudgetExceeded(self.spent_usd, self.budget_cap_usd, estimated_next_call_usd)
