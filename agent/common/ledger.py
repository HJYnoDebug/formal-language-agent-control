"""Thread-safe cost ledger with a hard budget cap, and the error raised when the model service keeps failing."""
import collections
import json
import threading
import time

from agent.harness.run_manifest import BudgetExceeded  # noqa: F401  (re-exported)


class ServiceFailure(Exception):
    pass


class Ledger:
    """The cap is checked before every call (in-flight estimates are reserved).
    spent: cost already persisted (the executor keeps it in an append-only done.jsonl).
    journal_path: optional append-only calls.jsonl, one line per billed call with the cell key and the attempt id,
    flushed at once, so that money spent inside an episode that never finished (killed process, budget stop) is still
    counted when the run resumes (2026-10-17, freeze log changes after the freeze 3)."""

    def __init__(self, cap_usd, spent_usd=0.0, journal_path=None):
        self.cap = cap_usd
        self.spent = spent_usd
        self.lock = threading.Lock()
        self.inflight = 0.0
        self.committed = 0.0      # spent in running cells, not yet persisted
        self.local = threading.local()
        self.by_attempt = collections.defaultdict(float)
        self.journal = open(journal_path, "a", encoding="utf-8") if journal_path else None

    def begin(self, key, attempt):
        """The episode run by this thread from now on (each episode runs in one worker thread)."""
        self.local.cell = (key, attempt)

    def reserve(self, est):
        with self.lock:
            total = self.spent + self.committed + self.inflight
            if total + est > self.cap:
                raise BudgetExceeded(total, self.cap, est)
            self.inflight += est

    def settle(self, est, actual):
        with self.lock:
            self.inflight -= est
            self.committed += actual
            key, attempt = getattr(self.local, "cell", (None, None))
            self.by_attempt[attempt] += actual
            if self.journal is not None and actual:
                self.journal.write(json.dumps({"key": key, "attempt": attempt, "cost_usd": actual,
                                               "t": round(time.time(), 3)}) + "\n")
                self.journal.flush()

    def attempt_cost(self, attempt):
        with self.lock:
            return self.by_attempt.get(attempt, 0.0)

    def persist(self, amount):
        with self.lock:
            self.committed -= amount
            self.spent += amount

    def close(self):
        with self.lock:
            if self.journal is not None:
                self.journal.close()
                self.journal = None
