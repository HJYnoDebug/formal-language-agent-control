# Help-desk agent testbed: procedure languages and partial controllers

A testbed for one question: how much of a history-dependent rule must be held outside a tool-calling LLM agent for
the rule to be kept, and what does that cost?

## Introduction

A tool-calling agent runs a simulated help desk. Tickets arrive during a working day, and each needs two tool steps
(`verify`, then `refund`, `replace` or `update_address`). Escalated tickets must be started at once, so they
interrupt the work in progress and put tickets on hold. One sentence of the system prompt, the procedure, decides
which ticket on hold may be resumed. The agent decides everything else: which ticket to work on, when to look
something up, when to switch.

The agent's trajectory is projected onto a word of events (arrive, step, hold, resume, close), and the procedure
becomes a formal language over such words. Four procedures that differ only in that one sentence span the Chomsky
hierarchy:

| Rule | Which ticket on hold may be resumed | Language class | Matched storage |
| --- | --- | --- | --- |
| PAR | alphabetically first if the number of refunds so far is even, else last | regular | one bit |
| BAL | first if refunds ≥ replacements so far, else last | one-counter | a counter |
| LIFO | the one most recently put on hold | context-free | a stack |
| FIFO | the one earliest put on hold | context-sensitive | a restricted queue |

A recognizer of the procedure language holds a configuration, updates it with every event and judges which resume
is legal. The control conditions hand these parts to the runtime one at a time:

- **C0**: the procedure sentence only; **R**: plus a description of the method.
- **N**, **M−**, **M**: a storage device the agent keeps itself: a free notebook, a device one class too weak, the
  matched device.
- **C1**: the matched device, kept up to date by the runtime and shown every turn.
- **C2**: C1 plus the legal ticket named.
- **E**: a gate that blocks an illegal resume, with nothing shown; **C3**: C2 plus the gate.
- **P** (a placebo display of the same length) and **T** (written reasoning before every step) are also implemented.

## Partial control and the workflow boundary

The controls form a ladder from a bare prompt towards a workflow, but none of them reaches it. Even under C3, which
holds the rule's state, names the legal ticket and blocks every illegal resume, the controller governs one kind of
decision only: which ticket on hold is resumed. The agent still decides which ticket to work on, when to start a
waiting ticket, when to look something up and when to switch, so it remains an agent. At the far end, the reference
condition **W** is a workflow: a program that schedules every step by the procedure and runs no model
(`runner.run_workflow`), the baseline for completion and cost.

Two limits mark the boundary:

- **A gate enforces safety, not progress.** It can block a resume that would leave the procedure language, but it
  cannot make the agent finish its tickets. That liveness property is only monitored: after a lapse the agent is
  reminded that tickets are still on hold, without being told which.
- **Only decisions fixed by the rule are handed over.** The resume decision is determined by the rule and the
  history, so a controller can hold it; decisions that depend on information the designer does not have stay with
  the agent. Handing over more than the rule requires turns the agent into a workflow.

## Design principles

- **One change per comparison.** Each planned pair of conditions differs in exactly one thing: storage, its class,
  who updates it, the judgment, the gate, the method text, or the length of the display.
- **The procedure judges one decision only.** Starting a waiting ticket or continuing the current one is always
  legal, so the agent always has a legal move. Environment errors (wrong step order, wrong customer id) are reported
  and never counted as violations.
- **The screen never gives the answer away.** The tools list tickets on hold alphabetically and show neither when nor
  in which order they were put on hold; the generator balances categories and codes so that no visible attribute
  predicts the legal ticket.
- **Only history-dependent decisions count.** A diagnostic point is a violable resume at which no policy that reads
  only the current window is always right.
- **Days are generated, fixed and shared.** A day is a deterministic function of its key, with a fixed number of
  tickets and steps, so context length and cost do not grow with the depth; every model, rule and control runs on
  the same days.
- **Neutral prompts.** No worked example, no words such as "queue" or "first come, first served"; a device
  description says what the device is, never when or how to use it.
- **Thresholds are derived, not chosen.** Every threshold is computed in code from a few fixed conventions
  (`agent/desk/prereg.py`), and an offline acceptance suite with simulated agents checks the platform before any
  paid run.
- **The judge matches the formal definitions.** The platform's judgments are checked against explicit grammars of
  the four languages on all short words (`analysis/verify_grammars.py`).

## Layout

| Path | Contents |
| --- | --- |
| `agent/desk/` | the platform: day generator, rules and judge, environment, prompts, devices, episode runner, run plans, preregistered constants, offline acceptance, comprehension probe, run entry point |
| `agent/common/`, `agent/harness/` | model client, tool-call backend, budget ledger, interruption-safe executor, run manifests, backups of run directories |
| `analysis/` | the main analysis (`desk_analysis.py`), its figures (`desk_figures.py`), the checks of the grammars and of the automaton sizes (`verify_grammars.py`, `verify_state_counts.py`) |
| `config/model_prices.json` | model ids, providers and prices |
| `tests/` | regression tests, all offline (a fake backend replaces the model) |

## Running

Python 3.11 with `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib` and `pytest` (`requirements.txt`). From the
repository root:

```
python -m pytest tests                               # offline regression tests
python -m agent.desk.acceptance                      # offline acceptance with simulated agents
python -m agent.desk.run --plan smoke --run d_smoke --budget 0.3 --backend fake:view:compliant
```

Real model calls use `--backend openrouter` and read the API key from the file named by the environment variable
`API_KEY_PATH`; the key is never printed or stored in the repository. Run directories are written to `runs/`, which
is not tracked.
