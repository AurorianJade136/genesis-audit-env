# Design note

## Overview
`generator` → `data/<ep>/` (what the agent may see) and `truth/<ep>/truth.json` (answer key).
`AuditEnv` serves the visible data through five actions. When the agent submits, the env hands the
findings to an **evaluator object** and returns only the reward. The matcher/reward code lives in the
evaluator side.

## Isolation (what is and is not guaranteed)
* The answer key is never inside the agent-visible episode directory and `AuditEnv` never opens it.
* On-disk episodes are scored by `SubprocessEvaluator`, which runs `python -m src.evaluator` as a
  **separate process**; the key is loaded only there. The harness (`src.runner.make_env`) creates the
  evaluator and hands the env an opaque object; the agent interface is just `reset`/`step`.
* No intermediate scoring: `step` returns reward 0 until `submit` (or timeout). `info` never carries TP/FP/FN/F1;
  those exist only in `evaluator.last_result` / `env.evaluator_report()`, for the harness after the run.
* Seed-generated episodes (`reset(seed=…)`) use an in-process `LocalEvaluator` (the key lives inside that object,
  not on the env). This is a convenience for training and tests, a weaker boundary than the subprocess.
* Limitation: this is process isolation on one machine, not a sandbox. The keys are plain files in the repo,
  so a policy that is handed the repo path could read them; agents must only receive the env object.

## Episode generator
Each seed yields a different company: period (month), time zone (fixed offsets), base currency, 1–3 foreign
currencies (incl. JPY with exponent 0 and KWD with exponent 3), a daily FX table (±0.4–1.8 % random walk),
7–12 vendors, 20–34 regular invoices plus distractors, ~90–200 ledger rows, mixed UTC/local timestamps, and
three record orderings (chronological, by entity, shuffled). Amounts, dates, IDs and FX rates all vary per seed.
Error counts for the four types and the two interacting scenarios are configured **independently**;
`{"clean": True}` gives a zero-error company. The 20 fixed episodes use disjoint seeds
(train 1001–1010, dev 2001–2010) and different profiles (incl. 4 clean ones).

### Error definitions
| type | what the books show | expected amount |
|---|---|---|
| `duplicate_invoice` | an invoice paid twice, **or** the same invoice entered twice under two IDs | invoice total (invoice currency) |
| `unallocated_or_partial_payment` | payable cleared via `5900_Purchase_Adjustments` although less was paid, **or** a payment never applied to its invoice | shortfall (partial) / payment amount (unapplied), invoice currency |
| `period_cutoff_error` | invoice issued before the period (or after it) but booked inside it | invoice total |
| `bad_fx_date_or_rate` | invoice or payment booked at the rate of another date (policy: rate on invoice issue date) | \|booked − correct\| in base currency |

Interacting scenarios (one issue each, with `secondary_types`):
1. **partial payment + stale FX** at the period boundary (cash booked at an old rate, shortfall plugged to 5900);
2. **double payment of a cut-off invoice** (prior-period invoice booked in this period, then paid twice).
A finding matches an interacting issue if its type is the primary *or* a secondary type.

### Distractors (clean, but look suspicious)
same vendor/amount with different descriptions weeks apart; legitimate instalments; legitimate partial payment with the
balance left open; invoices issued seconds after the period start / before its end, shown in UTC so a naive date
comparison disagrees with the local period; a payment one minute before period end.

## Matching and reward
* Match = same type (or secondary type), same affected-ID set, same currency, amount within `amount_tolerance_minor`,
  and `evidence_ids` ⊇ one of `valid_evidence_sets`.
* One-to-one **maximum bipartite matching** (augmenting paths): order of findings does not matter; duplicate findings
  count as false positives.
* `F1` over matches; both empty ⇒ F1 = 1 (a clean company is answered by silence).
* `R = clip(2·F1 − 1 − 0.002·S, −1, 1)`; `S` = number of non-submit steps taken.
* Submission limits (30 findings, 10 evidence IDs, 1000-char explanation, types, integer amounts) are validated before scoring.

### Assumptions to check against the task statement
* Malformed submissions are **rejected without ending the episode** (the step is still spent).
* Hitting the 100-step budget without submitting ends the episode as `truncated` and is scored as an empty submission.
* Truth `affected_ids` are compared as sets; `explanation` is not scored.

## Policies (`policies/`)
1. `random_explorer` – random inspections, then submits nothing (≈ −1 on dirty companies, ≈ +0.9 on clean ones).
2. `reconciliation` – pages all tables with cursors and applies deterministic rules; reaches F1 = 1.0 on all 20 episodes
   and on the original hand-written sample (it reads only the observation, never the key).
3. `flag_everything` – attacker reporting 30 invoices with guessed evidence; scores −1.0 everywhere.
4. `oracle` – built from the key on the evaluator side; submits the perfect answer in one step (reward 1.0).

## Tests (`tests/`, 200+ cases)
generator (reproducibility, difference between episodes, balance, key↔data consistency, coverage of clean /
independent / interacting episodes), matcher (match, evidence, tolerance, duplicates, clean company, order-independence,
reward formula, validation limits), environment (contract, cursors, FX access, all actions, calculator safety,
timeout, repeated submission, isolation / no leakage, subprocess evaluator parity, seed mode), policies, trajectories.

## Known limitations
* The reconciliation baseline encodes the generator's error mechanics; it demonstrates solvability, not general ability.
* No DST time zones; one policy variant (`fx_rate_date = invoice_issue_date`).
* Isolation is process-level (see above).
