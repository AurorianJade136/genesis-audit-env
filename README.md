# Genesis Audit Env

A reinforcement-learning environment in which an agent audits a small synthetic company's books
(ledger, invoices, payments, FX table) and submits the financial errors it can prove.
Episodes are generated from fixed seeds; scoring is done by a closed evaluator the agent cannot reach.

* 4 error types: `duplicate_invoice`, `unallocated_or_partial_payment`, `period_cutoff_error`, `bad_fx_date_or_rate`
* 2 interacting scenarios (partial payment + stale FX; double payment of a cut-off invoice)
* clean companies (zero errors) and believable distractors
* reward `R = clip(2·F1 − 1 − 0.002·S, −1, 1)`, `S` = inspection steps

## Quick start

Requirements: Python 3.10+ and `pytest` (everything else is standard library).

```bash
pip install -r requirements.txt
python verify.py          # ONE command: generate -> record trajectories -> tests -> benchmark -> demo
```

Individual steps:

```bash
python -m src.generator                                  # regenerate data/ (10 train + 10 dev) and truth/
python -m pytest -q                                      # tests (environment, matcher, reward, policies, trajectories)
python demo.py [episode]                                 # four policies on one company (default dev_01)
python -m src.runner run --policy reconciliation --episode dev_01
python -m src.runner benchmark                           # all policies x all 20 episodes
python -m src.runner record --out trajectories.jsonl     # real trajectories (4 policies x 3 episodes)
```

## Container

```bash
docker build -t genesis-audit-env .
docker run --rm genesis-audit-env            # runs verify.py inside the container
# or: make docker
```

## Using the environment

```python
from src.runner import make_env            # builds env + closed evaluator (separate process)
env, evaluator = make_env("train_01")      # the agent only ever touches `env`
obs, info = env.reset(seed=0, config=None)

obs, reward, terminated, truncated, info = env.step(
    {"action": "list_records", "table": "ledger", "limit": 25})
obs, reward, terminated, truncated, info = env.step(
    {"action": "list_records", "table": "ledger", "cursor": obs["next_cursor"]})

findings = [{
    "type": "duplicate_invoice",
    "affected_ids": ["INV-1001", "pay_001", "pay_003"],
    "claimed_amount_minor": 12000000,
    "currency": "KZT",
    "evidence_ids": ["doc_inv_01", "doc_pay_01", "doc_pay_03", "ld_0014"],
    "explanation": "Invoice paid twice.",
}]
obs, reward, terminated, truncated, info = env.step({"action": "submit", "findings": findings})
```

Fresh companies can also be generated from a seed, with independent error switches:

```python
from src.env import AuditEnv
env = AuditEnv()
obs, info = env.reset(seed=42, config={"errors": {"duplicate_invoice": 1, "bad_fx_date_or_rate": 2},
                                       "interacting": {"partial_fx": 1}})
# config={"clean": True} gives a company with zero errors
```

### Actions

| action | arguments | notes |
|---|---|---|
| `list_records` | `table` (`ledger`/`invoices`/`payments`/`fx`), `cursor`, `limit` (≤25), `filter` | cursor paging via `next_cursor`; offsets are rejected |
| `get_record` | `table`, `id` (row_id / doc_id / invoice_id / payment_id / fx_id) | |
| `get_document` | `doc_id` | invoice or payment document |
| `calculator` | `expression` | exact decimal; `+ - * / // %`, `abs round min max` |
| `submit` | `findings` (≤30) | ends the episode; reward is returned here |

Limits: 100 steps per episode (running out of steps without submitting ends the episode as
`truncated` and scores it as an empty submission); ≤10 `evidence_ids` per finding; `explanation` ≤1000 chars.
Nothing is accepted after the episode ends. Malformed submissions are rejected with an error and cost a step.

## Layout

```
src/generator.py      seeded company generator + the 20 fixed episode profiles
src/env.py            AuditEnv (reset/step, cursors, calculator, step budget)
src/matcher.py        maximum bipartite matcher, tolerance, validation, reward
src/evaluator.py      closed evaluator (separate process) - the only code that reads answer keys
src/trajectory.py     trajectory recorder (called inside env.step)
src/runner.py         harness: make_env, run policies, benchmark, record
policies/             4 policy adapters: random_explorer, reconciliation, flag_everything, oracle
data/<episode>/       agent-visible files (train_01..10, dev_01..10, episode_001 = original sample)
truth/<episode>/      answer keys, kept OUT of the agent-visible data directory
tests/                pytest suite
trajectories.jsonl    recorded trajectories (real actions/observations)
DESIGN.md             design note
```

See `DESIGN.md` for the isolation model, error definitions, matching rules and known limitations.
