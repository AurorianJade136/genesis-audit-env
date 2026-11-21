# AuditEnv: A Financial Audit Environment for AI Agents

> A pure-Python, gym-style environment where an agent investigates a company's books and must find the financial errors hidden in the ledger, invoices, payments and FX data.

<!-- Replace the placeholders below before submitting -->
**Team:** _your team name_ | **Hackathon:** _hackathon name_ | **Track:** _track_

---

## Overview

Auditing is a needle-in-a-haystack task: errors are rarely visible in a single record. They show up only when ledger entries, invoices, payments and FX rates are cross-checked against each other.

`AuditEnv` turns this into a reproducible environment. An agent explores the books through a small action API, then submits a list of findings. Findings are scored against a hidden ground truth.

**Key properties**

- **Pure Python.** Standard library only, with no gymnasium dependency.
- **Strict truth isolation.** The agent can never see the answer key (see [Isolation](#truth-isolation)).
- **Step-limited.** 100 steps per episode, which forces efficient investigation.
- **Realistic, interacting errors.** The issues include duplicate payments, partial payments hidden by plug adjustments, period cutoff errors and FX rate errors, and some of them compound.
- **Multi-currency.** KZT, USD and EUR, with amounts stored in minor units.

---

## Project Structure

```
.
├── env.py            # AuditEnv class (this submission's core)
├── matcher.py        # match(truth_issues, findings, steps_taken) -> scoring
├── README.md
└── <episode_dir>/
    ├── manifest.json     # visible: episode metadata
    ├── ledger.csv        # visible: general ledger entries
    ├── invoices.jsonl    # visible: invoices
    ├── payments.jsonl    # visible: payments
    ├── fx.csv            # visible: FX rates
    └── truth.json        # HIDDEN: ground-truth issues (never shown to the agent)
```

---

## Quick Start

```python
from env import AuditEnv

env = AuditEnv("path/to/episode_dir")
obs = env.reset()
# {"task": "Find the financial errors in this company.",
#  "manifest": {...}, "steps_remaining": 100}

# Browse up to 20 rows at a time
obs, reward, terminated, truncated, info = env.step(
    {"action": "list_records", "table": "ledger", "offset": 0, "limit": 20}
)

# Drill into one record
obs, reward, terminated, truncated, info = env.step(
    {"action": "get_record", "table": "invoices", "doc_id": "INV-1001"}
)

# Submit findings (format defined by matcher.py); this ends the episode
obs, reward, terminated, truncated, info = env.step(
    {"action": "submit", "findings": [...]}
)
```

---

## Action Space

| Action | Parameters | Returns |
|---|---|---|
| `list_records` | `table` (`ledger` \| `invoices` \| `payments`), optional `offset`, `limit` (max 20) | A page of rows plus `total` and `count` |
| `get_record` | `table`, and either `row_id` or `doc_id` | A single row |
| `submit` | `findings` (list) | Scored result. **Ends the episode** |

Every call to `step()` consumes one step, including errors and unknown actions.

### Step Return Value

`step()` returns `(observation, reward, terminated, truncated, info)`:

| Situation | reward | terminated | truncated |
|---|---|---|---|
| Normal query | `0.0` | `False` | `False` |
| Unknown / invalid action | `0.0` | `False` | `False` |
| `submit` | from `matcher.match` | `True` | `False` |
| Step 101 or later | `0.0` | `True` | `True` |

---

## Truth Isolation

The agent must never see `truth.json`. The design guarantees this structurally.

- `__init__` loads **only** `manifest.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl` and `fx.csv`. No truth attribute exists on the instance.
- `reset()`, `list_records` and `get_record` operate solely on that visible data.
- `truth.json` is opened in **exactly one place**, the private `_load_truth_issues()`, called only from the `submit` branch of `step()`. The result is passed to `match(...)` and then discarded.

---

## Injected Error Catalog

Each episode contains hidden issues the agent is expected to find. The reference episode contains the five below. Amounts are in minor units.

| ID | Type | Entity | Impact |
|---|---|---|---|
| `iss_01` | Duplicate settlement | `ent_0417` | 12,000,000 KZT over-credited to cash |
| `iss_02` | Partial payment / unsupported adjustment | `ent_0522` | 38,440,000 KZT booked to `5900_Purchase_Adjustments` |
| `iss_03` | Period cutoff error | `ent_0719` | 4,500,000 KZT expense and payable overstated |
| `iss_04` | Wrong FX rate date | `ent_0801` | 1,400,000 KZT unapplied AP credit |
| `iss_05` | Interacting partial payment and FX error | `ent_0903` | 22,280,000 KZT plug (21,200,000 shortfall + 1,080,000 FX) |

### Duplicate Settlement (`iss_01`)

Invoice `INV-1001` (`ent_0417`) is issued for `12,000,000` KZT minor (`ld_0005` debits `5100_Services_Expense` and `ld_0006` credits `2000_Accounts_Payable` for `12,000,000`), but it is settled twice in full: first by `pay_001` (`12,000,000` KZT minor on October 12, debited to `2000_Accounts_Payable` in `ld_0009` and credited to `1000_Cash` in `ld_0010`), and again by `pay_003` (`12,000,000` KZT minor on October 19, debited to `2000_Accounts_Payable` in `ld_0014` and credited to `1000_Cash` in `ld_0015`). Because `2000_Accounts_Payable` was credited once for `12,000,000` KZT minor and debited twice for `12,000,000` KZT minor (`24,000,000` KZT minor total), the account is left with a `12,000,000` KZT minor debit balance for `ent_0417` and cash is over-credited by the duplicate payment.

### Partial Payment / Unsupported Adjustment (`iss_02`)

Invoice `INV-1002` (`ent_0522`) is issued for `200,000` USD minor and booked at the October 8 FX rate of `480.50` KZT/USD as `96,100,000` KZT minor (`ld_0007` debits `5200_Software_Expense` and `ld_0008` credits `2000_Accounts_Payable` for `96,100,000`), whereas payment `pay_002` on October 15 covers only `120,000` USD minor, leaving `80,000` USD minor unpaid. Rather than leaving the `80,000` USD minor (`38,440,000` KZT minor at `480.50`) open, `ld_0011` debits the full `96,100,000` KZT minor from `2000_Accounts_Payable` while `ld_0012` credits `1000_Cash` for `57,660,000` KZT minor (`120,000 × 480.50`) and `ld_0013` books the remaining `38,440,000` KZT minor (`80,000 × 480.50`) to `5900_Purchase_Adjustments` without a valid adjustment.

### Period Cutoff Error (`iss_03`)

Invoice `INV-1006` (`ent_0719`) is issued on `2026-09-25`, prior to `period_start` (`2026-10-01T00:00:00+05:00`), for `4,500,000` KZT minor, yet the ledger records it on `2026-10-02` in `ld_0021` (debiting `5100_Services_Expense` for `4,500,000` KZT minor) and `ld_0022` (crediting `2000_Accounts_Payable` for `4,500,000` KZT minor). Posting this September expense into October overstates October expenses and period-end payables by `4,500,000` KZT minor.

### Wrong FX Rate Date (`iss_04`)

Under `policy.json` (which sets the FX rate date to the invoice issue date), `INV-1004` (`ent_0801`, `100,000` USD minor issued `2026-10-22`) is booked at the `2026-10-22` rate of `482.00` KZT/USD as `48,200,000` KZT minor (`ld_0023` debits `5200_Software_Expense` and `ld_0024` credits `2000_Accounts_Payable` for `48,200,000`). However, payment `pay_004` (`100,000` USD minor on `2026-10-23`) is converted using the `2026-09-01` rate of `468.00` KZT/USD in `ld_0025` (debiting `2000_Accounts_Payable` for `46,800,000` KZT minor) and `ld_0026` (crediting `1000_Cash` for `46,800,000` KZT minor), leaving an unapplied `1,400,000` KZT minor credit balance (`48,200,000 - 46,800,000`) in `2000_Accounts_Payable`.

### Interacting Partial Payment & FX Error (`iss_05`)

Invoice `INV-1005` (`ent_0903`, `100,000` EUR minor issued `2026-10-27`) is booked at `530.00` KZT/EUR as `53,000,000` KZT minor (`ld_0027` debits `5100_Services_Expense` and `ld_0028` credits `2000_Accounts_Payable` for `53,000,000`), while payment `pay_005` pays only `60,000` EUR minor (`40,000` EUR minor shortfall). In `ld_0029`–`ld_0031`, `2000_Accounts_Payable` is debited in full for `53,000,000` KZT minor, `1000_Cash` is credited for `30,720,000` KZT minor using the wrong `512.00` FX rate instead of `530.00` (`31,800,000` KZT minor, a `1,080,000` KZT minor FX understatement), and `5900_Purchase_Adjustments` is credited with a `22,280,000` KZT minor plug, combining the `21,200,000` KZT minor true `40,000` EUR minor shortfall (`40,000 × 530.00`) and the `1,080,000` KZT minor FX error.

---

## Why This Is Hard

- **Cross-source reasoning.** No single table reveals the error. The agent must reconcile ledger lines against invoices, payments and FX rates.
- **Plugs hide errors.** Balancing entries to `5900_Purchase_Adjustments` make the books balance while concealing the real problem (`iss_02`, `iss_05`).
- **Compounding issues.** A partial payment and an FX error can sit inside the same entries (`iss_05`).
- **Limited budget.** With only 100 steps and 20 rows per page, the agent must investigate selectively.

---

## Requirements

- Python 3.8+
- Standard library only

---

## License

_Add your license here._
