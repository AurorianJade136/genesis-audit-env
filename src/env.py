"""Gym-style audit environment.

    env = AuditEnv(episode_dir, evaluator=evaluator)        # fixed episode on disk
    obs, info = env.reset(seed=0, config=None)
    obs, reward, terminated, truncated, info = env.step(action)

or, generating a fresh company from a seed (no files involved):

    env = AuditEnv()
    obs, info = env.reset(seed=7, config={"errors": {"duplicate_invoice": 1}})

Actions (dicts): list_records, get_record, get_document, calculator, submit.

Isolation: this class never opens an answer key.  Scoring is delegated to an evaluator
object (see ``src/evaluator.py``) that lives outside the agent-facing interface; the
default evaluator for on-disk episodes runs in a separate process.
"""
from __future__ import annotations

import ast
import copy
import csv
import json
import os
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation

from src.evaluator import LocalEvaluator
from src.generator import build_episode
from src.matcher import validate_findings

MAX_STEPS = 100
MAX_PAGE_SIZE = 25

TABLES = {  # name -> (key field used for cursors, fields usable with get_record)
    "ledger": ("row_id", ("row_id",)),
    "invoices": ("doc_id", ("doc_id", "invoice_id")),
    "payments": ("doc_id", ("doc_id", "payment_id")),
    "fx": ("fx_id", ("fx_id",)),
}
INSPECTION_ACTIONS = ("list_records", "get_record", "get_document", "calculator")

TASK = (
    "You are an auditor. Inspect the books of this company for the period in the manifest and "
    "submit every financial error you can prove. Error types: duplicate_invoice, "
    "unallocated_or_partial_payment, period_cutoff_error, bad_fx_date_or_rate. A clean company "
    "has no errors: submitting nothing is then correct. Each finding needs type, affected_ids, "
    "claimed_amount_minor (integer, minor units), currency, evidence_ids and an explanation. "
    "Findings are scored once, when you submit."
)
ACTION_SPEC = {
    "list_records": {"table": "ledger|invoices|payments|fx", "cursor": "optional, from next_cursor",
                     "limit": f"1..{MAX_PAGE_SIZE}", "filter": "optional {field: value}"},
    "get_record": {"table": "ledger|invoices|payments|fx", "id": "row_id / doc_id / invoice_id / payment_id / fx_id"},
    "get_document": {"doc_id": "an invoice or payment document id"},
    "calculator": {"expression": "numbers, + - * / // %, parentheses, abs(), round(), min(), max()"},
    "submit": {"findings": "list of up to 30 findings; ends the episode"},
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_visible(episode_dir: str) -> dict:
    p = lambda n: os.path.join(episode_dir, n)  # noqa: E731
    return {
        "manifest": _read_json(p("manifest.json")),
        "policy": _read_json(p("policy.json")) if os.path.exists(p("policy.json")) else {},
        "ledger": _read_csv(p("ledger.csv")),
        "invoices": _read_jsonl(p("invoices.jsonl")),
        "payments": _read_jsonl(p("payments.jsonl")),
        "fx": _read_csv(p("fx.csv")),
    }


def normalize_visible(v: dict) -> dict:
    """Same types whether the episode came from disk (all strings) or from the generator."""
    ledger = []
    for row in v["ledger"]:
        r = dict(row)
        for k in ("debit_minor", "credit_minor"):
            r[k] = int(r[k])
        for k, val in list(r.items()):
            if val in ("null", ""):
                r[k] = None
        ledger.append(r)
    fx = []
    for row in v["fx"]:
        r = dict(row)
        r["fx_id"] = f"fx_{r['date']}_{r['source_currency']}_{r['target_currency']}"
        fx.append(r)
    return {"manifest": v["manifest"], "policy": v.get("policy", {}), "ledger": ledger,
            "invoices": list(v["invoices"]), "payments": list(v["payments"]), "fx": fx}


# --------------------------------------------------------------------------- #
# safe calculator
# --------------------------------------------------------------------------- #
def _round_half_up(x, ndigits=0):
    q = Decimal(1).scaleb(-int(ndigits))
    return x.quantize(q, rounding=ROUND_HALF_UP)


_FUNCS = {"abs": abs, "round": _round_half_up, "min": min, "max": max}


def safe_calc(expr: str) -> Decimal:
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expr) > 200:
        raise ValueError("expression too long")
    tree = ast.parse(expr.strip(), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            d = Decimal(str(n.value))
            if not d.is_finite():
                raise ValueError("non-finite number")
            return d
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = n.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return a / b
            if isinstance(op, ast.FloorDiv):
                return a // b
            if isinstance(op, ast.Mod):
                return a % b
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS
                and not n.keywords):
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("unsupported expression")

    return ev(tree)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class AuditEnv:
    def __init__(self, episode_dir: str | None = None, evaluator=None, recorder=None,
                 max_steps: int = MAX_STEPS, page_size: int = MAX_PAGE_SIZE):
        if episode_dir is not None and evaluator is None:
            raise ValueError("an evaluator is required for on-disk episodes "
                             "(see src.runner.make_env)")
        self._episode_dir = episode_dir
        self._evaluator = evaluator
        self._recorder = recorder
        self._max_steps = int(max_steps)
        self._page_size = min(int(page_size), MAX_PAGE_SIZE)
        self._ready = False
        self._done = False
        self._truncated = False
        self.steps_taken = 0
        self.episode_id = None

    # ---- harness-only helpers (not part of the agent interface) ----------- #
    def evaluator_report(self):
        """Closed-evaluation diagnostics for the *harness* after the run."""
        return None if self._evaluator is None else self._evaluator.last_result

    # ---- gym API ----------------------------------------------------------- #
    def reset(self, seed=None, config=None):
        config = dict(config or {})
        if "max_steps" in config:
            self._max_steps = int(config.pop("max_steps"))
        if self._episode_dir is not None:
            visible = load_visible(self._episode_dir)
        else:
            seed = 0 if seed is None else int(seed)
            visible, truth = build_episode(seed, config or None)
            self._evaluator = LocalEvaluator(truth)  # answer key stays inside the evaluator
        v = normalize_visible(visible)
        self._manifest, self._policy = v["manifest"], v["policy"]
        self._rows = {name: v[name] for name in TABLES}
        self._index = {name: {r[TABLES[name][0]]: i for i, r in enumerate(rows)}
                       for name, rows in self._rows.items()}
        self.episode_id = self._manifest.get("episode_id")
        self.steps_taken = 0
        self._done = False
        self._truncated = False
        self._ready = True
        obs = {
            "task": TASK,
            "manifest": copy.deepcopy(self._manifest),
            "policy": copy.deepcopy(self._policy),
            "tables": {name: {"rows": len(rows), "key": TABLES[name][0],
                              "fields": list(rows[0].keys()) if rows else []}
                       for name, rows in self._rows.items()},
            "actions": ACTION_SPEC,
            "max_steps": self._max_steps,
            "page_size": self._page_size,
            "steps_remaining": self._max_steps,
        }
        return obs, {"episode_id": self.episode_id, "seed": seed}

    def step(self, action):
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        if self._done:  # nothing is accepted after the episode ended
            return ({"error": "episode is over; call reset()", "steps_remaining": 0},
                    0.0, True, self._truncated, {})
        self.steps_taken += 1
        kind = action.get("action") if isinstance(action, dict) else None
        if kind == "submit":
            obs, reward, terminated, truncated, info = self._submit(action)
        else:
            obs = self._dispatch(kind, action)
            reward, terminated, truncated, info = 0.0, False, False, {}
        if not terminated and self.steps_taken >= self._max_steps:
            # budget gone without an accepted submission: scored as an empty submission
            truncated = True
            reward = self._evaluate([], inspection_steps=self.steps_taken)
            obs = dict(obs, truncated_reason="max_steps: no submission within budget")
            info = {"reason": "max_steps"}
        if terminated or truncated:
            self._done = True
            self._truncated = truncated
        obs["steps_remaining"] = max(0, self._max_steps - self.steps_taken)
        if self._recorder is not None:
            self._recorder.record(self.episode_id, self.steps_taken, action, obs, reward,
                                  terminated, truncated)
        return obs, reward, terminated, truncated, info

    # ---- actions ------------------------------------------------------------ #
    def _dispatch(self, kind, action):
        if not isinstance(action, dict):
            return {"error": "action must be a dict"}
        if kind == "list_records":
            return self._list_records(action)
        if kind == "get_record":
            return self._get_record(action)
        if kind == "get_document":
            return self._get_document(action)
        if kind == "calculator":
            return self._calculator(action)
        return {"error": f"unknown action: {kind!r}"}

    def _table_name(self, action):
        name = action.get("table", action.get("source"))
        return name if name in TABLES else None

    def _list_records(self, action):
        name = self._table_name(action)
        if name is None:
            return {"error": f"unknown table {action.get('table')!r}; use one of {list(TABLES)}"}
        if "offset" in action:
            return {"error": "offset is not supported; pass the next_cursor of the previous page"}
        rows, key = self._rows[name], TABLES[name][0]
        try:
            limit = int(action.get("limit", self._page_size))
        except (TypeError, ValueError):
            return {"error": "limit must be an integer"}
        limit = max(1, min(limit, self._page_size))
        start = 0
        cursor = action.get("cursor")
        if cursor is not None:
            prefix, _, last = str(cursor).partition(":")
            if prefix != name or last not in self._index[name]:
                return {"error": "invalid cursor for this table"}
            start = self._index[name][last] + 1
        flt = action.get("filter") or {}
        if not isinstance(flt, dict):
            return {"error": "filter must be an object {field: value}"}

        def ok(r):
            return all(str(r.get(k)) == str(v) for k, v in flt.items())

        page, last_idx = [], None
        i = start
        while i < len(rows) and len(page) < limit:
            if ok(rows[i]):
                page.append(copy.deepcopy(rows[i]))
                last_idx = i
            i += 1
        more = any(ok(r) for r in rows[i:]) if last_idx is not None else False
        return {"table": name, "records": page, "count": len(page),
                "total": sum(1 for r in rows if ok(r)),
                "next_cursor": f"{name}:{rows[last_idx][key]}" if more else None}

    def _find(self, name, value):
        for field in TABLES[name][1]:
            for r in self._rows[name]:
                if str(r.get(field)) == str(value):
                    return r
        return None

    def _get_record(self, action):
        name = self._table_name(action)
        if name is None:
            return {"error": f"unknown table {action.get('table')!r}; use one of {list(TABLES)}"}
        value = None
        for k in ("id", "record_id", "row_id", "doc_id", "invoice_id", "payment_id", "fx_id"):
            if k in action:
                value = action[k]
                break
        if value is None:
            return {"error": "get_record requires an id"}
        rec = self._find(name, value)
        if rec is None:
            return {"error": f"record not found: {value!r}"}
        return {"table": name, "record": copy.deepcopy(rec)}

    def _get_document(self, action):
        value = action.get("doc_id")
        if not value:
            return {"error": "doc_id required"}
        rec = self._find("invoices", value) or self._find("payments", value)
        if rec is None:
            return {"error": f"document not found: {value!r}"}
        return {"document": copy.deepcopy(rec)}

    def _calculator(self, action):
        expr = action.get("expression", "")
        try:
            res = safe_calc(expr)
        except (ValueError, SyntaxError, TypeError, InvalidOperation, DivisionByZero, ArithmeticError):
            return {"error": "invalid math expression"}
        exact = format(res.normalize(), "f") if res != 0 else "0"
        number = int(res) if res == res.to_integral_value() else float(res)
        return {"expression": expr, "result": number, "result_text": exact}

    # ---- submission --------------------------------------------------------- #
    def _evaluate(self, findings, inspection_steps):
        return self._evaluator.score(findings, inspection_steps)["reward"]

    def _submit(self, action):
        findings = action.get("findings", [])
        err = validate_findings(findings)
        if err:  # malformed submissions are rejected, the step is still spent
            return {"error": f"invalid submission: {err}"}, 0.0, False, False, {}
        inspection_steps = self.steps_taken - 1  # the submit itself is not an inspection
        reward = self._evaluate(findings, inspection_steps)
        return ({"result": "submitted", "findings_received": len(findings)},
                reward, True, False, {"submitted": True})
