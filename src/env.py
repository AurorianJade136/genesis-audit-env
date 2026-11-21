import csv
import json
import os
from src.matcher import match

MAX_STEPS = 100
MAX_LIST_ROWS = 20

class AuditEnv:
    def __init__(self, episode_dir):
        self.episode_dir = episode_dir
        self.manifest = self._read_json(os.path.join(episode_dir, "manifest.json"))
        self.ledger = self._read_csv(os.path.join(episode_dir, "ledger.csv"))
        self.invoices = self._read_jsonl(os.path.join(episode_dir, "invoices.jsonl"))
        self.payments = self._read_jsonl(os.path.join(episode_dir, "payments.jsonl"))
        self.fx = self._read_csv(os.path.join(episode_dir, "fx.csv"))
        self.steps_taken = 0

    @staticmethod
    def _read_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_csv(path):
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _read_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_truth_issues(self):
        truth = self._read_json(os.path.join(self.episode_dir, "truth.json"))
        return truth.get("issues", []) if isinstance(truth, dict) else truth

    def reset(self):
        self.steps_taken = 0
        return {
            "task": "Find the financial errors in this company.",
            "manifest": self.manifest,
            "steps_remaining": MAX_STEPS,
        }

    def step(self, action):
        self.steps_taken += 1
        if self.steps_taken > MAX_STEPS:
            return {"error": "out of steps"}, 0.0, True, True, {}

        if not isinstance(action, dict):
            return self._error("action must be a dict")

        kind = action.get("action")
        if kind == "list_records":
            return self._list_records(action)
        if kind == "get_record":
            return self._get_record(action)
        if kind == "submit":
            return self._submit(action)

        return self._error(f"unknown action: {kind!r}")

    def _remaining(self):
        return max(0, MAX_STEPS - self.steps_taken)

    def _error(self, message):
        return {"error": message, "steps_remaining": self._remaining()}, 0.0, False, False, {}

    def _table(self, name):
        return {"ledger": self.ledger, "invoices": self.invoices, "payments": self.payments}.get(name)

    def _list_records(self, action):
        name = action.get("table", action.get("source"))
        rows = self._table(name)
        if rows is None:
            return self._error(f"unknown table: {name!r}")

        try:
            offset = max(0, int(action.get("offset", 0)))
            limit = int(action.get("limit", MAX_LIST_ROWS))
        except (TypeError, ValueError):
            return self._error("offset and limit must be integers")
        limit = max(0, min(limit, MAX_LIST_ROWS))

        page = rows[offset:offset + limit]
        obs = {"table": name, "records": page, "count": len(page), "total": len(rows), "steps_remaining": self._remaining()}
        return obs, 0.0, False, False, {}

    def _get_record(self, action):
        name = action.get("table", action.get("source"))
        rows = self._table(name)
        if rows is None:
            return self._error(f"unknown table: {name!r}")

        if "row_id" in action:
            key, value = "row_id", action["row_id"]
        elif "doc_id" in action:
            key, value = "doc_id", action["doc_id"]
        else:
            return self._error("get_record requires row_id or doc_id")

        for row in rows:
            if str(row.get(key)) == str(value):
                return {"table": name, "record": row, "steps_remaining": self._remaining()}, 0.0, False, False, {}

        return self._error(f"record not found: {key}={value!r}")

    def _submit(self, action):
        findings = action.get("findings", [])
        truth_issues = self._load_truth_issues() # Only loaded here
        result = match(truth_issues, findings, self.steps_taken)
        
        return {"result": "Submitted successfully"}, result["reward"], True, False, result