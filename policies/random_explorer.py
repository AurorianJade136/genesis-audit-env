"""Baseline 1 -- random explorer: pokes at random records, then submits nothing."""
from __future__ import annotations

import random

from policies.base import Policy


class RandomExplorerPolicy(Policy):
    name = "random_explorer"

    def __init__(self, seed: int = 0, min_steps: int = 10, max_steps: int = 40):
        self.rng = random.Random(seed)
        self.min_steps, self.max_steps = min_steps, max_steps

    def reset(self, obs, info):
        self.n_explore = self.rng.randint(self.min_steps, self.max_steps)
        self.taken = 0
        self.page = obs.get("page_size", 25)
        self.ids = {"ledger": [], "invoices": [], "payments": [], "fx": []}
        self.docs = []

    def _remember(self, obs):
        table = obs.get("table")
        for r in obs.get("records", []):
            key = {"ledger": "row_id", "fx": "fx_id"}.get(table, "doc_id")
            self.ids[table].append(r[key])
            if key == "doc_id":
                self.docs.append(r["doc_id"])

    def act(self, obs):
        self._remember(obs)
        if self.taken >= self.n_explore:
            return {"action": "submit", "findings": []}
        self.taken += 1
        choice = self.rng.choice(["list", "list", "get", "doc", "calc"])
        table = self.rng.choice(list(self.ids))
        if choice == "get" and self.ids[table]:
            return {"action": "get_record", "table": table, "id": self.rng.choice(self.ids[table])}
        if choice == "doc" and self.docs:
            return {"action": "get_document", "doc_id": self.rng.choice(self.docs)}
        if choice == "calc":
            return {"action": "calculator",
                    "expression": f"{self.rng.randint(1, 9999)} * {self.rng.randint(1, 99)} / 7"}
        return {"action": "list_records", "table": table, "limit": self.rng.randint(1, self.page)}
