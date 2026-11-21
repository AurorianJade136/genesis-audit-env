"""Baseline 4 -- oracle policy for the *evaluator side*.

It is built by the harness from the answer key and never exposed to agents.  It submits the
perfect answer immediately (zero inspection steps), so it must score exactly 1.0 and is used to
check that the reward ceiling is reachable and that the key is self-consistent.
"""
from __future__ import annotations

import json

from policies.base import Policy


class OraclePolicy(Policy):
    name = "oracle"

    def __init__(self, truth: dict | str):
        if isinstance(truth, str):
            with open(truth, "r", encoding="utf-8") as f:
                truth = json.load(f)
        self._issues = truth.get("issues", [])

    def act(self, obs):
        findings = [{
            "type": t["type"],
            "affected_ids": list(t["affected_ids"]),
            "claimed_amount_minor": t["expected_amount_minor"],
            "currency": t["currency"],
            "evidence_ids": list(t["valid_evidence_sets"][0]),
            "explanation": f"Oracle answer for {t.get('issue_id')}.",
        } for t in self._issues]
        return {"action": "submit", "findings": findings}
