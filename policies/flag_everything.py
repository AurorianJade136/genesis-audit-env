"""Baseline 3 -- flag-everything attacker: reads the invoices and reports (almost) all of
them as errors, with guessed amounts and no real evidence.  Must score about -1."""
from __future__ import annotations

from policies.base import GeneratorPolicy, scan_table
from src.matcher import ERROR_TYPES, MAX_FINDINGS


class FlagEverythingPolicy(GeneratorPolicy):
    name = "flag_everything"

    def run(self, obs, info):
        invoices = yield from scan_table("invoices", obs["page_size"])
        findings = []
        for i, inv in enumerate(invoices[:MAX_FINDINGS]):
            findings.append({
                "type": ERROR_TYPES[i % len(ERROR_TYPES)],
                "affected_ids": [inv["invoice_id"]],
                "claimed_amount_minor": inv["total_minor"],
                "currency": inv["currency"],
                "evidence_ids": [inv["doc_id"]],
                "explanation": "Suspicious invoice.",
            })
        yield {"action": "submit", "findings": findings}
