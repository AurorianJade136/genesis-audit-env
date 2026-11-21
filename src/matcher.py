"""Finding matcher and terminal reward.

* One-to-one **maximum bipartite matching** (augmenting paths), so the score never
  depends on the order of the findings.
* ``amount_tolerance_minor`` from the answer key is applied to the amount.
* Submission limits are validated here (30 findings, 10 evidence ids, 1000 chars).

Reward:  R = clip(2*F1 - 1 - 0.002*S, -1, 1)   with S = number of inspection steps.
"""
from __future__ import annotations

ERROR_TYPES = (
    "duplicate_invoice",
    "unallocated_or_partial_payment",
    "period_cutoff_error",
    "bad_fx_date_or_rate",
)
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_AFFECTED_IDS = 20
MAX_EXPLANATION_CHARS = 1000
STEP_PENALTY = 0.002


def validate_findings(findings) -> str | None:
    """Return ``None`` if the submission is well-formed, else a human-readable error."""
    if not isinstance(findings, list):
        return "findings must be a list"
    if len(findings) > MAX_FINDINGS:
        return f"too many findings ({len(findings)} > {MAX_FINDINGS})"
    for i, f in enumerate(findings):
        where = f"findings[{i}]"
        if not isinstance(f, dict):
            return f"{where} must be an object"
        if f.get("type") not in ERROR_TYPES:
            return f"{where}.type must be one of {list(ERROR_TYPES)}"
        ids = f.get("affected_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            return f"{where}.affected_ids must be a non-empty list of strings"
        if len(ids) > MAX_AFFECTED_IDS:
            return f"{where}.affected_ids has more than {MAX_AFFECTED_IDS} ids"
        amt = f.get("claimed_amount_minor")
        if isinstance(amt, bool) or not isinstance(amt, int) or amt < 0:
            return f"{where}.claimed_amount_minor must be a non-negative integer (minor units)"
        cur = f.get("currency")
        if not isinstance(cur, str) or not cur:
            return f"{where}.currency must be a non-empty string"
        ev = f.get("evidence_ids", [])
        if not isinstance(ev, list) or not all(isinstance(x, str) for x in ev):
            return f"{where}.evidence_ids must be a list of strings"
        if len(ev) > MAX_EVIDENCE_IDS:
            return f"{where}.evidence_ids has more than {MAX_EVIDENCE_IDS} ids"
        expl = f.get("explanation", "")
        if not isinstance(expl, str):
            return f"{where}.explanation must be a string"
        if len(expl) > MAX_EXPLANATION_CHARS:
            return f"{where}.explanation is longer than {MAX_EXPLANATION_CHARS} characters"
    return None


def is_match(finding: dict, truth: dict, tolerance: int = 0) -> bool:
    """Does one finding correctly describe one hidden issue?"""
    types = {truth.get("type"), *truth.get("secondary_types", [])}
    if finding.get("type") not in types:
        return False
    if set(finding.get("affected_ids", [])) != set(truth.get("affected_ids", [])):
        return False
    if finding.get("currency") != truth.get("currency"):
        return False
    claimed = finding.get("claimed_amount_minor")
    if isinstance(claimed, bool) or not isinstance(claimed, int):
        return False
    if abs(claimed - truth.get("expected_amount_minor", 0)) > tolerance:
        return False
    evidence = set(finding.get("evidence_ids", []))
    return any(set(s) <= evidence for s in truth.get("valid_evidence_sets", []))


def max_matching(findings, truth_issues, tolerance: int = 0) -> list[tuple[int, int]]:
    """Maximum one-to-one matching (finding index, truth index) via augmenting paths."""
    adj = [
        [j for j, t in enumerate(truth_issues) if is_match(f, t, tolerance)]
        for f in findings
    ]
    match_of_truth: dict[int, int] = {}

    def try_assign(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_of_truth or try_assign(match_of_truth[j], seen):
                match_of_truth[j] = i
                return True
        return False

    for i in range(len(findings)):
        try_assign(i, set())
    return sorted((i, j) for j, i in match_of_truth.items())


def f1_score(tp: int, fp: int, fn: int) -> float:
    if tp + fp + fn == 0:  # clean company, nothing reported
        return 1.0
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def reward_from_f1(f1: float, inspection_steps: int) -> float:
    return max(-1.0, min(1.0, 2 * f1 - 1 - STEP_PENALTY * inspection_steps))


def match(truth_issues, findings, inspection_steps: int, tolerance: int = 0) -> dict:
    pairs = max_matching(findings, truth_issues, tolerance)
    tp = len(pairs)
    fp = len(findings) - tp
    fn = len(truth_issues) - tp
    f1 = f1_score(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "f1": f1,
        "reward": reward_from_f1(f1, inspection_steps),
        "inspection_steps": inspection_steps,
        "matched_issue_ids": [truth_issues[j].get("issue_id", j) for _, j in pairs],
    }
