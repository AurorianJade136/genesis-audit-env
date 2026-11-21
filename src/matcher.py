"""Matcher / scorer for the AI financial auditor RL environment.

Pure Python, no external dependencies.
"""

_MISSING = object()


def _is_match(truth, finding):
    """Return True if `finding` is a true positive for `truth`."""
    # 1. Same type
    if truth.get("type") != finding.get("type"):
        return False

    # 2. Identical set of affected_ids (order-insensitive)
    try:
        if set(truth.get("affected_ids") or []) != set(finding.get("affected_ids") or []):
            return False
    except TypeError:  # unhashable ids
        return False

    # 3. Exact amount match (tolerance 0); both values must be present
    expected = truth.get("expected_amount_minor", _MISSING)
    claimed = finding.get("claimed_amount_minor", _MISSING)
    if expected is _MISSING or claimed is _MISSING or expected is None or claimed is None:
        return False
    if expected != claimed:
        return False

    # 4. Same currency
    if truth.get("currency") != finding.get("currency"):
        return False

    # 5. Finding's evidence_ids must contain at least one full acceptable set
    try:
        evidence = set(finding.get("evidence_ids") or [])
        acceptable_sets = truth.get("acceptable_evidence_sets") or []
        return any(set(ev_set).issubset(evidence) for ev_set in acceptable_sets)
    except TypeError:
        return False


def match(truth_issues, findings, inspection_steps):
    """Score agent findings against ground-truth issues.

    Matching is strictly 1:1 (maximum bipartite matching), so duplicate
    findings for the same truth issue cannot score more than once.

    Returns {"tp", "fp", "fn", "f1", "reward"}.
    """
    truth_issues = truth_issues or []
    findings = findings or []

    # Build adjacency: finding index -> list of truth indices it can match
    adjacency = [
        [t_idx for t_idx, truth in enumerate(truth_issues) if _is_match(truth, finding)]
        for finding in findings
    ]

    # Maximum bipartite matching (augmenting paths) so the 1:1 assignment is
    # optimal and independent of finding order.
    truth_to_finding = {}

    def try_assign(f_idx, seen):
        for t_idx in adjacency[f_idx]:
            if t_idx in seen:
                continue
            seen.add(t_idx)
            if t_idx not in truth_to_finding or try_assign(truth_to_finding[t_idx], seen):
                truth_to_finding[t_idx] = f_idx
                return True
        return False

    for f_idx in range(len(findings)):
        try_assign(f_idx, set())

    tp = len(truth_to_finding)
    fp = len(findings) - tp
    fn = len(truth_issues) - tp

    denominator = 2 * tp + fp + fn
    if denominator == 0:
        f1 = 1.0  # both truth and findings empty
    else:
        f1 = (2 * tp) / denominator  # truth empty + findings present -> 0.0

    reward = max(-1.0, min(1.0, (2 * f1) - 1 - (0.002 * inspection_steps)))

    return {"tp": tp, "fp": fp, "fn": fn, "f1": f1, "reward": reward}

if __name__ == "__main__":
    # 1. Test empty
    res = match([], [], 10)
    assert res["f1"] == 1.0, "Empty should be F1=1"
    
    # 2. Test a perfect match from your truth.json
    truth = [{
        "issue_id": "iss_01",
        "type": "duplicate_invoice",
        "affected_ids": ["INV-1001", "pay_001", "pay_003"],
        "expected_amount_minor": 12000000,
        "currency": "KZT",
        "acceptable_evidence_sets": [["doc_inv_01", "doc_pay_01", "doc_pay_03", "ld_0014"]]
    }]
    finding = [{
        "type": "duplicate_invoice",
        "affected_ids": ["pay_003", "INV-1001", "pay_001"], # Note: different order, should still match
        "claimed_amount_minor": 12000000,
        "currency": "KZT",
        "evidence_ids": ["doc_inv_01", "doc_pay_01", "doc_pay_03", "ld_0014", "extra_id"],
        "explanation": "Found a duplicate"
    }]
    res2 = match(truth, finding, 10)
    print("Perfect match result:", res2)
    assert res2["tp"] == 1 and res2["fp"] == 0, "Perfect match failed"
    
    print("Matcher works perfectly!")