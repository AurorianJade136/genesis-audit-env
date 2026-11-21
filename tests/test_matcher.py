import copy

import pytest

from src.matcher import (MAX_EVIDENCE_IDS, MAX_EXPLANATION_CHARS, MAX_FINDINGS, f1_score,
                         is_match, match, max_matching, reward_from_f1, validate_findings)

T1 = {"issue_id": "a", "type": "duplicate_invoice", "affected_ids": ["INV-1", "pay_1", "pay_2"],
      "expected_amount_minor": 1000, "currency": "KZT",
      "valid_evidence_sets": [["d1", "d2", "l1"], ["d1", "d2", "l2"]]}
T2 = {"issue_id": "b", "type": "period_cutoff_error", "affected_ids": ["INV-2"],
      "expected_amount_minor": 500, "currency": "USD", "valid_evidence_sets": [["d3", "l3"]]}
T3 = {"issue_id": "c", "type": "unallocated_or_partial_payment", "secondary_types": ["bad_fx_date_or_rate"],
      "interacting": True, "affected_ids": ["INV-3", "pay_3"], "expected_amount_minor": 40,
      "currency": "EUR", "valid_evidence_sets": [["d4", "d5", "l4"]]}


def F(t, **over):
    f = {"type": t["type"], "affected_ids": list(t["affected_ids"]),
         "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
         "evidence_ids": list(t["valid_evidence_sets"][0]), "explanation": "e"}
    f.update(over)
    return f


def test_correct_match_scores_one():
    r = match([T1, T2], [F(T1), F(T2)], 0)
    assert (r["tp"], r["fp"], r["fn"], r["f1"], r["reward"]) == (2, 0, 0, 1.0, 1.0)


def test_alternative_evidence_set_is_accepted_and_extra_evidence_is_fine():
    assert is_match(F(T1, evidence_ids=["d1", "d2", "l2"]), T1)
    assert is_match(F(T1, evidence_ids=["d1", "d2", "l1", "zzz", "d9"]), T1)


def test_wrong_evidence_is_not_a_match():
    assert not is_match(F(T1, evidence_ids=["d1", "l1"]), T1)          # missing doc
    assert not is_match(F(T1, evidence_ids=[]), T1)
    r = match([T1], [F(T1, evidence_ids=["d1"])], 0)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1) and r["f1"] == 0.0


@pytest.mark.parametrize("over", [
    {"type": "bad_fx_date_or_rate"}, {"affected_ids": ["INV-1", "pay_1"]},
    {"currency": "USD"}, {"claimed_amount_minor": 1001},
])
def test_each_field_must_match(over):
    assert not is_match(F(T1, **over), T1)


def test_affected_ids_compared_as_a_set():
    assert is_match(F(T1, affected_ids=["pay_2", "INV-1", "pay_1"]), T1)


def test_amount_tolerance_is_applied():
    off_by_one = F(T1, claimed_amount_minor=1001)
    assert not is_match(off_by_one, T1, tolerance=0)
    assert is_match(off_by_one, T1, tolerance=1)
    assert not is_match(F(T1, claimed_amount_minor=1003), T1, tolerance=1)
    assert match([T1], [off_by_one], 0, tolerance=1)["tp"] == 1


def test_interacting_issue_accepts_primary_or_secondary_type():
    assert is_match(F(T3), T3)
    assert is_match(F(T3, type="bad_fx_date_or_rate"), T3)
    assert not is_match(F(T3, type="duplicate_invoice"), T3)


def test_duplicate_findings_count_as_false_positives():
    r = match([T1], [F(T1), F(T1)], 0)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 0)
    assert r["f1"] == pytest.approx(2 * 0.5 * 1 / 1.5)


def test_empty_findings_on_clean_company_is_perfect_and_on_dirty_company_is_zero():
    assert match([], [], 0)["f1"] == 1.0
    assert match([], [], 0)["reward"] == 1.0
    r = match([T1, T2], [], 0)
    assert (r["tp"], r["fp"], r["fn"], r["f1"], r["reward"]) == (0, 0, 2, 0.0, -1.0)


def test_findings_on_clean_company_are_penalised():
    r = match([], [F(T1)], 0)
    assert r["f1"] == 0.0 and r["reward"] == -1.0


def test_matching_is_order_independent_and_maximum_not_greedy():
    # Two truths that both accept finding A (loose evidence) but only A fits truth X.
    X = {"issue_id": "x", "type": "duplicate_invoice", "affected_ids": ["I"], "expected_amount_minor": 5,
         "currency": "KZT", "valid_evidence_sets": [["e1"]]}
    Y = {"issue_id": "y", "type": "duplicate_invoice", "affected_ids": ["I"], "expected_amount_minor": 5,
         "currency": "KZT", "valid_evidence_sets": [["e1"], ["e2"]]}
    A = {"type": "duplicate_invoice", "affected_ids": ["I"], "claimed_amount_minor": 5,
         "currency": "KZT", "evidence_ids": ["e1", "e2"]}
    B = dict(A, evidence_ids=["e2"])             # only fits Y
    # greedy in order [A, B] would give A->X (first free) fine; order [A,B] with truths [Y,X] would
    # give A->Y and then B has nothing.  Maximum matching finds 2 either way.
    for truths in ([X, Y], [Y, X]):
        for fs in ([A, B], [B, A]):
            assert len(max_matching(fs, truths)) == 2


def test_reward_formula_and_step_penalty_and_clipping():
    assert reward_from_f1(1.0, 0) == 1.0
    assert reward_from_f1(1.0, 10) == pytest.approx(1 - 0.02)
    assert reward_from_f1(0.5, 25) == pytest.approx(-0.05)
    assert reward_from_f1(0.0, 100) == -1.0           # clipped
    assert reward_from_f1(1.0, -50) == 1.0            # clipped
    assert match([T1], [F(T1)], 40)["reward"] == pytest.approx(1 - 0.08)


def test_f1_edge_cases():
    assert f1_score(0, 0, 0) == 1.0
    assert f1_score(0, 3, 0) == 0.0
    assert f1_score(2, 0, 2) == pytest.approx(2 * 1 * 0.5 / 1.5)


# ------------------------------ validation ------------------------------------ #
def test_valid_submission_passes():
    assert validate_findings([F(T1), F(T2)]) is None
    assert validate_findings([]) is None


def test_limits_are_enforced():
    ok = [F(T2) for _ in range(MAX_FINDINGS)]
    assert validate_findings(ok) is None
    assert "too many" in validate_findings(ok + [F(T2)])
    assert "evidence_ids" in validate_findings([F(T2, evidence_ids=[f"e{i}" for i in range(MAX_EVIDENCE_IDS + 1)])])
    assert validate_findings([F(T2, evidence_ids=[f"e{i}" for i in range(MAX_EVIDENCE_IDS)])]) is None
    assert "explanation" in validate_findings([F(T2, explanation="x" * (MAX_EXPLANATION_CHARS + 1))])
    assert validate_findings([F(T2, explanation="x" * MAX_EXPLANATION_CHARS)]) is None


@pytest.mark.parametrize("bad", [
    "not a list", [1], [{"type": "nonsense"}],
    [{"type": "duplicate_invoice", "affected_ids": [], "claimed_amount_minor": 1, "currency": "KZT"}],
    [{"type": "duplicate_invoice", "affected_ids": ["a"], "claimed_amount_minor": 1.5, "currency": "KZT"}],
    [{"type": "duplicate_invoice", "affected_ids": ["a"], "claimed_amount_minor": True, "currency": "KZT"}],
    [{"type": "duplicate_invoice", "affected_ids": ["a"], "claimed_amount_minor": -1, "currency": "KZT"}],
    [{"type": "duplicate_invoice", "affected_ids": ["a"], "claimed_amount_minor": 1, "currency": ""}],
    [{"type": "duplicate_invoice", "affected_ids": ["a"], "claimed_amount_minor": 1, "currency": "KZT",
      "evidence_ids": "d1"}],
])
def test_malformed_submissions_are_rejected(bad):
    assert validate_findings(bad) is not None


def test_match_does_not_mutate_inputs():
    truths, fs = [copy.deepcopy(T1)], [F(T1)]
    snap = copy.deepcopy((truths, fs))
    match(truths, fs, 3)
    assert (truths, fs) == snap
