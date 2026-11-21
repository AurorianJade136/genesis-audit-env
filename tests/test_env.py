import json
import os

import pytest

from src.env import MAX_PAGE_SIZE, MAX_STEPS, AuditEnv, safe_calc
from src.evaluator import LocalEvaluator, SubprocessEvaluator
from src.trajectory import KEYS, TrajectoryRecorder
from tests.helpers import DATA, TRUTH, finding, make_env, truth_of


def listing(env, table, **kw):
    return env.step({"action": "list_records", "table": table, **kw})[0]


def scan(env, table, **kw):
    rows, cursor = [], None
    while True:
        a = {"action": "list_records", "table": table, "limit": 7, **kw}
        if cursor:
            a["cursor"] = cursor
        obs = env.step(a)[0]
        rows += obs["records"]
        cursor = obs["next_cursor"]
        if not cursor:
            return rows


# ------------------------------- contract -------------------------------------- #
def test_reset_returns_obs_and_info_and_accepts_seed_and_config():
    env, _ = make_env()
    out = env.reset(seed=3, config={"max_steps": 50})
    assert isinstance(out, tuple) and len(out) == 2
    obs, info = out
    assert info["episode_id"] == "dev_01" and info["seed"] == 3
    assert obs["max_steps"] == 50 and obs["steps_remaining"] == 50
    assert {"task", "manifest", "policy", "tables", "actions"} <= set(obs)
    assert set(obs["tables"]) == {"ledger", "invoices", "payments", "fx"}


def test_step_returns_five_tuple_and_requires_reset():
    env, _ = make_env()
    with pytest.raises(RuntimeError):
        env.step({"action": "list_records", "table": "ledger"})
    env.reset()
    out = env.step({"action": "list_records", "table": "ledger"})
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert reward == 0.0 and terminated is False and truncated is False and info == {}


def test_all_five_actions_work():
    env, _ = make_env()
    env.reset()
    inv = listing(env, "invoices", limit=1)["records"][0]
    assert "record" in env.step({"action": "get_record", "table": "invoices", "id": inv["invoice_id"]})[0]
    assert env.step({"action": "get_document", "doc_id": inv["doc_id"]})[0]["document"] == inv
    assert env.step({"action": "calculator", "expression": "(1200 + 300) * 2 / 4"})[0]["result"] == 750
    obs, r, term, *_ = env.step({"action": "submit", "findings": []})
    assert term is True


def test_get_record_by_each_id_field():
    env, _ = make_env()
    env.reset()
    led = listing(env, "ledger", limit=1)["records"][0]
    assert env.step({"action": "get_record", "table": "ledger", "id": led["row_id"]})[0]["record"] == led
    pay = listing(env, "payments", limit=1)["records"][0]
    for key in ("doc_id", "payment_id"):
        assert env.step({"action": "get_record", "table": "payments", "id": pay[key]})[0]["record"] == pay
    fx = listing(env, "fx", limit=1)["records"][0]
    assert env.step({"action": "get_record", "table": "fx", "id": fx["fx_id"]})[0]["record"] == fx


def test_fx_table_is_accessible_with_ids_matching_evidence_format():
    env, _ = make_env("train_01")
    env.reset()
    obs = listing(env, "fx", limit=3)
    assert obs["total"] > 20
    r = obs["records"][0]
    assert r["fx_id"] == f"fx_{r['date']}_{r['source_currency']}_{r['target_currency']}"
    assert set(r) >= {"date", "source_currency", "target_currency", "rate"}


def test_ledger_numbers_are_ints_and_nulls_are_none():
    env, _ = make_env()
    env.reset()
    for row in listing(env, "ledger")["records"]:
        assert isinstance(row["debit_minor"], int) and isinstance(row["credit_minor"], int)
    rows = scan(env, "ledger")
    assert any(r["entity_id"] is None for r in rows)


# ------------------------------- cursors ---------------------------------------- #
@pytest.mark.parametrize("table", ["ledger", "invoices", "payments", "fx"])
def test_cursor_paging_is_complete_without_overlap(table):
    env, _ = make_env("train_06")
    obs0, _ = env.reset()
    rows = scan(env, table)
    keys = [r[obs0["tables"][table]["key"]] for r in rows]
    assert len(keys) == len(set(keys)) == obs0["tables"][table]["rows"]


def test_cursor_is_stable_and_page_size_capped():
    env, _ = make_env("train_06")
    env.reset()
    p1 = listing(env, "ledger", limit=5)
    again = listing(env, "ledger", limit=5)
    p1.pop('steps_remaining'), again.pop('steps_remaining')
    assert p1 == again                                           # same page, same cursor
    assert listing(env, "ledger", limit=10_000)["count"] == MAX_PAGE_SIZE
    a = listing(env, "ledger", limit=5, cursor=p1["next_cursor"])
    b = listing(env, "ledger", limit=10, cursor=p1["next_cursor"])
    assert b["records"][:5] == a["records"]                      # independent of page size


def test_bad_cursor_offset_and_table_are_errors_not_crashes():
    env, _ = make_env()
    env.reset()
    assert "error" in listing(env, "ledger", cursor="ledger:nope")
    assert "error" in listing(env, "ledger", cursor="invoices:doc_inv_001")
    assert "error" in listing(env, "ledger", offset=5)
    assert "error" in listing(env, "nonsense")
    assert "error" in env.step({"action": "get_record", "table": "ledger", "id": "zzz"})[0]
    assert "error" in env.step({"action": "get_document", "doc_id": "zzz"})[0]
    assert "error" in env.step({"action": "fly"})[0]
    assert "error" in env.step("not a dict")[0]


def test_filter():
    env, _ = make_env("train_06")
    env.reset()
    inv = listing(env, "invoices", limit=1)["records"][0]
    obs = listing(env, "ledger", filter={"invoice_id": inv["invoice_id"]})
    assert obs["count"] >= 2 and all(r["invoice_id"] == inv["invoice_id"] for r in obs["records"])


# ------------------------------- calculator ------------------------------------- #
@pytest.mark.parametrize("expr,expected", [("1+2*3", 7), ("10/4", 2.5), ("7//2", 3), ("-5 + 8", 3),
                                           ("round(2.5)", 3), ("abs(-4)", 4), ("max(1, 9, 3)", 9)])
def test_calculator_ok(expr, expected):
    assert float(safe_calc(expr)) == expected


@pytest.mark.parametrize("expr", ["__import__('os').system('id')", "2**9999999", "open('x')",
                                  "().__class__", "1/0", "a+1", "", "lambda: 1", "[1]", "x" * 300,
                                  "1e999999", "True + 1"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    env, _ = make_env()
    env.reset()
    obs = env.step({"action": "calculator", "expression": expr})[0]
    assert obs.get("error") == "invalid math expression"


def test_calculator_is_exact_decimal():
    env, _ = make_env()
    env.reset()
    assert env.step({"action": "calculator", "expression": "0.1 + 0.2"})[0]["result_text"] == "0.3"


# ------------------------------- termination ------------------------------------ #
def test_submit_terminates_and_later_actions_are_rejected():
    env, _ = make_env()
    env.reset()
    obs, r, term, trunc, info = env.step({"action": "submit", "findings": []})
    assert term and not trunc
    for act in ({"action": "list_records", "table": "ledger"}, {"action": "submit", "findings": []}):
        obs2, r2, term2, trunc2, _ = env.step(act)           # repeated submission / use after end
        assert "error" in obs2 and r2 == 0.0 and term2 is True
    steps = env.steps_taken
    env.step({"action": "calculator", "expression": "1+1"})
    assert env.steps_taken == steps                           # rejected actions cost nothing


def test_reset_starts_a_fresh_episode_after_termination():
    env, _ = make_env()
    env.reset()
    env.step({"action": "submit", "findings": []})
    env.reset()
    obs, *_ = env.step({"action": "list_records", "table": "ledger"})
    assert "records" in obs and env.steps_taken == 1


def test_timeout_after_max_steps_truncates_and_scores_as_empty_submission():
    env, ev = make_env("dev_02")                              # clean company
    env.reset()
    for i in range(MAX_STEPS):
        obs, r, term, trunc, info = env.step({"action": "calculator", "expression": "1+1"})
        if i < MAX_STEPS - 1:
            assert not term and not trunc
    assert trunc and not term
    assert info == {"reason": "max_steps"}
    assert r == pytest.approx(1.0 - 0.002 * MAX_STEPS)        # silence on a clean company, minus steps
    assert ev.last_result["inspection_steps"] == MAX_STEPS
    assert "error" in env.step({"action": "calculator", "expression": "1+1"})[0]


def test_timeout_on_dirty_company_is_minus_one():
    env, _ = make_env("dev_01")
    env.reset()
    for _ in range(MAX_STEPS):
        out = env.step({"action": "calculator", "expression": "1"})
    assert out[3] and out[1] == -1.0


def test_submit_on_last_step_is_still_a_submission():
    env, _ = make_env("dev_02")
    env.reset()
    for _ in range(MAX_STEPS - 1):
        env.step({"action": "calculator", "expression": "1"})
    obs, r, term, trunc, _ = env.step({"action": "submit", "findings": []})
    assert term and not trunc
    assert r == pytest.approx(1.0 - 0.002 * (MAX_STEPS - 1))


def test_steps_remaining_counts_down():
    env, _ = make_env()
    obs, _ = env.reset()
    assert obs["steps_remaining"] == MAX_STEPS
    assert env.step({"action": "calculator", "expression": "1"})[0]["steps_remaining"] == MAX_STEPS - 1


# ------------------------------- rewards ---------------------------------------- #
def test_perfect_answer_vs_silence():
    env, ev = make_env("dev_01")
    t = truth_of("dev_01")
    env.reset()
    _, r_all, term, _, _ = env.step({"action": "submit", "findings": [finding(i) for i in t["issues"]]})
    assert term and r_all == 1.0
    env.reset()
    _, r_none, *_ = env.step({"action": "submit", "findings": []})
    assert r_none == -1.0 and r_all > r_none


def test_inspection_steps_are_penalised_but_submit_is_free():
    env, _ = make_env("dev_02")
    env.reset()
    for _ in range(10):
        env.step({"action": "list_records", "table": "ledger"})
    _, r, *_ = env.step({"action": "submit", "findings": []})
    assert r == pytest.approx(1.0 - 0.02)


def test_flag_everything_attack_is_punished_on_every_episode():
    for name in ("dev_01", "dev_02", "train_05"):
        env, _ = make_env(name)
        env.reset()
        junk = [{"type": "duplicate_invoice", "affected_ids": [f"INV-{i}"], "claimed_amount_minor": 1,
                 "currency": "KZT", "evidence_ids": [], "explanation": "x"} for i in range(30)]
        _, r, *_ = env.step({"action": "submit", "findings": junk})
        assert r == -1.0


def test_duplicate_finding_and_wrong_evidence_through_env():
    env, _ = make_env("dev_04")
    t = truth_of("dev_04")["issues"]
    env.reset()
    _, r_dup, *_ = env.step({"action": "submit",
                             "findings": [finding(i) for i in t] + [finding(t[0])]})
    env.reset()
    _, r_bad, *_ = env.step({"action": "submit",
                             "findings": [finding(i) for i in t[1:]] + [finding(t[0], evidence_ids=[])]})
    assert 0.0 < r_dup < 1.0
    assert r_bad < r_dup


def test_invalid_submission_is_rejected_but_episode_continues():
    env, _ = make_env()
    env.reset()
    bad = [{"type": "nonsense"}]
    obs, r, term, trunc, _ = env.step({"action": "submit", "findings": bad})
    assert "error" in obs and r == 0.0 and not term and not trunc
    too_many = [{"type": "duplicate_invoice", "affected_ids": ["a"], "claimed_amount_minor": 1,
                 "currency": "KZT"}] * 31
    assert "error" in env.step({"action": "submit", "findings": too_many})[0]
    _, _, term, *_ = env.step({"action": "submit", "findings": []})
    assert term


# ------------------------------- isolation -------------------------------------- #
def test_no_diagnostics_or_truth_leak_through_the_agent_interface():
    env, ev = make_env("dev_01")
    t = truth_of("dev_01")
    env.reset()
    seen = []
    for a in ({"action": "list_records", "table": "ledger"}, {"action": "calculator", "expression": "1"},
              {"action": "submit", "findings": [finding(t["issues"][0])]}):
        out = env.step(a)
        seen.append(out)
    blob = json.dumps(seen)
    for banned in ("\"tp\"", "\"fp\"", "\"fn\"", "\"f1\"", "valid_evidence_sets", "expected_amount_minor"):
        assert banned not in blob
    assert seen[-1][4] == {"submitted": True}
    assert ev.last_result["tp"] == 1                           # diagnostics exist, but harness-side only


def test_env_holds_no_answer_key_and_data_dir_has_none():
    env, _ = make_env("dev_01")
    env.reset()
    for attr, val in vars(env).items():
        assert "truth" not in attr.lower()
        assert "valid_evidence_sets" not in json.dumps(val, default=lambda o: "")
    assert not os.path.exists(os.path.join(DATA, "dev_01", "truth.json"))


def test_on_disk_episode_requires_an_evaluator():
    with pytest.raises(ValueError):
        AuditEnv(os.path.join(DATA, "dev_01"))


def test_subprocess_evaluator_agrees_with_local_one():
    t = truth_of("dev_04")
    findings = [finding(i) for i in t["issues"][:2]]
    local = LocalEvaluator(t).score(findings, 12)
    sub = SubprocessEvaluator(os.path.join(TRUTH, "dev_04", "truth.json")).score(findings, 12)
    assert sub == local


def test_subprocess_evaluator_runs_in_another_process_and_rejects_bad_input():
    ev = SubprocessEvaluator(os.path.join(TRUTH, "dev_01", "truth.json"))
    with pytest.raises(RuntimeError):
        ev.score([{"type": "nonsense"}], 0)
    with pytest.raises(FileNotFoundError):
        SubprocessEvaluator(os.path.join(TRUTH, "nope", "truth.json"))


# ------------------------------- seed mode -------------------------------------- #
def test_reset_with_seed_generates_reproducible_episodes():
    env = AuditEnv()
    o1, i1 = env.reset(seed=7, config={"errors": {"duplicate_invoice": 1}})
    led1 = scan(env, "ledger")
    o2, _ = env.reset(seed=7, config={"errors": {"duplicate_invoice": 1}})
    led2 = scan(env, "ledger")
    o3, _ = env.reset(seed=8, config={"errors": {"duplicate_invoice": 1}})
    led3 = scan(env, "ledger")
    assert o1 == o2 and led1 == led2 and led1 != led3
    assert i1["seed"] == 7


def test_seed_mode_scores_with_hidden_key_and_clean_config_rewards_silence():
    env = AuditEnv()
    env.reset(seed=1, config={"clean": True})
    _, r, term, *_ = env.step({"action": "submit", "findings": []})
    assert term and r == 1.0
    env.reset(seed=1, config={"errors": {"period_cutoff_error": 1}})
    _, r, *_ = env.step({"action": "submit", "findings": []})
    assert r == -1.0


def test_disk_and_seed_modes_expose_identical_observations(tmp_path):
    from src.generator import build_episode, write_episode
    cfg = {"errors": {"bad_fx_date_or_rate": 1}}
    v, t = build_episode(21, cfg, episode_id="gen_21")
    write_episode(v, t, str(tmp_path / "d"), str(tmp_path / "t"))
    disk = AuditEnv(str(tmp_path / "d"), evaluator=LocalEvaluator(t))
    mem = AuditEnv()
    od, _ = disk.reset()
    om, _ = mem.reset(seed=21, config=cfg)
    assert od["manifest"] == om["manifest"] and od["tables"] == om["tables"]
    for table in ("ledger", "invoices", "payments", "fx"):
        assert scan(disk, table) == scan(mem, table)


# ------------------------------- trajectories ----------------------------------- #
def test_recorder_writes_real_payloads_with_exactly_the_required_keys(tmp_path):
    path = tmp_path / "t.jsonl"
    env, _ = make_env("dev_02", recorder=TrajectoryRecorder(str(path), append=False))
    env.reset()
    env.step({"action": "list_records", "table": "payments", "limit": 2})
    env.step({"action": "calculator", "expression": "2+2"})
    env.step({"action": "submit", "findings": []})
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(recs) == 3
    assert all(tuple(r.keys()) == KEYS for r in recs)
    assert [r["step"] for r in recs] == [1, 2, 3]
    assert recs[0]["action"] == {"action": "list_records", "table": "payments", "limit": 2}
    assert len(recs[0]["observation"]["records"]) == 2          # the real payload, not a summary
    assert recs[1]["observation"]["result"] == 4
    assert recs[2]["terminated"] is True and recs[2]["reward"] == pytest.approx(1.0 - 0.004)
    assert all(r["episode_id"] == "dev_02" for r in recs)
