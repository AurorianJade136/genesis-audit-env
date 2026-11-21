import json
import os

import pytest

from policies import POLICY_NAMES, build_policy
from policies.reconciliation import analyze
from src.matcher import max_matching
from src.runner import RECORD_EPISODES, all_episode_names, make_env, run_policy
from tests.helpers import TRUTH, truth_of

ALL = all_episode_names()


def test_four_policy_adapters_are_registered():
    assert set(POLICY_NAMES) == {"random_explorer", "reconciliation", "flag_everything", "oracle"}
    for n in POLICY_NAMES:
        assert build_policy(n, truth_path=os.path.join(TRUTH, "dev_01", "truth.json")).name == n
    with pytest.raises(ValueError):
        build_policy("oracle")            # the oracle cannot be built without the evaluator-side key
    with pytest.raises(ValueError):
        build_policy("nope")


@pytest.mark.parametrize("name", ALL)
def test_oracle_gets_the_maximum_reward(name):
    r = run_policy("oracle", name, isolated=False)
    assert r["reward"] == 1.0 and r["steps"] == 1
    assert r["report"]["f1"] == 1.0


@pytest.mark.parametrize("name", ALL)
def test_flag_everything_attacker_is_crushed(name):
    r = run_policy("flag_everything", name, isolated=False)
    assert r["reward"] == -1.0 and r["report"]["tp"] == 0


@pytest.mark.parametrize("name", ALL)
def test_reconciliation_baseline_finds_every_issue_without_false_positives(name):
    r = run_policy("reconciliation", name, isolated=False)
    rep = r["report"]
    assert (rep["fp"], rep["fn"]) == (0, 0), rep
    assert r["reward"] > 0.9


def test_reconciliation_solves_the_original_hand_written_episode_too():
    r = run_policy("reconciliation", "episode_001", isolated=False)
    assert r["report"]["f1"] == 1.0


def test_random_explorer_is_deterministic_per_seed_and_bad_on_dirty_companies():
    a = run_policy("random_explorer", "dev_01", isolated=False, seed=4)
    b = run_policy("random_explorer", "dev_01", isolated=False, seed=4)
    assert a["steps"] == b["steps"] and a["reward"] == b["reward"]
    assert a["reward"] < -0.9
    clean = run_policy("random_explorer", "dev_02", isolated=False, seed=4)
    assert clean["reward"] > 0.5          # silence is right on a clean company


def test_policy_ordering_is_sensible():
    rewards = {p: sum(run_policy(p, e, isolated=False)["reward"] for e in ALL) / len(ALL)
               for p in POLICY_NAMES}
    assert rewards["oracle"] > rewards["reconciliation"] > rewards["random_explorer"] > rewards["flag_everything"]


def test_policies_work_with_the_isolated_subprocess_evaluator():
    r = run_policy("reconciliation", "train_06", isolated=True)
    assert r["report"]["f1"] == 1.0


def test_reconciliation_does_not_flag_distractors():
    # distractor-heavy clean episodes -> no findings at all
    for name in ("train_02", "train_07", "dev_02", "dev_07"):
        env, _ = make_env(name, isolated=False)
        obs, info = env.reset()
        pol = build_policy("reconciliation")
        pol.reset(obs, info)
        o = obs
        while True:
            a = pol.act(o)
            if a["action"] == "submit":
                assert a["findings"] == []
                break
            o = env.step(a)[0]


def test_analyze_is_pure_and_uses_only_visible_data():
    env, _ = make_env("dev_04", isolated=False)
    obs, _ = env.reset()
    tables = {}
    for t in ("invoices", "payments", "fx", "ledger"):
        rows, cur = [], None
        while True:
            a = {"action": "list_records", "table": t, **({"cursor": cur} if cur else {})}
            o = env.step(a)[0]
            rows += o["records"]
            cur = o["next_cursor"]
            if not cur:
                break
        tables[t] = rows
    f1 = analyze(obs["manifest"], tables)
    f2 = analyze(obs["manifest"], tables)
    assert f1 == f2
    truth = truth_of("dev_04")
    assert len(max_matching(f1, truth["issues"], truth["amount_tolerance_minor"])) == len(truth["issues"])
