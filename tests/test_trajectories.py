import json
import os
from collections import defaultdict

from policies import POLICY_NAMES
from src.runner import RECORD_EPISODES
from src.trajectory import KEYS
from tests.helpers import ROOT

PATH = os.path.join(ROOT, "trajectories.jsonl")


def load():
    with open(PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def test_file_exists_under_the_required_name():
    assert os.path.isfile(PATH)
    assert not os.path.isdir(os.path.join(ROOT, "trajectories"))


def test_every_record_has_exactly_the_seven_required_keys_and_types():
    recs = load()
    assert len(recs) > 50
    for r in recs:
        assert tuple(r) == KEYS
        assert isinstance(r["episode_id"], str) and isinstance(r["step"], int)
        assert isinstance(r["action"], dict) and isinstance(r["observation"], dict)
        assert isinstance(r["reward"], (int, float))
        assert isinstance(r["terminated"], bool) and isinstance(r["truncated"], bool)


def test_trajectories_are_real_not_placeholder_strings():
    recs = load()
    assert not any(r["episode_id"] == "ep_demo_001" for r in recs)
    assert any("records" in r["observation"] and len(r["observation"]["records"]) > 5 for r in recs)
    assert any(r["action"].get("action") == "submit" and r["action"]["findings"] for r in recs)


def test_each_run_is_a_consecutive_episode_that_ends_once():
    runs, cur = [], []
    for r in load():
        if r["step"] == 1 and cur:
            runs.append(cur)
            cur = []
        cur.append(r)
    runs.append(cur)
    assert len(runs) == len(POLICY_NAMES) * len(RECORD_EPISODES)
    for run in runs:
        assert [r["step"] for r in run] == list(range(1, len(run) + 1))
        assert all(not (r["terminated"] or r["truncated"]) for r in run[:-1])
        assert run[-1]["terminated"] or run[-1]["truncated"]
        assert run[-1]["reward"] != 0.0 or run[-1]["episode_id"].startswith("dev_")
        assert all(r["reward"] == 0.0 for r in run[:-1])
