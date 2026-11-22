"""Paired invariance checks: permute ledger order + rename all IDs.

PDF §10: oracle and reconciliation must earn identical reward on the
original and transformed episode. 20 paired checks, 15 pts.
"""
import pytest

from src import transform
from src.runner import run_policy


EPISODES = [f"train_{i:02d}" for i in range(1, 11)] + [f"dev_{i:02d}" for i in range(1, 11)]


@pytest.mark.parametrize("name", EPISODES)
def test_oracle_reward_invariant(name):
    orig = run_policy("oracle", name)
    assert orig["reward"] == pytest.approx(1.0, abs=1e-6)
    transform.apply(name, name + "_pair", seed=42)
    pair = run_policy("oracle", name + "_pair")
    assert pair["reward"] == pytest.approx(orig["reward"], abs=1e-6), (
        f"{name}: oracle {orig['reward']} -> {pair['reward']}"
    )


@pytest.mark.parametrize("name", EPISODES[:5])
def test_reconciliation_reward_invariant(name):
    orig = run_policy("reconciliation", name)
    transform.apply(name, name + "_pair", seed=42)
    pair = run_policy("reconciliation", name + "_pair")
    assert pair["reward"] == pytest.approx(orig["reward"], abs=1e-3), (
        f"{name}: reconciliation {orig['reward']} -> {pair['reward']}"
    )