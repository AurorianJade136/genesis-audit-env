import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env import AuditEnv

def test_reset_and_isolation():
    env = AuditEnv("data/episode_001")
    obs = env.reset()
    assert "task" in obs
    assert not hasattr(env, "truth_issues")

def test_attack_penalty():
    env = AuditEnv("data/episode_001")
    env.reset()
    bad_findings = [{"type": "duplicate_invoice", "affected_ids": ["INV-1003"], "claimed_amount_minor": 5800000, "currency": "KZT", "evidence_ids": [], "explanation": "Fake"}]
    obs, reward, term, trunc, info = env.step({"action": "submit", "findings": bad_findings})
    assert reward <= -0.9