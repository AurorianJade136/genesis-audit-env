import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.env import AuditEnv

def run():
    env = AuditEnv("data/episode_001")
    env.reset()
    # Flag every single row in the ledger
    bad_findings = []
    for row in env.ledger:
        bad_findings.append({"type": "duplicate_invoice", "affected_ids": [row["row_id"]], "claimed_amount_minor": 0, "currency": "KZT", "evidence_ids": [], "explanation": "spam"})
    _, reward, _, _, _ = env.step({"action": "submit", "findings": bad_findings})
    print(f"Flag Everything Policy Reward: {reward}")

if __name__ == "__main__":
    run()