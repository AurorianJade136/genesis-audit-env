from src.env import AuditEnv

print("\n=== STARTING AI AUDITOR SIMULATION ===\n")

env = AuditEnv("data/episode_001")
obs = env.reset()
print("TASK:", obs["task"])
print("BUDGET:", obs["steps_remaining"], "steps\n")

print("--- AGENT ACTION: list_records(payments) ---")
obs, r, term, trunc, info = env.step({"action": "list_records", "table": "payments", "limit": 5})
for p in obs.get("records", [])[:3]:
    print(f"Found payment: {p.get('payment_id')} for invoice {p.get('invoice_refs')}")
    
print("\n--- AGENT ACTION: submit(5 findings) ---")

findings = [
    {
        "type": "duplicate_invoice",
        "affected_ids": ["INV-1001", "pay_001", "pay_003"],
        "claimed_amount_minor": 12000000,
        "currency": "KZT",
        "evidence_ids": ["doc_inv_01", "doc_pay_01", "doc_pay_03", "ld_0014"],
        "explanation": "Invoice INV-1001 was paid twice."
    },
    {
        "type": "unallocated_or_partial_payment",
        "affected_ids": ["INV-1002", "pay_002"],
        "claimed_amount_minor": 80000,
        "currency": "USD",
        "evidence_ids": ["doc_inv_02", "doc_pay_02", "ld_0011", "ld_0013"],
        "explanation": "Partial payment covered only 120k of 200k, but payable was cleared."
    },
    {
        "type": "period_cutoff_error",
        "affected_ids": ["INV-1006"],
        "claimed_amount_minor": 4500000,
        "currency": "KZT",
        "evidence_ids": ["doc_inv_06", "ld_0021"],
        "explanation": "September invoice booked in October."
    },
    {
        "type": "bad_fx_date_or_rate",
        "affected_ids": ["INV-1004", "pay_004"],
        "claimed_amount_minor": 1400000,
        "currency": "KZT",
        "evidence_ids": ["doc_inv_04", "doc_pay_04", "ld_0025", "fx_2026-10-22_USD_KZT"],
        "explanation": "Payment converted at wrong date's FX rate."
    },
    {
        "type": "unallocated_or_partial_payment",
        "affected_ids": ["INV-1005", "pay_005"],
        "claimed_amount_minor": 40000,
        "currency": "EUR",
        "evidence_ids": ["doc_inv_05", "doc_pay_05", "ld_0029", "ld_0030", "ld_0031", "fx_2026-10-27_EUR_KZT"],
        "explanation": "Partial payment in EUR at period boundary with wrong FX."
    }
]

obs, reward, term, trunc, info = env.step({"action": "submit", "findings": findings})

print(f"\n>>> EVALUATION RESULTS <<<")
print(f"Reward: {reward:.3f}")
print(f"True Positives: {info.get('tp', 0)}")
print(f"False Positives: {info.get('fp', 0)}")
print(f"False Negatives: {info.get('fn', 0)}")
print(f"F1 Score: {info.get('f1', 0.0):.3f}")

print("\n\n=== ATTACK SIMULATION (Flag Everything) ===")
env2 = AuditEnv("data/episode_001")
env2.reset()

bad_findings = [
    {"type": "duplicate_invoice", "affected_ids": ["INV-1003"], "claimed_amount_minor": 5800000, "currency": "KZT", "evidence_ids": [], "explanation": "Fake error"},
    {"type": "unallocated_or_partial_payment", "affected_ids": ["ld_0003"], "claimed_amount_minor": 3000000, "currency": "KZT", "evidence_ids": [], "explanation": "Fake error"}
]

_, bad_reward, *_ = env2.step({"action": "submit", "findings": bad_findings})
print(f"The 'Flag Everything' agent gets a reward of: {bad_reward:.3f}")
print("The scoring system successfully punished the blind attack.")
print("\n=== SIMULATION ENDED ===")