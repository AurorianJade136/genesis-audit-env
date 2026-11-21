"""Short demo: the same company audited by four policies.  Nothing is written to disk.

    python demo.py [episode]        # default: dev_01
"""
import sys

from policies import POLICY_NAMES
from src.runner import make_env, run_policy

episode = sys.argv[1] if len(sys.argv) > 1 else "dev_01"
env, _ = make_env(episode)
obs, info = env.reset(seed=0)
print(f"\n=== Episode {info['episode_id']} ===")
m = obs["manifest"]
print(f"base currency {m['base_currency']}, period {m['period_start'][:10]} .. {m['period_end'][:10]} "
      f"({m['timezone']}), ledger rows {m['counts']['ledger_rows']}, documents {m['counts']['documents']}")
print(f"step budget: {obs['max_steps']}   tables: " +
      ", ".join(f"{t}={v['rows']}" for t, v in obs["tables"].items()))

print("\nWhat an agent sees (first 2 invoices):")
obs2, *_ = env.step({"action": "list_records", "table": "invoices", "limit": 2})
for r in obs2["records"]:
    print(f"  {r['invoice_id']}  {r['currency']} {r['total_minor']}  issued {r['issued_at']}")
print(f"  next_cursor = {obs2['next_cursor']}")

print(f"\n{'policy':<17}{'reward':>8}{'steps':>7}{'TP':>4}{'FP':>4}{'FN':>4}{'F1':>7}")
for name in POLICY_NAMES:
    r = run_policy(name, episode)
    rep = r["report"]            # closed-evaluator diagnostics, available to the harness only
    print(f"{name:<17}{r['reward']:>8.3f}{r['steps']:>7}{rep['tp']:>4}{rep['fp']:>4}{rep['fn']:>4}{rep['f1']:>7.2f}")
print("\nThe attacker (flag_everything) is punished; the oracle is the ceiling; "
      "reconciliation is a real baseline that reads only what the env exposes.")
