# Design Note

## Architecture
- **Generator:** Creates synthetic companies with balanced ledgers.
- **Environment (`AuditEnv`):** Implements reset/step, limiting the agent to 100 steps. It allows list, get, calculator, and submit actions.
- **Matcher:** Enforces 1:1 matching with strict evidence sets and calculates the F1 terminal reward.

## Error Types & Isolation
- We simulate 4 error types + 1 interacting error (partial payment with bad FX at the period boundary).
- **Isolation:** The environment NEVER loads `truth.json` in its `__init__` or `reset`. It is only loaded locally inside the `submit` step, used for scoring, and immediately deleted from memory to prevent leaks.

## Limitations
- The matcher is greedy (not maximum bipartite matching).
- The generator is currently hardcoded to episode 001.