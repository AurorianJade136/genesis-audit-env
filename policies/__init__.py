"""Four baseline policy adapters (see DESIGN.md)."""
from policies.base import GeneratorPolicy, Policy
from policies.flag_everything import FlagEverythingPolicy
from policies.oracle import OraclePolicy
from policies.random_explorer import RandomExplorerPolicy
from policies.reconciliation import ReconciliationPolicy

POLICY_NAMES = ("random_explorer", "reconciliation", "flag_everything", "oracle")


def build_policy(name: str, seed: int = 0, truth_path: str | None = None) -> Policy:
    if name == "random_explorer":
        return RandomExplorerPolicy(seed=seed)
    if name == "reconciliation":
        return ReconciliationPolicy()
    if name == "flag_everything":
        return FlagEverythingPolicy()
    if name == "oracle":
        if truth_path is None:
            raise ValueError("the oracle policy needs the answer key (evaluator side only)")
        return OraclePolicy(truth_path)
    raise ValueError(f"unknown policy {name!r}; choose from {POLICY_NAMES}")


__all__ = ["Policy", "GeneratorPolicy", "RandomExplorerPolicy", "ReconciliationPolicy",
           "FlagEverythingPolicy", "OraclePolicy", "build_policy", "POLICY_NAMES"]
