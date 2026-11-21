import json
import os

from src.env import AuditEnv
from src.evaluator import LocalEvaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TRUTH = os.path.join(ROOT, "truth")


def truth_of(name):
    with open(os.path.join(TRUTH, name, "truth.json"), encoding="utf-8") as f:
        return json.load(f)


def make_env(name="dev_01", **kw):
    ev = LocalEvaluator(truth_of(name))
    return AuditEnv(os.path.join(DATA, name), evaluator=ev, **kw), ev


def finding(t, **over):
    """A finding copied from an answer-key issue."""
    f = {"type": t["type"], "affected_ids": list(t["affected_ids"]),
         "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
         "evidence_ids": list(t["valid_evidence_sets"][0]), "explanation": "x"}
    f.update(over)
    return f
