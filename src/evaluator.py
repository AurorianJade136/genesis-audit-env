"""Closed evaluator: the only code that ever reads an answer key.

The environment (agent side) holds an *evaluator object* and calls
``evaluator.score(findings, inspection_steps)``.  It receives back a reward, but the
diagnostics (TP/FP/FN/F1) stay inside the evaluator (``last_result``) for the harness.

Two implementations:

* ``SubprocessEvaluator`` -- runs ``python -m src.evaluator`` in a **separate process**
  that loads the answer key.  The agent process never opens the key.  (default for runs)
* ``LocalEvaluator``      -- same logic in-process, for generated-on-the-fly episodes
  (``reset(seed=...)``) and fast tests.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from src.matcher import match, validate_findings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_truth(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_against_truth(truth: dict, findings, inspection_steps: int) -> dict:
    err = validate_findings(findings)
    if err:
        raise ValueError(err)
    return match(truth.get("issues", []), findings, inspection_steps,
                 int(truth.get("amount_tolerance_minor", 0)))


class LocalEvaluator:
    def __init__(self, truth: dict):
        self._truth = truth
        self.last_result: dict | None = None

    @classmethod
    def from_path(cls, path: str) -> "LocalEvaluator":
        return cls(load_truth(path))

    def score(self, findings, inspection_steps: int) -> dict:
        self.last_result = score_against_truth(self._truth, findings, inspection_steps)
        return self.last_result


class SubprocessEvaluator:
    def __init__(self, truth_path: str, python: str | None = None, timeout: float = 60.0):
        self._truth_path = os.path.abspath(truth_path)
        self._python = python or sys.executable
        self._timeout = timeout
        self.last_result: dict | None = None
        if not os.path.isfile(self._truth_path):
            raise FileNotFoundError(f"answer key not found: {truth_path}")

    def score(self, findings, inspection_steps: int) -> dict:
        payload = json.dumps({"findings": findings, "inspection_steps": inspection_steps})
        proc = subprocess.run(
            [self._python, "-m", "src.evaluator", "--truth", self._truth_path],
            input=payload, capture_output=True, text=True, cwd=ROOT, timeout=self._timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"evaluator failed: {proc.stderr.strip()}")
        self.last_result = json.loads(proc.stdout)
        return self.last_result


def main():
    ap = argparse.ArgumentParser(description="Closed evaluator process (stdin JSON -> stdout JSON).")
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    req = json.load(sys.stdin)
    res = score_against_truth(load_truth(args.truth), req["findings"], int(req["inspection_steps"]))
    json.dump(res, sys.stdout)


if __name__ == "__main__":
    main()
