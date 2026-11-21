"""One command to rebuild and check everything:   python verify.py

1. regenerate the 20 episodes from their fixed seeds
2. record real trajectories (4 policies x 3 episodes) into trajectories.jsonl
3. run the full test-suite
4. print the benchmark table
5. run the demo
"""
import subprocess
import sys

STEPS = [
    ("generate episodes", [sys.executable, "-m", "src.generator"]),
    ("record trajectories", [sys.executable, "-m", "src.runner", "record", "--out", "trajectories.jsonl"]),
    ("run tests", [sys.executable, "-m", "pytest", "-q"]),
    ("benchmark", [sys.executable, "-m", "src.runner", "benchmark"]),
    ("demo", [sys.executable, "demo.py"]),
]

for title, cmd in STEPS:
    print(f"\n##### {title}: {' '.join(cmd[1:])}", flush=True)
    code = subprocess.call(cmd)
    if code != 0:
        sys.exit(f"FAILED: {title} (exit {code})")
print("\nALL CHECKS PASSED")
