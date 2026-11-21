"""Harness: builds environments with a closed evaluator, runs policies, records trajectories.

    python -m src.runner run --policy reconciliation --episode dev_01
    python -m src.runner benchmark                       # all policies x all 20 episodes
    python -m src.runner record --out trajectories.jsonl # real trajectories for the submission
"""
from __future__ import annotations

import argparse
import os
import sys

from policies import POLICY_NAMES, build_policy
from src.env import AuditEnv
from src.evaluator import LocalEvaluator, SubprocessEvaluator
from src.generator import episode_specs
from src.trajectory import TrajectoryRecorder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TRUTH = os.path.join(ROOT, "truth")
RECORD_EPISODES = ("dev_01", "dev_02", "dev_04")  # mixed errors / clean / interacting


def make_env(name: str, data_root: str = DATA, truth_root: str = TRUTH, isolated: bool = True,
             recorder=None, **kwargs):
    """Return ``(env, evaluator)``.  The harness keeps the evaluator; the agent only gets env."""
    truth_path = os.path.join(truth_root, name, "truth.json")
    evaluator = SubprocessEvaluator(truth_path) if isolated else LocalEvaluator.from_path(truth_path)
    env = AuditEnv(os.path.join(data_root, name), evaluator=evaluator, recorder=recorder, **kwargs)
    return env, evaluator


def run_episode(env: AuditEnv, policy, seed: int = 0) -> dict:
    obs, info = env.reset(seed=seed)
    policy.reset(obs, info)
    total, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
    return {"reward": total, "steps": steps, "terminated": terminated, "truncated": truncated,
            "report": env.evaluator_report()}


def run_policy(policy_name: str, episode: str, recorder=None, isolated: bool = True, seed: int = 0):
    env, evaluator = make_env(episode, isolated=isolated, recorder=recorder)
    truth_path = os.path.join(TRUTH, episode, "truth.json")
    policy = build_policy(policy_name, seed=seed, truth_path=truth_path)
    return run_episode(env, policy, seed=seed)


def all_episode_names():
    return [n for n, _, _ in episode_specs()]


def cmd_run(args):
    rec = TrajectoryRecorder(args.trajectories, append=False) if args.trajectories else None
    res = run_policy(args.policy, args.episode, recorder=rec, isolated=not args.local_evaluator)
    r = res["report"]
    print(f"{args.policy} on {args.episode}: reward={res['reward']:.3f} steps={res['steps']} "
          f"TP={r['tp']} FP={r['fp']} FN={r['fn']} F1={r['f1']:.3f}")


def cmd_benchmark(args):
    names = all_episode_names()
    print(f"{'policy':<17}{'split':<7}{'mean reward':>12}{'mean F1':>9}{'mean steps':>12}")
    for pol in POLICY_NAMES:
        for split in ("train", "dev"):
            eps = [n for n in names if n.startswith(split)]
            res = [run_policy(pol, e, isolated=not args.local_evaluator) for e in eps]
            n = len(res)
            print(f"{pol:<17}{split:<7}{sum(r['reward'] for r in res) / n:>12.3f}"
                  f"{sum(r['report']['f1'] for r in res) / n:>9.3f}"
                  f"{sum(r['steps'] for r in res) / n:>12.1f}")


def cmd_record(args):
    rec = TrajectoryRecorder(args.out, append=False)
    for pol in POLICY_NAMES:
        for ep in RECORD_EPISODES:
            res = run_policy(pol, ep, recorder=rec, isolated=not args.local_evaluator)
            print(f"recorded {pol:<16} {ep}: reward={res['reward']:.3f} steps={res['steps']}")
    print(f"Wrote {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--policy", choices=POLICY_NAMES, required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--trajectories")
    p.add_argument("--local-evaluator", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("benchmark")
    p.add_argument("--local-evaluator", action="store_true")
    p.set_defaults(fn=cmd_benchmark)
    p = sub.add_parser("record")
    p.add_argument("--out", default="trajectories.jsonl")
    p.add_argument("--local-evaluator", action="store_true")
    p.set_defaults(fn=cmd_record)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
