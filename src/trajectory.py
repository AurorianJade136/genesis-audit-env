"""Trajectory recording.  The environment calls ``record`` inside ``step()`` so the file
contains the real action and observation payloads, never hand-written summaries."""
from __future__ import annotations

import json
import os

KEYS = ("episode_id", "step", "action", "observation", "reward", "terminated", "truncated")


class TrajectoryRecorder:
    def __init__(self, path: str, append: bool = True):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        if not append:
            open(path, "w", encoding="utf-8").close()

    def record(self, episode_id, step, action, observation, reward, terminated, truncated):
        rec = {
            "episode_id": episode_id, "step": step, "action": action,
            "observation": observation, "reward": reward,
            "terminated": bool(terminated), "truncated": bool(truncated),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
