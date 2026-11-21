"""Common policy interface.

A policy is a tiny adapter between an agent and the environment:

    policy.reset(obs, info)      # called once with the reset() output
    action = policy.act(obs)     # called with the latest observation, returns an action dict

``GeneratorPolicy`` lets a policy be written as a straight-line coroutine:
``obs = yield action`` -- convenient for multi-page scans.
"""
from __future__ import annotations


class Policy:
    name = "policy"

    def reset(self, obs, info):  # pragma: no cover - trivial
        pass

    def act(self, obs):  # pragma: no cover - interface
        raise NotImplementedError


class GeneratorPolicy(Policy):
    def run(self, obs, info):  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self, obs, info):
        self._gen = self.run(obs, info)
        self._started = False

    def act(self, obs):
        try:
            if not self._started:
                self._started = True
                return next(self._gen)
            return self._gen.send(obs)
        except StopIteration:
            return {"action": "submit", "findings": []}


def scan_table(table: str, page_size: int):
    """Sub-coroutine: page through a whole table with cursors.  ``rows = yield from scan_table(...)``."""
    rows, cursor = [], None
    while True:
        action = {"action": "list_records", "table": table, "limit": page_size}
        if cursor:
            action["cursor"] = cursor
        obs = yield action
        if "error" in obs:
            return rows
        rows += obs["records"]
        cursor = obs.get("next_cursor")
        if not cursor:
            return rows
