import csv
import hashlib
import json
import os

import pytest

from src.generator import (ERROR_TYPES, TRAIN_SEED_BASE, DEV_SEED_BASE, build_episode,
                           episode_specs, generate_all, write_episode)
from tests.helpers import DATA, TRUTH, truth_of

NAMES = [n for n, _, _ in episode_specs()]


def read(name, f):
    with open(os.path.join(DATA, name, f), encoding="utf-8") as fh:
        return fh.read()


def ledger(name):
    with open(os.path.join(DATA, name, "ledger.csv"), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_twenty_episodes_exist_and_answer_keys_are_outside_agent_dirs():
    assert len(NAMES) == 20
    for n in NAMES:
        assert os.path.isdir(os.path.join(DATA, n))
        assert not os.path.exists(os.path.join(DATA, n, "truth.json")), "answer key leaked into data/"
        assert os.path.isfile(os.path.join(TRUTH, n, "truth.json"))


def test_same_seed_is_reproducible_and_different_seed_differs():
    a = build_episode(5, {"errors": {"duplicate_invoice": 1}})
    b = build_episode(5, {"errors": {"duplicate_invoice": 1}})
    c = build_episode(6, {"errors": {"duplicate_invoice": 1}})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_committed_data_matches_the_generator(tmp_path):
    generate_all(str(tmp_path / "d"), str(tmp_path / "t"))
    for n in NAMES:
        for f in ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"):
            assert read(n, f) == open(tmp_path / "d" / n / f, encoding="utf-8").read(), (n, f)
        assert truth_of(n) == json.load(open(tmp_path / "t" / n / "truth.json"))


def test_episodes_are_genuinely_different_companies():
    fingerprints = set()
    amounts, currencies, tzs, periods = set(), set(), set(), set()
    for n in NAMES:
        fingerprints.add(hashlib.sha256(read(n, "ledger.csv").encode()).hexdigest())
        m = json.loads(read(n, "manifest.json"))
        currencies.add(m["base_currency"])
        tzs.add(m["timezone"])
        periods.add(m["period_start"][:7])
        for line in read(n, "invoices.jsonl").splitlines():
            amounts.add(json.loads(line)["total_minor"])
    assert len(fingerprints) == 20
    assert len(amounts) > 300          # amounts are not shared between episodes
    assert len(currencies) >= 3 and len(tzs) >= 4 and len(periods) >= 8


def test_train_and_dev_use_disjoint_seeds():
    seeds = [s for _, s, _ in episode_specs()]
    assert len(set(seeds)) == 20
    assert all(TRAIN_SEED_BASE < s < DEV_SEED_BASE for s in seeds[:10])
    assert all(s > DEV_SEED_BASE for s in seeds[10:])


@pytest.mark.parametrize("name", NAMES)
def test_ledger_is_balanced_large_and_manifest_consistent(name):
    rows = ledger(name)
    assert sum(int(r["debit_minor"]) for r in rows) == sum(int(r["credit_minor"]) for r in rows)
    assert len(rows) >= 90
    m = json.loads(read(name, "manifest.json"))
    assert m["episode_id"] == name
    assert m["counts"]["ledger_rows"] == len(rows)
    n_docs = len(read(name, "invoices.jsonl").splitlines()) + len(read(name, "payments.jsonl").splitlines())
    assert m["counts"]["documents"] == n_docs
    assert truth_of(name)["episode_id"] == name            # truth id == manifest id


@pytest.mark.parametrize("name", NAMES)
def test_every_id_in_the_answer_key_exists_in_the_data(name):
    row_ids = {r["row_id"] for r in ledger(name)}
    inv = [json.loads(l) for l in read(name, "invoices.jsonl").splitlines()]
    pay = [json.loads(l) for l in read(name, "payments.jsonl").splitlines()]
    fx_ids = set()
    with open(os.path.join(DATA, name, "fx.csv"), encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            fx_ids.add(f"fx_{r['date']}_{r['source_currency']}_{r['target_currency']}")
    docs = {d["doc_id"] for d in inv + pay}
    entities = {d["invoice_id"] for d in inv} | {d["payment_id"] for d in pay}
    for iss in truth_of(name)["issues"]:
        assert set(iss["affected_ids"]) <= entities
        assert iss["valid_evidence_sets"]
        for s in iss["valid_evidence_sets"]:
            assert set(s) <= row_ids | docs | fx_ids


def test_profile_coverage_clean_independent_errors_and_two_interacting_scenarios():
    clean, flavors, types_per_ep = 0, set(), []
    for n in NAMES:
        t = truth_of(n)
        if not t["issues"]:
            clean += 1
        types_per_ep.append({i["type"] for i in t["issues"]})
        for i in t["issues"]:
            if i.get("interacting"):
                flavors.add(i["flavor"])
    assert clean >= 4                                             # zero-error companies
    assert {"partial_payment_with_stale_fx", "double_payment_of_cutoff_invoice"} <= flavors
    for et in ERROR_TYPES:                                         # every type appears ...
        assert any(et in s for s in types_per_ep)
        assert any(s and et not in s for s in types_per_ep)        # ... and is switchable off
    assert len({frozenset(s) for s in types_per_ep}) >= 8          # many different combinations


def test_distractors_are_present_but_not_in_the_answer_key():
    t = truth_of("train_01")
    assert t["clean_cases"]
    affected = {a for i in t["issues"] for a in i["affected_ids"]}
    assert not (set(t["clean_cases"]) & affected)


def test_amount_formats_and_tolerance_are_in_key():
    for n in NAMES:
        t = truth_of(n)
        assert isinstance(t["amount_tolerance_minor"], int)
        for i in t["issues"]:
            assert isinstance(i["expected_amount_minor"], int) and i["expected_amount_minor"] > 0


def test_write_episode_roundtrip(tmp_path):
    v, t = build_episode(11, {"clean": True}, episode_id="x")
    write_episode(v, t, str(tmp_path / "d"), str(tmp_path / "t"))
    assert (tmp_path / "d" / "ledger.csv").exists() and (tmp_path / "t" / "truth.json").exists()
    assert not (tmp_path / "d" / "truth.json").exists()
    assert t["issues"] == []
