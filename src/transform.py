"""Paired-check transform: shuffle ledger rows and rename all IDs consistently."""
import csv, json, os, random, shutil


def _clean(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def _id(prefix, old, mapping):
    if old in (None, "", "null"):
        return old
    if old not in mapping:
        mapping[old] = f"{prefix}{len(mapping) + 1:05d}"
    return mapping[old]


def apply(src_name, out_name, seed=42):
    sd = os.path.join("data", src_name)
    st = os.path.join("truth", src_name)
    od = os.path.join("data", out_name)
    ot = os.path.join("truth", out_name)
    _clean(od); _clean(ot)
    rng = random.Random(seed)
    m = {"ld": {}, "ent": {}, "INV": {}, "pay": {}, "doc_inv": {}, "doc_pay": {}}

    with open(os.path.join(sd, "ledger.csv"), newline="") as f:
        r = csv.DictReader(f); fields = r.fieldnames; rows = list(r)
    for row in rows:
        row["row_id"] = _id("ld_", row["row_id"], m["ld"])
        row["entity_id"] = _id("ent_", row["entity_id"], m["ent"])
        row["invoice_id"] = _id("INV-", row["invoice_id"], m["INV"])
        if row.get("payment_id") not in (None, "", "null"):
            row["payment_id"] = _id("pay_", row["payment_id"], m["pay"])
    rng.shuffle(rows)
    with open(os.path.join(od, "ledger.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    with open(os.path.join(sd, "invoices.jsonl")) as f:
        invs = [json.loads(l) for l in f if l.strip()]
    for inv in invs:
        inv["doc_id"] = _id("doc_inv_", inv["doc_id"], m["doc_inv"])
        inv["invoice_id"] = _id("INV-", inv["invoice_id"], m["INV"])
        inv["entity_id"] = _id("ent_", inv["entity_id"], m["ent"])
    with open(os.path.join(od, "invoices.jsonl"), "w") as f:
        for inv in invs: f.write(json.dumps(inv) + "\n")

    with open(os.path.join(sd, "payments.jsonl")) as f:
        pays = [json.loads(l) for l in f if l.strip()]
    for p in pays:
        p["doc_id"] = _id("doc_pay_", p["doc_id"], m["doc_pay"])
        p["payment_id"] = _id("pay_", p["payment_id"], m["pay"])
        p["entity_id"] = _id("ent_", p["entity_id"], m["ent"])
        p["invoice_refs"] = [_id("INV-", r, m["INV"]) for r in p["invoice_refs"]]
    with open(os.path.join(od, "payments.jsonl"), "w") as f:
        for p in pays: f.write(json.dumps(p) + "\n")

    for fn in ("fx.csv", "policy.json"):
        shutil.copy(os.path.join(sd, fn), os.path.join(od, fn))

    with open(os.path.join(sd, "manifest.json")) as f:
        man = json.load(f)
    man["episode_id"] = out_name
    with open(os.path.join(od, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)

    def rn(x):
        for mp in m.values():
            if x in mp: return mp[x]
        return x

    with open(os.path.join(st, "truth.json")) as f:
        truth = json.load(f)
    for issue in truth.get("issues", []):
        issue["affected_ids"] = [rn(i) for i in issue["affected_ids"]]
        issue["valid_evidence_sets"] = [[rn(e) for e in es] for es in issue["valid_evidence_sets"]]
    if "clean_cases" in truth:
        truth["clean_cases"] = [rn(c) for c in truth["clean_cases"]]
    truth["episode_id"] = out_name
    with open(os.path.join(ot, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)

    with open(os.path.join(ot, "rename_map.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m