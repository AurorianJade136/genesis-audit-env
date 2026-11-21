"""Baseline 2 -- simple reconciliation algorithm.

Pages through every table with cursors, then applies five deterministic rules using only
what the environment shows (manifest, policy, ledger, documents, FX table):

1. duplicate_invoice        an invoice paid more than its total, or the same invoice
                            (vendor, currency, total, description) entered twice
2. partial / unallocated    a payable cleared with a 5900 plug although less was paid, or a
                            payment with no invoice reference
3. period_cutoff_error      issued outside the period but booked inside it
4. bad_fx_date_or_rate      booked base amount differs from amount x rate on the issue date
5. interacting cases are reported once, under the primary type (duplicate beats cutoff,
   partial beats fx)
"""
from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from policies.base import GeneratorPolicy, scan_table

AP = "2000_Accounts_Payable"
ADJ = "5900_Purchase_Adjustments"
ROUNDING_SLACK = 2  # minor units of the base currency


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _conv(amount, src, dst, rate, exps) -> int:
    if src == dst:
        return amount
    major = Decimal(amount) / (Decimal(10) ** exps[src])
    out = major * rate * (Decimal(10) ** exps[dst])
    return int(out.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def analyze(manifest: dict, tables: dict) -> list[dict]:
    """Pure function: tables -> findings."""
    base = manifest["base_currency"]
    exps = manifest["currency_exponents"]
    p_start, p_end = _dt(manifest["period_start"]), _dt(manifest["period_end"])
    tz = p_start.tzinfo
    invoices = {i["invoice_id"]: i for i in tables["invoices"]}
    payments = tables["payments"]
    fx = {(r["source_currency"], r["target_currency"], r["date"]): Decimal(r["rate"])
          for r in tables["fx"]}
    by_inv: dict[str, list[dict]] = {}
    by_pay: dict[str, list[dict]] = {}
    for r in tables["ledger"]:
        if r["invoice_id"]:
            by_inv.setdefault(r["invoice_id"], []).append(r)
        if r["payment_id"]:
            by_pay.setdefault(r["payment_id"], []).append(r)
    pays_of: dict[str, list[dict]] = {}
    for p in payments:
        for ref in p.get("invoice_refs", []):
            pays_of.setdefault(ref, []).append(p)

    def issue_date(inv):
        return _dt(inv["issued_at"]).astimezone(tz).date().isoformat()

    def fx_id(inv):
        return f"fx_{issue_date(inv)}_{inv['currency']}_{base}"

    def rate_of(inv):
        if inv["currency"] == base:
            return Decimal(1)
        return fx.get((inv["currency"], base, issue_date(inv)))

    def inv_rows(inv_id):
        return [r["row_id"] for r in by_inv.get(inv_id, []) if not r["payment_id"]]

    findings: list[dict] = []
    flagged_double: set[str] = set()
    flagged_partial: set[str] = set()
    flagged_dup_record: set[str] = set()

    def add(type_, affected, amount, cur, evidence, text):
        findings.append({
            "type": type_, "affected_ids": affected, "claimed_amount_minor": int(amount),
            "currency": cur, "evidence_ids": list(dict.fromkeys(evidence))[:10],
            "explanation": text[:900],
        })

    # 1a. one invoice paid more than once / more than its total
    for inv_id, plist in pays_of.items():
        inv = invoices.get(inv_id)
        if inv is None or len(plist) < 2:
            continue
        paid = sum(p["amount_minor"] for p in plist)
        if paid > inv["total_minor"] and all(p["amount_minor"] == inv["total_minor"] for p in plist[:2]):
            plist = sorted(plist, key=lambda p: _dt(p["timestamp"]))
            second = plist[1]
            ev = [inv["doc_id"]] + [p["doc_id"] for p in plist[:2]]
            ev += [r["row_id"] for r in by_pay.get(second["payment_id"], [])]
            note = ""
            issued = _dt(inv["issued_at"])
            if issued < p_start or issued > p_end:  # interacting: also a cut-off problem
                ev += inv_rows(inv_id)
                note = " The invoice was also issued outside the period (cut-off)."
            add("duplicate_invoice", [inv_id] + [p["payment_id"] for p in plist[:2]],
                inv["total_minor"], inv["currency"], ev,
                f"{inv_id} was paid {len(plist)} times; payment {second['payment_id']} duplicates it."
                + note)
            flagged_double.add(inv_id)

    # 1b. the same invoice entered twice under two ids
    groups: dict[tuple, list[dict]] = {}
    for inv in invoices.values():
        key = (inv["entity_id"], inv["currency"], inv["total_minor"],
               tuple(li["description"] for li in inv["line_items"]))
        groups.setdefault(key, []).append(inv)
    for grp in groups.values():
        if len(grp) < 2:
            continue
        grp = sorted(grp, key=lambda i: _dt(i["issued_at"]))
        first, later = grp[0], grp[1]
        if (_dt(later["issued_at"]) - _dt(first["issued_at"])).days <= 7:
            ev = [first["doc_id"], later["doc_id"]] + inv_rows(later["invoice_id"])
            add("duplicate_invoice", [first["invoice_id"], later["invoice_id"]],
                later["total_minor"], later["currency"], ev,
                f"{later['invoice_id']} repeats {first['invoice_id']} (same vendor, amount, lines).")
            flagged_dup_record.add(later["invoice_id"])

    # 2a. payable cleared with an adjustment plug although less was paid
    handled_payments: set[str] = set()
    for p in payments:
        refs = p.get("invoice_refs", [])
        if len(refs) != 1 or refs[0] not in invoices:
            continue
        inv = invoices[refs[0]]
        rows = by_pay.get(p["payment_id"], [])
        if any(r["account"] == ADJ and r["credit_minor"] > 0 for r in rows) \
                and p["amount_minor"] < inv["total_minor"]:
            ev = [inv["doc_id"], p["doc_id"]] + [r["row_id"] for r in rows]
            if inv["currency"] != base:
                ev.append(fx_id(inv))
            short = inv["total_minor"] - p["amount_minor"]
            add("unallocated_or_partial_payment", [inv["invoice_id"], p["payment_id"]], short,
                inv["currency"], ev,
                f"{p['payment_id']} paid {p['amount_minor']} of {inv['total_minor']} but the payable "
                f"was cleared in full via {ADJ}.")
            flagged_partial.add(inv["invoice_id"])
            handled_payments.add(p["payment_id"])

    # 2b. payments never applied to an invoice
    paid_ids = {ref for p in payments for ref in p.get("invoice_refs", [])}
    for p in payments:
        if p.get("invoice_refs"):
            continue
        cands = [i for i in invoices.values()
                 if i["invoice_id"] not in paid_ids and i["currency"] == p["currency"]
                 and i["total_minor"] == p["amount_minor"]
                 and (not p.get("entity_id") or i["entity_id"] == p["entity_id"])]
        if cands:
            inv = cands[0]
            ev = [inv["doc_id"], p["doc_id"]] + [r["row_id"] for r in by_pay.get(p["payment_id"], [])]
            add("unallocated_or_partial_payment", [inv["invoice_id"], p["payment_id"]],
                p["amount_minor"], p["currency"], ev,
                f"{p['payment_id']} matches {inv['invoice_id']} but was never applied to it.")
            handled_payments.add(p["payment_id"])

    # 3. cut-off: issued outside the period, booked inside it
    for inv in invoices.values():
        if inv["invoice_id"] in flagged_double:
            continue
        issued = _dt(inv["issued_at"])
        outside = issued < p_start or issued > p_end
        booked = [r for r in by_inv.get(inv["invoice_id"], []) if not r["payment_id"]
                  and p_start <= _dt(r["timestamp"]) <= p_end]
        if outside and booked:
            add("period_cutoff_error", [inv["invoice_id"]], inv["total_minor"], inv["currency"],
                [inv["doc_id"]] + inv_rows(inv["invoice_id"]),
                f"{inv['invoice_id']} issued {inv['issued_at']} is outside the period but booked inside it.")

    # 4. FX: booked base amount vs amount x rate on the invoice issue date
    for inv in invoices.values():
        if inv["currency"] == base:
            continue
        rate = rate_of(inv)
        if rate is None:
            continue
        plist = pays_of.get(inv["invoice_id"], [])
        pay_docs = [p["doc_id"] for p in plist]
        pay_ids = [p["payment_id"] for p in plist]
        expected = _conv(inv["total_minor"], inv["currency"], base, rate, exps)
        ap_cr = [r for r in by_inv.get(inv["invoice_id"], [])
                 if r["account"] == AP and r["credit_minor"] > 0 and not r["payment_id"]]
        if ap_cr and abs(ap_cr[0]["credit_minor"] - expected) > ROUNDING_SLACK:
            add("bad_fx_date_or_rate", [inv["invoice_id"]] + pay_ids,
                abs(ap_cr[0]["credit_minor"] - expected), base,
                [inv["doc_id"]] + pay_docs + inv_rows(inv["invoice_id"]) + [fx_id(inv)],
                f"{inv['invoice_id']} booked at a rate other than the {issue_date(inv)} rate.")
        for p in plist:
            if p["payment_id"] in handled_payments:
                continue
            ap_dr = [r for r in by_pay.get(p["payment_id"], [])
                     if r["account"] == AP and r["debit_minor"] > 0]
            if not ap_dr:
                continue
            exp_p = _conv(p["amount_minor"], inv["currency"], base, rate, exps)
            diff = abs(ap_dr[0]["debit_minor"] - exp_p)
            if diff > ROUNDING_SLACK and inv["invoice_id"] not in flagged_partial:
                add("bad_fx_date_or_rate", [inv["invoice_id"], p["payment_id"]], diff, base,
                    [inv["doc_id"], p["doc_id"]] + [r["row_id"] for r in by_pay[p["payment_id"]]]
                    + [fx_id(inv)],
                    f"{p['payment_id']} settled {inv['invoice_id']} at a rate other than the "
                    f"{issue_date(inv)} rate.")
    return findings


class ReconciliationPolicy(GeneratorPolicy):
    name = "reconciliation"

    def run(self, obs, info):
        tables = {}
        for t in ("invoices", "payments", "fx", "ledger"):
            tables[t] = yield from scan_table(t, obs["page_size"])
        yield {"action": "submit", "findings": analyze(obs["manifest"], tables)}
