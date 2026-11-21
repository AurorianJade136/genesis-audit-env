"""Synthetic company generator for the Genesis audit environment.

Every episode is a *different* small company: its own period, time zone, base
currency, foreign currencies, FX table, vendors, invoices, payments and ledger.
The four error types are configured independently and two "interacting" error
scenarios are supported.  Clean companies (zero errors) are valid episodes.

Agent-visible files go to ``data/<name>/`` and the hidden answer key goes to
``truth/<name>/truth.json`` -- never into the agent-visible directory.

Usage:
    python -m src.generator                # regenerate the 10 train + 10 dev episodes
    python -m src.generator --out data --truth truth

Library:
    visible, truth = build_episode(seed, config)
"""
from __future__ import annotations

import argparse
import calendar
import csv
import itertools
import json
import os
import random
import shutil
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

SCHEMA_VERSION = "1.0.0"

ERROR_TYPES = (
    "duplicate_invoice",
    "unallocated_or_partial_payment",
    "period_cutoff_error",
    "bad_fx_date_or_rate",
)
INTERACTING = ("partial_fx", "duplicate_cutoff")

CURRENCY_EXPONENTS = {"KZT": 2, "USD": 2, "EUR": 2, "GBP": 2, "JPY": 0, "CNY": 2, "AED": 2, "KWD": 3}
USD_PER_UNIT = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "KZT": 1 / 485, "JPY": 1 / 150,
    "CNY": 1 / 7.2, "AED": 0.2723, "KWD": 3.25,
}
# typical invoice size in *major* units
AMOUNT_RANGE_MAJOR = {
    "KZT": (150_000, 4_000_000), "USD": (400, 15_000), "EUR": (400, 15_000),
    "GBP": (300, 12_000), "JPY": (60_000, 2_000_000), "CNY": (3_000, 100_000),
    "AED": (1_500, 50_000), "KWD": (150, 5_000),
}
# fixed-offset zones only (no DST) so that every timestamp is unambiguous
TIMEZONES = {
    "Asia/Almaty": 5, "Asia/Tashkent": 5, "Asia/Dubai": 4,
    "Asia/Singapore": 8, "Asia/Tokyo": 9, "UTC": 0,
}
SERVICES = [
    "office cleaning services", "consulting services", "software licence",
    "cloud hosting subscription", "facility maintenance", "legal advisory",
    "marketing campaign", "equipment servicing", "courier and logistics",
    "engineering design services", "security monitoring", "translation services",
    "accounting support", "training workshop", "network maintenance",
]
EXPENSE_ACCOUNTS = [
    "5100_Services_Expense", "5200_Software_Expense",
    "5500_Maintenance_Expense", "5600_Marketing_Expense",
]
DISTRACTOR_KINDS = (
    "recurring_similar", "installments", "partial_open",
    "tz_boundary_start", "tz_boundary_end", "month_end_payment",
)
VAT_RATE = Decimal("0.16")

LEDGER_FIELDS = [
    "row_id", "timestamp", "account", "debit_minor", "credit_minor",
    "currency", "entity_id", "invoice_id", "payment_id",
]
FX_FIELDS = ["date", "source_currency", "target_currency", "rate"]


# --------------------------------------------------------------------------- #
# money helpers
# --------------------------------------------------------------------------- #
def convert_minor(amount_minor: int, src: str, dst: str, rate: Decimal) -> int:
    """Convert a minor-unit amount using ``rate`` (dst units per 1 src unit), half-up."""
    major = Decimal(amount_minor) / (Decimal(10) ** CURRENCY_EXPONENTS[src])
    out = major * Decimal(rate) * (Decimal(10) ** CURRENCY_EXPONENTS[dst])
    return int(out.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _fmt_rate(r: float) -> Decimal:
    return Decimal(repr(r)).quantize(Decimal("0.0001") if r >= 1 else Decimal("0.00000001"))


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def resolve_config(config, rng: random.Random) -> dict:
    """Fill every missing option from ``rng``.  Error counts are independent."""
    cfg = dict(config or {})
    explicit = cfg.get("clean") or "errors" in cfg or "interacting" in cfg
    zeros = {t: 0 for t in ERROR_TYPES}
    izeros = {k: 0 for k in INTERACTING}
    if cfg.get("clean"):
        errors, inter = dict(zeros), dict(izeros)
    elif explicit:
        errors = {t: int(cfg.get("errors", {}).get(t, 0)) for t in ERROR_TYPES}
        inter = {k: int(cfg.get("interacting", {}).get(k, 0)) for k in INTERACTING}
    elif rng.random() < 0.2:
        errors, inter = dict(zeros), dict(izeros)
    else:
        errors = {t: rng.choice([0, 0, 1, 1, 2]) for t in ERROR_TYPES}
        inter = {k: rng.choice([0, 0, 1]) for k in INTERACTING}

    base = cfg.get("base_currency") or rng.choices(["KZT", "USD", "EUR"], [6, 2, 2])[0]
    pool = [c for c in CURRENCY_EXPONENTS if c != base]
    foreign = list(cfg.get("foreign_currencies") or rng.sample(pool, rng.randint(1, 3)))
    out = {
        "errors": errors,
        "interacting": inter,
        "base_currency": base,
        "foreign_currencies": foreign,
        "timezone": cfg.get("timezone") or rng.choice(list(TIMEZONES)),
        "year": int(cfg.get("year", 2026)),
        "month": int(cfg.get("month") or rng.randint(1, 12)),
        "n_vendors": int(cfg.get("n_vendors") or rng.randint(7, 12)),
        "n_invoices": int(cfg.get("n_invoices") or rng.randint(20, 34)),
        "distractors": cfg.get("distractors") if cfg.get("distractors") is not None else rng.randint(4, 8),
        "ordering": cfg.get("ordering") or rng.choice(["chronological", "by_entity", "shuffled"]),
        "amount_tolerance_minor": (
            cfg["amount_tolerance_minor"] if cfg.get("amount_tolerance_minor") is not None
            else rng.choice([0, 0, 1])
        ),
    }
    if out["base_currency"] in out["foreign_currencies"]:
        raise ValueError("base currency cannot also be a foreign currency")
    if out["timezone"] not in TIMEZONES:
        raise ValueError(f"unknown timezone {out['timezone']!r}")
    return out


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
class _Builder:
    def __init__(self, seed: int, config, episode_id: str):
        self.seed = seed
        self.episode_id = episode_id
        self.rng = random.Random(seed)
        self.cfg = resolve_config(config, self.rng)
        self.base = self.cfg["base_currency"]
        self.foreign = self.cfg["foreign_currencies"]
        self.tz_name = self.cfg["timezone"]
        self.tz = timezone(timedelta(hours=TIMEZONES[self.tz_name]))
        y, m = self.cfg["year"], self.cfg["month"]
        self.ndays = calendar.monthrange(y, m)[1]
        self.p_start = datetime(y, m, 1, 0, 0, 0, tzinfo=self.tz)
        self.p_end = datetime(y, m, self.ndays, 23, 59, 59, tzinfo=self.tz)
        self._used_ids: set[str] = set()
        self.txns: list[dict] = []
        self.invoices: list[dict] = []
        self.payments: list[dict] = []
        self.issues: list[dict] = []
        self.clean_cases: list[str] = []
        self._build_fx()
        self._build_vendors()

    # ---- primitives ------------------------------------------------------- #
    def uid(self, prefix: str, lo: int, hi: int, width: int) -> str:
        while True:
            v = f"{prefix}{self.rng.randint(lo, hi):0{width}d}"
            if v not in self._used_ids:
                self._used_ids.add(v)
                return v

    def _build_fx(self):
        self.fx: dict[tuple[str, str], Decimal] = {}
        start = self.p_start.date() - timedelta(days=14)
        end = self.p_end.date() + timedelta(days=6)
        for cur in self.foreign:
            r = USD_PER_UNIT[cur] / USD_PER_UNIT[self.base] * self.rng.uniform(0.97, 1.03)
            d = start
            while d <= end:
                self.fx[(cur, d.isoformat())] = _fmt_rate(r)
                r *= 1 + self.rng.uniform(0.004, 0.018) * self.rng.choice((-1, 1))
                d += timedelta(days=1)

    def _build_vendors(self):
        self.vendors: list[tuple[str, str]] = []
        n = self.cfg["n_vendors"]
        for i in range(n):
            ent = self.uid("ent_", 100, 9999, 4)
            if i < len(self.foreign):
                cur = self.foreign[i]
            elif i == len(self.foreign):
                cur = self.base
            else:
                cur = self.base if self.rng.random() < 0.45 else self.rng.choice(self.foreign)
            self.vendors.append((ent, cur))

    def vendor(self, foreign: bool | None = None):
        pool = self.vendors
        if foreign is True:
            pool = [v for v in pool if v[1] != self.base]
        elif foreign is False:
            pool = [v for v in pool if v[1] == self.base]
        return self.rng.choice(pool)

    def local_date(self, dt: datetime) -> str:
        return dt.astimezone(self.tz).date().isoformat()

    def rate_for(self, cur: str, issue_dt: datetime) -> Decimal:
        if cur == self.base:
            return Decimal(1)
        return self.fx[(cur, self.local_date(issue_dt))]

    def fx_id(self, cur: str, issue_dt: datetime) -> str:
        return f"fx_{self.local_date(issue_dt)}_{cur}_{self.base}"

    def conv(self, amount_minor: int, cur: str, rate: Decimal) -> int:
        if cur == self.base:
            return amount_minor
        return convert_minor(amount_minor, cur, self.base, rate)

    def wrong_rate(self, cur: str, issue_dt: datetime, direction: str | None = None) -> Decimal:
        """A real rate from some *other* date that differs by >= 1%."""
        correct = self.rate_for(cur, issue_dt)
        ds = self.local_date(issue_dt)
        cands = [
            r for (c, d), r in self.fx.items()
            if c == cur and d != ds and abs(r / correct - 1) >= Decimal("0.01")
        ]
        if direction == "lower":
            lower = [r for r in cands if r < correct]
            cands = lower or cands
        if not cands:
            return _fmt_rate(float(correct) * 0.97)
        return self.rng.choice(cands)

    def rand_dt(self, day_lo: int, day_hi: int, hour_lo: int = 8, hour_hi: int = 17) -> datetime:
        day = self.rng.randint(day_lo, day_hi)
        return self.p_start + timedelta(
            days=day, hours=self.rng.randint(hour_lo, hour_hi), minutes=self.rng.randint(0, 59)
        )

    def day_index(self, dt: datetime) -> int:
        return (dt - self.p_start).days

    def rand_total(self, cur: str, cents: bool = False) -> int:
        lo, hi = AMOUNT_RANGE_MAJOR[cur]
        e = CURRENCY_EXPONENTS[cur]
        step = 100 if hi >= 100_000 else 10
        major = int(round(self.rng.uniform(lo / step, hi / step))) * step
        major = max(major, step)
        minor = major * 10 ** e
        if cents and e > 0 and self.rng.random() < 0.25:
            minor += self.rng.randint(1, 10 ** e - 1)
        return minor

    def round_down(self, cur: str, minor: int, step_major: int = 10) -> int:
        unit = 10 ** CURRENCY_EXPONENTS[cur] * step_major
        out = (minor // unit) * unit
        return out if out > 0 else 10 ** CURRENCY_EXPONENTS[cur]

    def ts(self, dt: datetime, utc: bool = False) -> str:
        return (dt.astimezone(timezone.utc) if utc else dt.astimezone(self.tz)).isoformat()

    def boundary_safe(self, dt: datetime) -> bool:
        return self.p_start + timedelta(days=2) < dt < self.p_end - timedelta(days=2)

    # ---- ledger ----------------------------------------------------------- #
    def add_txn(self, dt: datetime, rows: list[dict], entity: str | None):
        ts = self.ts(dt)
        full = []
        for r in rows:
            row = {
                "row_id": None, "timestamp": ts, "account": r["account"],
                "debit_minor": r.get("debit", 0), "credit_minor": r.get("credit", 0),
                "currency": self.base, "entity_id": entity,
                "invoice_id": r.get("invoice"), "payment_id": r.get("payment"),
            }
            full.append(row)
        self.txns.append({"dt": dt, "entity": entity, "rows": full, "seq": len(self.txns)})
        return full

    # ---- documents -------------------------------------------------------- #
    def make_invoice(self, vendor, issue_dt, *, book_dt=None, total_minor=None, desc=None,
                     lines=None, vat=False, book_rate=None, due_days=None, utc=False,
                     cents=True) -> dict:
        ent, cur = vendor
        if lines is None:
            net = total_minor if total_minor is not None else self.rand_total(cur, cents=cents)
            n_lines = self.rng.choice([1, 1, 2, 3])
            parts = []
            left = net
            for i in range(n_lines - 1):
                p = int(left * self.rng.uniform(0.25, 0.6))
                parts.append(p)
                left -= p
            parts.append(left)
            first = desc or self.rng.choice(SERVICES)
            descs = [first] + [self.rng.choice(SERVICES) for _ in range(n_lines - 1)]
            lines = list(zip(descs, parts))
        net_total = sum(a for _, a in lines)
        vat_minor = 0
        if vat:
            vat_minor = int((Decimal(net_total) * VAT_RATE).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        total = net_total + vat_minor
        items = [
            {"line": i + 1, "description": d, "quantity": 1, "unit_price_minor": a, "amount_minor": a}
            for i, (d, a) in enumerate(lines)
        ]
        if vat:
            items.append({"line": len(items) + 1, "description": "VAT 16%", "quantity": 1,
                          "unit_price_minor": vat_minor, "amount_minor": vat_minor})
        inv_id = self.uid("INV-", 1000, 9999, 4)
        doc_id = self.uid("doc_inv_", 1, 999, 3)
        due = (issue_dt + timedelta(days=due_days or self.rng.choice([14, 30]))).replace(
            hour=23, minute=59, second=59)
        inv = {
            "doc_id": doc_id, "invoice_id": inv_id, "entity_id": ent,
            "issued_at": self.ts(issue_dt, utc=utc), "due_at": self.ts(due),
            "currency": cur, "total_minor": total, "line_items": items,
        }
        rate = book_rate if book_rate is not None else self.rate_for(cur, issue_dt)
        ap_base = self.conv(total, cur, rate)
        vat_base = self.conv(vat_minor, cur, rate) if vat else 0
        exp_base = ap_base - vat_base
        rows = [{"account": self.rng.choice(EXPENSE_ACCOUNTS), "debit": exp_base,
                 "invoice": inv_id}]
        if vat:
            rows.append({"account": "1500_VAT_Input", "debit": vat_base, "invoice": inv_id})
        rows.append({"account": "2000_Accounts_Payable", "credit": ap_base, "invoice": inv_id})
        full = self.add_txn(book_dt or issue_dt, rows, ent)
        rec = {
            "doc": inv, "id": inv_id, "doc_id": doc_id, "ent": ent, "cur": cur,
            "total": total, "rate": rate, "ap_base": ap_base, "issue_dt": issue_dt,
            "lines": lines, "exp_row": full[0], "ap_row": full[-1],
            "rows": full, "paid_base": 0,
        }
        self.invoices.append(rec)
        return rec

    def make_payment(self, rec, pay_dt, amount_minor, *, ap_debit=None, cash=None, rate=None,
                     unapplied=False, utc=False) -> dict:
        cur = rec["cur"]
        rate = rec["rate"] if rate is None else rate
        if ap_debit is None:
            ap_debit = self.conv(amount_minor, cur, rate)
        if cash is None:
            cash = ap_debit
        pid = self.uid("pay_", 100, 9999, 4)
        doc_id = self.uid("doc_pay_", 1, 999, 3)
        pay = {
            "doc_id": doc_id, "payment_id": pid, "timestamp": self.ts(pay_dt, utc=utc),
            "currency": cur, "amount_minor": amount_minor,
            "invoice_refs": [] if unapplied else [rec["id"]], "entity_id": rec["ent"],
        }
        if unapplied:
            rows = [
                {"account": "1400_Unapplied_Payments", "debit": ap_debit, "payment": pid},
                {"account": "1000_Cash", "credit": cash, "payment": pid},
            ]
        else:
            rows = [
                {"account": "2000_Accounts_Payable", "debit": ap_debit,
                 "invoice": rec["id"], "payment": pid},
                {"account": "1000_Cash", "credit": cash, "invoice": rec["id"], "payment": pid},
            ]
            if cash != ap_debit:
                assert cash < ap_debit
                rows.append({"account": "5900_Purchase_Adjustments", "credit": ap_debit - cash,
                             "invoice": rec["id"], "payment": pid})
        full = self.add_txn(pay_dt, rows, rec["ent"])
        prec = {"doc": pay, "id": pid, "doc_id": doc_id, "dt": pay_dt, "rows": full,
                "ap_row": full[0], "cash_row": full[1],
                "adj_row": full[2] if len(full) > 2 else None, "ap_debit": ap_debit, "cash": cash}
        self.payments.append(prec)
        rec["paid_base"] += ap_debit
        return prec

    def add_issue(self, type_, affected, amount, currency, groups, **extra):
        self.issues.append({"type": type_, "affected": affected, "amount": amount,
                            "currency": currency, "groups": groups, "extra": extra})

    # ---- normal activity -------------------------------------------------- #
    def build_normal(self, n: int):
        nd = self.ndays
        for _ in range(n):
            vendor = self.rng.choice(self.vendors)
            issue_dt = self.rand_dt(1, nd - 5)
            utc = self.rng.random() < 0.15 and self.boundary_safe(issue_dt)
            rec = self.make_invoice(vendor, issue_dt, vat=self.rng.random() < 0.3, utc=utc)
            day = self.day_index(issue_dt)
            if self.rng.random() < 0.62 and day <= nd - 4:
                pd_ = self.rand_dt(day + 1, min(day + 12, nd - 1))
                put = self.rng.random() < 0.12 and self.boundary_safe(pd_)
                self.make_payment(rec, pd_, rec["total"], utc=put)

    def build_misc(self):
        lo, hi = AMOUNT_RANGE_MAJOR[self.base]
        e10 = 10 ** CURRENCY_EXPONENTS[self.base]
        rent = int(self.rng.uniform(lo * 0.5, lo) // 1000 * 1000 or lo) * e10
        dt = self.p_start + timedelta(days=2, hours=9)
        self.add_txn(dt, [{"account": "5300_Rent_Expense", "debit": rent},
                          {"account": "1000_Cash", "credit": rent}], None)
        for _ in range(self.rng.randint(2, 5)):
            amt = int(self.rng.uniform(lo * 0.01, lo * 0.05)) * e10 // 10 * 10 or e10
            acct = self.rng.choice(["5400_Bank_Fees", "5700_Utilities", "5800_Office_Supplies"])
            self.add_txn(self.rand_dt(0, self.ndays - 1, 9, 17),
                         [{"account": acct, "debit": amt}, {"account": "1000_Cash", "credit": amt}], None)
        cash_out = sum(r["credit_minor"] for t in self.txns for r in t["rows"]
                       if r["account"] == "1000_Cash")
        capital = (int(cash_out * 1.4) // (10 ** 7) + 1) * 10 ** 7
        self.add_txn(self.p_start + timedelta(hours=9),
                     [{"account": "1000_Cash", "debit": capital},
                      {"account": "3000_Equity", "credit": capital}], None)

    # ---- distractors (look suspicious, are clean) ------------------------- #
    def add_distractor(self, kind: str):
        nd = self.ndays
        v = self.rng.choice(self.vendors)
        if kind == "recurring_similar":
            d = self.rng.randint(1, nd - 16)
            svc = self.rng.choice(SERVICES)
            total = self.rand_total(v[1])
            a = self.make_invoice(v, self.rand_dt(d, d), total_minor=total, desc=f"{svc} - week 1", due_days=14)
            b = self.make_invoice(v, self.rand_dt(d + 7, d + 10), total_minor=total, desc=f"{svc} - week 2", due_days=14)
            for r in (a, b):
                dd = self.day_index(r["issue_dt"])
                self.make_payment(r, self.rand_dt(dd + 2, min(dd + 8, nd - 1)), r["total"])
            self.clean_cases += [a["id"], b["id"]]
        elif kind == "installments":
            d = self.rng.randint(1, nd - 14)
            r = self.make_invoice(v, self.rand_dt(d, d), cents=False)
            p1 = self.round_down(r["cur"], int(r["total"] * self.rng.uniform(0.4, 0.6)))
            p2 = r["total"] - p1
            first = self.make_payment(r, self.rand_dt(d + 2, d + 5), p1)
            self.make_payment(r, self.rand_dt(d + 8, min(d + 12, nd - 1)), p2,
                              ap_debit=r["ap_base"] - first["ap_debit"])
            self.clean_cases.append(r["id"])
        elif kind == "partial_open":
            d = self.rng.randint(1, nd - 10)
            r = self.make_invoice(v, self.rand_dt(d, d), cents=False, due_days=45)
            p = self.round_down(r["cur"], int(r["total"] * self.rng.uniform(0.3, 0.6)))
            self.make_payment(r, self.rand_dt(d + 2, d + 6), p)
            self.clean_cases.append(r["id"])
        elif kind == "tz_boundary_start":
            dt = self.p_start + timedelta(minutes=self.rng.randint(20, 50))
            r = self.make_invoice(v, dt, utc=True)
            self.clean_cases.append(r["id"])
        elif kind == "tz_boundary_end":
            dt = self.p_end - timedelta(minutes=self.rng.randint(20, 50))
            r = self.make_invoice(v, dt, utc=True)
            self.clean_cases.append(r["id"])
        elif kind == "month_end_payment":
            d = nd - self.rng.randint(5, 8)
            r = self.make_invoice(v, self.rand_dt(d, d), cents=False)
            self.make_payment(r, self.p_end - timedelta(minutes=1), r["total"])
            self.clean_cases.append(r["id"])
        else:
            raise ValueError(kind)

    # ---- single-type error injections ------------------------------------- #
    def inject_duplicate(self):
        nd = self.ndays
        v = self.rng.choice(self.vendors)
        if self.rng.random() < 0.6:  # flavour A: one invoice paid twice
            d = self.rng.randint(1, nd - 12)
            r = self.make_invoice(v, self.rand_dt(d, d))
            d1 = d + self.rng.randint(1, 5)
            d2 = min(d1 + self.rng.randint(2, 6), nd - 1)
            p1 = self.make_payment(r, self.rand_dt(d1, d1), r["total"])
            p2 = self.make_payment(r, self.rand_dt(d2, d2), r["total"])
            self.add_issue(
                "duplicate_invoice", [r["id"], p1["id"], p2["id"]], r["total"], r["cur"],
                [[r["doc_id"]], [p1["doc_id"]], [p2["doc_id"]], [p2["ap_row"], p2["cash_row"]]],
                flavor="double_payment")
        else:  # flavour B: the same invoice entered twice under two ids
            d = self.rng.randint(1, nd - 12)
            dt1 = self.rand_dt(d, d)
            a = self.make_invoice(v, dt1, cents=False)
            dt2 = dt1 + timedelta(hours=self.rng.randint(2, 50), minutes=self.rng.randint(1, 59))
            b = self.make_invoice(v, dt2, lines=a["lines"])
            for r in (a, b):
                dd = self.day_index(r["issue_dt"])
                self.make_payment(r, self.rand_dt(dd + 2, min(dd + 7, nd - 1)), r["total"])
            self.add_issue(
                "duplicate_invoice", [a["id"], b["id"]], a["total"], a["cur"],
                [[a["doc_id"]], [b["doc_id"]], [b["exp_row"], b["ap_row"]]],
                flavor="duplicate_invoice_record")

    def inject_partial(self):
        nd = self.ndays
        v = self.rng.choice(self.vendors)
        if self.rng.random() < 0.65:  # partial payment, payable cleared with a plug
            d = self.rng.randint(1, nd - 8)
            r = self.make_invoice(v, self.rand_dt(d, d), cents=False)
            p = self.round_down(r["cur"], int(r["total"] * self.rng.uniform(0.4, 0.85)))
            ap_debit = r["ap_base"]
            cash = self.conv(p, r["cur"], r["rate"])
            pay = self.make_payment(r, self.rand_dt(d + 2, d + 6), p, ap_debit=ap_debit, cash=cash)
            self.add_issue(
                "unallocated_or_partial_payment", [r["id"], pay["id"]], r["total"] - p, r["cur"],
                [[r["doc_id"]], [pay["doc_id"]], [pay["ap_row"]], [pay["adj_row"]]],
                flavor="partial_plug")
        else:  # payment never applied to the invoice
            d = self.rng.randint(1, nd - 14)
            r = self.make_invoice(v, self.rand_dt(d, d), cents=False, due_days=7)
            pay = self.make_payment(r, self.rand_dt(d + 3, d + 6), r["total"], unapplied=True)
            self.add_issue(
                "unallocated_or_partial_payment", [r["id"], pay["id"]], r["total"], r["cur"],
                [[r["doc_id"]], [pay["doc_id"]], [pay["ap_row"], pay["cash_row"]]],
                flavor="unapplied_payment")

    def inject_cutoff(self):
        nd = self.ndays
        v = self.rng.choice(self.vendors)
        if self.rng.random() < 0.6:  # prior-period invoice booked in this period
            issue_dt = self.rand_dt(-self.rng.randint(1, 10), -1)
            month_name = calendar.month_name[issue_dt.month]
            desc = f"{month_name} {self.rng.choice(SERVICES)}"
            r = self.make_invoice(v, issue_dt, book_dt=self.rand_dt(0, 2), desc=desc, cents=False)
            flavor = "prior_period"
        else:  # next-period invoice booked early
            issue_dt = self.rand_dt(nd, nd + 4)
            r = self.make_invoice(v, issue_dt, book_dt=self.rand_dt(nd - 3, nd - 1), cents=False)
            flavor = "next_period"
        self.add_issue("period_cutoff_error", [r["id"]], r["total"], r["cur"],
                       [[r["doc_id"]], [r["exp_row"], r["ap_row"]]], flavor=flavor)

    def inject_fx(self):
        nd = self.ndays
        v = self.vendor(foreign=True)
        d = self.rng.randint(1, nd - 8)
        issue_dt = self.rand_dt(d, d)
        cur = v[1]
        correct = self.rate_for(cur, issue_dt)
        wrong = self.wrong_rate(cur, issue_dt)
        pay_dt = self.rand_dt(d + 2, d + 6)
        if self.rng.random() < 0.5:  # invoice booked at a stale rate
            r = self.make_invoice(v, issue_dt, book_rate=wrong, cents=False)
            pay = self.make_payment(r, pay_dt, r["total"], ap_debit=self.conv(r["total"], cur, correct))
            diff = abs(r["ap_base"] - self.conv(r["total"], cur, correct))
            groups = [[r["doc_id"]], [pay["doc_id"]], [r["exp_row"], r["ap_row"]],
                      [self.fx_id(cur, issue_dt)]]
            flavor = "invoice_booked_wrong_rate"
        else:  # payment booked at a stale rate
            r = self.make_invoice(v, issue_dt, cents=False)
            pay = self.make_payment(r, pay_dt, r["total"],
                                    ap_debit=self.conv(r["total"], cur, wrong))
            diff = abs(self.conv(r["total"], cur, wrong) - r["ap_base"])
            groups = [[r["doc_id"]], [pay["doc_id"]], [pay["ap_row"], pay["cash_row"]],
                      [self.fx_id(cur, issue_dt)]]
            flavor = "payment_booked_wrong_rate"
        self.add_issue("bad_fx_date_or_rate", [r["id"], pay["id"]], diff, self.base, groups,
                       flavor=flavor)

    # ---- interacting scenarios -------------------------------------------- #
    def inject_partial_fx(self):
        """Foreign invoice, partial payment at the period boundary booked at a stale rate."""
        nd = self.ndays
        v = self.vendor(foreign=True)
        cur = v[1]
        d = nd - self.rng.randint(4, 6)
        issue_dt = self.rand_dt(d, d)
        r = self.make_invoice(v, issue_dt, cents=False)
        p = self.round_down(cur, int(r["total"] * self.rng.uniform(0.5, 0.8)))
        stale = self.wrong_rate(cur, issue_dt, direction="lower")
        cash = self.conv(p, cur, stale)
        pay_dt = self.p_end - timedelta(minutes=self.rng.randint(30, 150))
        pay = self.make_payment(r, pay_dt, p, ap_debit=r["ap_base"], cash=cash)
        fx_diff = self.conv(p, cur, r["rate"]) - cash
        self.add_issue(
            "unallocated_or_partial_payment", [r["id"], pay["id"]], r["total"] - p, cur,
            [[r["doc_id"]], [pay["doc_id"]], [pay["ap_row"]], [pay["cash_row"]], [pay["adj_row"]],
             [self.fx_id(cur, issue_dt)]],
            secondary_types=["bad_fx_date_or_rate"], interacting=True,
            components={
                "shortfall_minor": r["total"] - p, "shortfall_currency": cur,
                "fx_difference_minor": fx_diff, "fx_difference_currency": self.base,
                "correct_rate": format(r["rate"], "f"), "booked_rate": format(stale, "f"),
            },
            flavor="partial_payment_with_stale_fx")

    def inject_duplicate_cutoff(self):
        """Prior-period invoice booked in this period *and* paid twice."""
        nd = self.ndays
        v = self.rng.choice(self.vendors)
        issue_dt = self.rand_dt(-self.rng.randint(2, 8), -2)
        desc = f"{calendar.month_name[issue_dt.month]} {self.rng.choice(SERVICES)}"
        r = self.make_invoice(v, issue_dt, book_dt=self.rand_dt(0, 1), desc=desc, cents=False)
        d1 = self.rng.randint(3, 8)
        d2 = min(d1 + self.rng.randint(3, 8), nd - 1)
        p1 = self.make_payment(r, self.rand_dt(d1, d1), r["total"])
        p2 = self.make_payment(r, self.rand_dt(d2, d2), r["total"])
        self.add_issue(
            "duplicate_invoice", [r["id"], p1["id"], p2["id"]], r["total"], r["cur"],
            [[r["doc_id"]], [p1["doc_id"]], [p2["doc_id"]], [p2["ap_row"], p2["cash_row"]],
             [r["exp_row"], r["ap_row"]]],
            secondary_types=["period_cutoff_error"], interacting=True,
            flavor="double_payment_of_cutoff_invoice")

    # ---- assembly --------------------------------------------------------- #
    def build(self):
        cfg = self.cfg
        self.build_normal(cfg["n_invoices"])
        for i in range(cfg["distractors"]):
            self.add_distractor(self.rng.choice(DISTRACTOR_KINDS))
        e = cfg["errors"]
        for _ in range(e["duplicate_invoice"]):
            self.inject_duplicate()
        for _ in range(e["unallocated_or_partial_payment"]):
            self.inject_partial()
        for _ in range(e["period_cutoff_error"]):
            self.inject_cutoff()
        for _ in range(e["bad_fx_date_or_rate"]):
            self.inject_fx()
        for _ in range(cfg["interacting"]["partial_fx"]):
            self.inject_partial_fx()
        for _ in range(cfg["interacting"]["duplicate_cutoff"]):
            self.inject_duplicate_cutoff()
        self.build_misc()
        return self._finalize()

    def _finalize(self):
        rng = self.rng
        txns = list(self.txns)
        mode = self.cfg["ordering"]
        if mode == "chronological":
            txns.sort(key=lambda t: (t["dt"], t["seq"]))
        elif mode == "by_entity":
            txns.sort(key=lambda t: (t["entity"] or "~", t["dt"], t["seq"]))
        else:
            rng.shuffle(txns)
        width = max(4, len(str(sum(len(t["rows"]) for t in txns))))
        n = 0
        ledger = []
        for t in txns:
            for row in t["rows"]:
                n += 1
                row["row_id"] = f"ld_{n:0{width}d}"
                ledger.append({k: row[k] for k in LEDGER_FIELDS})

        invoices = [r["doc"] for r in self.invoices]
        payments = [p["doc"] for p in self.payments]
        if mode == "shuffled":
            rng.shuffle(invoices)
            rng.shuffle(payments)
        else:
            invoices.sort(key=lambda d: d["doc_id"])
            payments.sort(key=lambda d: d["doc_id"])

        fx_rows = [
            {"date": d, "source_currency": c, "target_currency": self.base, "rate": format(r, "f")}
            for (c, d), r in sorted(self.fx.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        ]
        used_cur = {self.base} | {i["currency"] for i in invoices}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "base_currency": self.base,
            "period_start": self.ts(self.p_start),
            "period_end": self.ts(self.p_end),
            "timezone": self.tz_name,
            "currency_exponents": {c: CURRENCY_EXPONENTS[c] for c in sorted(used_cur)},
            "counts": {"ledger_rows": len(ledger), "documents": len(invoices) + len(payments)},
            "rounding": {"mode": "half_up", "unit": "minor"},
        }
        policy = {"fx_rate_date": "invoice_issue_date", "period_boundaries": "inclusive",
                  "rounding": "half_up"}

        def rid(x):
            return x["row_id"] if isinstance(x, dict) else x

        issues = []
        for i, iss in enumerate(self.issues, 1):
            sets = []
            for combo in itertools.product(*iss["groups"]):
                s = sorted({rid(x) for x in combo})
                if s not in sets:
                    sets.append(s)
            out = {"issue_id": f"iss_{i:02d}", "type": iss["type"]}
            ex = dict(iss["extra"])
            if "secondary_types" in ex:
                out["secondary_types"] = ex.pop("secondary_types")
            if ex.pop("interacting", False):
                out["interacting"] = True
            out["affected_ids"] = iss["affected"]
            out["expected_amount_minor"] = iss["amount"]
            out["currency"] = iss["currency"]
            if "components" in ex:
                out["components"] = ex.pop("components")
            out["flavor"] = ex.pop("flavor", None)
            out["valid_evidence_sets"] = sets
            issues.append(out)
        truth = {
            "episode_id": self.episode_id,
            "schema_version": SCHEMA_VERSION,
            "amount_tolerance_minor": self.cfg["amount_tolerance_minor"],
            "issues": issues,
            "clean_cases": sorted(self.clean_cases),
        }
        visible = {"manifest": manifest, "policy": policy, "ledger": ledger,
                   "invoices": invoices, "payments": payments, "fx": fx_rows}
        return visible, truth


def build_episode(seed: int, config=None, episode_id: str | None = None):
    """Deterministically build one company.  Returns ``(visible, truth)`` dicts."""
    return _Builder(seed, config, episode_id or f"gen_{seed}").build()


# --------------------------------------------------------------------------- #
# disk output
# --------------------------------------------------------------------------- #
def write_episode(visible: dict, truth: dict, data_dir: str, truth_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)

    def dump(name, obj):
        with open(os.path.join(data_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")

    dump("manifest.json", visible["manifest"])
    dump("policy.json", visible["policy"])
    with open(os.path.join(data_dir, "ledger.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LEDGER_FIELDS)
        for row in visible["ledger"]:
            w.writerow(["null" if row[k] is None else row[k] for k in LEDGER_FIELDS])
    with open(os.path.join(data_dir, "fx.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FX_FIELDS)
        for row in visible["fx"]:
            w.writerow([row[k] for k in FX_FIELDS])
    for name, key in (("invoices.jsonl", "invoices"), ("payments.jsonl", "payments")):
        with open(os.path.join(data_dir, name), "w", encoding="utf-8") as f:
            for row in visible[key]:
                f.write(json.dumps(row) + "\n")
    with open(os.path.join(truth_dir, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------- #
# the 20 fixed episodes (seeds are part of the submission: reproducible)
# --------------------------------------------------------------------------- #
TRAIN_SEED_BASE = 1000
DEV_SEED_BASE = 2000


def _p(dup=0, part=0, cut=0, fx=0, a=0, b=0, **kw):
    d = {"errors": {"duplicate_invoice": dup, "unallocated_or_partial_payment": part,
                    "period_cutoff_error": cut, "bad_fx_date_or_rate": fx},
         "interacting": {"partial_fx": a, "duplicate_cutoff": b}}
    d.update(kw)
    return d


TRAIN_PROFILES = [
    _p(1, 1, 1, 1, 1, 0, base_currency="KZT", timezone="Asia/Almaty", month=10),
    _p(clean=True, distractors=8, month=3),
    _p(dup=2, month=5),
    _p(part=2, fx=2, base_currency="USD", month=7),
    _p(cut=2, b=1, base_currency="EUR", timezone="Asia/Dubai"),
    _p(1, 1, 1, 1, 1, 1, n_invoices=34),
    _p(clean=True, distractors=6, base_currency="USD", timezone="Asia/Tokyo"),
    _p(cut=1, fx=3, month=2),
    _p(1, 1, 0, 0, 1, 1, timezone="Asia/Singapore"),
    _p(2, 1, 2, 1, 0, 0, month=12),
]
DEV_PROFILES = [
    _p(1, 1, 1, 1, 1, 1, base_currency="KZT", month=10, n_invoices=30),
    _p(clean=True, distractors=8, base_currency="EUR", month=6),
    _p(part=1, fx=1, timezone="UTC"),
    _p(dup=2, cut=1, b=1, month=11),
    _p(fx=2, a=1, base_currency="USD", month=1),
    _p(cut=3, month=8),
    _p(clean=True, distractors=7, timezone="Asia/Tashkent", n_invoices=34),
    _p(1, 2, 0, 1, 0, 0, base_currency="EUR", month=4),
    _p(a=2, b=0, month=9),
    _p(1, 1, 1, 1, 0, 0, timezone="Asia/Dubai", month=5),
]


def episode_specs():
    for i, prof in enumerate(TRAIN_PROFILES, 1):
        yield f"train_{i:02d}", TRAIN_SEED_BASE + i, prof
    for i, prof in enumerate(DEV_PROFILES, 1):
        yield f"dev_{i:02d}", DEV_SEED_BASE + i, prof


def generate_all(data_root: str = "data", truth_root: str = "truth") -> list[str]:
    names = []
    for name, seed, prof in episode_specs():
        for root in (data_root, truth_root):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        visible, truth = build_episode(seed, prof, episode_id=name)
        write_episode(visible, truth, os.path.join(data_root, name), os.path.join(truth_root, name))
        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--truth", default="truth")
    args = ap.parse_args()
    names = generate_all(args.out, args.truth)
    print(f"Generated {len(names)} episodes into {args.out}/ (answer keys in {args.truth}/).")


if __name__ == "__main__":
    main()
