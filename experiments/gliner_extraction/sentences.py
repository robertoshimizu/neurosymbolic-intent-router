"""Transfer sentences for the GLiNER extraction experiment, reviewed by a human on 2026-09-26.

Expected spans are copied verbatim from the sentence. `value` is the dollar amount the rules
should end up with for the first request: None when the sentence gives no dollar figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

ALLOWLIST = ("external bank account", "savings account")


@dataclass(frozen=True)
class Case:
    text: str
    amounts: tuple[str, ...]
    payees: tuple[str, ...]
    value: Decimal | None
    probe: str
    sources: tuple[str, ...] = ()


CASES: tuple[Case, ...] = (
    Case("Send $500 to my external bank account.", ("$500",), ("external bank account",), Decimal("500"), "control"),
    Case(
        "Wire five hundred dollars to my external bank account.",
        ("five hundred dollars",),
        ("external bank account",),
        Decimal("500"),
        "amount in words",
    ),
    Case("Transfer $1.5k to my savings account.", ("$1.5k",), ("savings account",), Decimal("1500"), "wrong value"),
    Case("Send 500 USD to my external bank account.", ("500 USD",), ("external bank account",), Decimal("500"), "currency code"),
    Case("Send €200 to my external bank account.", ("€200",), ("external bank account",), None, "foreign currency"),
    Case(
        "Move $2,000 from my savings account to my external bank account.",
        ("$2,000",),
        ("external bank account",),
        Decimal("2000"),
        "source vs destination",
        sources=("savings account",),
    ),
    Case(
        "Send $300 to my external bank account and $200 to my savings account.",
        ("$300", "$200"),
        ("external bank account", "savings account"),
        Decimal("300"),
        "two pairs",
    ),
    Case(
        "I paid $50 yesterday; now send $500 to my external bank account.",
        ("$500",),
        ("external bank account",),
        Decimal("500"),
        "background amount",
    ),
    Case("Send $500 to my external account.", ("$500",), ("external account",), Decimal("500"), "payee paraphrase"),
    Case("Send $500 to John Smith.", ("$500",), ("John Smith",), Decimal("500"), "payee not on allowlist"),
    Case("Send $500 to account 12345678.", ("$500",), ("account 12345678",), Decimal("500"), "account number"),
    Case("Send money to my external bank account.", (), ("external bank account",), None, "no amount"),
    Case(
        "Send half of my balance to my external bank account.",
        ("half of my balance",),
        ("external bank account",),
        None,
        "relative amount",
    ),
    Case("Don't send $500 to my external bank account.", ("$500",), ("external bank account",), Decimal("500"), "prohibition"),
    Case("Send my external bank account $750.25.", ("$750.25",), ("external bank account",), Decimal("750.25"), "word order"),
)
