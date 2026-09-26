"""Policy rules on an already-chosen action. No model is involved."""

from __future__ import annotations

from decimal import Decimal

import pytest

from policy import ParsedRequest, Session, evaluate_action, to_dollars, to_request



def _session(*, is_authenticated: bool = True, role: str = "customer", status: str = "active") -> Session:
    return Session(
        session_id="s",
        is_authenticated=is_authenticated,
        role=role,
        status=status,
        account_id="a",
        payee_allowlist=(),
    )


def test_unknown_payee_does_not_bind() -> None:
    parsed = to_request(("$5,000",), ("external bank account",), ("savings vault",))
    assert parsed.amount == Decimal("5000")
    assert parsed.payee is None


def test_unauthenticated_wire_denied() -> None:
    session = _session(is_authenticated=False, role="unauthenticated", status="inactive")
    allowed, reason = evaluate_action(
        "wire_transfer_funds", session, Decimal("0"), ParsedRequest(amount=Decimal("5000"), payee=None)
    )
    assert allowed is False
    assert "not authenticated" in reason


def test_delete_denied_for_customer() -> None:
    allowed, reason = evaluate_action(
        "delete_account", _session(), Decimal("100"), ParsedRequest(amount=None, payee=None)
    )
    assert allowed is False
    assert "admin" in reason


@pytest.mark.parametrize(
    ("span", "dollars"),
    [
        ("$500", "500"),
        ("$1,500", "1500"),
        ("$ 750.25", "750.25"),
        ("500 USD", "500"),
        ("$1,500 dollars", "1500"),
        ("five hundred dollars", "500"),
        ("one thousand five hundred", "1500"),
        ("one thousand, five hundred dollars", "1500"),
        ("fifteen hundred bucks", "1500"),
        ("twelve hundred and fifty", "1250"),
        ("$0", "0"),
    ],
)
def test_to_dollars_reads_plain_figures_and_number_words(span: str, dollars: str) -> None:
    assert to_dollars(span) == Decimal(dollars)


@pytest.mark.parametrize(
    "span",
    ["$1.5k", "$1 thousand", "a thousand", "€200", "500", "$1,50", "$1.999", "-$500", "half of my balance", "money", ""],
)
def test_to_dollars_raises_instead_of_guessing(span: str) -> None:
    with pytest.raises(ValueError):
        to_dollars(span)


ALLOWLIST = ("external bank account", "savings account")


def test_to_request_reads_one_amount_and_one_allowlisted_payee() -> None:
    parsed = to_request(("one thousand five hundred dollars",), ("External Bank Account",), ALLOWLIST)
    assert parsed.amount == Decimal("1500")
    assert parsed.payee == "external bank account"


@pytest.mark.parametrize(
    "amounts",
    [(), ("$50", "$500"), ("$1.5k",), ("half",)],
    ids=["none", "two", "shorthand", "unreadable"],
)
def test_to_request_leaves_the_amount_empty_unless_one_is_readable(amounts: tuple[str, ...]) -> None:
    assert to_request(amounts, ("external bank account",), ALLOWLIST).amount is None


@pytest.mark.parametrize(
    "payees",
    [(), ("savings account", "external bank account"), ("John Smith",), ("external account",)],
    ids=["none", "source-and-destination", "not-allowlisted", "paraphrase"],
)
def test_to_request_leaves_the_payee_empty_unless_one_is_allowlisted(payees: tuple[str, ...]) -> None:
    assert to_request(("$500",), payees, ALLOWLIST).payee is None


def test_unreadable_amount_is_denied_with_a_reason() -> None:
    session = _session()
    allowed, reason = evaluate_action(
        "wire_transfer_funds", session, Decimal("10000"), to_request(("$1.5k",), ("external bank account",), ALLOWLIST)
    )
    assert allowed is False
    assert "amount is missing or invalid" in reason
