"""Policy rules on an already-chosen action. No model is involved."""

from __future__ import annotations

from decimal import Decimal

from policy import Session, evaluate_action, parse_request

WIRE_UTTERANCE = "I want to send $5,000 to my external bank account."


def _session(*, is_authenticated: bool = True, role: str = "customer", status: str = "active") -> Session:
    return Session(
        session_id="s",
        is_authenticated=is_authenticated,
        role=role,
        status=status,
        account_id="a",
        payee_allowlist=(),
    )


def test_parse_amount_with_comma_and_dollar() -> None:
    parsed = parse_request(WIRE_UTTERANCE, ("external bank account",))
    assert parsed.amount == Decimal("5000")
    assert parsed.payee == "external bank account"


def test_unknown_payee_does_not_bind() -> None:
    parsed = parse_request(WIRE_UTTERANCE, ("savings vault",))
    assert parsed.amount == Decimal("5000")
    assert parsed.payee is None


def test_guest_wire_denied() -> None:
    session = _session(is_authenticated=False, role="guest", status="inactive")
    allowed, reason = evaluate_action(
        "wire_transfer_funds", session, Decimal("0"), parse_request("send $5,000", ())
    )
    assert allowed is False
    assert "not signed in" in reason


def test_delete_denied_for_customer() -> None:
    allowed, reason = evaluate_action(
        "delete_account", _session(), Decimal("100"), parse_request("delete my account", ())
    )
    assert allowed is False
    assert "admin" in reason
