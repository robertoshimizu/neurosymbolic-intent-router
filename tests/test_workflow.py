"""Tests for the request workflow: the router decides, the state machine executes."""

from __future__ import annotations

from decimal import Decimal

from policy import Outcome, Session, parse_request
from router import Decision
from transfer import Ledger
from workflow import RequestWorkflow

WIRE_500 = "Send $500 to my external bank account."


def _session(role: str = "customer") -> Session:
    return Session(
        session_id="s",
        is_authenticated=True,
        role=role,
        status="active",
        account_id="acct",
        payee_allowlist=("external bank account",),
    )


def _ledger(balance: str) -> Ledger:
    ledger = Ledger()
    ledger.set_balance("acct", Decimal(balance))
    return ledger


def _decision(action: str, outcome: Outcome, text: str = WIRE_500) -> Decision:
    return Decision(
        action=action,
        outcome=outcome,
        reason="decided by policy",
        parsed=parse_request(text, ("external bank account",)),
        scores={},
        permissions={},
    )


def _run(decision: Decision, session: Session, ledger: Ledger, request_id: str = "r1") -> RequestWorkflow:
    workflow = RequestWorkflow(decision, session, ledger, request_id)
    workflow.send("decided")
    return workflow


def test_denied_request_is_refused_and_never_executes() -> None:
    ledger = _ledger("10000")
    workflow = _run(_decision("wire_transfer_funds", "deny"), _session(), ledger)
    assert workflow.refused.is_active
    assert ledger.get_balance("acct") == Decimal("10000")


def test_approved_wire_completes_with_one_debit() -> None:
    ledger = _ledger("10000")
    workflow = _run(_decision("wire_transfer_funds", "execute"), _session(), ledger)
    assert workflow.completed.is_active
    assert ledger.get_balance("acct") == Decimal("9500")
    assert ledger.has_settled("r1")


def test_wire_fails_when_funds_vanish_after_the_decision() -> None:
    ledger = _ledger("10000")
    decision = _decision("wire_transfer_funds", "execute")
    ledger.set_balance("acct", Decimal("100"))
    workflow = _run(decision, _session(), ledger)
    assert workflow.failed.is_active
    assert ledger.get_balance("acct") == Decimal("100")


def test_closed_account_blocks_a_later_wire() -> None:
    ledger = _ledger("10000")
    closing = _run(_decision("delete_account", "execute", "Close this account."), _session("admin"), ledger, "r1")
    assert closing.completed.is_active
    assert ledger.is_closed("acct")

    wire = _run(_decision("wire_transfer_funds", "execute"), _session(), ledger, "r2")
    assert wire.failed.is_active
    assert ledger.get_balance("acct") == Decimal("10000")


def test_closing_twice_fails_the_second_time() -> None:
    ledger = _ledger("0")
    session = _session("admin")
    _run(_decision("delete_account", "execute", "Close this account."), session, ledger, "r1")
    again = _run(_decision("delete_account", "execute", "Close this account."), session, ledger, "r2")
    assert again.failed.is_active
