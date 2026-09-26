"""Tests for the request workflow: routing (models interpret, rules decide) is one state; executing re-checks facts."""

from __future__ import annotations

from decimal import Decimal

from contracts import Classifier, IntentRank
from policy import Session
from transfer import Ledger
from workflow import RequestWorkflow

WIRE_500 = "Send $500 to my external bank account."
CLOSE = "Close this account."


class _FixedClassifier(Classifier):
    def __init__(self, action: str | None) -> None:
        self._action = action

    def classify(self, text: str) -> IntentRank | None:
        if self._action is None:
            return None
        return IntentRank(action=self._action, scores={}, confidence=0.9, source="fake", request_count="one")


class _DrainAfterRouting:
    """Listener: the balance drops after routing has decided, before execution starts."""

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def on_exit_routing(self) -> None:
        self._ledger.set_balance("acct", Decimal("100"))


def _session(role: str = "customer", *, signed_in: bool = True) -> Session:
    return Session(
        session_id="s",
        is_authenticated=signed_in,
        role=role,
        status="active",
        account_id="acct",
        payee_allowlist=("external bank account",),
    )


def _ledger(balance: str) -> Ledger:
    ledger = Ledger()
    ledger.set_balance("acct", Decimal(balance))
    return ledger


def _run(
    text: str, action: str | None, session: Session, ledger: Ledger,
    request_id: str = "r1", listener: object | None = None,
) -> RequestWorkflow:
    workflow = RequestWorkflow(text, session, ledger, request_id, classifier=_FixedClassifier(action))
    if listener is not None:
        workflow.add_listener(listener)
    workflow.send("start")
    return workflow


def test_denied_request_is_refused_and_never_executes() -> None:
    ledger = _ledger("10000")
    workflow = _run(WIRE_500, "wire_transfer_funds", _session(signed_in=False), ledger)
    assert workflow.path == ["received", "routing", "refused"]
    assert ledger.get_balance("acct") == Decimal("10000")


def test_unavailable_classifier_is_refused_in_routing() -> None:
    workflow = _run(WIRE_500, None, _session(), _ledger("10000"))
    assert workflow.path == ["received", "routing", "refused"]
    assert workflow.note == "classifier unavailable"


def test_approved_wire_completes_with_one_debit() -> None:
    ledger = _ledger("10000")
    workflow = _run(WIRE_500, "wire_transfer_funds", _session(), ledger)
    assert workflow.path == ["received", "routing", "executing", "completed"]
    assert ledger.get_balance("acct") == Decimal("9500")
    assert ledger.has_settled("r1")


def test_wire_fails_when_funds_vanish_after_routing() -> None:
    ledger = _ledger("10000")
    workflow = _run(WIRE_500, "wire_transfer_funds", _session(), ledger, listener=_DrainAfterRouting(ledger))
    assert workflow.path == ["received", "routing", "executing", "failed"]
    assert ledger.get_balance("acct") == Decimal("100")


def test_closed_account_blocks_a_later_wire() -> None:
    ledger = _ledger("10000")
    closing = _run(CLOSE, "delete_account", _session("admin"), ledger, "r1")
    assert closing.completed.is_active
    assert ledger.is_closed("acct")

    wire = _run(WIRE_500, "wire_transfer_funds", _session(), ledger, "r2")
    assert wire.failed.is_active
    assert ledger.get_balance("acct") == Decimal("10000")


def test_closing_twice_fails_the_second_time() -> None:
    ledger = _ledger("0")
    _run(CLOSE, "delete_account", _session("admin"), ledger, "r1")
    again = _run(CLOSE, "delete_account", _session("admin"), ledger, "r2")
    assert again.failed.is_active
