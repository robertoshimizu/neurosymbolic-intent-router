"""Tests for the action router and wire-transfer ledger."""

from __future__ import annotations

import os
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from router import (
    ACTION_CATALOG,
    Decision,
    IntentRank,
    decide,
    route,
)
from jev import JevClassifier
from minilm import MiniLMExplainer
from policy import Session
from transfer import Ledger, WireTransfer, confirm_or_refuse


def _orthonormal_catalog() -> dict[str, np.ndarray]:
    """Injected vectors: wire is nearest to the query, FAQ is distant."""
    return {
        "wire_transfer_funds": np.array([1.0, 0.0, 0.0, 0.0]),
        "view_account_balance": np.array([0.0, 1.0, 0.0, 0.0]),
        "delete_account": np.array([0.0, 0.0, 1.0, 0.0]),
        "view_public_faq": np.array([0.0, 0.0, 0.0, 1.0]),
    }


WIRE_UTTERANCE = "I want to send $5,000 to my external bank account."


def _session(
    session_id: str,
    *,
    is_authenticated: bool = True,
    role: str = "customer",
    status: str = "active",
    account_id: str = "acct",
    payee_allowlist: tuple[str, ...] = ("external bank account",),
) -> Session:
    return Session(
        session_id=session_id,
        is_authenticated=is_authenticated,
        role=role,
        status=status,
        account_id=account_id,
        payee_allowlist=payee_allowlist,
    )


def _ledger(balance: Decimal, account_id: str = "acct") -> Ledger:
    ledger = Ledger()
    ledger.set_balance(account_id, balance)
    return ledger


def _decide_wire(session: Session, ledger: Ledger) -> Decision:
    return decide(
        WIRE_UTTERANCE,
        session,
        ledger,
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.9)),
    )


def test_insufficient_funds_suggests_balance_view() -> None:
    session = _session("zero")
    decision = _decide_wire(session, _ledger(Decimal("0")))
    assert decision.outcome == "deny"
    assert "insufficient funds" in decision.reason
    assert decision.suggestion == "view_account_balance"


def test_confirm_then_settle_debits_once() -> None:
    ledger = _ledger(Decimal("10000"))
    transfer = WireTransfer(
        ledger=ledger,
        account_id="acct",
        amount=Decimal("5000"),
        transfer_id="wire-1",
    )
    transfer.send("request_confirmation")
    assert confirm_or_refuse(transfer) is True
    transfer.send("submit")
    transfer.send("settle")
    assert ledger.get_balance("acct") == Decimal("5000")
    assert ledger.has_settled("wire-1")

    # Second debit with the same transfer id must not move money again.
    moved = ledger.debit("acct", Decimal("5000"), "wire-1")
    assert moved is False
    assert ledger.get_balance("acct") == Decimal("5000")


def test_confirm_refused_when_balance_drained() -> None:
    ledger = _ledger(Decimal("10000"))
    transfer = WireTransfer(
        ledger=ledger,
        account_id="acct",
        amount=Decimal("5000"),
        transfer_id="wire-2",
    )
    transfer.send("request_confirmation")
    ledger.set_balance("acct", Decimal("0"))
    assert confirm_or_refuse(transfer) is False
    assert transfer.awaiting_confirmation.is_active
    assert ledger.get_balance("acct") == Decimal("0")


def _never(*_args) -> None:
    raise AssertionError("must not run")


def _jev_rank(action: str, confidence: float) -> IntentRank:
    scores = {name: 0.05 for name in ACTION_CATALOG}
    scores["none"] = 0.05
    scores[action] = confidence
    return IntentRank(
        action=action,
        scores=scores,
        confidence=confidence,
        source="jev",
    )


def _classifier(rank: IntentRank | None, labels=None) -> SimpleNamespace:
    return SimpleNamespace(classify=lambda _text: rank, label=labels or _never)


def test_jev_none_stops_before_policy() -> None:
    decision = decide(
        "What is the capital of Portugal?",
        _session("funded"),
        _ledger(Decimal("10000")),
        classifier=_classifier(_jev_rank("none", 0.91)),
    )
    assert decision.outcome == "deny"
    assert decision.reason == "no matching action"
    assert decision.action == "none"
    assert decision.permissions == {}
    assert decision.source == "jev"


def test_jev_wire_still_needs_confirmation() -> None:
    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.86)),
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "needs_confirmation"
    assert decision.source == "jev"
    assert decision.description_match is None


def test_minilm_disagreement_does_not_override_jev() -> None:
    faq_query = np.array([0.02, 0.05, 0.10, 0.98])
    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.86)),
        explainer=MiniLMExplainer(SimpleNamespace(
            embed=lambda _text: faq_query, action_embeddings=_orthonormal_catalog)),
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "needs_confirmation"
    match = decision.description_match
    assert match is not None
    assert not match.agrees
    assert match.nearest == "view_public_faq"
    assert "description_gap=nearest_differs" in decision.rule_trace


def test_classifier_failure_denies_without_guessing() -> None:
    def _boom(_utterance: str) -> IntentRank:
        raise RuntimeError("classifier down")

    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        classifier=SimpleNamespace(classify=_boom, label=_never),
    )
    assert decision.action == "none"
    assert decision.outcome == "deny"
    assert decision.reason == "classifier unavailable"


@pytest.mark.integration
def test_live_jev_abstains_on_unrelated_sentence() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY is not set")
    decision = decide(
        "What is the capital of Portugal?",
        _session("funded"),
        _ledger(Decimal("10000")),
        classifier=JevClassifier(),
    )
    assert decision.source == "jev"
    assert decision.reason == "no matching action"
    assert decision.outcome == "deny"


CLOSE_AND_WIRE = "Close this account and send $500 to my external bank account."
CLOSE = "Close this account."
WIRE_500 = "Send $500 to my external bank account."
BALANCE = "Show my balance."
LABELS = {CLOSE: "delete_account", WIRE_500: "wire_transfer_funds", BALANCE: "view_account_balance"}


def _labels(requests: tuple[str, ...]) -> list[IntentRank]:
    return [_jev_rank(LABELS.get(text, "none"), 0.95) for text in requests]


def _route(utterance: str, session: Session, ledger: Ledger, *, count: str | None, split, labeler=_labels, action: str = "none") -> Decision:
    rank = replace(_jev_rank(action, 0.9), request_count=count)
    return route(utterance, session, ledger, splitter=SimpleNamespace(split=split), classifier=_classifier(rank, labeler))


def test_route_orders_money_movement_before_deletion() -> None:
    ledger = _ledger(Decimal("10000"))
    decision = _route(CLOSE_AND_WIRE, _session("admin", role="admin"), ledger, count="several", split=lambda _u: (CLOSE, WIRE_500))
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "needs_confirmation"
    assert decision.follow_ups == (CLOSE,)
    assert ledger.get_balance("acct") == Decimal("10000")


def test_route_denied_first_request_offers_no_follow_ups() -> None:
    decision = _route(CLOSE_AND_WIRE, _session("admin", role="admin"), _ledger(Decimal("0")), count="several", split=lambda _u: (CLOSE, WIRE_500))
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "deny"
    assert decision.follow_ups == ()


def test_route_one_request_uses_the_single_jev_rank() -> None:
    decision = _route(
        WIRE_UTTERANCE, _session("funded"), _ledger(Decimal("10000")),
        count="one", split=_never, labeler=_never, action="wire_transfer_funds",
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "needs_confirmation"
    assert decision.follow_ups == ()


def test_route_failed_split_denies_without_guessing() -> None:
    def _down(_utterance: str) -> tuple[str, ...]:
        raise RuntimeError("Ollama request failed")

    decision = _route(CLOSE_AND_WIRE, _session("funded"), _ledger(Decimal("10000")), count="several", split=_down)
    assert decision.reason == "several requests could not be separated"


def test_route_failed_labeling_denies_without_guessing() -> None:
    decision = _route(
        CLOSE_AND_WIRE, _session("funded"), _ledger(Decimal("10000")),
        count="several", split=lambda _u: (CLOSE, WIRE_500), labeler=lambda _r: None,
    )
    assert decision.reason == "several requests could not be labeled"


def test_route_none_label_is_ordered_last() -> None:
    def _unsure_balance(requests: tuple[str, ...]) -> list[IntentRank]:
        return [_jev_rank("none" if text == BALANCE else LABELS[text], 0.95) for text in requests]

    decision = _route(
        "Show my balance and send $500 to my external bank account.", _session("funded"), _ledger(Decimal("10000")),
        count="several", split=lambda _u: (BALANCE, WIRE_500), labeler=_unsure_balance,
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.follow_ups == (BALANCE,)
