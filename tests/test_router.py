"""Tests for the action router and wire-transfer ledger."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from typing import NoReturn

import numpy as np
import pytest

from contracts import (
    Classifier,
    DescriptionMatch,
    Explainer,
    Extraction,
    Extractor,
    IntentRank,
    Labeler,
    RequestCount,
    Splitter,
)
from adapters.jev import JevClassifier
from adapters.minilm import ActionEmbedder, MiniLMExplainer
from policy import ACTION_CATALOG
from adapters.python_policy import PythonPolicy
from router import Decision, decide, route
from policy import Session
from transfer import Ledger, WireTransfer, authorize_or_refuse


def _orthonormal_catalog() -> dict[str, np.ndarray]:
    """Injected vectors: wire is nearest to the query, FAQ is distant."""
    return {
        "wire_transfer_funds": np.array([1.0, 0.0, 0.0, 0.0]),
        "view_account_balance": np.array([0.0, 1.0, 0.0, 0.0]),
        "delete_account": np.array([0.0, 0.0, 1.0, 0.0]),
        "view_public_faq": np.array([0.0, 0.0, 0.0, 1.0]),
    }


WIRE_UTTERANCE = "I want to send $5,000 to my external bank account."
POLICY = PythonPolicy()


class _SpanExtractor(Extractor):
    """Fake extractor: returns the known spans that occur in the text, in no particular order."""

    AMOUNTS = ("$5,000", "$500")
    PAYEES = ("external bank account",)

    def extract(self, text: str) -> Extraction | None:
        return Extraction(
            text=text,
            amounts=tuple(span for span in self.AMOUNTS if span in text),
            payees=tuple(span for span in self.PAYEES if span in text),
            source="fake",
        )


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
        policy=POLICY,
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.9)),
        extractor=_SpanExtractor(),
    )


def test_insufficient_funds_suggests_balance_view() -> None:
    session = _session("zero")
    decision = _decide_wire(session, _ledger(Decimal("0")))
    assert decision.outcome == "deny"
    assert "insufficient funds" in decision.reason
    assert decision.suggestion == "view_account_balance"


def test_no_suggestion_the_policy_would_deny() -> None:
    session = _session("unauthenticated", is_authenticated=False)
    decision = _decide_wire(session, _ledger(Decimal("0")))
    assert decision.outcome == "deny"
    assert "insufficient funds" in decision.reason
    assert decision.suggestion is None


def test_authorize_then_settle_debits_once() -> None:
    ledger = _ledger(Decimal("10000"))
    transfer = WireTransfer(
        ledger=ledger,
        account_id="acct",
        amount=Decimal("5000"),
        transfer_id="wire-1",
    )
    assert authorize_or_refuse(transfer) is True
    transfer.send("submit")
    transfer.send("settle")
    assert ledger.get_balance("acct") == Decimal("5000")
    assert ledger.has_settled("wire-1")

    # Second debit with the same transfer id must not move money again.
    moved = ledger.debit("acct", Decimal("5000"), "wire-1")
    assert moved is False
    assert ledger.get_balance("acct") == Decimal("5000")


def test_authorize_refused_when_balance_drained() -> None:
    ledger = _ledger(Decimal("10000"))
    transfer = WireTransfer(
        ledger=ledger,
        account_id="acct",
        amount=Decimal("5000"),
        transfer_id="wire-2",
    )
    ledger.set_balance("acct", Decimal("0"))
    assert authorize_or_refuse(transfer) is False
    assert transfer.drafted.is_active
    assert ledger.get_balance("acct") == Decimal("0")


def _never(*_args: object) -> NoReturn:
    raise AssertionError("must not run")


class _FakeClassifier(Classifier):
    def __init__(self, classify: Callable[[str], IntentRank | None]) -> None:
        self._classify = classify

    def classify(self, text: str) -> IntentRank | None:
        return self._classify(text)


class _FakeLabeler(Labeler):
    def __init__(self, label: Callable[[tuple[str, ...]], list[IntentRank] | None]) -> None:
        self._label = label

    def label(self, texts: tuple[str, ...]) -> list[IntentRank] | None:
        return self._label(texts)


class _FakeSplitter(Splitter):
    def __init__(self, split: Callable[[str], tuple[str, ...]]) -> None:
        self._split = split

    def split(self, text: str) -> tuple[str, ...]:
        return self._split(text)


class _FixedEmbedder(ActionEmbedder):
    """Fixed vectors, so the real MiniLMExplainer runs without the model."""

    def __init__(self, query: np.ndarray, actions: dict[str, np.ndarray]) -> None:
        super().__init__()
        self._query = query
        self._actions = actions

    def embed(self, text: str) -> np.ndarray:
        return self._query

    def action_embeddings(self, catalog: dict[str, str] | None = None) -> dict[str, np.ndarray]:
        return self._actions


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


def _classifier(rank: IntentRank | None) -> _FakeClassifier:
    return _FakeClassifier(lambda _text: rank)


def test_jev_none_stops_before_policy() -> None:
    decision = decide(
        "What is the capital of Portugal?",
        _session("funded"),
        _ledger(Decimal("10000")),
        policy=POLICY,
        classifier=_classifier(_jev_rank("none", 0.91)),
        extractor=_SpanExtractor(),
    )
    assert decision.outcome == "deny"
    assert decision.reason == "no matching action"
    assert decision.action == "none"
    assert decision.permissions == {}
    assert decision.source == "jev"


def test_jev_wire_is_judged_by_policy() -> None:
    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        policy=POLICY,
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.86)),
        extractor=_SpanExtractor(),
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "execute"
    assert decision.source == "jev"
    assert decision.description_match is None


def test_minilm_disagreement_does_not_override_jev() -> None:
    faq_query = np.array([0.02, 0.05, 0.10, 0.98])
    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        policy=POLICY,
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.86)),
        explainer=MiniLMExplainer(_FixedEmbedder(faq_query, _orthonormal_catalog())),
        extractor=_SpanExtractor(),
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "execute"
    match = decision.description_match
    assert match is not None
    assert not match.agrees
    assert match.nearest == "view_public_faq"
    assert "description_gap=nearest_differs" in decision.rule_trace


def test_explainer_failure_keeps_the_decision() -> None:
    class _BrokenExplainer(Explainer):
        def explain(self, text: str, action: str) -> DescriptionMatch | None:
            raise RuntimeError("model failed to load")

    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        policy=POLICY,
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.86)),
        explainer=_BrokenExplainer(),
        extractor=_SpanExtractor(),
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "execute"
    assert decision.description_match is None
    assert "explainer=unavailable" in decision.rule_trace


def test_classifier_failure_denies_without_guessing() -> None:
    def _boom(_utterance: str) -> IntentRank:
        raise RuntimeError("classifier down")

    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        policy=POLICY,
        classifier=_FakeClassifier(_boom),
        extractor=_SpanExtractor(),
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
        policy=POLICY,
        classifier=JevClassifier(),
        extractor=_SpanExtractor(),
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


def _route(
    utterance: str, session: Session, ledger: Ledger, *,
    count: RequestCount | None,
    split: Callable[[str], tuple[str, ...]],
    labeler: Callable[[tuple[str, ...]], list[IntentRank] | None] = _labels,
    action: str = "none",
) -> Decision:
    rank = replace(_jev_rank(action, 0.9), request_count=count)
    return route(
        utterance, session, ledger, policy=POLICY,
        classifier=_classifier(rank), labeler=_FakeLabeler(labeler), splitter=_FakeSplitter(split),
        extractor=_SpanExtractor(),
    )


def test_route_orders_money_movement_before_deletion() -> None:
    ledger = _ledger(Decimal("10000"))
    decision = _route(CLOSE_AND_WIRE, _session("admin", role="admin"), ledger, count="several", split=lambda _u: (CLOSE, WIRE_500))
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "execute"
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
    assert decision.outcome == "execute"
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


class _RepeatingOrderPolicy(PythonPolicy):
    """A reasoner whose order repeats one request and drops the other."""

    def order(self, actions: tuple[str, ...]) -> tuple[int, ...]:
        return (0,) * len(actions)


def test_route_invalid_order_denies_without_guessing() -> None:
    rank = replace(_jev_rank("none", 0.9), request_count="several")
    decision = route(
        CLOSE_AND_WIRE, _session("funded"), _ledger(Decimal("10000")), policy=_RepeatingOrderPolicy(),
        classifier=_classifier(rank), labeler=_FakeLabeler(_labels), splitter=_FakeSplitter(lambda _u: (CLOSE, WIRE_500)),
    )
    assert decision.outcome == "deny"
    assert decision.reason == "several requests could not be ordered"


def test_route_none_label_is_ordered_last() -> None:
    def _unsure_balance(requests: tuple[str, ...]) -> list[IntentRank]:
        return [_jev_rank("none" if text == BALANCE else LABELS[text], 0.95) for text in requests]

    decision = _route(
        "Show my balance and send $500 to my external bank account.", _session("funded"), _ledger(Decimal("10000")),
        count="several", split=lambda _u: (BALANCE, WIRE_500), labeler=_unsure_balance,
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.follow_ups == (BALANCE,)


class _BrokenExtractor(Extractor):
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def extract(self, text: str) -> Extraction | None:
        if self.mode == "raises":
            raise RuntimeError("model failed to load")
        if self.mode == "other-text":
            return Extraction(text="Send $500.", amounts=("$500",), payees=(), source="fake")
        return None


@pytest.mark.parametrize("mode", ["unavailable", "raises", "other-text"])
def test_wire_is_denied_when_nothing_can_be_read(mode: str) -> None:
    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        policy=POLICY,
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.9)),
        extractor=_BrokenExtractor(mode),
    )
    assert decision.outcome == "deny"
    assert "amount is missing or invalid" in decision.reason
    assert "extractor=unavailable" in decision.rule_trace


def test_wire_with_two_amounts_is_denied() -> None:
    decision = decide(
        "I paid $500 yesterday; now send $5,000 to my external bank account.",
        _session("funded"),
        _ledger(Decimal("10000")),
        policy=POLICY,
        classifier=_classifier(_jev_rank("wire_transfer_funds", 0.9)),
        extractor=_SpanExtractor(),
    )
    assert decision.outcome == "deny"
    assert "amount is missing or invalid" in decision.reason
