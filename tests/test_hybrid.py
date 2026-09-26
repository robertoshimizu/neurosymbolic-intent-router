"""Tests for the hybrid action router and wire-transfer ledger."""

from __future__ import annotations

import os
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from hybrid import (
    ACTION_CATALOG,
    ActionEmbedder,
    Decision,
    IntentRank,
    Session,
    decide,
    evaluate_action,
    parse_request,
    route,
)
from transfer import Ledger, WireTransfer, confirm_or_refuse


def _orthonormal_catalog() -> dict[str, np.ndarray]:
    """Injected vectors: wire is nearest to the query, FAQ is distant."""
    return {
        "wire_transfer_funds": np.array([1.0, 0.0, 0.0, 0.0]),
        "view_account_balance": np.array([0.0, 1.0, 0.0, 0.0]),
        "delete_account": np.array([0.0, 0.0, 1.0, 0.0]),
        "view_public_faq": np.array([0.0, 0.0, 0.0, 1.0]),
    }


WIRE_QUERY = np.array([0.98, 0.10, 0.05, 0.02])
AMBIGUOUS_QUERY = np.array([0.70, 0.69, 0.0, 0.0])
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


def _decide_wire(
    session: Session,
    ledger: Ledger,
    *,
    query_vector: np.ndarray = WIRE_QUERY,
    **kwargs,
) -> Decision:
    return decide(
        WIRE_UTTERANCE,
        session,
        ledger,
        query_vector=query_vector,
        action_embeddings=_orthonormal_catalog(),
        **kwargs,
    )


def test_parse_amount_with_comma_and_dollar() -> None:
    parsed = parse_request(
        WIRE_UTTERANCE,
        ("external bank account",),
    )
    assert parsed.amount == Decimal("5000")
    assert parsed.payee == "external bank account"


def test_unknown_payee_does_not_bind() -> None:
    parsed = parse_request(
        WIRE_UTTERANCE,
        ("savings vault",),
    )
    assert parsed.amount == Decimal("5000")
    assert parsed.payee is None


def test_guest_wire_denied() -> None:
    session = _session(
        "guest",
        is_authenticated=False,
        role="guest",
        status="inactive",
        account_id="a",
        payee_allowlist=(),
    )
    allowed, reason = evaluate_action(
        "wire_transfer_funds",
        session,
        Decimal("0"),
        parse_request("send $5,000", ()),
    )
    assert allowed is False
    assert "not signed in" in reason


def test_delete_denied_for_customer() -> None:
    session = _session("cust", account_id="a", payee_allowlist=())
    allowed, reason = evaluate_action(
        "delete_account",
        session,
        Decimal("100"),
        parse_request("delete my account", ()),
    )
    assert allowed is False
    assert "admin" in reason


def test_insufficient_funds_suggests_balance_view() -> None:
    session = _session("zero")
    decision = _decide_wire(session, _ledger(Decimal("0")))
    assert decision.outcome == "deny"
    assert "insufficient funds" in decision.reason
    assert decision.suggestion == "view_account_balance"


def test_ambiguous_margin_denied() -> None:
    session = _session("funded")
    decision = _decide_wire(
        session,
        _ledger(Decimal("10000")),
        query_vector=AMBIGUOUS_QUERY,
        min_margin=0.08,
    )
    assert decision.outcome == "deny"
    assert "ambiguous" in decision.reason


def test_minilm_far_from_every_action_denied() -> None:
    # Nearest action is the always-allowed FAQ at cosine ~0.33, below MIN_SCORE.
    decision = decide(
        "Tell me a joke.",
        _session("funded"),
        _ledger(Decimal("10000")),
        query_vector=np.array([-0.5, -0.5, -0.5, 0.3]),
        action_embeddings=_orthonormal_catalog(),
    )
    assert decision.action == "view_public_faq"
    assert decision.outcome == "deny"
    assert decision.reason == "similarity below minimum confidence"


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


@pytest.mark.integration
def test_minilm_ranks_wire_highest_for_canonical_sentence() -> None:
    embedder = ActionEmbedder()
    query = embedder.embed(WIRE_UTTERANCE)
    action_vecs = embedder.action_embeddings(ACTION_CATALOG)
    from hybrid import cosine_scores

    scores = cosine_scores(query, action_vecs)
    top = max(scores, key=scores.get)
    assert top == "wire_transfer_funds"
    ranked = sorted(scores.values(), reverse=True)
    assert ranked[0] - ranked[1] >= 0.08 or ranked[0] >= 0.45


def test_disk_cached_embed_does_not_load_model(tmp_path) -> None:
    text = "cached utterance"
    writer = ActionEmbedder(embedding_cache_dir=tmp_path)
    path = writer._query_cache_path(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    reader = ActionEmbedder(embedding_cache_dir=tmp_path)
    vector = reader.embed(text)
    assert reader._model is None
    np.testing.assert_array_equal(vector, [1.0, 0.0, 0.0])


def test_disk_cached_actions_do_not_load_model(tmp_path) -> None:
    catalog = {"wire_transfer_funds": "Send money", "view_public_faq": "FAQ"}
    writer = ActionEmbedder(embedding_cache_dir=tmp_path)
    npz_path, meta_path = writer._action_cache_paths(catalog)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        wire_transfer_funds=np.array([1.0, 0.0], dtype=np.float64),
        view_public_faq=np.array([0.0, 1.0], dtype=np.float64),
    )
    meta_path.write_text(
        '{"model": "%s", "actions": ["wire_transfer_funds", "view_public_faq"]}'
        % writer.model_name
    )

    reader = ActionEmbedder(embedding_cache_dir=tmp_path)
    vectors = reader.action_embeddings(catalog)
    assert reader._model is None
    assert set(vectors) == set(catalog)


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


def test_jev_none_stops_before_policy() -> None:
    decision = decide(
        "What is the capital of Portugal?",
        _session("funded"),
        _ledger(Decimal("10000")),
        intent_ranker=lambda _utterance: _jev_rank("none", 0.91),
    )
    assert decision.outcome == "deny"
    assert decision.reason == "no matching action"
    assert decision.action == "none"
    assert decision.permissions == {}
    assert decision.source == "jev"


def test_jev_low_confidence_stops_before_policy() -> None:
    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        intent_ranker=lambda _utterance: _jev_rank("wire_transfer_funds", 0.2),
    )
    assert decision.outcome == "deny"
    assert decision.reason == "no matching action"
    assert decision.permissions == {}


def test_jev_wire_still_needs_confirmation() -> None:
    decision = decide(
        WIRE_UTTERANCE,
        _session("funded"),
        _ledger(Decimal("10000")),
        intent_ranker=lambda _utterance: _jev_rank("wire_transfer_funds", 0.86),
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
        query_vector=faq_query,
        action_embeddings=_orthonormal_catalog(),
        intent_ranker=lambda _utterance: _jev_rank("wire_transfer_funds", 0.86),
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "needs_confirmation"
    match = decision.description_match
    assert match is not None
    assert not match.agrees
    assert match.nearest == "view_public_faq"
    assert "description_gap=nearest_differs" in decision.rule_trace


def test_jev_failure_falls_back_to_minilm() -> None:
    def _boom(_utterance: str) -> IntentRank:
        raise RuntimeError("ranker down")

    decision = _decide_wire(
        _session("funded"),
        _ledger(Decimal("10000")),
        intent_ranker=_boom,
    )
    assert decision.source == "minilm"
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "needs_confirmation"


@pytest.mark.integration
def test_live_jev_abstains_on_unrelated_sentence() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("TYPESAFE_API_KEY is not set")
    decision = decide(
        "What is the capital of Portugal?",
        _session("funded"),
        _ledger(Decimal("10000")),
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
    return route(utterance, session, ledger, splitter=split, labeler=labeler, intent_ranker=lambda _u: rank)


def _never(*_args) -> None:
    raise AssertionError("must not run")


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


def test_route_low_confidence_label_is_ordered_last() -> None:
    def _unsure_balance(requests: tuple[str, ...]) -> list[IntentRank]:
        return [_jev_rank(LABELS[text], 0.3 if text == BALANCE else 0.95) for text in requests]

    decision = _route(
        "Show my balance and send $500 to my external bank account.", _session("funded"), _ledger(Decimal("10000")),
        count="several", split=lambda _u: (BALANCE, WIRE_500), labeler=_unsure_balance,
    )
    assert decision.action == "wire_transfer_funds"
    assert decision.follow_ups == (BALANCE,)
