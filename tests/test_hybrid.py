"""Tests for the hybrid action router and wire-transfer ledger."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hybrid import (
    ACTION_CATALOG,
    ActionEmbedder,
    Decision,
    Session,
    decide,
    evaluate_action,
    parse_request,
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


def test_insufficient_funds_denied() -> None:
    session = _session("thin", account_id="a")
    parsed = parse_request(
        "send $5,000 to external bank account",
        session.payee_allowlist,
    )
    allowed, reason = evaluate_action(
        "wire_transfer_funds", session, Decimal("1"), parsed
    )
    assert allowed is False
    assert "insufficient funds" in reason


def test_funded_allowlisted_wire_permitted() -> None:
    session = _session("funded", account_id="a")
    parsed = parse_request(
        "send $5,000 to external bank account",
        session.payee_allowlist,
    )
    allowed, reason = evaluate_action(
        "wire_transfer_funds", session, Decimal("10000"), parsed
    )
    assert allowed is True
    assert "permitted" in reason


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


def test_denied_top_intent_does_not_become_faq() -> None:
    session = _session(
        "guest",
        is_authenticated=False,
        role="guest",
        status="inactive",
        payee_allowlist=(),
    )
    decision = _decide_wire(session, _ledger(Decimal("0")))
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "deny"
    assert "not signed in" in decision.reason
    assert decision.action != "view_public_faq"


def test_insufficient_funds_suggests_balance_view() -> None:
    session = _session("zero")
    decision = _decide_wire(session, _ledger(Decimal("0")))
    assert decision.outcome == "deny"
    assert "insufficient funds" in decision.reason
    assert decision.suggestion == "view_account_balance"


def test_allowed_wire_needs_confirmation() -> None:
    session = _session("funded")
    decision = _decide_wire(session, _ledger(Decimal("10000")))
    assert decision.action == "wire_transfer_funds"
    assert decision.outcome == "needs_confirmation"


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
