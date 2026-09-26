"""Tests for the Jev classifier adapter."""

from __future__ import annotations

from types import SimpleNamespace

from jev import JevClassifier


def _answer(choice: str, confidence: float) -> SimpleNamespace:
    return SimpleNamespace(choice=choice, confidence=confidence, probabilities={choice: confidence})


def _classifier(**answers: SimpleNamespace) -> JevClassifier:
    classifier = JevClassifier()
    classifier._ask = lambda _state, _questions: SimpleNamespace(answers=answers)
    return classifier


def test_unsure_action_and_count_become_none() -> None:
    rank = _classifier(
        action=_answer("wire_transfer_funds", 0.2),
        request_count=_answer("several", 0.4),
    ).classify("Send $500 to my external bank account.")
    assert rank.action == "none"
    assert rank.confidence == 0.2
    assert rank.request_count is None


def test_unsure_label_becomes_none_and_sure_label_is_kept() -> None:
    ranks = _classifier(
        request_1=_answer("view_account_balance", 0.3),
        request_2=_answer("wire_transfer_funds", 0.95),
    ).label(("Show my balance.", "Send $500 to my external bank account."))
    assert [r.action for r in ranks] == ["none", "wire_transfer_funds"]
