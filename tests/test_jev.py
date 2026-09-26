"""Tests for the Jev classifier adapter."""

from __future__ import annotations

from typesafe_sdk import ChoiceAnswer

from jev import JevClassifier


def _answer(choice: str, confidence: float) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=choice, confidence=confidence, probabilities={choice: confidence})


class _CannedJev(JevClassifier):
    """Real adapter logic; only the network call is replaced."""

    def __init__(self, **answers: ChoiceAnswer) -> None:
        super().__init__()
        self._answers = answers

    def _ask(
        self, state: str, questions: dict[str, tuple[str, dict[str, str]]]
    ) -> dict[str, ChoiceAnswer] | None:
        return self._answers


def _classifier(**answers: ChoiceAnswer) -> JevClassifier:
    return _CannedJev(**answers)


def test_unsure_action_and_count_become_none() -> None:
    rank = _classifier(
        action=_answer("wire_transfer_funds", 0.2),
        request_count=_answer("several", 0.4),
    ).classify("Send $500 to my external bank account.")
    assert rank is not None
    assert rank.action == "none"
    assert rank.confidence == 0.2
    assert rank.request_count is None


def test_unsure_label_becomes_none_and_sure_label_is_kept() -> None:
    ranks = _classifier(
        request_1=_answer("view_account_balance", 0.3),
        request_2=_answer("wire_transfer_funds", 0.95),
    ).label(("Show my balance.", "Send $500 to my external bank account."))
    assert ranks is not None
    assert [r.action for r in ranks] == ["none", "wire_transfer_funds"]
