"""Contract tests: every adapter must honour the meaning of its role, not only its signature.

To check a new adapter, add it to the role's fixture params. Signatures are
checked by pyright, which `test_pyright_reports_no_errors` runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast, get_args

import numpy as np
import pytest
import typesafe_sdk
from typesafe_sdk import ChoiceAnswer

import intent_understanding
from contracts import Classifier, Explainer, IntentRank, Labeler, RequestCount, Splitter
from intent_understanding import MedGemmaSplitter
from jev import JevClassifier
from minilm import ActionEmbedder, MiniLMExplainer
from policy import ACTION_CATALOG, NONE_ACTION

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = {*ACTION_CATALOG, NONE_ACTION}


def test_pyright_reports_no_errors() -> None:
    """A signature that breaks a contract fails the test run, not a user's sentence."""
    result = subprocess.run(
        [sys.executable, "-m", "pyright", "--outputjson"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    report = json.loads(result.stdout)
    errors = [d for d in report["generalDiagnostics"] if d["severity"] == "error"]
    assert not errors, "\n".join(
        f"{d['file']}:{d['range']['start']['line'] + 1}: {d['message']}" for d in errors
    )


# --- IntentRank: whatever a model returns, values outside the contract never reach the router.


@pytest.mark.parametrize(
    "field",
    [
        {"action": "transfer_all_money"},
        {"confidence": 1.5},
        {"confidence": float("nan")},
        {"request_count": cast(Any, "two")},
        {"scores": cast(Any, {"wire_transfer_funds": "high"})},
    ],
    ids=["unknown-action", "confidence-above-1", "confidence-nan", "unknown-count", "text-score"],
)
def test_intent_rank_rejects_values_outside_the_contract(field: dict[str, Any]) -> None:
    valid: dict[str, Any] = {
        "action": "wire_transfer_funds",
        "scores": {"wire_transfer_funds": 0.9},
        "confidence": 0.9,
        "source": "test",
        "request_count": "one",
    }
    with pytest.raises(ValueError):
        IntentRank(**{**valid, **field})


# --- Classifier and Labeler: None means unavailable, never an exception or a guess.


class _FailingClient:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("service down")


@pytest.fixture(params=["jev-no-key", "jev-call-fails"])
def unavailable_classifier(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> JevClassifier:
    if request.param == "jev-no-key":
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder-not-a-real-key")
        monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", _FailingClient)
    return JevClassifier()


def test_unavailable_classifier_returns_none(unavailable_classifier: Classifier) -> None:
    assert unavailable_classifier.classify("Send $500 to my external bank account.") is None


def test_unavailable_labeler_returns_none(unavailable_classifier: Labeler) -> None:
    assert unavailable_classifier.label(("Show my balance.", "Close this account.")) is None


def _answer(choice: str, confidence: float) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=choice, confidence=confidence, probabilities={choice: confidence})


@pytest.fixture(params=["jev"])
def answering_classifier(monkeypatch: pytest.MonkeyPatch) -> JevClassifier:
    """The real adapter; only the network call returns canned answers."""
    answers = {
        "action": _answer("wire_transfer_funds", 0.9),
        "request_count": _answer("one", 0.9),
        "request_1": _answer("view_account_balance", 0.9),
        "request_2": _answer("delete_account", 0.9),
    }
    monkeypatch.setattr(
        JevClassifier, "_ask",
        lambda _self, _state, questions: {name: answers[name] for name in questions},
    )
    return JevClassifier()


@pytest.fixture(params=["jev"])
def off_contract_classifier(monkeypatch: pytest.MonkeyPatch) -> JevClassifier:
    """The model answers confidently with an action that is not in the catalog."""
    answers = {
        "action": _answer("transfer_all_money", 0.9),
        "request_count": _answer("one", 0.9),
        "request_1": _answer("transfer_all_money", 0.9),
    }
    monkeypatch.setattr(
        JevClassifier, "_ask",
        lambda _self, _state, questions: {name: answers[name] for name in questions},
    )
    return JevClassifier()


def test_off_contract_answer_counts_as_unavailable(off_contract_classifier: Classifier) -> None:
    assert off_contract_classifier.classify("Move everything out.") is None


def test_off_contract_label_counts_as_unavailable(off_contract_classifier: Labeler) -> None:
    assert off_contract_classifier.label(("Move everything out.",)) is None


def test_classifier_answers_within_the_catalog(answering_classifier: Classifier) -> None:
    rank = answering_classifier.classify("Send $500 to my external bank account.")
    assert rank is not None
    assert rank.action in ACTIONS
    assert rank.request_count is None or rank.request_count in get_args(RequestCount)


def test_labeler_answers_once_per_request_in_order(answering_classifier: Labeler) -> None:
    ranks = answering_classifier.label(("Show my balance.", "Close this account."))
    assert ranks is not None
    assert [r.action for r in ranks] == ["view_account_balance", "delete_account"]


# --- Splitter: failure raises (the router then denies); success keeps the written order.


@pytest.fixture(params=["medgemma"])
def unreachable_splitter() -> Splitter:
    return MedGemmaSplitter(endpoint="http://127.0.0.1:9/api/chat", timeout_s=1)


def test_unreachable_splitter_raises_instead_of_inventing_requests(unreachable_splitter: Splitter) -> None:
    with pytest.raises(Exception):
        unreachable_splitter.split("Close this account and send $500.")


def test_splitter_keeps_written_order_and_uses_its_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_chat(content: str, *, model: str, endpoint: str, timeout_s: float, as_json: bool) -> str:
        calls.append((model, endpoint))
        return "Close this account.\n\nSend $500 to my external bank account.\n"

    monkeypatch.setattr(intent_understanding, "_chat", _fake_chat)
    splitter: Splitter = MedGemmaSplitter(model="other-model", endpoint="http://example.invalid/api/chat")
    assert splitter.split("Close this account and send $500 to my external bank account.") == (
        "Close this account.",
        "Send $500 to my external bank account.",
    )
    assert calls == [("other-model", "http://example.invalid/api/chat")]


# --- Explainer: display only; a chosen action far from its description is a gap.


class _FixedEmbedder(ActionEmbedder):
    def __init__(self, query: np.ndarray) -> None:
        super().__init__()
        self._query = query

    def embed(self, text: str) -> np.ndarray:
        return self._query

    def action_embeddings(self, catalog: dict[str, str] | None = None) -> dict[str, np.ndarray]:
        # A fifth axis lets the query be nearest to the wire at any cosine.
        return {name: np.eye(5)[i] for i, name in enumerate(ACTION_CATALOG)}


def _query_with_wire_cosine(cosine: float) -> np.ndarray:
    rest = 0.1
    return np.array([cosine, rest, rest, rest, np.sqrt(1 - cosine**2 - 3 * rest**2)])


@pytest.mark.parametrize(("cosine", "gap"), [(0.46, False), (0.44, True)])
def test_explainer_flags_a_gap_below_its_floor(cosine: float, gap: bool) -> None:
    assert list(ACTION_CATALOG)[0] == "wire_transfer_funds"
    explainer: Explainer = MiniLMExplainer(_FixedEmbedder(_query_with_wire_cosine(cosine)))
    match = explainer.explain("Send $500 to my external bank account.", "wire_transfer_funds")
    assert match is not None
    assert match.chosen == "wire_transfer_funds"
    assert match.nearest == "wire_transfer_funds"
    assert match.description_gap is gap
