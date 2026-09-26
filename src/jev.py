"""Jev (TypeSafe system_one) as the router's Classifier and Labeler."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import ChoiceAnswer

from contracts import Classifier, IntentRank, Labeler, RequestCount
from policy import ACTION_CATALOG, NONE_ACTION

# TYPESAFE_API_KEY comes from the project .env, never from code.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

# Jev confidence is a probability, not a cosine. Docs treat values under 0.5 as unsure.
MIN_CONFIDENCE = 0.5
NONE_CRITERION = "The utterance is not a request for any of these banking actions"
REQUEST_COUNT_CRITERIA = {
    "none": "The text asks the system to do nothing",
    "one": "The text asks the system to do exactly one thing",
    "several": "The text asks the system to do two or more different things",
}


class JevClassifier(Classifier, Labeler):
    """Answers below min_confidence come back as `none`, so the router never sees Jev's scale."""

    def __init__(self, min_confidence: float = MIN_CONFIDENCE) -> None:
        self.min_confidence = min_confidence

    def classify(self, text: str) -> IntentRank | None:
        """Action and request count in one call. None means unavailable."""
        response = self._ask(
            text,
            {
                "action": (
                    "Which banking action is the user asking for? "
                    "Choose none if the request is not one of these actions.",
                    {**ACTION_CATALOG, NONE_ACTION: NONE_CRITERION},
                ),
                "request_count": (
                    "How many distinct things does the user ask the system to do? "
                    "A mention of a future action that is only context does not count as a request.",
                    REQUEST_COUNT_CRITERIA,
                ),
            },
        )
        if response is None:
            return None
        count = response["request_count"]
        choice = count.choice
        request_count: RequestCount | None = None
        if float(count.confidence) >= self.min_confidence and choice in ("none", "one", "several"):
            request_count = choice
        return replace(self._to_rank(response["action"]), request_count=request_count)

    def label(self, texts: tuple[str, ...]) -> list[IntentRank] | None:
        """One call, one catalog question per text. None means unavailable."""
        state = "\n".join(f"Request {i}: {text}" for i,
                          text in enumerate(texts, start=1))
        response = self._ask(
            state,
            {
                f"request_{i}": (
                    f"Which banking action does Request {i} ask for? Choose none if it is not one of these actions.",
                    {**ACTION_CATALOG, NONE_ACTION: NONE_CRITERION},
                )
                for i in range(1, len(texts) + 1)
            },
        )
        if response is None:
            return None
        return [self._to_rank(response[f"request_{i}"]) for i in range(1, len(texts) + 1)]

    def _ask(
        self, state: str, questions: dict[str, tuple[str, dict[str, str]]]
    ) -> dict[str, ChoiceAnswer] | None:
        """One system_one call. None means unavailable, including an answer that is not a choice."""
        if not os.environ.get("TYPESAFE_API_KEY"):
            return None
        try:
            from typesafe_sdk import Choice, TypeSafeClient

            with TypeSafeClient() as client:
                response = client.system_one(
                    state=state,
                    questions={
                        name: Choice(instructions=instructions, criteria=criteria)
                        for name, (instructions, criteria) in questions.items()
                    },
                )
        except Exception:
            print("Jev unavailable.", flush=True)
            return None
        answers = {name: answer for name, answer in response.answers.items()
                   if isinstance(answer, ChoiceAnswer)}
        return answers if answers.keys() == questions.keys() else None

    def _to_rank(self, answer: ChoiceAnswer) -> IntentRank:
        confidence = answer.confidence
        return IntentRank(
            action=answer.choice if confidence >= self.min_confidence else NONE_ACTION,
            scores=dict(answer.probabilities),
            confidence=confidence,
            source="jev",
        )
