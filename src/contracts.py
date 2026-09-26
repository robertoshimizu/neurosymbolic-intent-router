"""The router's contracts: the roles a model adapter must implement, and the types they exchange.

Adapters subclass these Protocols explicitly. A missing method fails when the
adapter is created; a wrong signature fails `uv run pyright`.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol

RequestCount = Literal["none", "one", "several"]


@dataclass(frozen=True)
class IntentRank:
    """A classifier's proposal. Policy has not run yet."""

    action: str
    scores: dict[str, float]
    confidence: float
    source: str
    request_count: RequestCount | None = None


@dataclass(frozen=True)
class DescriptionMatch:
    """Closeness of the utterance to the action descriptions.

    The classifier still chooses the action. This only explains that choice.
    """

    chosen: str
    nearest: str
    nearest_score: float
    chosen_score: float
    similarities: dict[str, float]
    description_gap: bool

    @property
    def agrees(self) -> bool:
        return self.nearest == self.chosen


class Classifier(Protocol):
    """Picks one catalog action and counts the requests. Unsure answers are `none`; None means unavailable."""

    @abstractmethod
    def classify(self, text: str) -> IntentRank | None: ...


class Labeler(Protocol):
    """Picks one catalog action per request. Unsure answers are `none`; None means unavailable."""

    @abstractmethod
    def label(self, texts: tuple[str, ...]) -> list[IntentRank] | None: ...


class Splitter(Protocol):
    """Separates a sentence into its requests, in the order written."""

    @abstractmethod
    def split(self, text: str) -> tuple[str, ...]: ...


class Explainer(Protocol):
    """Display-only note on a chosen action. Never changes the decision."""

    @abstractmethod
    def explain(self, text: str, action: str) -> DescriptionMatch | None: ...
