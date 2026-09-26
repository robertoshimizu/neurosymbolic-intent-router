"""The router's contracts: the roles a model adapter must implement, and the types they exchange.

Adapters subclass these Protocols explicitly. A missing method fails when the
adapter is created; a wrong signature fails `uv run pyright`. A model value
outside the contract fails when an IntentRank is built.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol, get_args

from policy import ACTION_CATALOG, NONE_ACTION

RequestCount = Literal["none", "one", "several"]


@dataclass(frozen=True)
class IntentRank:
    """A classifier's reading of the text: which catalog intent it expresses. Not an action; policy has not run yet."""

    action: str
    scores: dict[str, float]
    confidence: float
    source: str
    request_count: RequestCount | None = None

    def __post_init__(self) -> None:
        """Models can return anything; reject values outside the contract before the router sees them."""
        if self.action != NONE_ACTION and self.action not in ACTION_CATALOG:
            raise ValueError(f"action {self.action!r} is not in the catalog")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence!r} is outside 0..1")
        if self.request_count is not None and self.request_count not in get_args(RequestCount):
            raise ValueError(f"request_count {self.request_count!r} is not one of {get_args(RequestCount)}")
        if not all(isinstance(score, (int, float)) for score in self.scores.values()):
            raise ValueError("scores must be numbers")


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
    """Reads which catalog intent the text expresses and counts the requests. Unsure answers are `none`; None means unavailable."""

    @abstractmethod
    def classify(self, text: str) -> IntentRank | None: ...


class Labeler(Protocol):
    """Reads which catalog intent each request expresses. Unsure answers are `none`; None means unavailable."""

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
