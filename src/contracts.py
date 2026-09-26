"""The router's contracts: the roles a model or reasoner adapter must implement, and the types they exchange.

Adapters subclass these Protocols explicitly. A missing method fails when the
adapter is created; a wrong signature fails `uv run pyright`. A model value
outside the contract fails when an IntentRank is built; a reasoner answer
outside the contract fails when a PolicyResult is built.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol, get_args

from policy import ACTION_CATALOG, NONE_ACTION, ParsedRequest, Session

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
class Extraction:
    """An extractor's reading of one request: amount and payee spans copied from its text. Not values; the rules convert them."""

    text: str
    amounts: tuple[str, ...]
    payees: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        """A span the text does not contain was invented, not read; reject it before the rules see it."""
        for span in (*self.amounts, *self.payees):
            if not isinstance(span, str) or not span or span not in self.text:
                raise ValueError(f"span {span!r} is not in the text")


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


class Extractor(Protocol):
    """Reads the amount and payee spans of one request. None means unavailable."""

    @abstractmethod
    def extract(self, text: str) -> Extraction | None: ...


class Explainer(Protocol):
    """Display-only note on a chosen action. Never changes the decision."""

    @abstractmethod
    def explain(self, text: str, action: str) -> DescriptionMatch | None: ...


@dataclass(frozen=True)
class PolicyResult:
    """A reasoner's verdict on one action. Denied: every reason, in the reasoner's order, and at most one
    permitted action to offer instead. Allowed: the permit message."""

    allowed: bool
    reasons: tuple[str, ...]
    suggestion: str | None = None

    def __post_init__(self) -> None:
        """Reasoners can return anything; reject values outside the contract before the router sees them."""
        if not isinstance(self.allowed, bool):
            raise ValueError(f"allowed {self.allowed!r} is not a bool")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise ValueError("reasons must be a non-empty tuple")
        if not all(isinstance(reason, str) and reason for reason in self.reasons):
            raise ValueError("reasons must be non-empty strings")
        if self.suggestion is not None:
            if self.allowed:
                raise ValueError("an allowed action carries no suggestion")
            if self.suggestion not in ACTION_CATALOG:
                raise ValueError(f"suggestion {self.suggestion!r} is not in the catalog")


class Policy(Protocol):
    """Judges one action against session facts, and orders several requests. The router does not know which reasoner answers."""

    @abstractmethod
    def evaluate(
        self, action: str, session: Session, balance: Decimal, parsed: ParsedRequest
    ) -> PolicyResult: ...

    @abstractmethod
    def order(self, actions: tuple[str, ...]) -> tuple[int, ...]:
        """Positions of `actions` in the order they should be handled. Ties keep the written order."""
        ...
