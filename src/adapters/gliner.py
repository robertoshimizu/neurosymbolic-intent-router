"""GLiNER2.5 as the router's Extractor: amount and payee spans, read from the text, never converted here."""

from __future__ import annotations

from typing import Protocol, cast

from contracts import Extraction, Extractor

MODEL_NAME = "fastino/gliner2.5-base-v1"
# No source-account label: in the experiment it fired on the destination; two payee spans deny instead.
LABELS = {
    "amount": "money amount the user asks to send",
    "payee": "destination the money is sent to",
}


class EntityModel(Protocol):
    """The one gliner2 call this adapter uses."""

    def extract_entities(self, text: str, entity_types: dict[str, str]) -> dict[str, object]: ...


class GLiNERExtractor(Extractor):
    """Loads the model on first use. A failed load, call or answer raises; the router then reads nothing and denies."""

    def __init__(self, model_name: str = MODEL_NAME, *, model: EntityModel | None = None) -> None:
        self.model_name = model_name
        self._model = model

    def _load(self) -> EntityModel:
        if self._model is None:
            from gliner2 import AutoExtractor

            self._model = cast(EntityModel, AutoExtractor.from_pretrained(self.model_name))
        return self._model

    def extract(self, text: str) -> Extraction:
        entities = self._load().extract_entities(text, LABELS).get("entities")
        if not isinstance(entities, dict):
            raise ValueError("model answer has no entities")
        return Extraction(
            text=text,
            amounts=_span_texts(entities.get("amount", [])),
            payees=_span_texts(entities.get("payee", [])),
            source="gliner",
        )


def _span_texts(items: object) -> tuple[str, ...]:
    """gliner2 returns plain strings, or dicts with a text field when asked for spans or confidence."""
    if not isinstance(items, list):
        raise ValueError("entities must be a list")
    texts = [item.get("text") if isinstance(item, dict) else item for item in items]
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("every entity must be text")
    return tuple(cast(list[str], texts))
