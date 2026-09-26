"""GLiNER as the Extractor: its answer becomes spans of the text, and the rules turn them into facts."""

from __future__ import annotations

from decimal import Decimal

import pytest

from adapters.gliner import GLiNERExtractor
from policy import to_request

ALLOWLIST = ("external bank account", "savings account")


class _FakeModel:
    def __init__(self, answer: dict[str, object]) -> None:
        self.answer = answer

    def extract_entities(self, text: str, entity_types: dict[str, str]) -> dict[str, object]:
        return self.answer


def test_extractor_keeps_amount_and_payee_spans() -> None:
    model = _FakeModel({"entities": {"amount": ["$500"], "payee": [{"text": "external bank account"}]}})
    extraction = GLiNERExtractor(model=model).extract("Send $500 to my external bank account.")
    assert extraction.amounts == ("$500",)
    assert extraction.payees == ("external bank account",)


@pytest.mark.parametrize(
    "answer",
    [{}, {"entities": {"amount": "$500"}}, {"entities": {"amount": [5]}}, {"entities": {"amount": ["$5,000"]}}],
    ids=["no-entities", "not-a-list", "not-text", "invented-span"],
)
def test_malformed_or_invented_answer_raises(answer: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        GLiNERExtractor(model=_FakeModel(answer)).extract("Send $500 to my external bank account.")


@pytest.mark.gliner
@pytest.mark.parametrize(
    ("text", "amount", "payee"),
    [
        ("Send one thousand five hundred dollars to my external bank account.", Decimal("1500"), "external bank account"),
        ("Transfer $1.5k to my savings account.", None, "savings account"),
        ("Move $2,000 from my savings account to my external bank account.", Decimal("2000"), None),
    ],
)
def test_gliner_reads_facts_the_rules_accept_or_refuse(text: str, amount: Decimal | None, payee: str | None) -> None:
    extraction = GLiNERExtractor().extract(text)
    parsed = to_request(extraction.amounts, extraction.payees, ALLOWLIST)
    assert (parsed.amount, parsed.payee) == (amount, payee)
