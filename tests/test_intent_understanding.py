"""Isolated sentence readings. Live cases call local MedGemma and print its JSON."""

from __future__ import annotations

import json

import pytest

from intent_understanding import SECTIONS, parse_model_json, prompt_for, read_sentence

SUPPLIER = (
    "The supplier stopped shipping shortly after Banco X withdrew the credit line. "
    "Management subsequently shifted production to its Mexican subsidiary."
)

SENTENCES = (
    "The nurse arrived after the patient fell.",
    "She left because the train was late.",
    "How much money do I have before I wire funds out?",
    "Close this account and send the remaining cash to my external bank account.",
    "What is the capital of Portugal?",
    SUPPLIER,
)


def test_prompt_appends_only_the_text() -> None:
    prompt = prompt_for("What is the capital of Portugal?")
    assert prompt.endswith('TEXT TO ANALYZE: What is the capital of Portugal?')
    assert "Banco X" not in prompt


def test_parser_keeps_items_without_a_statement_field() -> None:
    raw = """
    {"entities": [{"entity": "Portugal", "type": "Country"}],
     "relationships": [{"relation": "capital_of", "source": "Portugal", "target": "UNKNOWN"}],
     "events": []}
    """
    payload = parse_model_json(raw)
    assert payload["entities"][0]["entity"] == "Portugal"
    assert payload["relationships"][0]["target"] == "UNKNOWN"


@pytest.mark.integration
@pytest.mark.parametrize("sentence", SENTENCES)
def test_medgemma_reads_sentence(sentence: str) -> None:
    reading = read_sentence(sentence)
    print(f"\nSENTENCE: {sentence}")
    print(json.dumps(reading.raw, indent=2, ensure_ascii=False))
    assert isinstance(reading.raw, dict)
    assert any(isinstance(reading.raw.get(name), list) for name in SECTIONS)
