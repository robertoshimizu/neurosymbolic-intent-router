"""Isolated sentence readings. Live cases call local MedGemma and only check shape."""

from __future__ import annotations

import pytest

from intent_understanding import parse_model_json, read_sentence, reading_from_payload

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


def test_established_drops_inferred_and_unknown() -> None:
    reading = reading_from_payload(
        {
            "claims": [
                {"statement": "A patient fell.", "status": "EXPLICIT", "kind": "event"},
                {
                    "statement": "The fall caused the arrival.",
                    "status": "INFERRED",
                    "kind": "causal",
                },
                {"statement": "The nurse was on duty.", "status": "UNKNOWN", "kind": "concept"},
            ]
        }
    )
    established = reading.established()
    assert len(established) == 1
    assert established[0]["statement"] == "A patient fell."


def test_parse_model_json_rejects_a_list() -> None:
    with pytest.raises(ValueError):
        parse_model_json("[]")


@pytest.mark.integration
@pytest.mark.parametrize("sentence", SENTENCES)
def test_medgemma_returns_claims(sentence: str) -> None:
    reading = read_sentence(sentence)
    print(f"\nSENTENCE: {sentence}")
    for claim in reading.claims:
        print(f"  {claim['status']:<9} {claim['kind']:<12} {claim['statement']}")
    assert isinstance(reading.claims, tuple)
