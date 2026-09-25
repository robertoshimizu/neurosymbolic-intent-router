"""Isolated sentence readings. Live cases call local MedGemma and print its JSON."""

from __future__ import annotations

import json

import pytest

from intent_understanding import (
    parse_model_json,
    parse_split,
    prompt_for,
    read_sentence,
    split_requests,
)

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

# Human approved omitting unsupported causes and requiring explicit question,
# request, and prospective context with flexible field names. Other criteria
# remain drafts pending human review. These are not model
# instructions or automated semantic assertions. Equivalent JSON is acceptable;
# judge meaning without requiring a particular field name or wording.
# Record each requirement as met, missed, or unclear, and each forbidden
# assertion as absent, present, or unclear. JSON shape checks alone are not a pass.
REVIEW_CRITERIA: dict[str, dict[str, tuple[str, ...]]] = {
    SENTENCES[0]: {
        "must_preserve": (
            "The patient fell; the nurse arrived; the fall preceded arrival.",
        ),
        "must_not_assert": (
            "The fall caused arrival, even as an INFERRED explanation.",
            "An injury, treatment, or the nurse's motivation as established fact.",
        ),
    },
    SENTENCES[1]: {
        "must_preserve": (
            "She left; the train was late; the text explicitly links lateness to leaving.",
        ),
        "must_not_assert": (
            "An identity for 'she' or an unstated destination.",
        ),
    },
    SENTENCES[2]: {
        "must_preserve": (
            "Explicit question context about available money, with the amount unknown.",
            "Explicit prospective context for the wire and the 'before' relation.",
        ),
        "must_not_assert": (
            "A completed transfer or an instruction to execute a transfer now.",
            "An amount, destination, or available balance not supplied by the text.",
        ),
    },
    SENTENCES[3]: {
        "must_preserve": (
            "Explicit request context for both closing and sending remaining cash.",
            "The external bank account is the requested transfer destination.",
            "The amount is described as remaining cash, without a numeric value.",
        ),
        "must_not_assert": (
            "Either requested event has already happened.",
            "A numeric amount, account identifier, or explicit 'before'/'after' relation.",
        ),
    },
    SENTENCES[4]: {
        "must_preserve": (
            "Portugal and explicit question context asking for its unspecified capital.",
        ),
        "must_not_assert": (
            "Lisbon or any other supplied answer: this task reads rather than answers.",
        ),
    },
    SUPPLIER: {
        "must_preserve": (
            "The supplier stopped shipping; Banco X withdrew the credit line.",
            "Management shifted production to a Mexican subsidiary.",
            "Withdrawal preceded the shipping stop, with a short interval.",
            "'Subsequently' places the production shift later in the described sequence.",
        ),
        "must_not_assert": (
            "A specific shipped item or a uniquely identified credit-line holder.",
            "An unsupported ownership link or uniquely resolved antecedent for 'its'.",
            "Financing dependence or physical location in Mexico as established fact.",
            "Withdrawal caused the shipping stop, even as an INFERRED explanation.",
        ),
    },
}

# Remaining human review:
# - Which optional implications of 'shifted production' should be retained?
# Missing optional implications are not failures under these draft criteria.


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


def _raw_text(value: object) -> str:
    """Flatten model JSON for flexible-field semantic smoke checks."""
    if isinstance(value, dict):
        return " ".join(f"{key} {_raw_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_raw_text(item) for item in value)
    return str(value)


def _section_items(raw: dict, fragment: str) -> list:
    return [
        value
        for key, value in raw.items()
        if fragment in key.lower().replace("_", " ") and isinstance(value, list)
    ]


def assert_action_reading(sentence: str, raw: dict) -> None:
    text = _raw_text(raw).lower()
    if sentence == SENTENCES[0]:
        assert all(word in text for word in ("nurse", "patient"))
        assert "after" in text or "before" in text
        assert not any(_section_items(raw, "causal"))
    elif sentence == SENTENCES[1]:
        assert all(word in text for word in ("she", "train", "late"))
        assert any(_section_items(raw, "causal"))
    elif sentence == SENTENCES[2]:
        assert all(word in text for word in ("question", "prospective", "before"))
        assert "unknown" in text
    elif sentence == SENTENCES[3]:
        assert all(word in text for word in ("close", "send", "cash", "external bank"))
        assert "prospective" in text or "request" in text
    elif sentence == SENTENCES[4]:
        assert all(word in text for word in ("portugal", "capital", "question", "unknown"))
        assert "lisbon" not in text
    elif sentence == SUPPLIER:
        assert all(
            word in text
            for word in ("supplier", "banco x", "credit line", "shipping", "production", "subsidiary")
        )
        assert "shortly" in text or "after" in text
        assert "subsequently" in text or "after" in text
        assert not any(_section_items(raw, "causal"))


@pytest.mark.integration
@pytest.mark.parametrize("sentence", SENTENCES)
def test_medgemma_reads_sentence(sentence: str) -> None:
    reading = read_sentence(sentence)
    print(f"\nSENTENCE: {sentence}")
    print("DRAFT HUMAN REVIEW CRITERIA (not an automated semantic verdict):")
    print(json.dumps(REVIEW_CRITERIA[sentence], indent=2, ensure_ascii=False))
    print(json.dumps(reading.raw, indent=2, ensure_ascii=False))
    assert isinstance(reading.raw, dict)
    assert_action_reading(sentence, reading.raw)


def test_parse_split_keeps_one_request_per_nonblank_line() -> None:
    raw = "\n  Send the remaining cash to my external bank account.\n\nClose this account.  \n"
    assert parse_split(raw) == (
        "Send the remaining cash to my external bank account.",
        "Close this account.",
    )


# sentence -> words each split line must contain, in the order written.
SPLIT_CASES: dict[str, tuple[tuple[str, ...], ...]] = {
    SENTENCES[3]: (("close", "account"), ("send", "remaining cash", "external bank account")),
    "Show my balance and then wire $500 to my external bank account.": (
        ("balance",),
        ("wire", "$500", "external bank account"),
    ),
    "I want to send $5,000 to my external bank account.": (
        ("send", "$5,000", "external bank account"),
    ),
    "Don't close my account, just show me my balance.": (("balance",),),
}


@pytest.mark.integration
@pytest.mark.parametrize("sentence", SPLIT_CASES)
def test_medgemma_splits_requests(sentence: str) -> None:
    lines = split_requests(sentence)
    print(f"\nSENTENCE: {sentence}")
    for line in lines:
        print(f"  - {line}")
    expected = SPLIT_CASES[sentence]
    assert len(lines) == len(expected)
    for line, words in zip(lines, expected):
        assert all(word in line.lower() for word in words)
    assert not any("close" in line.lower() for line in lines if "don't" in sentence.lower())
