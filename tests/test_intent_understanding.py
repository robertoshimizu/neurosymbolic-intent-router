"""MedGemma request splitting. Live cases call local Ollama."""

from __future__ import annotations

import pytest

from intent_understanding import parse_split, split_requests


def test_parse_split_keeps_one_request_per_nonblank_line() -> None:
    raw = "\n  Send the remaining cash to my external bank account.\n\nClose this account.  \n"
    assert parse_split(raw) == (
        "Send the remaining cash to my external bank account.",
        "Close this account.",
    )


# sentence -> words each split line must contain, in the order written.
SPLIT_CASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "Close this account and send the remaining cash to my external bank account.": (
        ("close", "account"),
        ("send", "remaining cash", "external bank account"),
    ),
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
