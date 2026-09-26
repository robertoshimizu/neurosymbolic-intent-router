"""Isolated experiment: does GLiNER2.5 find the amount and payee where parse_request fails?

GLiNER returns spans of the user's text; parse_request returns a Decimal and an allowlisted
label. Only the span shape is asserted. Hits, misses and time are printed, because they are
the result. Nothing in src/ uses this file.

    uv run --group experiments pytest experiments/gliner_extraction -m gliner -s
    GLINER_MODEL=fastino/gliner2.5-small-v1 uv run --group experiments pytest experiments/gliner_extraction -m gliner -s
"""

from __future__ import annotations

import os
import re
import time
from decimal import Decimal
from typing import Any

import pytest

from policy import ParsedRequest
from sentences import ALLOWLIST, CASES, Case

# The regex parser src/policy.py used before the extractor replaced it: the baseline this experiment measured.
AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")


def parse_request(utterance: str, payee_allowlist: tuple[str, ...]) -> ParsedRequest:
    match = AMOUNT_RE.search(utterance)
    amount = Decimal(match.group(1).replace(",", "")) if match else None
    lower = utterance.lower()
    payee = next((label for label in payee_allowlist if label.lower() in lower), None)
    return ParsedRequest(amount=amount, payee=payee)

pytestmark = pytest.mark.gliner

MODEL = os.environ.get("GLINER_MODEL", "fastino/gliner2.5-base-v1")
AMOUNT_AND_PAYEE = {
    "amount": "money amount the user asks to send",
    "payee": "destination the money is sent to",
}
# Both label sets run: does asking for the source pull the destination away from `payee`?
LABEL_SETS = {
    "amount+payee": AMOUNT_AND_PAYEE,
    "amount+payee+source": {**AMOUNT_AND_PAYEE, "source_account": "account the money is taken from"},
}


@pytest.fixture(scope="module")
def model() -> Any:
    gliner2 = pytest.importorskip("gliner2")
    return gliner2.AutoExtractor.from_pretrained(MODEL)


def spans(text: str, result: dict[str, Any], labels: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """Check each span points into the text, then keep its text per label."""
    entities = result["entities"]
    assert set(entities) <= set(labels)
    found: dict[str, tuple[str, ...]] = {}
    for label in labels:
        items = entities.get(label, [])
        for item in items:
            assert text[item["start"] : item["end"]] == item["text"]
            assert 0.0 <= item["confidence"] <= 1.0
        found[label] = tuple(f'{item["text"]} ({item["confidence"]:.2f})' for item in items)
    return found


def hit(found: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    """Exact match: the same span texts, in any order, and nothing extra."""
    return sorted(f.rsplit(" (", 1)[0] for f in found) == sorted(expected)


def baseline(case: Case) -> tuple[str, bool]:
    """parse_request against the first request: its dollar value and its allowlisted payee."""
    parsed = parse_request(case.text, ALLOWLIST)
    if parsed.amount == case.value:
        amount = "ok"
    elif parsed.amount is None:
        amount = "missing"
    else:
        amount = f"WRONG {parsed.amount}"
    first_payee = case.payees[0] if case.payees and case.payees[0] in ALLOWLIST else None
    return amount, parsed.payee == first_payee


@pytest.mark.parametrize("label_set", LABEL_SETS)
def test_gliner_against_parse_request(model: Any, label_set: str) -> None:
    labels = LABEL_SETS[label_set]
    totals = {"amount": 0, "payee": 0, "source_account": 0, "base_amount": 0, "base_wrong": 0, "base_payee": 0}
    elapsed = 0.0
    for number, case in enumerate(CASES, 1):
        start = time.perf_counter()
        result = model.extract_entities(case.text, labels, include_confidence=True, include_spans=True)
        elapsed += time.perf_counter() - start
        found = spans(case.text, result, labels)
        expected = {"amount": case.amounts, "payee": case.payees, "source_account": case.sources}
        marks = {label: hit(found[label], expected[label]) for label in labels}
        base_amount, base_payee = baseline(case)
        for label, ok in marks.items():
            totals[label] += ok
        totals["base_amount"] += base_amount == "ok"
        totals["base_wrong"] += base_amount.startswith("WRONG")
        totals["base_payee"] += base_payee

        print(f"\n#{number} [{case.probe}] {case.text}")
        for label, ok in marks.items():
            print(f"  gliner {label:<15} {'hit ' if ok else 'MISS'} found={list(found[label])} expected={list(expected[label])}")
        print(f"  parse_request  amount={base_amount} payee={'ok' if base_payee else 'MISS'}")

    n = len(CASES)
    print(f"\nMODEL {MODEL}, labels {label_set}: {n} sentences, {elapsed / n * 1000:.0f} ms per extraction after loading")
    print("  gliner  " + "  ".join(f"{label} {totals[label]}/{n}" for label in labels))
    print(f"  parse_request  amount {totals['base_amount']}/{n} ({totals['base_wrong']} wrong values)  payee {totals['base_payee']}/{n}")
