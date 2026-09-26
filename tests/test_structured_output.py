"""Isolated experiment: can a local Ollama model return valid structured output for the router's tasks?

Two approaches, same schemas, same sentences:
- native: Ollama's `format=<JSON schema>`, which constrains decoding to the schema.
- instructor: Instructor over Ollama's OpenAI-compatible endpoint; validates after
  generation and asks again on failure.

Only the shape is asserted. Accuracy, retries and time are printed, because
they vary between runs. Nothing in src/ uses this file.

    uv run pytest tests/test_structured_output.py -m ollama -s
    OLLAMA_MODELS="medgemma:27b,qwen3.8:27b" uv run pytest tests/test_structured_output.py -m ollama -s
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import instructor
import pytest
from openai import OpenAI
from pydantic import BaseModel, Field, JsonValue, ValidationError, field_validator

from policy import ACTION_CATALOG, NONE_ACTION
from test_intent_understanding import SPLIT_CASES

pytestmark = [pytest.mark.integration, pytest.mark.ollama]

OLLAMA = "http://127.0.0.1:11434"
MODELS = [m.strip() for m in os.environ.get("OLLAMA_MODELS", "medgemma:27b").split(",") if m.strip()]
ACTIONS = sorted({*ACTION_CATALOG, NONE_ACTION})
ACTION_ENUM: list[JsonValue] = [*ACTIONS]

# sentence -> the action a correct classifier picks. Clear-cut cases only.
CLASSIFY_CASES: dict[str, str] = {
    "I want to send $5,000 to my external bank account.": "wire_transfer_funds",
    "Show my balance.": "view_account_balance",
    "Close this account.": "delete_account",
    "Where can I read answers to common questions about transfers?": "view_public_faq",
    "What is the capital of Portugal?": NONE_ACTION,
    "How much money do I have before I wire funds out?": "view_account_balance",
    "Don't close my account, just show me my balance.": "view_account_balance",
}


class ActionChoice(BaseModel):
    action: str = Field(json_schema_extra={"enum": ACTION_ENUM})

    @field_validator("action")
    @classmethod
    def _in_catalog(cls, value: str) -> str:
        if value not in ACTIONS:
            raise ValueError(f"{value!r} is not one of {ACTIONS}")
        return value


class SplitRequests(BaseModel):
    requests: list[str] = Field(min_length=1)


Schema = TypeVar("Schema", bound=BaseModel)


def _classify_prompt(text: str) -> str:
    catalog = "\n".join(f"- {name}: {description}" for name, description in ACTION_CATALOG.items())
    return (
        "Which banking action is the user asking for? Choose exactly one action from this list, "
        f"or {NONE_ACTION} if the text is not a request for any of them.\n"
        f"{catalog}\n- {NONE_ACTION}: The text is not a request for any of these banking actions\n"
        f"Answer as JSON matching this schema: {json.dumps(ActionChoice.model_json_schema())}\n"
        f"TEXT: {text}"
    )


def _split_prompt(text: str) -> str:
    return (
        "Rewrite the text below as the separate things it asks the system to do. Write each request as one "
        "short, self-contained imperative sentence, using the text's own words. Keep amounts, currencies, "
        "accounts, and destinations exactly as written, and do not add any that are not written. Leave out "
        "things the text says not to do, background, and future actions mentioned only as context. Do not "
        "answer, merge, reorder, or add requests. If the text asks for exactly one thing, return exactly one request. "
        f"Answer as JSON matching this schema: {json.dumps(SplitRequests.model_json_schema())}\n"
        f"TEXT: {text}"
    )


@dataclass
class Outcome:
    valid: bool
    correct: bool
    retries: int
    seconds: float
    shown: str


def _native(model: str, prompt: str, schema: type[Schema]) -> tuple[Schema | None, int]:
    """Ollama's own structured output: the JSON schema constrains decoding."""
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "format": schema.model_json_schema(),
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        content = json.loads(response.read().decode()).get("message", {}).get("content", "")
    try:
        return schema.model_validate_json(content), 0
    except ValidationError:
        return None, 0


def _instructor(model: str, prompt: str, schema: type[Schema]) -> tuple[Schema | None, int]:
    """Instructor: plain JSON mode, validated afterwards, asked again on a validation error. Thinking off."""
    client = instructor.from_openai(
        OpenAI(base_url=f"{OLLAMA}/v1", api_key="ollama", timeout=300), mode=instructor.Mode.JSON
    )
    errors: list[Exception] = []
    client.on("parse:error", errors.append)
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=schema,
            messages=[{"role": "user", "content": prompt}],
            max_retries=2,
            temperature=0,
            reasoning_effort="none",  # same as the native call's "think": False
        )
    except Exception:
        return None, len(errors)
    return result, len(errors)


Runner = Callable[[str, str, type[Any]], tuple[Any, int]]
RUNNERS: dict[str, Runner] = {"native": _native, "instructor": _instructor}


@pytest.fixture(params=MODELS)
def model(request: pytest.FixtureRequest) -> str:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as response:
            installed = {m["name"] for m in json.loads(response.read().decode())["models"]}
    except (urllib.error.URLError, OSError):
        pytest.skip("Ollama is not reachable")
    name: str = request.param
    if name not in installed:
        pytest.skip(f"{name} is not installed in Ollama")
    return name


def _report(title: str, outcomes: dict[str, Outcome]) -> None:
    print(f"\n{title}")
    for sentence, o in outcomes.items():
        mark = "ok " if o.correct else ("bad" if o.valid else "INV")
        print(f"  [{mark}] {o.seconds:6.1f}s retries={o.retries}  {sentence[:52]:<52} -> {o.shown}")
    n = len(outcomes)
    print(
        f"  valid {sum(o.valid for o in outcomes.values())}/{n}"
        f"  correct {sum(o.correct for o in outcomes.values())}/{n}"
        f"  retries {sum(o.retries for o in outcomes.values())}"
        f"  time {sum(o.seconds for o in outcomes.values()):.1f}s"
    )


@pytest.mark.parametrize("approach", list(RUNNERS))
def test_classify_returns_a_catalog_action(model: str, approach: str) -> None:
    outcomes: dict[str, Outcome] = {}
    for sentence, expected in CLASSIFY_CASES.items():
        start = time.perf_counter()
        choice, retries = RUNNERS[approach](model, _classify_prompt(sentence), ActionChoice)
        outcomes[sentence] = Outcome(
            valid=choice is not None,
            correct=choice is not None and choice.action == expected,
            retries=retries,
            seconds=time.perf_counter() - start,
            shown=choice.action if choice is not None else "invalid",
        )
    _report(f"classify | {approach} | {model}", outcomes)
    assert all(o.valid for o in outcomes.values()), "every answer must parse into a catalog action"


@pytest.mark.parametrize("approach", list(RUNNERS))
def test_split_returns_a_list_of_requests(model: str, approach: str) -> None:
    outcomes: dict[str, Outcome] = {}
    for sentence, expected in SPLIT_CASES.items():
        start = time.perf_counter()
        split, retries = RUNNERS[approach](model, _split_prompt(sentence), SplitRequests)
        requests = split.requests if split is not None else []
        correct = len(requests) == len(expected) and all(
            all(word in line.lower() for word in words) for line, words in zip(requests, expected)
        )
        outcomes[sentence] = Outcome(
            valid=split is not None,
            correct=correct,
            retries=retries,
            seconds=time.perf_counter() - start,
            shown=" | ".join(requests) if split is not None else "invalid",
        )
    _report(f"split | {approach} | {model}", outcomes)
    assert all(o.valid for o in outcomes.values()), "every answer must parse into a list of requests"
