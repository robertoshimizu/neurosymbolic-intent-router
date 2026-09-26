"""Ask local MedGemma what a sentence supports. No application catalog."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

SECTIONS = (
    "entities",
    "relationships",
    "events",
    "causal_relationships",
    "concepts",
    "implicit_facts",
    "temporal_relationships",
)

ACTION_FOCUS = """Prioritize an action-oriented reading for a downstream application. Identify every possible request or question, including coordinated requests, and preserve the speech act (question, request, prohibition, assertion, or mere mention). For each mentioned event, distinguish completed/asserted, prospective, hypothetical, conditional, or unknown status. Preserve action arguments such as amount, currency, account, and destination only when the text supplies them; otherwise mark them unknown or missing. Do not answer questions or convert prospective mentions into instructions. Use clear JSON fields of your choice for these distinctions. """

INSTRUCTIONS = """Extract the entities, relationships, events, causal relationships, concepts, implicit facts, and temporal relationships from the text below. For every extracted statement, distinguish its epistemic status as EXPLICIT if it is directly stated in the text, ENTAILED if it necessarily follows from the text, INFERRED if it is a plausible interpretation but is not necessarily true, or UNKNOWN if it cannot be established from the available text. Preserve uncertainty and ambiguity rather than resolving them through assumptions. Explicitly represent question and request context and whether a mentioned event is prospective rather than asserted to have occurred; attach these distinctions to the relevant relationships or events using clear JSON fields of your choice. An explicitly mentioned event is not necessarily an asserted occurrence or a request to perform it. Preserve what a question asks and its unknown answer without answering it. Do not use external or general world knowledge to turn plausible explanations into facts. Report a causal relationship only when the text expresses a causal connection, preserving any uncertainty expressed about that connection. When the text expresses only temporal order, report that order and leave causal relationships empty. Words such as "after," "shortly after," and "subsequently" express time, not cause. Do not supply causal explanations from temporal order, proximity, correlation, or world knowledge, even labeled INFERRED. Do not invent ownership, organizational relationships, motivations, dependencies, intentions, capabilities, locations, financial relationships, or other unstated facts. Extract higher-level concepts and patterns only when they are supported by the text, clearly distinguishing them from literal entities and events. Resolve pronouns and references only when their antecedents are sufficiently supported; otherwise preserve the ambiguity. Prefer precision over completeness: omitting an uncertain extraction is better than presenting speculation as fact. Keep the output concise, avoid explaining your reasoning, do not repeat information across sections unnecessarily, and return only the structured extraction. Never assume that two entities belong to the same organization merely because they appear in the same passage. Never assign ownership, employment, management, financing, or pronoun antecedents unless the linguistic evidence uniquely supports that assignment. If multiple antecedents are possible, mark the relationship AMBIGUOUS rather than selecting one. TEXT TO ANALYZE: """


def prompt_for(text: str) -> str:
    """Same instructions for every text. Only the analyzed passage changes."""
    return f"{ACTION_FOCUS}{INSTRUCTIONS}{text.strip()}"


@dataclass(frozen=True)
class Reading:
    raw: dict

    def section(self, name: str) -> list:
        value = self.raw.get(name, [])
        return value if isinstance(value, list) else []


def parse_model_json(text: str) -> dict:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model JSON must be an object")
    return payload


def read_sentence(
    text: str,
    *,
    model: str = "medgemma:27b",
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    timeout_s: float = 180,
) -> Reading:
    """One user message, thinking off, matching `ollama run --think=false`."""
    content = _chat(prompt_for(text), model=model, endpoint=endpoint, timeout_s=timeout_s, as_json=True)
    return Reading(raw=parse_model_json(content))


SPLIT_INSTRUCTIONS = """Rewrite the text below as the separate things it asks the system to do. Write each request as one short, self-contained imperative sentence on its own line, using the text's own words. Keep amounts, currencies, accounts, and destinations exactly as written, and do not add any that are not written. Leave out things the text says not to do, background, and future actions that are mentioned only as context. Do not answer, merge, reorder, or add requests. If the text asks for exactly one thing, return exactly one line. Return only the lines, with no numbering, bullets, or other text. TEXT: """


def parse_split(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def split_requests(
    text: str,
    *,
    model: str = "medgemma:27b",
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    timeout_s: float = 180,
) -> tuple[str, ...]:
    """Separate plain requests, in the order written. The router orders them, not the model."""
    content = _chat(
        f"{SPLIT_INSTRUCTIONS}{text.strip()}",
        model=model,
        endpoint=endpoint,
        timeout_s=timeout_s,
        as_json=False,
    )
    return parse_split(content)


class MedGemmaSplitter:
    """The router's Splitter: MedGemma on local Ollama."""

    def __init__(
        self,
        *,
        model: str = "medgemma:27b",
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout_s: float = 180,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def split(self, text: str) -> tuple[str, ...]:
        return split_requests(text, model=self.model, endpoint=self.endpoint, timeout_s=self.timeout_s)


def _chat(content: str, *, model: str, endpoint: str, timeout_s: float, as_json: bool) -> str:
    payload: dict = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": content}],
        "options": {"temperature": 0},
    }
    if as_json:
        payload["format"] = "json"
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            envelope = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama request failed") from exc
    return envelope.get("message", {}).get("content", "")
