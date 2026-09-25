"""Ask a local model what a sentence supports. No application catalog."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

ESTABLISHED = frozenset({"EXPLICIT", "ENTAILED"})

PROMPT = """Extract the entities, relationships, events, causal relationships, concepts, implicit facts, and temporal relationships from the text. For every extracted statement, distinguish its epistemic status as EXPLICIT if it is directly stated in the text, ENTAILED if it necessarily follows from the text, INFERRED if it is a plausible interpretation but is not necessarily true, or UNKNOWN if it cannot be established from the available text. Preserve uncertainty and ambiguity rather than resolving them through assumptions. Do not use external or general world knowledge to turn plausible explanations into facts. Do not infer causality merely from temporal sequence, proximity, correlation, or words such as "after," "shortly after," or "subsequently"; report causality as explicit only when the text establishes a causal relationship, otherwise classify it as INFERRED. Do not invent ownership, organizational relationships, motivations, dependencies, intentions, capabilities, locations, financial relationships, or other unstated facts. Extract higher-level concepts and patterns only when they are supported by the text, clearly distinguishing them from literal entities and events. Resolve pronouns and references only when their antecedents are sufficiently supported; otherwise preserve the ambiguity. Prefer precision over completeness: omitting an uncertain extraction is better than presenting speculation as fact. Keep the output concise, avoid explaining your reasoning, do not repeat information across sections unnecessarily, and return only the structured extraction. Never assume that two entities belong to the same organization merely because they appear in the same passage. Never assign ownership, employment, management, financing, or pronoun antecedents unless the linguistic evidence uniquely supports that assignment. If multiple antecedents are possible, mark the relationship AMBIGUOUS rather than selecting one.

Return JSON only, with this shape:
{"claims": [{"statement": "<short claim>", "status": "<status>", "kind": "<kind>"}]}

status is one of: EXPLICIT, ENTAILED, INFERRED, UNKNOWN, AMBIGUOUS
kind is one of: entity, relationship, event, causal, concept, implicit_fact, temporal, intent
"""


@dataclass(frozen=True)
class Reading:
    claims: tuple[dict, ...]

    def established(self) -> tuple[dict, ...]:
        return tuple(
            claim
            for claim in self.claims
            if str(claim.get("status", "")).upper() in ESTABLISHED
        )


def parse_model_json(text: str) -> dict:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model JSON must be an object")
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    return payload


def reading_from_payload(payload: dict) -> Reading:
    claims = []
    for item in payload.get("claims", []):
        if isinstance(item, dict) and item.get("statement"):
            claims.append(
                {
                    "statement": str(item["statement"]),
                    "status": str(item.get("status", "")).upper(),
                    "kind": str(item.get("kind", "")).lower(),
                }
            )
    return Reading(claims=tuple(claims))


def read_sentence(
    sentence: str,
    *,
    model: str = "medgemma:27b",
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    timeout_s: float = 180,
) -> Reading:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": sentence},
            ],
            "options": {"temperature": 0},
        }
    ).encode()
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
    payload = parse_model_json(envelope.get("message", {}).get("content", ""))
    return reading_from_payload(payload)
