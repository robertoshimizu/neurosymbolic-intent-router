"""Split a sentence into separate requests with local MedGemma."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from contracts import Splitter

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


class MedGemmaSplitter(Splitter):
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
