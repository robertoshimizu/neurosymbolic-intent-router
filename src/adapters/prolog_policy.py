"""The Policy role in SWI-Prolog: the rules in policy.pl, queried through janus.

The SWI-Prolog.app runtime needs DYLD_FALLBACK_LIBRARY_PATH set when the
process starts, so run with `uv run --env-file .env`.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import janus_swi as janus

from contracts import Policy, PolicyResult
from policy import ParsedRequest, Session

RULES_FILE = Path(__file__).with_name("policy.pl")


class PrologPolicy(Policy):
    """Asserts one request's facts, asks policy:denials/3 for every reason an action is denied and
    policy:suggestion/3 for what to offer, then forgets the facts. policy:ordered/2 orders several requests."""

    def __init__(self, rules_file: Path = RULES_FILE) -> None:
        janus.consult(str(rules_file))

    def evaluate(
        self, action: str, session: Session, balance: Decimal, parsed: ParsedRequest
    ) -> PolicyResult:
        request = f"r{uuid4().hex}"
        try:
            self._assert_facts(request, session, balance, parsed)
            return self._judge(action, request)
        finally:
            janus.query_once("policy:forget(R)", {"R": request})

    def _assert_facts(self, request: str, session: Session, balance: Decimal, parsed: ParsedRequest) -> None:
        """One Prolog fact per known value. Values are bound, never spliced into Prolog text."""
        facts: list[tuple[str, dict[str, object]]] = [
            ("policy:account_status(R, V)", {"V": session.status}),
            ("policy:role(R, V)", {"V": session.role}),
            ("policy:balance(R, V)", {"V": Fraction(balance)}),
        ]
        if session.is_authenticated:
            facts.append(("policy:authenticated(R)", {}))
        if parsed.amount is not None:
            facts.append(("policy:amount(R, V)", {"V": Fraction(parsed.amount)}))
        if parsed.payee is not None:
            facts.append(("policy:payee(R, V)", {"V": parsed.payee}))
        for fact, values in facts:
            janus.query_once(f"assertz({fact})", {"R": request, **values})

    def _judge(self, action: str, request: str) -> PolicyResult:
        answer = janus.query_once("policy:denials(A, R, Reasons)", {"A": action, "R": request})
        if not answer["truth"]:
            raise RuntimeError(f"policy:denials/3 failed for {action!r}")
        reasons = answer["Reasons"]
        if not isinstance(reasons, list):
            raise ValueError(f"policy:denials/3 returned {reasons!r}, not a list")
        if reasons:
            offer = janus.query_once("policy:suggestion(A, R, S)", {"A": action, "R": request})
            return PolicyResult(
                allowed=False,
                reasons=tuple(reasons),
                suggestion=offer["S"] if offer["truth"] else None,
            )
        permitted = janus.query_once("policy:permitted_message(A, M)", {"A": action})
        if not permitted["truth"]:
            raise RuntimeError(f"policy:permitted_message/2 has no entry for {action!r}")
        return PolicyResult(allowed=True, reasons=(permitted["M"],))

    def order(self, actions: tuple[str, ...]) -> tuple[int, ...]:
        answer = janus.query_once("policy:ordered(A, P)", {"A": list(actions)})
        if not answer["truth"]:
            raise RuntimeError("policy:ordered/2 failed")
        positions = answer["P"]
        if not isinstance(positions, list) or not all(isinstance(p, int) for p in positions):
            raise ValueError(f"policy:ordered/2 returned {positions!r}, not a list of positions")
        return tuple(positions)
