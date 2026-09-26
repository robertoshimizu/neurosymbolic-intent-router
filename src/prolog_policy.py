"""The Policy role in SWI-Prolog: the rules in policy.pl, queried through janus.

The SWI-Prolog.app runtime needs DYLD_FALLBACK_LIBRARY_PATH set when the
process starts, so run with `uv run --env-file .env`.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import janus_swi as janus

from contracts import Policy, PolicyResult
from policy import ParsedRequest, Session

RULES_FILE = Path(__file__).with_name("policy.pl")


class PrologPolicy(Policy):
    """Asks policy:denials/3 for every reason an action is denied, policy:suggestion/3 for what to offer,
    and policy:ordered/2 for the handling order of several requests."""

    def __init__(self, rules_file: Path = RULES_FILE) -> None:
        janus.consult(str(rules_file))

    def evaluate(
        self, action: str, session: Session, balance: Decimal, parsed: ParsedRequest
    ) -> PolicyResult:
        facts = {
            "authenticated": session.is_authenticated,
            "status": session.status,
            "role": session.role,
            "amount": None if parsed.amount is None else Fraction(parsed.amount),
            "payee": parsed.payee,
            "balance": Fraction(balance),
        }
        answer = janus.query_once("policy:denials(A, F, R)", {"A": action, "F": facts})
        if not answer["truth"]:
            raise RuntimeError(f"policy:denials/3 failed for {action!r}")
        reasons = answer["R"]
        if not isinstance(reasons, list):
            raise ValueError(f"policy:denials/3 returned {reasons!r}, not a list")
        if reasons:
            offer = janus.query_once("policy:suggestion(A, F, S)", {"A": action, "F": facts})
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
