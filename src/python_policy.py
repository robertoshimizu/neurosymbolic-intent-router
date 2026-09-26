"""The Policy role in plain Python: the RULES table in policy.py."""

from __future__ import annotations

from decimal import Decimal

from contracts import Policy, PolicyResult
from policy import RULES, ParsedRequest, Session, denials


class PythonPolicy(Policy):
    """The reference reasoner. Other reasoners must agree with it."""

    def evaluate(
        self, action: str, session: Session, balance: Decimal, parsed: ParsedRequest
    ) -> PolicyResult:
        reasons = denials(action, session, balance, parsed)
        if reasons:
            return PolicyResult(allowed=False, reasons=reasons)
        return PolicyResult(allowed=True, reasons=(RULES[action].permitted,))
