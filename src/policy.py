"""Banking domain and policy rules. No model code: the action arrives already chosen."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from transfer import Ledger

NONE_ACTION = "none"

ACTION_CATALOG: dict[str, str] = {
    "wire_transfer_funds": "Send money by wire transfer to an external bank account",
    "view_account_balance": "Show the current account balance",
    "delete_account": "Permanently delete the user account",
    "view_public_faq": "Open the public frequently asked questions page",
}


# Logical order for several requests: reads, then money movement, then deletion.
# Ties keep the written order. `none` goes last so a real action runs first.
ACTION_PRECEDENCE: dict[str, int] = {
    "view_public_faq": 0,
    "view_account_balance": 0,
    "wire_transfer_funds": 1,
    "delete_account": 2,
}

AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")

Outcome = Literal["deny", "execute"]


@dataclass(frozen=True)
class Session:
    """Identity and policy facts loaded by session id. Not taken from the utterance."""

    session_id: str
    is_authenticated: bool
    role: str
    status: str
    account_id: str
    payee_allowlist: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedRequest:
    amount: Decimal | None
    payee: str | None


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    reason: str
    parsed: ParsedRequest
    suggestion: str | None = None


def parse_request(utterance: str, payee_allowlist: tuple[str, ...]) -> ParsedRequest:
    """Extract a dollar amount and a registered payee label from the sentence."""
    amount: Decimal | None = None
    match = AMOUNT_RE.search(utterance)
    if match:
        amount = Decimal(match.group(1).replace(",", ""))

    payee: str | None = None
    lower = utterance.lower()
    for label in payee_allowlist:
        if label.lower() in lower:
            payee = label
            break

    return ParsedRequest(amount=amount, payee=payee)


def evaluate_action(
    action: str,
    session: Session,
    balance: Decimal,
    parsed: ParsedRequest,
) -> tuple[bool, str]:
    """Return (allowed, reason) for one catalog action against session facts."""
    if action == "wire_transfer_funds":
        if not session.is_authenticated:
            return False, "caller is not authenticated"
        if session.status != "active":
            return False, "account is not active"
        if parsed.amount is None or parsed.amount <= 0:
            return False, "transfer amount is missing or invalid"
        if parsed.payee is None:
            return False, "payee is not on the allowlist"
        if parsed.amount > balance:
            return False, "insufficient funds for the requested amount"
        return True, "wire transfer permitted"

    if action == "delete_account":
        if not (session.is_authenticated and session.role == "admin"):
            return False, "delete requires an authenticated admin"
        return True, "account deletion permitted"

    if action == "view_account_balance":
        if not session.is_authenticated:
            return False, "caller is not authenticated"
        return True, "balance view permitted"

    if action == "view_public_faq":
        return True, "public faq always allowed"

    return False, "unknown action denied"


def decide(text: str, action: str, session: Session, ledger: Ledger) -> Verdict:
    """Judge one already-chosen action: deny, or execute."""
    parsed = parse_request(text, session.payee_allowlist)
    if action == NONE_ACTION:
        return Verdict("deny", "no matching action", parsed)
    allowed, reason = evaluate_action(action, session, ledger.get_balance(session.account_id), parsed)
    if not allowed:
        suggestion = (
            "view_account_balance"
            if action == "wire_transfer_funds" and "insufficient funds" in reason
            else None
        )
        return Verdict("deny", reason, parsed, suggestion)
    return Verdict("execute", reason, parsed)
