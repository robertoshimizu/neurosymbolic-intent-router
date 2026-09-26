"""Banking domain and the Python policy rules. No model code: the action arrives already chosen."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Literal

NONE_ACTION = "none"

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


# A check returns a denial reason, or None when it passes.
Check = Callable[[Session, Decimal, ParsedRequest], str | None]


def require_authenticated(session: Session, balance: Decimal, parsed: ParsedRequest) -> str | None:
    return None if session.is_authenticated else "caller is not authenticated"


def require_active(session: Session, balance: Decimal, parsed: ParsedRequest) -> str | None:
    return None if session.status == "active" else "account is not active"


def require_amount(session: Session, balance: Decimal, parsed: ParsedRequest) -> str | None:
    if parsed.amount is None or parsed.amount <= 0:
        return "transfer amount is missing or invalid"
    return None


def require_payee(session: Session, balance: Decimal, parsed: ParsedRequest) -> str | None:
    return None if parsed.payee is not None else "payee is not on the allowlist"


def require_funds(session: Session, balance: Decimal, parsed: ParsedRequest) -> str | None:
    if parsed.amount is not None and parsed.amount > balance:
        return "insufficient funds for the requested amount"
    return None


def require_admin(session: Session, balance: Decimal, parsed: ParsedRequest) -> str | None:
    if session.is_authenticated and session.role == "admin":
        return None
    return "delete requires an authenticated admin"


@dataclass(frozen=True)
class ActionRule:
    """Everything the rules know about one action. Checks run in order; the first failure denies."""

    description: str
    precedence: int
    checks: tuple[Check, ...]
    permitted: str


# Adding an action means adding one entry here.
# Precedence orders several requests: reads, then money movement, then deletion.
RULES: dict[str, ActionRule] = {
    "wire_transfer_funds": ActionRule(
        "Send money by wire transfer to an external bank account",
        1,
        (require_authenticated, require_active, require_amount, require_payee, require_funds),
        "wire transfer permitted",
    ),
    "view_account_balance": ActionRule(
        "Show the current account balance",
        0,
        (require_authenticated,),
        "balance view permitted",
    ),
    "delete_account": ActionRule(
        "Permanently delete the user account",
        2,
        (require_admin,),
        "account deletion permitted",
    ),
    "view_public_faq": ActionRule(
        "Open the public frequently asked questions page",
        0,
        (),
        "public faq always allowed",
    ),
}

ACTION_CATALOG: dict[str, str] = {action: rule.description for action, rule in RULES.items()}

# Ties keep the written order. `none` is absent, so callers sort it last.
ACTION_PRECEDENCE: dict[str, int] = {action: rule.precedence for action, rule in RULES.items()}


def denials(
    action: str,
    session: Session,
    balance: Decimal,
    parsed: ParsedRequest,
) -> tuple[str, ...]:
    """Every reason the action is denied, in check order. Empty means permitted."""
    rule = RULES.get(action)
    if rule is None:
        return ("unknown action denied",)
    reasons = (check(session, balance, parsed) for check in rule.checks)
    return tuple(reason for reason in reasons if reason is not None)


def evaluate_action(
    action: str,
    session: Session,
    balance: Decimal,
    parsed: ParsedRequest,
) -> tuple[bool, str]:
    """Return (allowed, first reason) for one catalog action against session facts."""
    reasons = denials(action, session, balance, parsed)
    if reasons:
        return False, reasons[0]
    return True, RULES[action].permitted
