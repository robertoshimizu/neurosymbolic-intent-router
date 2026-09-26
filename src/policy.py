"""Banking domain and the Python policy rules. No model code: the action arrives already chosen."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Literal

from text_to_num import text2num

NONE_ACTION = "none"

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


# "$1,500", "$750.25", "1500 USD", "500 dollars": a whole figure with a dollar sign or word, nothing else.
PLAIN_DOLLARS_RE = re.compile(
    r"(?:\$\s*(?P<a>\d{1,3}(?:,\d{3})+|\d+)(?P<a_cents>\.\d{1,2})?(?:\s*(?:usd|dollars?|bucks))?"
    r"|(?P<b>\d{1,3}(?:,\d{3})+|\d+)(?P<b_cents>\.\d{1,2})?\s*(?:usd|dollars?|bucks))"
)
DOLLAR_WORDS_RE = re.compile(r"\s+(?:usd|dollars?|bucks)$")


def to_dollars(span: str) -> Decimal:
    """Convert an extracted amount span to US dollars. Raises ValueError rather than guess.

    Accepts a plain dollar figure ("$1,500", "750.25 dollars") or number words with an optional
    dollar word ("one thousand, five hundred dollars"). Refuses scales and shorthand ("$1.5k",
    "$1 thousand", "a thousand"), other currencies, and anything text2num cannot read in full.
    """
    text = " ".join(span.strip().lower().split())
    plain = PLAIN_DOLLARS_RE.fullmatch(text)
    if plain:
        whole = plain.group("a") or plain.group("b")
        cents = plain.group("a_cents") or plain.group("b_cents") or ""
        return Decimal(whole.replace(",", "") + cents)
    words = DOLLAR_WORDS_RE.sub("", text).replace(",", "")
    try:
        return Decimal(text2num(words, "en"))
    except ValueError as exc:
        raise ValueError(f"amount {span!r} is neither a plain dollar figure nor a number in words") from exc


def to_request(
    amounts: tuple[str, ...], payees: tuple[str, ...], payee_allowlist: tuple[str, ...]
) -> ParsedRequest:
    """Turn extracted spans into the facts the rules check. Anything but exactly one readable amount
    and exactly one allowlisted payee leaves that fact empty, and the rules deny."""
    amount: Decimal | None = None
    if len(amounts) == 1:
        try:
            amount = to_dollars(amounts[0])
        except ValueError:
            amount = None

    payee: str | None = None
    if len(payees) == 1:
        wanted = payees[0].strip().lower()
        payee = next((label for label in payee_allowlist if label.lower() == wanted), None)

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
    return None if parsed.payee is not None else "payee is missing, ambiguous or not on the allowlist"


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

# Ties keep the written order. `none` and unknown actions go last.
ACTION_PRECEDENCE: dict[str, int] = {action: rule.precedence for action, rule in RULES.items()}


def ordered(actions: tuple[str, ...]) -> tuple[int, ...]:
    """Positions of `actions` by precedence: reads, then money movement, then deletion, then the rest."""
    last = len(ACTION_PRECEDENCE)
    return tuple(sorted(range(len(actions)), key=lambda i: ACTION_PRECEDENCE.get(actions[i], last)))


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


# When `action` fails `check`, offer `suggested`, but only if the policy permits it.
SUGGESTIONS: tuple[tuple[str, Check, str], ...] = (
    ("wire_transfer_funds", require_funds, "view_account_balance"),
)


def suggestion(
    action: str,
    session: Session,
    balance: Decimal,
    parsed: ParsedRequest,
) -> str | None:
    """The first permitted action to offer after a denial, or None."""
    for denied_action, check, suggested in SUGGESTIONS:
        if (
            action == denied_action
            and check(session, balance, parsed) is not None
            and not denials(suggested, session, balance, parsed)
        ):
            return suggested
    return None


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
