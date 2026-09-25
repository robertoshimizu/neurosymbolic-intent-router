"""Neuro-symbolic action router: MiniLM ranking + session-backed policy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv

from transfer import Ledger, WireTransfer, confirm_or_refuse

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_ENV_LOADED = False


def _load_env() -> None:
    """Load project .env so HF_TOKEN is available to the Hugging Face Hub client."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)
    _ENV_LOADED = True


_load_env()
MIN_SCORE = 0.45
MIN_MARGIN = 0.08

HIGH_STAKES_ACTIONS = frozenset({"wire_transfer_funds", "delete_account"})

ACTION_CATALOG: dict[str, str] = {
    "wire_transfer_funds": (
        "Send money by wire transfer to an external bank account"
    ),
    "view_account_balance": "Show the current account balance",
    "delete_account": "Permanently delete the user account",
    "view_public_faq": "Open the public frequently asked questions page",
}

AMOUNT_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")

Outcome = Literal["deny", "execute", "needs_confirmation"]


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
class Decision:
    action: str | None
    outcome: Outcome
    reason: str
    parsed: ParsedRequest
    scores: dict[str, float]
    permissions: dict[str, bool]
    rule_trace: list[str] = field(default_factory=list)
    suggestion: str | None = None


def parse_request(
    utterance: str, payee_allowlist: tuple[str, ...]
) -> ParsedRequest:
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
            return False, "caller is not signed in"
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
            return False, "caller is not signed in"
        return True, "balance view permitted"

    if action == "view_public_faq":
        return True, "public faq always allowed"

    return False, "unknown action denied"


def evaluate_permissions(
    session: Session,
    balance: Decimal,
    parsed: ParsedRequest,
    actions: list[str],
) -> tuple[dict[str, bool], dict[str, str]]:
    permissions: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for action in actions:
        allowed, reason = evaluate_action(action, session, balance, parsed)
        permissions[action] = allowed
        reasons[action] = reason
    return permissions, reasons


def cosine_scores(
    query_vector: np.ndarray, action_embeddings: dict[str, np.ndarray]
) -> dict[str, float]:
    query_norm = query_vector / np.linalg.norm(query_vector)
    scores: dict[str, float] = {}
    for action_id, action_vec in action_embeddings.items():
        action_norm = action_vec / np.linalg.norm(action_vec)
        scores[action_id] = float(np.dot(query_norm, action_norm))
    return scores


class ActionEmbedder:
    """Lazy MiniLM encoder with cached action-description embeddings."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._action_embeddings: dict[str, np.ndarray] | None = None

    def _load_model(self):
        if self._model is None:
            _load_env()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        vector = self._load_model().encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float64)

    def action_embeddings(
        self, catalog: dict[str, str] | None = None
    ) -> dict[str, np.ndarray]:
        catalog = catalog or ACTION_CATALOG
        if self._action_embeddings is None:
            self._action_embeddings = {
                action_id: self.embed(description)
                for action_id, description in catalog.items()
            }
        return self._action_embeddings


def decide(
    utterance: str,
    session: Session,
    ledger: Ledger,
    *,
    query_vector: np.ndarray | None = None,
    action_embeddings: dict[str, np.ndarray] | None = None,
    embedder: ActionEmbedder | None = None,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
) -> Decision:
    """Rank the utterance, judge the top raw intent, and return a Decision.

    A denied top intent stays denied. It is never replaced by FAQ.
    """
    if action_embeddings is None or query_vector is None:
        embedder = embedder or ActionEmbedder()
        action_embeddings = action_embeddings or embedder.action_embeddings()
        query_vector = (
            query_vector if query_vector is not None else embedder.embed(utterance)
        )

    scores = cosine_scores(query_vector, action_embeddings)
    ranked = sorted(scores, key=scores.get, reverse=True)
    top_action = ranked[0]
    top_score = scores[top_action]
    runner_up_score = scores[ranked[1]] if len(ranked) > 1 else 0.0

    balance = ledger.get_balance(session.account_id)
    parsed = parse_request(utterance, session.payee_allowlist)
    permissions, reasons = evaluate_permissions(
        session, balance, parsed, list(scores.keys())
    )

    rule_trace = [
        f"top_raw_intent={top_action} score={top_score:.4f}",
        f"runner_up_score={runner_up_score:.4f}",
        f"balance={balance}",
        f"parsed_amount={parsed.amount}",
        f"parsed_payee={parsed.payee}",
        f"policy[{top_action}]={reasons[top_action]}",
    ]

    def _decision(
        outcome: Outcome,
        reason: str,
        *,
        suggestion: str | None = None,
        trace_extra: list[str] | None = None,
    ) -> Decision:
        return Decision(
            action=top_action,
            outcome=outcome,
            reason=reason,
            parsed=parsed,
            scores=scores,
            permissions=permissions,
            rule_trace=rule_trace if not trace_extra else rule_trace + trace_extra,
            suggestion=suggestion,
        )

    if not permissions[top_action]:
        suggestion = None
        if (
            top_action == "wire_transfer_funds"
            and "insufficient funds" in reasons[top_action]
        ):
            suggestion = "view_account_balance"
        return _decision("deny", reasons[top_action], suggestion=suggestion)

    if top_score < min_score:
        return _decision(
            "deny",
            "similarity below minimum confidence",
            trace_extra=["confidence=below_min_score"],
        )

    if len(ranked) > 1 and (top_score - runner_up_score) < min_margin:
        return _decision(
            "deny",
            "ambiguous intent: margin below minimum",
            trace_extra=["confidence=below_min_margin"],
        )

    if top_action in HIGH_STAKES_ACTIONS:
        return _decision(
            "needs_confirmation",
            reasons[top_action],
            trace_extra=["high_stakes=confirmation_required"],
        )

    return _decision("execute", reasons[top_action])


def print_decision(utterance: str, session: Session, decision: Decision) -> None:
    print(f"\nUser Query: '{utterance}'")
    print(
        f"Context: Auth={session.is_authenticated}, Role='{session.role}', "
        f"Status='{session.status}', Account='{session.account_id}'"
    )
    print("\nAction Decision Matrix:")
    print(
        f"{'Candidate Action':<24} | {'Raw Similarity':<15} | {'Allowed?':<8}"
    )
    print("-" * 55)
    for act, score in sorted(
        decision.scores.items(), key=lambda item: item[1], reverse=True
    ):
        allowed_str = "YES" if decision.permissions[act] else "NO"
        print(f"{act:<24} | {score:<15.4f} | {allowed_str:<8}")

    print(
        f"\nDecision: action={decision.action} outcome={decision.outcome} "
        f"reason={decision.reason}"
    )
    if decision.suggestion:
        print(f"Suggestion: {decision.suggestion}")
    print()


def build_demo_world() -> tuple[dict[str, Session], Ledger]:
    ledger = Ledger()
    ledger.set_balance("acct-guest", Decimal("0"))
    ledger.set_balance("acct-zero", Decimal("0"))
    ledger.set_balance("acct-funded", Decimal("10000"))

    sessions = {
        "guest": Session(
            session_id="guest",
            is_authenticated=False,
            role="guest",
            status="inactive",
            account_id="acct-guest",
            payee_allowlist=(),
        ),
        "zero": Session(
            session_id="zero",
            is_authenticated=True,
            role="customer",
            status="active",
            account_id="acct-zero",
            payee_allowlist=("external bank account",),
        ),
        "funded": Session(
            session_id="funded",
            is_authenticated=True,
            role="customer",
            status="active",
            account_id="acct-funded",
            payee_allowlist=("external bank account",),
        ),
    }
    return sessions, ledger


if __name__ == "__main__":
    utterance = "I want to send $5,000 to my external bank account."
    sessions, ledger = build_demo_world()
    embedder = ActionEmbedder()

    for key in ("guest", "zero", "funded"):
        session = sessions[key]
        decision = decide(utterance, session, ledger, embedder=embedder)
        balance = ledger.get_balance(session.account_id)
        print(f"\n--- Session '{key}' (balance=${balance}) ---")
        print_decision(utterance, session, decision)

        if decision.outcome == "needs_confirmation" and decision.action == (
            "wire_transfer_funds"
        ):
            assert decision.parsed.amount is not None
            transfer = WireTransfer(
                ledger=ledger,
                account_id=session.account_id,
                amount=decision.parsed.amount,
                transfer_id="wire-demo-1",
            )
            transfer.send("request_confirmation")
            confirmed = confirm_or_refuse(transfer)
            print(f"Confirmation accepted: {confirmed}")
            if confirmed:
                transfer.send("submit")
                transfer.send("settle")
                print(
                    f"Settled. New balance: ${ledger.get_balance(session.account_id)}"
                )
