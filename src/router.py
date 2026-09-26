"""Neuro-symbolic action router: a Classifier picks the action, policy judges it."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal

import policy
from contracts import Classifier, DescriptionMatch, Explainer, IntentRank, Labeler, Splitter
from policy import (
    ACTION_CATALOG,
    ACTION_PRECEDENCE,
    NONE_ACTION,
    Outcome,
    ParsedRequest,
    Session,
    evaluate_action,
    parse_request,
)
from transfer import Ledger


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
    source: str | None = None
    description_match: DescriptionMatch | None = None
    follow_ups: tuple[str, ...] = ()


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


def _classify(utterance: str, classifier: Classifier | None) -> IntentRank | None:
    """None when there is no classifier or it is unavailable. Nothing guesses in its place."""
    if classifier is None:
        return None
    try:
        return classifier.classify(utterance)
    except Exception:
        return None


def _unavailable(utterance: str, session: Session) -> Decision:
    return Decision(
        action=NONE_ACTION,
        outcome="deny",
        reason="classifier unavailable",
        parsed=parse_request(utterance, session.payee_allowlist),
        scores={},
        permissions={},
        rule_trace=["classifier=unavailable", "abstain=before_policy"],
    )


def decide(
    utterance: str,
    session: Session,
    ledger: Ledger,
    *,
    classifier: Classifier | None = None,
    explainer: Explainer | None = None,
    rank: IntentRank | None = None,
) -> Decision:
    """Classify the utterance, judge the action, and return a Decision.

    No classifier answer, or a `none` action, stops before policy. A denied
    catalog intent stays denied and is never replaced by FAQ. A precomputed
    `rank` skips classifying.
    """
    rank = rank or _classify(utterance, classifier)
    if rank is None:
        return _unavailable(utterance, session)
    balance = ledger.get_balance(session.account_id)
    scores = rank.scores

    if rank.action == NONE_ACTION:
        return Decision(
            action=NONE_ACTION,
            outcome="deny",
            reason="no matching action",
            parsed=parse_request(utterance, session.payee_allowlist),
            scores=scores,
            permissions={},
            rule_trace=[
                f"source={rank.source} action={rank.action} confidence={rank.confidence:.4f}",
                "abstain=before_policy",
            ],
            source=rank.source,
        )

    top_action = rank.action
    description_match = None
    explainer_failed = False
    if explainer is not None and top_action in ACTION_CATALOG:
        try:
            description_match = explainer.explain(utterance, top_action)
        except Exception:
            # Display only: a broken explainer loses the note, never the decision.
            explainer_failed = True

    catalog_actions = [action for action in scores if action in ACTION_CATALOG]
    if top_action not in ACTION_CATALOG:
        catalog_actions = list(ACTION_CATALOG)
    verdict = policy.decide(utterance, top_action, session, ledger)
    parsed = verdict.parsed
    policy_reason = verdict.reason
    permissions, _reasons = evaluate_permissions(
        session, balance, parsed, catalog_actions or [top_action]
    )

    rule_trace = [
        f"source={rank.source}",
        f"top_raw_intent={top_action} confidence={rank.confidence:.4f}",
        f"balance={balance}",
        f"parsed_amount={parsed.amount}",
        f"parsed_payee={parsed.payee}",
        f"policy[{top_action}]={policy_reason}",
    ]
    if explainer_failed:
        rule_trace.append("explainer=unavailable")
    if description_match is not None:
        rule_trace.append(
            f"explainer_nearest={description_match.nearest} "
            f"score={description_match.nearest_score:.4f}"
        )
        rule_trace.append(
            f"explainer_chosen_score={description_match.chosen_score:.4f}"
        )
        if description_match.agrees and not description_match.description_gap:
            rule_trace.append("explainer_agrees=yes")
        elif not description_match.agrees:
            rule_trace.append("description_gap=nearest_differs")
        else:
            rule_trace.append("description_gap=chosen_score_low")

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
            source=rank.source,
            description_match=description_match,
        )

    if verdict.outcome == "deny":
        return _decision("deny", policy_reason, suggestion=verdict.suggestion)

    if verdict.outcome == "needs_confirmation":
        return _decision(
            "needs_confirmation",
            policy_reason,
            trace_extra=["high_stakes=confirmation_required"],
        )

    return _decision("execute", policy_reason)


def _deny_split(utterance: str, session: Session, reason: str, trace: list[str]) -> Decision:
    return Decision(
        action=NONE_ACTION,
        outcome="deny",
        reason=reason,
        parsed=parse_request(utterance, session.payee_allowlist),
        scores={},
        permissions={},
        rule_trace=["request_count=several", *trace],
    )


def route(
    utterance: str,
    session: Session,
    ledger: Ledger,
    *,
    classifier: Classifier | None = None,
    labeler: Labeler | None = None,
    splitter: Splitter | None = None,
    explainer: Explainer | None = None,
) -> Decision:
    """One classifier call. Several requests: the splitter separates them, the labeler labels them in one call,
    the rule orders, policy decides the first, and the rest are listed.
    Never mutates the ledger.
    """
    rank = _classify(utterance, classifier)
    if rank is None:
        return _unavailable(utterance, session)
    if rank.request_count != "several":
        return decide(utterance, session, ledger, explainer=explainer, rank=rank)

    try:
        requests = splitter.split(utterance) if splitter is not None else ()
    except Exception:
        requests = ()
    if len(requests) < 2:
        return _deny_split(utterance, session, "several requests could not be separated", [f"split_count={len(requests)}"])

    try:
        ranks = labeler.label(requests) if labeler is not None else None
    except Exception:
        ranks = None
    if ranks is None or len(ranks) != len(requests):
        return _deny_split(utterance, session, "several requests could not be labeled", [f"split_count={len(requests)}"])

    labels = [r.action for r in ranks]
    ordered = sorted(range(len(requests)), key=lambda i: ACTION_PRECEDENCE.get(
        labels[i], len(ACTION_PRECEDENCE)))
    head = ordered[0]
    first = decide(requests[head], session, ledger,
                   explainer=explainer, rank=ranks[head])
    trace = [
        "request_count=several",
        *(f"split[{i}]={requests[i]!r} label={labels[i]} confidence={ranks[i].confidence:.4f}" for i in range(len(requests))),
        f"logical_order={[labels[i] for i in ordered]}",
    ]
    follow_ups = () if first.outcome == "deny" else tuple(
        requests[i] for i in ordered[1:])
    return replace(first, rule_trace=trace + first.rule_trace, follow_ups=follow_ups)
