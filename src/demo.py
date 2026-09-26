"""Demo entry point: wires Jev, MedGemma, GLiNER, MiniLM and a policy reasoner into the router and runs three sessions.

    uv run --env-file .env python src/demo.py [--policy prolog|python]
"""

from __future__ import annotations

import argparse
from decimal import Decimal

from adapters.gliner import GLiNERExtractor
from adapters.medgemma import MedGemmaSplitter
from adapters.jev import JevClassifier
from adapters.minilm import MiniLMExplainer
from contracts import Policy
from policy import Session
from adapters.python_policy import PythonPolicy
from router import Decision
from transfer import Ledger
from workflow import RequestWorkflow


def build_policy(name: str) -> Policy:
    """The only place that knows which reasoner judges actions."""
    if name == "python":
        return PythonPolicy()
    from adapters.prolog_policy import PrologPolicy  # needs SWI-Prolog; imported only when chosen

    return PrologPolicy()


def print_decision(utterance: str, session: Session, decision: Decision) -> None:
    print(f"\nUser Query: '{utterance}'")
    print(
        f"Context: Auth={session.is_authenticated}, Role='{session.role}', "
        f"Status='{session.status}', Account='{session.account_id}'"
    )
    show_similarity = decision.description_match is not None
    print(f"\nClassifier: {decision.source}")
    print("\nAction Decision Matrix:")
    if show_similarity:
        print(
            f"{'Candidate Action':<24} | {'Classifier Score':<16} | "
            f"{'Explainer Score':<15} | {'Allowed?':<8}"
        )
        print("-" * 74)
    else:
        print(
            f"{'Candidate Action':<24} | {'Classifier Score':<16} | {'Allowed?':<8}"
        )
        print("-" * 56)
    for act, score in sorted(
        decision.scores.items(), key=lambda item: item[1], reverse=True
    ):
        if not decision.permissions:
            allowed_str = "—"
        else:
            allowed_str = "YES" if decision.permissions.get(
                act, False) else "NO"
        if decision.description_match is not None:
            explained = decision.description_match.similarities.get(act)
            explained_str = f"{explained:.4f}" if explained is not None else "—"
            print(
                f"{act:<24} | {score:<16.4f} | {explained_str:<15} | {allowed_str:<8}"
            )
        else:
            print(f"{act:<24} | {score:<16.4f} | {allowed_str:<8}")

    print(
        f"\nDecision: action={decision.action} outcome={decision.outcome} "
        f"reason={decision.reason}"
    )
    if decision.description_match is not None:
        match = decision.description_match
        if match.agrees and not match.description_gap:
            print(
                f"Explainer agrees: {match.nearest} score={match.chosen_score:.4f}"
            )
        else:
            print(
                "Description gap: "
                f"explainer nearest={match.nearest} ({match.nearest_score:.4f}); "
                f"chosen action score={match.chosen_score:.4f}"
            )
    if decision.follow_ups:
        print("You also asked (ask again to proceed):")
        for follow_up in decision.follow_ups:
            print(f"  - {follow_up}")
    if decision.suggestion:
        print(f"Suggestion: {decision.suggestion}")
    print()


def build_demo_world() -> tuple[dict[str, Session], Ledger]:
    ledger = Ledger()
    ledger.set_balance("acct-unauthenticated", Decimal("0"))
    ledger.set_balance("acct-zero", Decimal("0"))
    ledger.set_balance("acct-funded", Decimal("10000"))

    sessions = {
        "unauthenticated": Session(
            session_id="unauthenticated",
            is_authenticated=False,
            role="unauthenticated",
            status="inactive",
            account_id="acct-unauthenticated",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("prolog", "python"), default="prolog")
    policy_name = parser.parse_args().policy
    policy = build_policy(policy_name)
    print(f"Policy: {policy_name}")
    utterance = "Close this account and send $500 to my external bank account."
    sessions, ledger = build_demo_world()
    classifier = JevClassifier()
    explainer = MiniLMExplainer()
    splitter = MedGemmaSplitter()
    extractor = GLiNERExtractor()

    for key in ("unauthenticated", "zero", "funded"):
        session = sessions[key]
        balance = ledger.get_balance(session.account_id)
        workflow = RequestWorkflow(
            utterance, session, ledger, request_id=f"request-{key}",
            policy=policy, classifier=classifier, labeler=classifier, splitter=splitter,
            extractor=extractor, explainer=explainer,
        )
        workflow.send("start")
        print(f"\n--- Session '{key}' (balance=${balance}) ---")
        if workflow.decision is not None:
            print_decision(utterance, session, workflow.decision)
        print(f"Workflow: {' -> '.join(workflow.path)} ({workflow.note}). "
              f"Balance now ${ledger.get_balance(session.account_id)}")
