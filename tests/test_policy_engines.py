"""Every Policy adapter must agree with the reference PythonPolicy, verdict and reasons, in order.

Needs SWI-Prolog: uv run --env-file .env pytest -m prolog
"""

from __future__ import annotations

import itertools
from decimal import Decimal

import pytest

from contracts import Policy
from policy import ACTION_CATALOG, NONE_ACTION, ParsedRequest, Session
from python_policy import PythonPolicy

ACTIONS = (*ACTION_CATALOG, NONE_ACTION, "bogus_action")
AMOUNTS = (None, Decimal("0"), Decimal("-1"), Decimal("500"), Decimal("500.10"), Decimal("20000"))
BALANCES = (Decimal("0"), Decimal("500.10"), Decimal("10000"))


def _cases() -> list[tuple[str, Session, Decimal, ParsedRequest]]:
    cases = []
    for action, authenticated, role, status, amount, payee, balance in itertools.product(
        ACTIONS,
        (True, False),
        ("admin", "customer"),
        ("active", "frozen"),
        AMOUNTS,
        (None, "external bank account"),
        BALANCES,
    ):
        session = Session("s", authenticated, role, status, "acct")
        cases.append((action, session, balance, ParsedRequest(amount, payee)))
    return cases


@pytest.fixture(scope="module")
def prolog_policy() -> Policy:
    from prolog_policy import PrologPolicy  # needs SWI-Prolog; imported only by this test

    return PrologPolicy()


@pytest.mark.prolog
def test_prolog_policy_agrees_with_python_policy(prolog_policy: Policy) -> None:
    reference = PythonPolicy()
    cases = _cases()
    mismatches = [
        (action, session, balance, parsed, expected, got)
        for action, session, balance, parsed in cases
        if (expected := reference.evaluate(action, session, balance, parsed))
        != (got := prolog_policy.evaluate(action, session, balance, parsed))
    ]
    assert not mismatches, f"{len(mismatches)} of {len(cases)} cases differ; first: {mismatches[0]}"
