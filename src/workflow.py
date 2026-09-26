"""One user request, from its text to its effect on the ledger.

`routing` is one state that holds the neuro-symbolic step: the models interpret
the text, the rules decide the action. Its only exit is the rules' decision (deny or execute).
`executing` then re-checks the ledger's facts before anything moves.
"""

from __future__ import annotations

from statemachine import State, StateChart

from contracts import Classifier, Explainer, Labeler, Policy, Splitter
from policy import Session
from router import Decision, route
from transfer import Ledger, WireTransfer, authorize_or_refuse

READ_ONLY_ACTIONS = frozenset({"view_account_balance", "view_public_faq"})


class RequestWorkflow(StateChart):
    """received → routing → refused | executing → completed | failed."""

    allow_event_without_transition = False

    received = State(initial=True)
    routing = State()
    refused = State(final=True)
    executing = State()
    completed = State(final=True)
    failed = State(final=True)

    start = received.to(routing)
    routed = routing.to(refused, cond="is_denied") | routing.to(executing, cond="is_approved")
    finish = executing.to(completed)
    fail = executing.to(failed)

    def __init__(
        self,
        utterance: str,
        session: Session,
        ledger: Ledger,
        request_id: str,
        *,
        policy: Policy,
        classifier: Classifier | None = None,
        labeler: Labeler | None = None,
        splitter: Splitter | None = None,
        explainer: Explainer | None = None,
    ) -> None:
        self.utterance = utterance
        self.session = session
        self.ledger = ledger
        self.request_id = request_id
        self.policy = policy
        self.roles = {"classifier": classifier, "labeler": labeler, "splitter": splitter, "explainer": explainer}
        self.decision: Decision | None = None
        self.note = ""
        self.path: list[str] = []
        super().__init__()

    def on_enter_state(self, target: State) -> None:
        self.path.append(target.id)

    def on_enter_routing(self) -> None:
        """Neural interpret + symbolic decide, as one step. The rules' decision is the only way out."""
        self.decision = route(
            self.utterance, self.session, self.ledger,
            policy=self.policy, classifier=self.roles["classifier"], labeler=self.roles["labeler"],
            splitter=self.roles["splitter"], explainer=self.roles["explainer"],
        )
        self.send("routed")

    def is_denied(self) -> bool:
        return self.decision is not None and self.decision.outcome == "deny"

    def is_approved(self) -> bool:
        return self.decision is not None and self.decision.outcome == "execute"

    def on_enter_refused(self) -> None:
        self.note = self.decision.reason if self.decision is not None else "no decision"

    def on_enter_executing(self) -> None:
        try:
            ok, self.note = self._run()
        except Exception as exc:
            ok, self.note = False, f"execution error: {exc}"
        self.send("finish" if ok else "fail")

    def _run(self) -> tuple[bool, str]:
        """Carry out the approved action. Facts are re-checked here, not trusted from the decision."""
        assert self.decision is not None
        action = self.decision.action
        account = self.session.account_id
        if action in READ_ONLY_ACTIONS:
            return True, f"{action} shown"
        if action == "wire_transfer_funds":
            amount = self.decision.parsed.amount
            if amount is None:
                return False, "no amount to transfer"
            transfer = WireTransfer(ledger=self.ledger, account_id=account, amount=amount, transfer_id=self.request_id)
            if not authorize_or_refuse(transfer):
                return False, "account closed or funds no longer cover the transfer"
            transfer.send("submit")
            transfer.send("settle")
            return True, f"wire of ${amount} settled"
        if action == "delete_account":
            if self.ledger.is_closed(account):
                return False, "account already closed"
            self.ledger.close(account)
            return True, "account closed"
        return False, f"no executor for {action!r}"
