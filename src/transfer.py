"""In-memory ledger and wire-transfer statechart."""

from __future__ import annotations

from decimal import Decimal

from statemachine import State, StateChart
from statemachine.exceptions import TransitionNotAllowed


class Ledger:
    """Account balances with idempotent settlement by transfer id."""

    def __init__(self) -> None:
        self._balances: dict[str, Decimal] = {}
        self._settled: set[str] = set()

    def set_balance(self, account_id: str, amount: Decimal) -> None:
        self._balances[account_id] = amount

    def get_balance(self, account_id: str) -> Decimal:
        return self._balances.get(account_id, Decimal("0"))

    def has_settled(self, transfer_id: str) -> bool:
        return transfer_id in self._settled

    def debit(self, account_id: str, amount: Decimal, transfer_id: str) -> bool:
        """Debit once per transfer_id. Returns True if this call moved money."""
        if transfer_id in self._settled:
            return False
        balance = self.get_balance(account_id)
        if amount > balance:
            raise ValueError(
                f"insufficient funds: need {amount}, have {balance}"
            )
        self._balances[account_id] = balance - amount
        self._settled.add(transfer_id)
        return True


class WireTransfer(StateChart):
    """Lifecycle for a confirmed outbound wire against an in-memory ledger."""

    allow_event_without_transition = False

    drafted = State(initial=True)
    awaiting_confirmation = State()
    authorized = State()
    submitted = State()
    settled = State(final=True)
    declined = State(final=True)

    request_confirmation = drafted.to(awaiting_confirmation)
    confirm = awaiting_confirmation.to(authorized, cond="funds_available")
    submit = authorized.to(submitted)
    settle = submitted.to(settled)
    cancel = (
        drafted.to(declined)
        | awaiting_confirmation.to(declined)
        | authorized.to(declined)
    )

    def __init__(
        self,
        ledger: Ledger,
        account_id: str,
        amount: Decimal,
        transfer_id: str,
        **kwargs,
    ) -> None:
        self.ledger = ledger
        self.account_id = account_id
        self.amount = amount
        self.transfer_id = transfer_id
        super().__init__(**kwargs)

    def funds_available(self) -> bool:
        return self.ledger.get_balance(self.account_id) >= self.amount

    def on_enter_settled(self) -> None:
        self.ledger.debit(self.account_id, self.amount, self.transfer_id)


def confirm_or_refuse(transfer: WireTransfer) -> bool:
    """Confirm when funds cover the amount; otherwise leave state unchanged."""
    try:
        transfer.send("confirm")
        return True
    except TransitionNotAllowed:
        return False
