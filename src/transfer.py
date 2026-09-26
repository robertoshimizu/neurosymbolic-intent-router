"""In-memory ledger and wire-transfer statechart."""

from __future__ import annotations

from decimal import Decimal

from statemachine import State, StateChart
from statemachine.exceptions import TransitionNotAllowed


class Ledger:
    """Account balances with idempotent settlement by transfer id, and account closure."""

    def __init__(self) -> None:
        self._balances: dict[str, Decimal] = {}
        self._settled: set[str] = set()
        self._closed: set[str] = set()

    def close(self, account_id: str) -> None:
        self._closed.add(account_id)

    def is_closed(self, account_id: str) -> bool:
        return account_id in self._closed

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
    """Lifecycle for an approved outbound wire against an in-memory ledger."""

    allow_event_without_transition = False

    drafted = State(initial=True)
    authorized = State()
    submitted = State()
    settled = State(final=True)
    declined = State(final=True)

    # Funds are re-checked when the transfer is authorized, not when it was requested.
    authorize = drafted.to(authorized, cond=["account_open", "funds_available"])
    submit = authorized.to(submitted)
    settle = submitted.to(settled)
    cancel = (
        drafted.to(declined)
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

    def account_open(self) -> bool:
        return not self.ledger.is_closed(self.account_id)

    def funds_available(self) -> bool:
        return self.ledger.get_balance(self.account_id) >= self.amount

    def on_enter_settled(self) -> None:
        self.ledger.debit(self.account_id, self.amount, self.transfer_id)


def authorize_or_refuse(transfer: WireTransfer) -> bool:
    """Authorize when the account is open and funds cover the amount; otherwise leave state unchanged."""
    try:
        transfer.send("authorize")
        return True
    except TransitionNotAllowed:
        return False
