/*  Banking policy rules. Same rules as RULES in policy.py.

    Facts is a dict built by the Python adapter:
      authenticated: @true | @false   (janus form of Python bool)
      status, role:  atoms
      amount:        @none | an exact rational
      payee:         @none | a string
      balance:       an exact rational
*/

:- module(policy, [denials/3, permitted_message/2]).

action(wire_transfer_funds).
action(view_account_balance).
action(delete_account).
action(view_public_faq).

permitted_message(wire_transfer_funds,  "wire transfer permitted").
permitted_message(view_account_balance, "balance view permitted").
permitted_message(delete_account,       "account deletion permitted").
permitted_message(view_public_faq,      "public faq always allowed").

%!  denials(+Action, +Facts, -Reasons) is det.
%   Every reason Action is denied, in clause order. [] means permitted.
denials(Action, Facts, Reasons) :-
    findall(Reason, denied(Action, Facts, Reason), Reasons).

%   Clause order is the reported order of reasons.
denied(Action, _, "unknown action denied") :-
    \+ action(Action).

denied(wire_transfer_funds, F, "caller is not authenticated") :-
    \+ authenticated(F).
denied(wire_transfer_funds, F, "account is not active") :-
    \+ get_dict(status, F, active).
denied(wire_transfer_funds, F, "transfer amount is missing or invalid") :-
    \+ positive_amount(F).
denied(wire_transfer_funds, F, "payee is not on the allowlist") :-
    get_dict(payee, F, @none).
denied(wire_transfer_funds, F, "insufficient funds for the requested amount") :-
    get_dict(amount, F, Amount),
    number(Amount),
    get_dict(balance, F, Balance),
    Amount > Balance.

denied(view_account_balance, F, "caller is not authenticated") :-
    \+ authenticated(F).

denied(delete_account, F, "delete requires an authenticated admin") :-
    \+ ( authenticated(F), get_dict(role, F, admin) ).

authenticated(F) :-
    get_dict(authenticated, F, @true).

positive_amount(F) :-
    get_dict(amount, F, Amount),
    number(Amount),
    Amount > 0.
