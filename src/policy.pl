/*  Banking policy rules. Same rules as RULES in policy.py.

    Facts is a dict built by the Python adapter:
      authenticated: @true | @false   (janus form of Python bool)
      status, role:  atoms
      amount:        @none | an exact rational
      payee:         @none | a string
      balance:       an exact rational
*/

:- module(policy, [denials/3, permitted_message/2, suggestion/3, ordered/2]).

action(wire_transfer_funds).
action(view_account_balance).
action(delete_account).
action(view_public_faq).

%   Handling order for several requests: reads, then money movement, then deletion.
precedence(view_public_faq,      0).
precedence(view_account_balance, 0).
precedence(wire_transfer_funds,  1).
precedence(delete_account,       2).

permitted_message(wire_transfer_funds,  "wire transfer permitted").
permitted_message(view_account_balance, "balance view permitted").
permitted_message(delete_account,       "account deletion permitted").
permitted_message(view_public_faq,      "public faq always allowed").

%!  ordered(+Actions, -Positions) is det.
%   Positions (0-based) of Actions by precedence. Ties keep the written order;
%   none and unknown actions go last.
ordered(Actions, Positions) :-
    findall(Rank-Position,
            ( nth0(Position, Actions, Action), rank(Action, Rank) ),
            Keyed),
    msort(Keyed, Sorted),
    pairs_values(Sorted, Positions).

rank(Action, Rank) :-
    (   precedence(Action, Rank)
    ->  true
    ;   Rank = 99
    ).

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
    insufficient_funds(F).

denied(view_account_balance, F, "caller is not authenticated") :-
    \+ authenticated(F).

denied(delete_account, F, "delete requires an authenticated admin") :-
    \+ ( authenticated(F), get_dict(role, F, admin) ).

%!  suggestion(+Action, +Facts, -Suggested) is semidet.
%   A permitted action to offer after Action is denied. First match only.
suggestion(Action, Facts, Suggested) :-
    suggests(Action, Facts, Suggested),
    denials(Suggested, Facts, []),
    !.

suggests(wire_transfer_funds, F, view_account_balance) :-
    insufficient_funds(F).

insufficient_funds(F) :-
    get_dict(amount, F, Amount),
    number(Amount),
    get_dict(balance, F, Balance),
    Amount > Balance.

authenticated(F) :-
    get_dict(authenticated, F, @true).

positive_amount(F) :-
    get_dict(amount, F, Amount),
    number(Amount),
    Amount > 0.
