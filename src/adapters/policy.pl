/*  Banking policy rules. Same rules as RULES in policy.py.

    The Python adapter asserts facts about one request R, queries, then calls
    forget(R). A value the request does not have is simply not asserted:
      authenticated(R)          the caller is authenticated
      account_status(R, S)      S is an atom, e.g. active
      role(R, Role)             Role is an atom, e.g. customer, admin
      amount(R, A)              the parsed amount, an exact rational
      payee(R, P)               the payee, if it is on the allowlist
      balance(R, B)             the account balance, an exact rational
*/

:- module(policy, [denials/3, permitted_message/2, suggestion/3, ordered/2, forget/1,
                   authenticated/1, account_status/2, role/2, amount/2, payee/2, balance/2]).

:- dynamic authenticated/1, account_status/2, role/2, amount/2, payee/2, balance/2.

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

%   Clause order is the reported order of reasons.
denied(Action, _, "unknown action denied") :-
    \+ action(Action).

denied(wire_transfer_funds, R, "caller is not authenticated") :-
    \+ authenticated(R).
denied(wire_transfer_funds, R, "account is not active") :-
    \+ account_status(R, active).
denied(wire_transfer_funds, R, "transfer amount is missing or invalid") :-
    \+ positive_amount(R).
denied(wire_transfer_funds, R, "payee is missing, ambiguous or not on the allowlist") :-
    \+ payee(R, _).
denied(wire_transfer_funds, R, "insufficient funds for the requested amount") :-
    insufficient_funds(R).

denied(view_account_balance, R, "caller is not authenticated") :-
    \+ authenticated(R).

denied(delete_account, R, "delete requires an authenticated admin") :-
    \+ admin(R).

%   After a denial, offer an action only if it is itself permitted.
suggests(wire_transfer_funds, R, view_account_balance) :-
    insufficient_funds(R).

positive_amount(R) :-
    amount(R, Amount),
    Amount > 0.

insufficient_funds(R) :-
    amount(R, Amount),
    balance(R, Balance),
    Amount > Balance.

admin(R) :-
    authenticated(R),
    role(R, admin).

%!  denials(+Action, +R, -Reasons) is det.
%   Every reason Action is denied for request R, in clause order. [] means permitted.
denials(Action, R, Reasons) :-
    findall(Reason, denied(Action, R, Reason), Reasons).

%!  suggestion(+Action, +R, -Suggested) is semidet.
%   The first permitted action to offer after Action is denied.
suggestion(Action, R, Suggested) :-
    suggests(Action, R, Suggested),
    denials(Suggested, R, []),
    !.

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

%!  forget(+R) is det.
%   Remove every fact about request R.
forget(R) :-
    retractall(authenticated(R)),
    retractall(account_status(R, _)),
    retractall(role(R, _)),
    retractall(amount(R, _)),
    retractall(payee(R, _)),
    retractall(balance(R, _)).
