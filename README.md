# neurosymbolic-intent-router

**Models interpret, rules decide.** Language models only read the text; symbolic rules choose and authorize every action.

`#MedGemma` `#Qwen3.8` `#MiniLM` `#SentenceTransformers` `#Jev` `#python-statemachine` `#StateMachine` `#FiniteStateMachine` `#NeuroSymbolic` `#NeuroSymbolicAI` `#IntentClassification` `#LLM` `#NLP` `#Ollama` `#Instructor` `#Pydantic` `#Pyright` `#Python` `#Prolog` `#SWIProlog` `#FailClosed` `#StructuredOutput`

## Why

A bank assistant must not act on a misread request. This project tests two hypotheses:

- **H1.** If language models only interpret the text (which intent, how many requests) and symbolic rules decide what runs, fewer wrong actions are executed than when a model chooses the action. The price is lower recall on requests the rules cannot verify.
- **H2.** If a state machine carries each request, the model never moves the workflow: only the rules' decision leaves `routing`, facts are re-checked when the action runs, and money moves once.

## How it works

```
                      user's sentence
                            │
                            ▼
┌────────────────────── STATE MACHINE ───────────────────────┐
│                                                            │
│ received                                                   │
│    │ start                                                 │
│    ▼                                                       │
│ ┌──────────────────── routing ────────────────────┐        │
│ │  NEURAL · interpret                             │        │
│ │    What is being asked? How many requests?      │        │
│ │  SYMBOLIC · decide                              │        │
│ │    Inside the contract? Which request first?    │        │
│ │    Allowed for this session and balance?        │        │
│ └────────────────────────┬────────────────────────┘        │
│                          │ decision                        │
│             ┌────────────┴────────────┐                    │
│           deny                     execute                 │
│             ▼                         ▼                    │
│          refused                  executing                │
│                            facts re-checked; debit once    │
│                                  ├──► completed            │
│                                  └──► failed               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

- **Neural, interpret.** Jev reads the intent and counts the requests. MedGemma splits a sentence with several requests. MiniLM shows how close the sentence is to each action's description, for display only.
- **Symbolic, decide.** A swappable reasoner, SWI-Prolog (`policy.pl`) by default or the same rules in Python, orders the requests, allows or denies the first, lists every reason for a denial, and may suggest a permitted alternative.
- **State machine, execute.** `RequestWorkflow` runs `received → routing → refused | executing → completed | failed`. A wire runs its own `WireTransfer`, which re-checks the balance and debits once.

Any doubt means deny: a model that is unsure, down or off-contract, an amount that cannot be parsed, or a reasoner that fails.

## Demo

One sentence, three bank customers. Output of `uv run --env-file .env python src/demo.py` on 2026-09-26, with the Prolog reasoner: one run, trimmed, not edited. It needs Jev (`TYPESAFE_API_KEY`), a local Ollama with `medgemma:27b`, and SWI-Prolog; see Run.

The sentence makes two requests. Jev reads it as `several`, MedGemma splits it into "Close this account." and "Send $500 to my external bank account.", Jev labels each one, and the precedence rule puts the wire first. The split itself is not printed; the follow-up line comes from it. A denial lists every reason the rules found, not only the first.

**User not authenticated ($0): refused for four reasons; no suggestion, because the rules would deny the balance view too.**

```
Policy: prolog

User Query: 'Close this account and send $500 to my external bank account.'
Context: Auth=False, Role='unauthenticated', Status='inactive', Account='acct-unauthenticated'

Classifier: jev

Action Decision Matrix:
Candidate Action         | Classifier Score | Explainer Score | Allowed?
--------------------------------------------------------------------------
wire_transfer_funds      | 0.9900           | 0.5360          | NO
none                     | 0.0100           | —               | NO
delete_account           | 0.0000           | 0.1422          | NO
view_public_faq          | 0.0000           | 0.0869          | YES
view_account_balance     | 0.0000           | 0.2191          | NO

Decision: action=wire_transfer_funds outcome=deny reason=caller is not authenticated; account is not active; payee is not on the allowlist; insufficient funds for the requested amount
Explainer agrees: wire_transfer_funds score=0.5360

Workflow: received -> routing -> refused (caller is not authenticated; account is not active; payee is not on the allowlist; insufficient funds for the requested amount). Balance now $0
```

**Customer with no money ($0): refused, insufficient funds; the rules suggest the balance view.**

```
User Query: 'Close this account and send $500 to my external bank account.'
Context: Auth=True, Role='customer', Status='active', Account='acct-zero'

Classifier: jev

Action Decision Matrix:
Candidate Action         | Classifier Score | Explainer Score | Allowed?
--------------------------------------------------------------------------
wire_transfer_funds      | 0.9900           | 0.5360          | NO
none                     | 0.0100           | —               | NO
delete_account           | 0.0000           | 0.1422          | NO
view_public_faq          | 0.0000           | 0.0869          | YES
view_account_balance     | 0.0000           | 0.2191          | YES

Decision: action=wire_transfer_funds outcome=deny reason=insufficient funds for the requested amount
Explainer agrees: wire_transfer_funds score=0.5360
Suggestion: view_account_balance

Workflow: received -> routing -> refused (insufficient funds for the requested amount). Balance now $0
```

**Funded customer ($10,000): the wire is allowed and settled; closing the account is listed, not run.**

```
User Query: 'Close this account and send $500 to my external bank account.'
Context: Auth=True, Role='customer', Status='active', Account='acct-funded'

Classifier: jev

Action Decision Matrix:
Candidate Action         | Classifier Score | Explainer Score | Allowed?
--------------------------------------------------------------------------
wire_transfer_funds      | 0.9900           | 0.5360          | YES
none                     | 0.0100           | —               | NO
view_account_balance     | 0.0000           | 0.2191          | YES
delete_account           | 0.0000           | 0.1422          | NO
view_public_faq          | 0.0000           | 0.0869          | YES

Decision: action=wire_transfer_funds outcome=execute reason=wire transfer permitted
Explainer agrees: wire_transfer_funds score=0.5360
You also asked (ask again to proceed):
  - Close this account.

Workflow: received -> routing -> executing -> completed (wire of $500 settled). Balance now $9500
```

The rules behind these outcomes, from `src/policy.pl`. For each request `R`, the Python adapter asserts facts such as `authenticated(R)`, `amount(R, 500)` and `balance(R, 10000)`; a value the request lacks is simply not asserted:

```prolog
%   Handling order for several requests: reads, then money movement, then deletion.
precedence(view_public_faq,      0).
precedence(view_account_balance, 0).
precedence(wire_transfer_funds,  1).
precedence(delete_account,       2).

denied(wire_transfer_funds, R, "caller is not authenticated") :-
    \+ authenticated(R).
denied(wire_transfer_funds, R, "account is not active") :-
    \+ account_status(R, active).
denied(wire_transfer_funds, R, "transfer amount is missing or invalid") :-
    \+ positive_amount(R).
denied(wire_transfer_funds, R, "payee is not on the allowlist") :-
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

%!  suggestion(+Action, +R, -Suggested) is semidet.
%   The first permitted action to offer after Action is denied.
suggestion(Action, R, Suggested) :-
    suggests(Action, R, Suggested),
    denials(Suggested, R, []),
    !.
```

What it shows: the models read the same sentence the same way for all three customers; only the rules and the ledger make the outcomes differ. The model's 0.99 confidence does not move money for the user who is not authenticated. `--policy python` prints the same decisions.

## Architecture

The router depends only on roles defined in `src/contracts.py`. `src/demo.py` chooses the implementations.

| Role | Job | Today |
|---|---|---|
| Classifier, Labeler | Intent and request count; a label for each split request | Jev (`jev.py`) |
| Splitter | The separate requests, as written | MedGemma on Ollama (`intent_understanding.py`) |
| Explainer | Closeness to each action description, display only | MiniLM (`minilm.py`) |
| Policy | Order of requests; allow or deny with every reason; suggestion | Prolog (`prolog_policy.py`, `policy.pl`) or Python (`python_policy.py`) |

```
┌──────────────────────────────────────────────────────────────────────────┐
│ demo.py        composition root: builds the adapters and passes them to  │
│                RequestWorkflow. The only file that names models and the  │
│                reasoner (--policy prolog | python).                      │
└────┬────────────────┬────────────────┬────────────────┬────────────────┬─┘
     │ builds         │ builds         │ builds         │ builds         │
     ▼                ▼                ▼                ▼                │
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐     │
│ jev.py       │ │ minilm.py    │ │ intent_under │ │ prolog_policy │     │
│ JevClassifier│ │ MiniLM-      │ │ standing.py  │ │ .py + .pl     │     │
│              │ │ Explainer    │ │ MedGemma-    │ │ python_policy │     │
│              │ │              │ │ Splitter     │ │ .py           │     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └───────┬───────┘     │
       │ implements     │ implements     │ implements      │ implements  │
       │ Classifier,    │ Explainer      │ Splitter        │ Policy      │
       │ Labeler        │                │                 │             │
       ▼                ▼                ▼                 ▼             │
┌──────────────────────────────────────────────────────────────────┐     │
│ contracts.py   «Protocol» Classifier · Labeler · Splitter ·      │     │
│                Explainer · Policy   (@abstractmethod)            │     │
│                IntentRank · PolicyResult · DescriptionMatch      │     │
│                imports only domain types from policy.py          │     │
└──────────────────────────────▲───────────────────────────────────┘     │
                               │ imports the roles                       │ calls, via workflow.py
                               │                                         ▼
┌──────────────────────────────┴───────────────────────────────────────────┐
│ router.py      route() · decide() · Decision                             │
│                imports no adapter; the Policy is passed in               │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ policy.py      catalog · Session · ParsedRequest · parse_request         │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ transfer.py    Ledger · WireTransfer (python-statemachine)               │
└──────────────────────────────────────────────────────────────────────────┘
```

- Model and reasoner answers are checked at the contract (`IntentRank`, `PolicyResult`). An answer outside it means deny.
- The Python rules are the reference. Prolog must match them on 1,728 policy cases and 252 orderings (`tests/test_policy_engines.py`).
- No fallback model and no human confirmation step: the router is judged on its own decision.

## Run

```bash
uv sync
uv run --env-file .env python src/demo.py                  # Prolog reasoner (default)
uv run --env-file .env python src/demo.py --policy python  # Python reasoner
uv run pytest -m "not integration and not prolog"          # unit tests, including pyright
uv run --env-file .env pytest -m prolog                    # Prolog agrees with Python
```

Needs `TYPESAFE_API_KEY` in `.env` (Jev), Ollama with `medgemma:27b`, and SWI-Prolog 9. With the macOS app, also put `DYLD_FALLBACK_LIBRARY_PATH=/Applications/SWI-Prolog.app/Contents/Frameworks` in `.env`. `--policy python` does not need Prolog.

What each test checks: [tests/README.md](tests/README.md).

## Experiments

- [Structured output from local models](docs/structured-output-experiment.md): native Ollama `format=<schema>` against Instructor, on MedGemma and Qwen.

## Status

Not measured yet: everything above comes from single runs on hand-picked sentences.

Next:
- a labelled evaluation set with a model-only baseline: precision, recall, and wrong actions executed;
- amounts in words and "the remaining cash" (amount parsing is still Python only, outside the reasoner);
- how stable Jev's count and MedGemma's split are across runs.
