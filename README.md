# neurosymbolic-intent-router

**Models interpret, rules decide.** Language models are used only to understand the text; they never choose an action.

`#MedGemma` `#Qwen3.8` `#MiniLM` `#SentenceTransformers` `#Jev` `#python-statemachine` `#StateMachine` `#FiniteStateMachine` `#NeuroSymbolic` `#NeuroSymbolicAI` `#IntentClassification` `#LLM` `#NLP` `#Ollama` `#Instructor` `#Pydantic` `#Pyright` `#Python` `#Prolog` `#SWIProlog` `#FailClosed` `#StructuredOutput`

## Goal

Turn a user's free-text request into an action in a workflow, with high precision: the workflow should not act on a misread request. The project explores how far a neuro-symbolic design gets there. Language models handle what language makes hard: what is being asked, and how many things. Symbolic components keep the authority: what is allowed, in what order, and how a transaction moves from start to finish. The long-term aim is to measure this: raise precision while giving up no more recall than the domain can afford.

Banking is the test domain, because the cost of a wrong action is obvious there.

## Hypothesis

**H1: Models interpret, rules decide.** A workflow in which language models only *interpret* the text (which intent it expresses, how many requests it makes, what each request says), and symbolic rules *decide* which action, if any, runs, acts wrongly less often than a workflow in which a model chooses the action. The price is lower recall on requests the rules cannot verify.

*Why we expect this:* language models are good at reading language, but their answers are not calibrated, can vary between runs, and can be confidently wrong. Rules cannot read language, but they are deterministic, auditable, and can refuse. Dividing the work plays to both strengths: the models do the reading, and only the rules turn that reading into an action.

**H2: The model never moves the workflow.** Choosing the right action is only half the risk. The other half is carrying it out: acting on a stale fact, running a step out of order, or moving money twice. An explicit finite state machine turns "only the rules' decision and the ledger's facts move the workflow" into structure. The neuro-symbolic step is one state, `routing`: the models interpret the text and the rules decide the action inside it, and its only exit is the rules' decision. So:
- events that are not allowed from the current state are rejected;
- facts are re-checked when the action runs, not when it was requested (a wire is authorized only if the account is still open and the funds still cover it);
- final states are final, and money moves once.

In many agent designs the model chooses the next step. Here the model's output is only an input to the rules inside `routing`, and the transition out of `routing` is chosen by the rules' decision alone.

**How we will test it (not done yet):**
- Compare with a model-only baseline on a labelled set of sentences.
- Measure precision and recall per action.
- Count **wrong actions executed**, the number that matters most.
- Count transitions the state machine rejected.

## Approach

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


- **Neural parts interpret.** A classifier reads which catalog intent the text expresses (or none) and counts the requests, a splitter separates a sentence with several requests, and an explainer shows how close the sentence is to each intent's description. None of them chooses or authorizes an action. Each is a swappable model behind a contract (Jev, MedGemma and MiniLM today).
- **Symbolic parts decide.**
  - Policy rules check the session and the ledger. The reasoner is swappable: SWI-Prolog (`policy.pl`) by default, or the same rules in Python.
  - A fixed precedence rule, part of the same reasoner, orders multiple requests.
  - Contract checks reject classifier and labeler output outside the catalog or outside its expected shape.
- **A state machine holds it all.** Every request goes through `RequestWorkflow`: `received` → `routing` → `refused` or `executing` → `completed` or `failed`. `routing` is one state in which the neural and symbolic parts work together (it runs `route()`), and the rules' decision is its only exit. While executing, a wire runs its own `WireTransfer` (drafted, authorized, submitted, settled), and a deletion closes the account in the ledger.

The design fails closed. A sentence is denied when a model is unsure, unavailable, or answers outside its contract, and when an amount cannot be parsed. That is a deliberate cost to recall: for a bank, refusing and asking again costs less than acting wrongly.

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

The **Allowed?** column is the symbolic side at work. The model scores are identical in all three runs; the permissions are not:

| Action | Not authenticated | No money | Funded |
|---|---|---|---|
| wire_transfer_funds | NO | NO | YES |
| view_account_balance | NO | YES | YES |
| view_public_faq | YES | YES | YES |
| delete_account | NO | NO | NO |

What it shows: the models read the same sentence the same way for all three customers; only the rules and the ledger make the outcomes differ. The model's 0.99 confidence does not move money for the user who is not authenticated. `--policy python` prints the same decisions.

## Status

> **Not yet measured.** Precision and recall have not been computed. The results below are single runs on a handful of hand-picked sentences; they show how the design behaves, not how well it performs.

Next steps:
- a labelled evaluation set, with a model-only baseline to compare against;
- parsing amounts written in words.

## Repository map

| Path | Role |
|---|---|
| `src/router.py` | `route()` and `decide()`: orchestration, with no model code |
| `src/contracts.py` | The roles models and reasoners must implement, and the checks on their output |
| `src/policy.py` | Catalog, precedence rule, session types, amount parsing, and the banking rules as a Python table (`RULES`, `SUGGESTIONS`) |
| `src/policy.pl` | The same banking rules in Prolog |
| `src/python_policy.py`, `src/prolog_policy.py` | Reasoner adapters (symbolic): `PythonPolicy` and `PrologPolicy` |
| `src/workflow.py` | `RequestWorkflow` state machine: from the sentence, through routing, to its effect |
| `src/transfer.py` | `WireTransfer` state machine, and the ledger (balances, once-only debits, closed accounts) |
| `src/jev.py`, `src/minilm.py`, `src/intent_understanding.py` | Model adapters (neural). |
| `src/demo.py` | Wires the models and the reasoner together and runs three sessions |
| `src/coffee.py` | A minimal `python-statemachine` example |

## Architecture

The router never names a model or a reasoner. `src/contracts.py` defines five roles as `Protocol`s, plus the types they exchange. It imports only domain types from `policy.py` (the catalog, `Session`, `ParsedRequest`). `IntentRank` rejects any model value outside the contract (an action not in the catalog, a confidence outside 0..1, an unknown request count, a non-numeric score), and `PolicyResult` rejects any reasoner answer outside it (no reasons, a non-string reason, a suggestion outside the catalog or on an allowed action), before the router sees them. Each model and each reasoner is an adapter in its own file that explicitly subclasses the role it fulfils. `src/demo.py` is the composition root: it builds the adapters and hands them to `RequestWorkflow`, which runs `route()` in its `routing` state. Swapping a model or a reasoner means writing a new adapter that subclasses the same role and changing one line in `demo.py`.

| Role | Contract | Today | File |
|---|---|---|---|
| Classifier | `classify(text)`: action (or `none`), confidence, request count. `None` means unavailable. | Jev | `src/jev.py` |
| Labeler | `label(texts)`: one action per request, in one call. `None` means unavailable. | Jev | `src/jev.py` |
| Splitter | `split(text)`: the separate requests, in the order written | MedGemma | `src/intent_understanding.py` |
| Explainer | `explain(text, action)`: a display-only note on the chosen action | MiniLM | `src/minilm.py` |
| Policy | `evaluate(action, session, balance, parsed)`: allowed or not, every reason, and a permitted action to suggest. `order(actions)`: the handling order of several requests, as positions | SWI-Prolog (default), Python | `src/prolog_policy.py`, `src/python_policy.py` |

**How the contract is enforced.** Python checks two different things, at two different times:
- **A missing method** fails when the adapter is created. Every role method is an `@abstractmethod`, and each adapter subclasses its role explicitly (`class JevClassifier(Classifier, Labeler)`). Leaving out `label` raises `TypeError: Can't instantiate abstract class`.
- **A wrong signature** fails `uv run pyright`. Python never checks signatures at runtime. Pyright reports "overrides class `Classifier` in an incompatible manner" for any mismatch in parameters or return type.

Both checks were confirmed by planting each violation in `jev.py`.

Arrows mean "imports". Everything points to `contracts.py`, and nothing in it points back. The adapters (the details) and `router.py` (the high-level rules) both depend on the same abstraction, and neither depends on the other. Two kinds of arrows are left out: the adapters read the catalog and the domain types from `policy.py`, and `demo.py` also imports `policy.py` and `transfer.py` to build the sessions and the ledger. `workflow.py` is not drawn either: it sits above the router, runs `route()` in its `routing` state, and uses `transfer.py` to carry out the decision. `demo.py` starts every request through it.

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

### Design decisions

- **Roles, not models.** `route()` takes a `classifier`, a `labeler`, a `splitter` and an `explainer`; `decide()` takes a `classifier` and an `explainer`. Tests pass small fakes that subclass the same roles, so they check the router's rules, not a model's judgment.
- **The reasoner is a role too.** `route()`, `decide()` and `RequestWorkflow` require a `policy`: leaving it out raises `TypeError`, and pyright flags it. A reasoner that errors denies with "policy unavailable". An OWL reasoner, or any other, would be one more adapter.
- **Python is the reference reasoner.** `PythonPolicy` stays, and `test_policy_engines.py` requires every other reasoner to return the same verdict, reasons (in order) and suggestion on 1,728 combinations of session, amount, payee and balance.
- **Contracts in their own module.** Adapters import `contracts.py`, not the router, so they can be written, tested and replaced without loading the router.
- **Classifying and labelling are separate roles.** A replacement model may classify one sentence well but not label a batch in one call. Jev happens to provide both, so the demo passes it twice.
- **Each adapter owns its calibration.** Jev's 0.5 confidence floor lives in `jev.py` and turns an unsure answer into `none`. MiniLM's 0.45 description-gap floor lives in `minilm.py`. The router sees only an action or `none`, so two scales never meet in one rule.
- **No fallback.** If the classifier is missing or down, the sentence is denied with "classifier unavailable". MiniLM used to rank in Jev's place, and it gave absurd sentences whichever action was least far away.
- **The explainer never votes.** MiniLM's scores are printed beside the classifier's, and a disagreement is labelled a description gap. It never changes the decision, and if it fails the decision goes ahead without the note (`explainer=unavailable` in the trace).
- **The splitter has one narrow job.** MedGemma only rewrites a sentence into separate requests. When it also chose actions, it copied the catalog descriptions and lost "$500", so labelling stays with the classifier.
- **One classifier call per sentence.** That call returns both the action and the request count. A second call happens only when there are several requests, and it labels all of them at once.
- **One table holds the rules.** Each action in `policy.py`'s `RULES` has its description, its precedence and its checks. Adding an action means adding one entry there and its clauses in `policy.pl`. `ACTION_CATALOG` and `ACTION_PRECEDENCE` are built from it.
- **Every reason, not the first.** A denial lists every rule that failed, in a fixed order. A caller who is not authenticated and has no money is told both.
- **The rules choose the suggestion.** After a denial, the reasoner may offer one other action, and only one it would itself permit: a wire denied for insufficient funds suggests the balance view, unless the caller may not see it. The router passes it through and does not read the reasons' text.
- **A rule, not a model, orders requests.** The reasoner's `order()` puts reads, then the wire, then deletion, then `none`; ties keep the written order. In Prolog these are `precedence/2` facts and `ordered/2`; in Python, `ACTION_PRECEDENCE`. It returns positions, not action names, so two requests for the same action stay distinct. An answer that is not each position exactly once is denied. Only the first request is decided. The rest are listed as follow-ups and never run on their own.
- **No human confirmation step.** It would catch the router's mistakes and hide them from any measurement. The router is judged on its own decision. Human-in-the-loop could be tested later as a separate hypothesis.
- **Neural and symbolic share one state.** `routing` runs the whole interpret-and-decide step, because a state needs one clear exit: the rules' decision, `deny` or `execute`. Splitting "interpret" and "decide" into separate states would give the models' reading its own transition. Inside `routing`, `route()` stays a pure function.
- **Policy is pure.** A reasoner judges an action that has already been chosen, from the session, the balance and the parsed request. It holds no model code and never writes to the ledger.
- **Keys stay with their adapter.** `jev.py` and `minilm.py` each load their own key from `.env`. The router loads nothing.

### Known limits of this design

- Pyright runs in `standard` mode, not `strict`. It checks every signature against the contracts, but it does not require every value to be typed.
- Amount parsing is still Python only, outside the swappable reasoner.
- The SWI-Prolog.app runtime on macOS finds its libraries only if `DYLD_FALLBACK_LIBRARY_PATH` is set when the process starts, so commands that load Prolog run with `uv run --env-file .env`.

## Run

```bash
uv sync
uv run --env-file .env python src/demo.py                  # Prolog reasoner (default)
uv run --env-file .env python src/demo.py --policy python  # Python reasoner
uv run python src/coffee.py
uv run pytest -m "not integration and not prolog"
uv run --env-file .env pytest -m prolog                    # Prolog agrees with Python
uv run pyright
```

The Prolog reasoner needs SWI-Prolog 9 and the `janus-swi` bridge (installed by `uv sync`). With the macOS app, `.env` must contain `DYLD_FALLBACK_LIBRARY_PATH=/Applications/SWI-Prolog.app/Contents/Frameworks`. `--policy python` does not load Prolog.

The demo calls TypeSafe Jev when `TYPESAFE_API_KEY` is set in `.env`. That file is gitignored. MiniLM weights and embedding vectors are cached under `.cache/`, which is also gitignored. Hugging Face downloads use `HF_TOKEN` from the same `.env` file. Neither value is printed.

The splitter calls local Ollama. It expects `medgemma:27b`.

```bash
uv run pytest tests/test_intent_understanding.py -m integration -s
```

## The router

`route()` in `src/router.py` is the router's entry point; `src/demo.py` is the program's. Jev, MedGemma and MiniLM appear below as the adapters that `demo.py` wires in today.

1. The classifier receives the sentence in one call. Jev answers two closed questions: which action (the four catalog actions plus `none`), and how many requests the sentence makes (`none`, `one`, `several`). Identity, role, and balance are not sent. If the classifier is missing or down, the sentence is denied with "classifier unavailable".
2. If the count is not `several`, the sentence goes straight to `decide()` with that answer. Most sentences stop here.
3. If the count is `several`, the splitter (MedGemma) rewrites the sentence as plain requests in the user's own words. It does not choose actions. The labeler (also Jev) then labels every request in one call.
4. The injected `Policy` orders the requests: reads, then the wire, then deletion, then `none`. Only the first goes to `decide()`. The others are listed as follow-ups and are not evaluated. If the first is denied, nothing is listed. If the split, the labelling or the ordering fails, the sentence is denied.

`decide()` then applies the rules.

1. A `none` action is denied with "no matching action". Banking rules do not run. For Jev, an answer under 0.5 confidence has already become `none` inside the adapter.
2. For a catalog action, the explainer (MiniLM) scores the sentence against the written action descriptions. The scores are printed beside the classifier's and never change the choice.
3. The injected `Policy` judges the action from the session, the balance and the parsed request, and lists every reason it is denied. A wire needs authentication, an active account, a parsed amount, an allowlisted payee, and enough balance. A deletion needs an authenticated admin. A wire denied for insufficient funds suggests the balance view, if the caller may see it. `route()` runs inside the `routing` state of `RequestWorkflow` (`src/workflow.py`). An allowed action then moves the workflow to `executing`, which re-checks the facts at that moment. A wire runs `WireTransfer` (drafted, authorized, submitted, settled); authorization requires the account to be open and the funds to cover the amount, and the ledger debits on settle, once per transfer id. A deletion closes the account, so later wires on it are refused. There is no human confirmation step: the router is judged on its own decision.

The chart shows calls at runtime. The Architecture section shows which module imports which.

```
                     user sentence
                           │
                           ▼
          ┌──────────────────────────────────┐
          │ route()                router.py │
          │ Classifier.classify: 1 call      │
          │   • action (catalog + none)      │
          │   • request_count                │
          │ None / error ──► DENY            │
          │   "classifier unavailable"       │
          └────────────────┬─────────────────┘
                           │
             request_count == "several"?
               │                        │
              no                       yes
               │                        ▼
               │       ┌──────────────────────────────────┐
               │       │ Splitter.split                   │
               │       │ → plain requests, as written     │
               │       └────────────────┬─────────────────┘
               │           fails or < 2 ──► DENY
               │                        ▼
               │       ┌──────────────────────────────────┐
               │       │ Labeler.label                    │
               │       │ 1 call, 1 question per request   │
               │       └────────────────┬─────────────────┘
               │           fails ──────────► DENY
               │                        ▼
               │       ┌──────────────────────────────────┐
               │       │ Policy.order     prolog|python   │
               │       │ reads → wire → delete → none     │
               │       │ first is decided; the rest       │
               │       │ become follow_ups                │
               │       └────────────────┬─────────────────┘
               │           invalid ────────► DENY
               │                        │ first request + its rank
               ▼                        ▼
          ┌──────────────────────────────────┐
          │ decide()               router.py │
          │ 1. action == none ──► DENY       │
          │    "no matching action"          │
          │ 2. Explainer.explain             │
          │    display only, no vote         │
          │ 3. parse_request       policy.py │
          │ 4. Policy.evaluate  prolog|python│
          │    session + balance ──► DENY    │
          │    (all reasons, suggestion)     │
          │    allowed ──► EXECUTE           │
          └────────────────┬─────────────────┘
                           ▼
          Decision (+ follow_ups if not denied)
                           │ exit of the routing state
                           ▼
          ┌──────────────────────────────────┐
          │ RequestWorkflow      workflow.py │
          │ routing ─ deny ────► refused     │
          │         └ execute ─► executing   │
          │ executing ─┬─► completed         │
          │            └─► failed            │
          │ wire: WireTransfer   transfer.py │
          │   drafted → authorized (account  │
          │   open, funds re-checked) →      │
          │   submitted → settled, debit once│
          │ delete: close account in ledger  │
          └──────────────────────────────────┘
```

On a clear sentence, "I want to send $5,000 to my external bank account.", the three sessions behave as follows. A user who is not authenticated is denied. A customer with no balance is denied for insufficient funds. A customer with $10,000 is allowed; the workflow settles the wire, leaving $5,000.

## What the router does not understand

Jev assigns one label from a list. It does not unpack a sentence that contains two requests, and a confident label can be only one reading of a mixed sentence. `route()` now handles that case by counting, splitting, and ordering, as described above.

These showed up in live checks before `route()` existed:

- "How much money do I have before I wire funds out?" Jev chose the balance view with very high confidence. MiniLM's nearest description was the wire. The balance view was executed.
- "Where can I read answers to common questions about transfers?" Jev chose the FAQ. MiniLM leaned toward the wire. The FAQ was executed.
- "Move five thousand dollars to my external bank account." Jev chose the wire, then policy denied it because the amount parser only accepts a `$` figure.
- "Close this account and send the remaining cash to my external bank account." Jev returned `none`.
- An unrelated sentence such as "What is the capital of Portugal?" correctly stopped at `none`.

With `route()`, one live run gave:

- "Close this account and send $500 to my external bank account." (admin) Jev counted several. MedGemma split it into "Close this account." and "Send $500 to my external bank account.". Jev labeled them delete (0.95) and wire (0.98). The rule put the wire first. It stopped at confirmation (a step since removed; the wire now executes), and "Close this account." was listed as a follow-up.
- "Close this account and send the remaining cash to my external bank account." The same split and order. The wire was denied because "remaining cash" is not an amount, so nothing was listed.
- "Show my balance and then wire $500 to my external bank account." The balance view executed, and the wire was listed.
- "How much money do I have before I wire funds out?" Jev counted one request. The balance view executed.
- "I want to send $5,000…" and "What is the capital of Portugal?" Jev counted one request each and gave the same results as before.

Each case was run once. How stable the count, the split, and the labels are across runs has not been measured.

TypeSafe's own guidance is that one choice should be a snap judgment, and that a sentence with several questions should be split into small questions whose answers the application combines. A low confidence is a handoff to a person or to a normal language model. Jev is not the component that finishes reading a complex sentence.

MiniLM makes the gap visible. Each cosine is closeness to a description we wrote. It is not a calibrated probability, and it must not cast a second vote. That is why it no longer ranks when Jev is down: as a fallback it gave an absurd sentence whichever action was least far away.

## Challenges

**A closed label set cannot represent a mixed sentence.** Adding `none` stops unrelated text. It does not split "close the account and send the cash" into two explicit requests. Forcing one winner hides the second request.
*Status: addressed, not proven.* Jev counts the requests, MedGemma splits, Jev labels each part, and a fixed rule picks the first. The second request is listed, not hidden. The evidence is one live run per sentence.

**Similarity and probability are different scales.** MiniLM's 0.12 and Jev's 0.98 do not mean the same thing. Printing them in one column, or letting a cosine floor run before the label is known, produced denials that talked about banking rules for a sentence that was not a banking request.
*Status: solved.* MiniLM only explains a Jev choice and never overrides it, and it no longer ranks on its own. Each adapter keeps its own floor, so the router only sees an action or `none`. Unit tests cover each rule.

**The amount in words and the amount in symbols diverged.** Jev treated "five thousand dollars" as a wire. The ledger rule never saw an amount, because the parser looked for `$5,000`.
*Status: not solved.* The parser still accepts only a `$` figure. "The remaining cash" is not resolved to a balance either.

**Rewriting a prompt that already worked made the readings worse.** This came from an earlier full-sentence reader, since removed. Its original instructions, sent as one message, already separated time from cause. Later drafts added a private JSON shape, moved the rules into a system message, and stacked extra bans ("never return an empty list", "do not attach a time to an event"). MedGemma followed the newest ban and dropped an earlier one. Questions came back empty, events were reduced to bare verbs, or the same events were copied into the causal section so the list would not be blank. A second bug hid good answers: the test kept only objects that contained a field named `statement`, so a correct payload in MedGemma's own shape was printed as empty.

*Status: a lesson, applied again.* The splitter has one narrow job. An attempt to make MedGemma also pick catalog actions made it copy the action descriptions in place of the user's words, and "$500" was lost. That job went back to Jev.

## Experiment: structured output from local models

*Run on 2026-09-26. Isolated in `tests/test_structured_output.py`; nothing in `src/` uses it.*

**The question.** LLMs sometimes return the wrong type. Should the project validate model output at runtime with pydantic, or force JSON with a library such as Instructor, Outlines, Guidance or PydanticAI?

**What was decided for the main program.** Validation happens once, where model output becomes an `IntentRank`, with a plain dataclass check (see Architecture). Pydantic is not a direct dependency of the contracts: there are five fields, and the catalog check needs custom code either way. The Jev SDK already validates its responses with pydantic. MedGemma's split output is plain text, which a schema cannot check for meaning.

**The three ways to get structured output.**
- **Validate afterwards and retry** (Instructor, Marvin, PydanticAI). The model writes freely, the output is checked, and on failure the model is asked again. This costs another model call per retry.
- **Constrain decoding with a grammar** (Outlines, Guidance). Tokens outside the schema are masked while the model generates, so the shape is valid on the first try. Ollama has this built in: since v0.5, a JSON schema passed as `format` becomes a grammar ([Clayton](https://blog.danielclayton.co.uk/posts/ollama-structured-outputs/), [llama.cpp](https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output), [Ollama docs](https://docs.ollama.com/capabilities/structured-outputs)).
- **Native provider features** (OpenAI Structured Outputs, Gemini JSON schema). These are cloud-only, so they don't apply here.

A valid shape is not a correct answer. "Let Me Speak Freely?" (EMNLP 2024 Industry Track) found that strict format constraints hurt reasoning but help classification ([arXiv 2408.02442](https://arxiv.org/abs/2408.02442)). Choosing one catalog action is a classification task.

**The experiment.** The same schemas and sentences were sent through two approaches on local Ollama 0.34.4 at temperature 0:
- **native:** Ollama's `format=<schema>`. The action field is an `enum` of the catalog plus `none`, so no other value can be generated.
- **instructor:** Instructor 1.17.0 in JSON mode through Ollama's OpenAI-compatible endpoint, validating with pydantic, with up to 2 retries.

There were two tasks: classify 7 clear-cut sentences into one catalog action, and split the 4 existing split cases into a list of requests. Only the shape is asserted; accuracy, retries and time are printed.

| Model | Task | native | instructor, thinking on | instructor, thinking off |
|---|---|---|---|---|
| medgemma:27b | classify (7) | 7/7 correct, 23.3 s | 7/7 correct, 27.2 s | not run (the model has no thinking mode) |
| medgemma:27b | split (4) | 4/4 correct, 12.1 s | 4/4 correct, 17.2 s | not run |
| qwen3.8:27b | classify (7) | 7/7 correct, 26.4 s | 7/7 correct, 94.6 s | 7/7 correct, 44.0 s |
| qwen3.8:27b | split (4) | 4/4 correct, 16.0 s | 4/4 correct, 77.1 s | 4/4 correct, 28.6 s |

Every answer in every run had a valid shape, and Instructor never retried. The first native call of each run includes loading the model (about 10 s).

**Thinking mode.** The native call sends `"think": false`. At first the Instructor call did not, so qwen reasoned before every answer. Ollama's OpenAI-compatible endpoint turns this off with `reasoning_effort="none"`. A direct check on qwen confirmed it: the output dropped from 32 tokens in 10 s to 3 tokens in 2 s. With thinking off, Instructor's time on qwen halved but stayed about twice native's (about 6 to 7 s per call against about 2.5 s). The reason for the remaining gap has not been measured. One untested difference is that Instructor's JSON mode asks only for valid JSON, while native passes the full schema.

**Conclusions.**
- For a local Ollama model, native `format=<schema>` is enough: equally reliable here, faster, no extra dependency, and nothing to retry. Instructor's retry loop never fired, so its main benefit went unused.
- Both models classified all 7 sentences correctly, including `none` for "capital of Portugal" and the "Don't close…" prohibition. Either could become a `Classifier` adapter, subject to the existing contracts and `IntentRank` checks. Neither gives a calibrated confidence like Jev's, so an adapter would need a fixed confidence or token probabilities.
- Both splitters kept "$500" and dropped the prohibited "close". "The remaining cash" is still passed along as text.

**Limits.** One run per model, 11 fairly clear sentences. This shows that the format is reliable, not that accuracy holds on mixed or tricky sentences or from one run to the next.

```bash
uv run pytest tests/test_structured_output.py -m ollama -s
OLLAMA_MODELS="medgemma:27b,qwen3.8:27b" uv run pytest tests/test_structured_output.py -m ollama -s
```

## Tests

`uv run pytest -m "not integration and not prolog"` runs the unit tests, including pyright, so one command checks both the contracts' signatures and their meaning. Integration tests call MiniLM, Jev, or MedGemma. To check a new adapter, add it to the role's fixture in `test_contracts.py`. Router tests use a fake classifier, splitter and explainer, so they check the router's rules, not the models' judgment.

| Test | Goal | Type |
|---|---|---|
| `test_policy.py` | | |
| parse_amount_with_comma_and_dollar | "$5,000" parses to 5000 and the payee binds | unit |
| unknown_payee_does_not_bind | A payee outside the allowlist does not bind | unit |
| unauthenticated_wire_denied | An unauthenticated user cannot wire | unit |
| delete_denied_for_customer | Only an admin may delete | unit |
| `test_router.py` | | |
| insufficient_funds_suggests_balance_view | A wire above the balance is denied and suggests the balance view | unit |
| no_suggestion_the_policy_would_deny | An unauthenticated caller with no funds is not offered the balance view | unit |
| authorize_then_settle_debits_once | Settlement debits once per transfer id | unit |
| authorize_refused_when_balance_drained | Authorization is refused when funds are gone | unit |
| jev_none_stops_before_policy | A `none` action is denied before policy | unit |
| jev_wire_is_judged_by_policy | A wire chosen by the classifier is executed only through policy | unit |
| minilm_disagreement_does_not_override_jev | The explainer never replaces the classifier's choice | unit |
| explainer_failure_keeps_the_decision | A broken explainer loses its note, never the decision | unit |
| classifier_failure_denies_without_guessing | A classifier that is down means deny, not a guess | unit |
| route_orders_money_movement_before_deletion | Close + wire: the wire goes first, close is listed | unit |
| route_denied_first_request_offers_no_follow_ups | A denied first request lists nothing | unit |
| route_one_request_uses_the_single_jev_rank | One request skips the split | unit |
| route_failed_split_denies_without_guessing | A failed split is denied | unit |
| route_failed_labeling_denies_without_guessing | A failed labeling is denied | unit |
| route_invalid_order_denies_without_guessing | A reasoner order that repeats or drops a request is denied | unit |
| route_none_label_is_ordered_last | A `none` label goes last | unit |
| live_jev_abstains_on_unrelated_sentence | Real Jev returns `none` for "capital of Portugal" | integration |
| `test_workflow.py` | | |
| denied_request_is_refused_and_never_executes | Path received → routing → refused; no money moves | unit |
| unavailable_classifier_is_refused_in_routing | A classifier that is down ends the request in `refused`, from `routing` | unit |
| approved_wire_completes_with_one_debit | Path received → routing → executing → completed; one debit | unit |
| wire_fails_when_funds_vanish_after_routing | The balance drops after routing decided; execution re-checks it and fails, no debit | unit |
| closed_account_blocks_a_later_wire | A deletion closes the account; a later wire fails | unit |
| closing_twice_fails_the_second_time | A closed account cannot be closed again | unit |
| `test_jev.py` | | |
| unsure_action_and_count_become_none | Jev answers under 0.5 become `none` and an unknown count | unit |
| unsure_label_becomes_none_and_sure_label_is_kept | The same floor applies to split-request labels | unit |
| `test_minilm.py` | | |
| disk_cached_embed_does_not_load_model | A cached query vector skips the model | unit |
| disk_cached_actions_do_not_load_model | Cached action vectors skip the model | unit |
| minilm_agrees_with_wire_for_canonical_sentence | Real MiniLM agrees with the wire and shows no gap | integration |
| `test_contracts.py` | | |
| pyright_reports_no_errors | Every adapter and fake matches its role's signatures; runs pyright | unit |
| intent_rank_rejects_values_outside_the_contract (5 cases) | Unknown action, confidence above 1 or NaN, unknown count, text score: all rejected | unit |
| off_contract_answer_counts_as_unavailable | A confident answer outside the catalog makes the classifier unavailable | unit |
| off_contract_label_counts_as_unavailable | The same for the labeler | unit |
| unavailable_classifier_returns_none (no key, call fails) | An unavailable classifier returns `None`, never raises or guesses | unit |
| unavailable_labeler_returns_none (no key, call fails) | The same for the labeler | unit |
| classifier_answers_within_the_catalog | Actions stay in the catalog plus `none`; the count is a valid `RequestCount` | unit |
| labeler_answers_once_per_request_in_order | One label per request, in the order given | unit |
| unreachable_splitter_raises_instead_of_inventing_requests | A splitter that cannot reach its model raises, so the router denies | unit |
| splitter_keeps_written_order_and_uses_its_settings | Requests come back in written order, from the configured model | unit |
| explainer_flags_a_gap_below_its_floor (0.46, 0.44) | MiniLM flags a gap just below 0.45 and not just above | unit |
| `test_policy_engines.py` | | |
| prolog_policy_agrees_with_python_policy | Prolog returns the same verdict, reasons in order, and suggestion as Python on 1,728 cases | integration (prolog) |
| prolog_policy_orders_requests_like_python_policy | Prolog orders all 252 sequences of 2 or 3 labels as Python does, ties included | integration (prolog) |
| `test_structured_output.py` | | |
| classify_returns_a_catalog_action (native, instructor) | A local Ollama model returns one catalog action in a valid shape; accuracy printed | integration (ollama) |
| split_returns_a_list_of_requests (native, instructor) | A local Ollama model returns a list of requests in a valid shape; accuracy printed | integration (ollama) |
| `test_intent_understanding.py` | | |
| parse_split_keeps_one_request_per_nonblank_line | The split parser keeps one request per line | unit |
| medgemma_splits_requests (4 sentences) | MedGemma splits in the user's words | integration |

Every unit test was checked by planting the bug it guards against and confirming that the test fails. For the contract tests, each of eleven planted bugs failed only the tests meant to catch it.

## What is not solved

The router uses MedGemma only to split a sentence Jev has counted as several requests. The split is plain text, and Jev labels it. Amounts in words and "the remaining cash" are still not parsed.
