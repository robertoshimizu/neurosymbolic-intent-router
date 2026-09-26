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

The **Allowed?** column is the symbolic side at work. The model scores are identical in all three runs; the permissions are not:

| Action | Not authenticated | No money | Funded |
|---|---|---|---|
| wire_transfer_funds | NO | NO | YES |
| view_account_balance | NO | YES | YES |
| view_public_faq | YES | YES | YES |
| delete_account | NO | NO | NO |

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

## Status

Not measured yet: everything above comes from single runs on hand-picked sentences.

Next:
- a labelled evaluation set with a model-only baseline: precision, recall, and wrong actions executed;
- amounts in words and "the remaining cash" (amount parsing is still Python only, outside the reasoner);
- how stable Jev's count and MedGemma's split are across runs.
