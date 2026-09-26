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

"Close this account and send $500 to my external bank account." for three customers (`uv run --env-file .env python src/demo.py`, one run on 2026-09-26). The models read it the same way every time: two requests, a wire (Jev 0.99) and a close. The Prolog rules put the wire first and judge it:

| | Not authenticated, $0 | Customer, $0 | Customer, $10,000 |
|---|---|---|---|
| Wire allowed | no | no | yes |
| Reasons | not authenticated; account not active; payee not on the allowlist; insufficient funds | insufficient funds | |
| Suggestion | none (the balance view is denied too) | view the balance | |
| Workflow | refused | refused | completed; balance $9,500; "Close this account." listed as a follow-up |

The same reading, three outcomes: only the rules and the ledger differ. `--policy python` gives the same decisions.

## Architecture

The router depends only on roles defined in `src/contracts.py`. `src/demo.py` chooses the implementations.

| Role | Job | Today |
|---|---|---|
| Classifier, Labeler | Intent and request count; a label for each split request | Jev (`jev.py`) |
| Splitter | The separate requests, as written | MedGemma on Ollama (`intent_understanding.py`) |
| Explainer | Closeness to each action description, display only | MiniLM (`minilm.py`) |
| Policy | Order of requests; allow or deny with every reason; suggestion | Prolog (`prolog_policy.py`, `policy.pl`) or Python (`python_policy.py`) |

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
