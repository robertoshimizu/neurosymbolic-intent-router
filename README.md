# State machine playground

This repository is a small experiment in routing a user's sentence to an action without letting the model authorize the action.

Two pieces of work sit side by side. They meet in one place: the demo passes MedGemma's splitter to `route()` for sentences that make several requests.

- `src/router.py`, `src/policy.py` and `src/transfer.py` route a banking sentence. A classifier proposes an action. Rules about the session and the ledger decide whether that action may run. A wire then moves through a state machine and debits an in-memory balance once.
- `src/intent_understanding.py` is a separate reading of a sentence. It asks a local model what the text supports. Only its `MedGemmaSplitter` is used by the router.

`src/coffee.py` is only a short example of the `python-statemachine` library.

## Architecture

The router never names a model. It defines three roles as small `Protocol`s and depends only on them. Each model is an adapter in its own file that fulfils one role. `src/demo.py` is the composition root: it builds the adapters and hands them to `route()`. Swapping a model means writing a new adapter and changing one line in `demo.py`.

| Role | Contract | Today | File |
|---|---|---|---|
| Classifier | `classify(text)`: action (or `none`), confidence, request count. `label(texts)`: one action per text. `None` means unavailable. | Jev | `src/jev.py` |
| Splitter | `split(text)`: the separate requests, in the order written | MedGemma | `src/intent_understanding.py` |
| Explainer | `explain(text, action)`: a display-only note on the chosen action | MiniLM | `src/minilm.py` |

Arrows mean "imports". The dependency inversion is in the middle band: the arrows run from the adapters (the details) into `router.py` (the high-level rules), because the adapters implement the router's roles and return its types. `router.py` imports only the domain below it, never an adapter. `MedGemmaSplitter` fits `Splitter` by shape alone, so it imports nothing from the router. Two kinds of arrows are left out: the adapters read the action catalog from `policy.py`, and `demo.py` also imports `policy.py` and `transfer.py` to build the sessions and the ledger.

```
┌──────────────────────────────────────────────────────────────────────┐
│ demo.py            composition root: builds the adapters and wires   │
│                    them into route(). The only file naming models.   │
└─────┬───────────────────┬────────────────────┬─────────────────┬─────┘
      │ builds            │ builds             │ builds          │ calls route(text,
      ▼                   ▼                    ▼                 │   classifier=,
┌───────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │   splitter=,
│ jev.py        │  │ minilm.py       │  │ intent_understanding │ │   explainer=)
│ JevClassifier │  │ MiniLMExplainer │  │ MedGemmaSplitter     │ │
└───────┬───────┘  └────────┬────────┘  └──────────┬───────────┘ │
        │ implements        │ implements           ┆ implements  │
        │ Classifier        │ Explainer            ┆ Splitter    │
        │ (imports router)  │ (imports router)     ┆ (no import: │
        │                   │                      ┆  structural)│
        ▼                   ▼                      ▼             ▼
┌──────────────────────────────────────────────────────────────────────┐
│ router.py          «Protocol» Classifier · Splitter · Explainer      │
│                    IntentRank · DescriptionMatch · Decision          │
│                    route() · decide()        imports no adapter      │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ policy.py          catalog · ACTION_PRECEDENCE · Session             │
│                    parse_request · policy.decide()    no model code  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ transfer.py        Ledger · WireTransfer (python-statemachine)       │
└──────────────────────────────────────────────────────────────────────┘
```

### Design decisions

- **Roles, not models.** `route()` and `decide()` take a `classifier`, a `splitter` and an `explainer`. Tests pass simple fakes, so they check the router's rules, not a model's judgment.
- **Each adapter owns its calibration.** Jev's 0.5 confidence floor lives in `jev.py` and turns an unsure answer into `none`. MiniLM's 0.45 description-gap floor lives in `minilm.py`. The router sees only an action or `none`, so two scales never meet in one rule.
- **No fallback.** If the classifier is missing or down, the sentence is denied with "classifier unavailable". MiniLM used to rank in Jev's place, and it gave absurd sentences whichever action was least far away.
- **The explainer never votes.** MiniLM's scores are printed beside the classifier's, and a disagreement is labelled a description gap. It never changes the decision.
- **The splitter has one narrow job.** MedGemma only rewrites a sentence into separate requests. When it also chose actions, it copied the catalog descriptions and lost "$500", so labelling stays with the classifier.
- **One classifier call per sentence.** That call returns both the action and the request count. A second call happens only when there are several requests, and it labels all of them at once.
- **A rule, not a model, orders requests.** `ACTION_PRECEDENCE` in `policy.py` puts reads, then the wire, then deletion, then `none`. Only the first request is decided. The rest are listed as follow-ups and never run on their own.
- **Policy is pure.** `policy.decide(text, action, session, ledger)` judges an action that has already been chosen, with no model code.
- **Keys stay with their adapter.** `jev.py` and `minilm.py` each load their own key from `.env`. The router loads nothing.

### Known limits of this design

- The Protocols and shared types live in `router.py`, so an adapter that imports them loads the router module too. A separate module for the contracts would remove that.
- `Classifier` bundles `classify` and `label`. A model that can only do one must still provide both.
- `request_count` is a plain string. A classifier that answers `"two"` instead of `"several"` would silently be treated as one request.
- A new action means editing `ACTION_CATALOG`, `ACTION_PRECEDENCE` and the `if action == ...` branches in `policy.py`. With four actions this was left as is.

## Run

```bash
uv sync
uv run python src/demo.py
uv run python src/coffee.py
uv run pytest -m "not integration"
```

The demo calls TypeSafe Jev when `TYPESAFE_API_KEY` is set in `.env`. That file is gitignored. MiniLM weights and embedding vectors are cached under `.cache/`, which is also gitignored. Hugging Face downloads use `HF_TOKEN` from the same `.env` file. Neither value is printed.

The sentence reader and the splitter call local Ollama. They expect `medgemma:27b`. The full reading does not use the router; `route()` uses only the splitter.

```bash
uv run pytest tests/test_intent_understanding.py -m integration -s
```

## The router

`route()` in `src/router.py` is the router's entry point; `src/demo.py` is the program's. Jev, MedGemma and MiniLM appear below as the adapters that `demo.py` wires in today.

1. The classifier receives the sentence in one call. Jev answers two closed questions: which action (the four catalog actions plus `none`), and how many requests the sentence makes (`none`, `one`, `several`). Identity, role, and balance are not sent. If the classifier is missing or down, the sentence is denied with "classifier unavailable".
2. If the count is not `several`, the sentence goes straight to `decide()` with that answer. Most sentences stop here.
3. If the count is `several`, the splitter (MedGemma) rewrites the sentence as plain requests in the user's own words. It does not choose actions. The classifier then labels every request in one call.
4. `ACTION_PRECEDENCE` orders the requests: reads, then the wire, then deletion, then `none`. Only the first goes to `decide()`. The others are listed as follow-ups and are not evaluated. If the first is denied, nothing is listed. If the split or the labelling fails, the sentence is denied.

`decide()` then applies the rules.

1. A `none` action is denied with "no matching action". Banking rules do not run. For Jev, an answer under 0.5 confidence has already become `none` inside the adapter.
2. For a catalog action, the explainer (MiniLM) scores the sentence against the written action descriptions. The scores are printed beside the classifier's and never change the choice.
3. `policy.decide()` uses the session and the ledger. A wire needs authentication, an active account, a parsed amount, an allowlisted payee, and enough balance. A deletion needs an authenticated admin. A wire or a deletion stops at confirmation. `WireTransfer` in `src/transfer.py` moves drafted, awaiting confirmation, authorized, submitted, settled. The ledger debits on settle, once per transfer id.

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
               │       │ Classifier.label                 │
               │       │ 1 call, 1 question per request   │
               │       └────────────────┬─────────────────┘
               │           fails ──────────► DENY
               │                        ▼
               │       ┌──────────────────────────────────┐
               │       │ ACTION_PRECEDENCE      policy.py │
               │       │ reads → wire → delete → none     │
               │       │ first is decided; the rest       │
               │       │ become follow_ups                │
               │       └────────────────┬─────────────────┘
               │                        │ first request + its rank
               ▼                        ▼
          ┌──────────────────────────────────┐
          │ decide()               router.py │
          │ 1. action == none ──► DENY       │
          │    "no matching action"          │
          │ 2. Explainer.explain             │
          │    display only, no vote         │
          │ 3. policy.decide()     policy.py │
          │    parse_request, then           │
          │    evaluate_action with          │
          │    session + ledger ──► DENY     │
          │    wire or delete ──► CONFIRM    │
          │    otherwise ──► EXECUTE         │
          └────────────────┬─────────────────┘
                           ▼
          Decision (+ follow_ups if not denied)
                           │
                           ▼
          print_decision()               demo.py
                           │
            needs confirmation + wire?
                           ▼
          ┌──────────────────────────────────┐
          │ WireTransfer         transfer.py │
          │ drafted → awaiting_confirmation  │
          │ → authorized → submitted →       │
          │ settled → Ledger.debit()         │
          │ (once per transfer id)           │
          └──────────────────────────────────┘
```

On a clear sentence, "I want to send $5,000 to my external bank account.", the three sessions behave as follows. A guest is denied because they are not signed in. A customer with no balance is denied for insufficient funds. A customer with $10,000 is asked to confirm, and settlement leaves $5,000.

## What the router does not understand

Jev assigns one label from a list. It does not unpack a sentence that contains two requests, and a confident label can be only one reading of a mixed sentence. `route()` now handles that case by counting, splitting, and ordering, as described above.

These showed up in live checks before `route()` existed:

- "How much money do I have before I wire funds out?" Jev chose the balance view with very high confidence. MiniLM's nearest description was the wire. The balance view was executed.
- "Where can I read answers to common questions about transfers?" Jev chose the FAQ. MiniLM leaned toward the wire. The FAQ was executed.
- "Move five thousand dollars to my external bank account." Jev chose the wire, then policy denied it because the amount parser only accepts a `$` figure.
- "Close this account and send the remaining cash to my external bank account." Jev returned `none`.
- An unrelated sentence such as "What is the capital of Portugal?" correctly stopped at `none`.

With `route()`, one live run gave:

- "Close this account and send $500 to my external bank account." (admin) Jev counted several. MedGemma split it into "Close this account." and "Send $500 to my external bank account.". Jev labeled them delete (0.95) and wire (0.98). The rule put the wire first. It stopped at confirmation, and "Close this account." was listed as a follow-up.
- "Close this account and send the remaining cash to my external bank account." The same split and order. The wire was denied because "remaining cash" is not an amount, so nothing was listed.
- "Show my balance and then wire $500 to my external bank account." The balance view executed, and the wire was listed.
- "How much money do I have before I wire funds out?" Jev counted one request. The balance view executed.
- "I want to send $5,000…" and "What is the capital of Portugal?" Jev counted one request each and gave the same results as before.

Each case was run once. How stable the count, the split, and the labels are across runs has not been measured.

TypeSafe's own guidance is that one choice should be a snap judgment, and that a sentence with several questions should be split into small questions whose answers the application combines. A low confidence is a handoff to a person or to a normal language model. Jev is not the component that finishes reading a complex sentence.

MiniLM makes the gap visible. Each cosine is closeness to a description we wrote. It is not a calibrated probability, and it must not cast a second vote. That is why it no longer ranks when Jev is down: as a fallback it gave an absurd sentence whichever action was least far away.

## The open task: read the sentence first

`src/intent_understanding.py` is the attempt to read a sentence before any router exists. The router uses only its `MedGemmaSplitter`, wired in by the demo; the full reading is not used by the router.

The prompt asks MedGemma 27B, through local Ollama with thinking turned off, to extract entities, relationships, events, causes, concepts, implicit facts, and temporal links. Each claim is marked:

- `EXPLICIT` when the text states it
- `ENTAILED` when it necessarily follows
- `INFERRED` when it is plausible but not necessary
- `UNKNOWN` when the text does not establish it
- `AMBIGUOUS` when a pronoun has more than one possible antecedent

Time order ("after", "shortly after", "subsequently") must not be rewritten as a cause. World knowledge must not promote a guess into a fact. The same instructions are used for every sentence. Only the passage after `TEXT TO ANALYZE:` changes.

The command-line form that produced a usable reading is one user message: those instructions, then the sentence. The code sends that same message. It stores the JSON MedGemma returns. It does not rename fields and it does not drop an item because the field is called `entity` or `event` instead of `statement`.

On the supplier passage, that reading keeps the withdrawal and the stopped shipping as events, keeps "shortly after" and "subsequently" as time, and marks a causal link between them as `INFERRED`. On "What is the capital of Portugal?" it returns Portugal and a capital link whose target is unknown, instead of an empty object.

## Challenges

**A closed label set cannot represent a mixed sentence.** Adding `none` stops unrelated text. It does not split "close the account and send the cash" into two explicit requests. Forcing one winner hides the second request.
*Status: addressed, not proven.* Jev counts the requests, MedGemma splits, Jev labels each part, and a fixed rule picks the first. The second request is listed, not hidden. The evidence is one live run per sentence.

**Similarity and probability are different scales.** MiniLM's 0.12 and Jev's 0.98 do not mean the same thing. Printing them in one column, or letting a cosine floor run before the label is known, produced denials that talked about banking rules for a sentence that was not a banking request.
*Status: solved.* MiniLM only explains a Jev choice and never overrides it, and it no longer ranks on its own. Each adapter keeps its own floor, so the router only sees an action or `none`. Unit tests cover each rule.

**The amount in words and the amount in symbols diverged.** Jev treated "five thousand dollars" as a wire. The ledger rule never saw an amount, because the parser looked for `$5,000`.
*Status: not solved.* The parser still accepts only a `$` figure. "The remaining cash" is not resolved to a balance either.

**Rewriting a prompt that already worked made the readings worse.** The original instructions, sent as one message, already separated time from cause. Later drafts added a private JSON shape, moved the rules into a system message, and stacked extra bans ("never return an empty list", "do not attach a time to an event"). MedGemma followed the newest ban and dropped an earlier one. Questions came back empty, events were reduced to bare verbs, or the same events were copied into the causal section so the list would not be blank. A second bug hid good answers: the test kept only objects that contained a field named `statement`, so a correct payload in MedGemma's own shape was printed as empty.

*Status: a lesson, applied again.* The splitter has one narrow job. An attempt to make MedGemma also pick catalog actions made it copy the action descriptions in place of the user's words, and "$500" was lost. That job went back to Jev.

**"Omit anything uncertain" and "always return something" cannot both be the loudest rule.** Precision is what keeps a guessed cause out. Treated as the only rule, it also deletes a question, because a question states no fact. The fix used here was to stop editing the instructions and to accept the model's JSON.
*Status: open.* The full reading is unchanged and is not used for routing.

## Tests

`uv run pytest -m "not integration"` runs the unit tests. Integration tests call MiniLM, Jev, or MedGemma. Router tests use a fake classifier, splitter and explainer, so they check the router's rules, not the models' judgment.

| Test | Goal | Type |
|---|---|---|
| `test_policy.py` | | |
| parse_amount_with_comma_and_dollar | "$5,000" parses to 5000 and the payee binds | unit |
| unknown_payee_does_not_bind | A payee outside the allowlist does not bind | unit |
| guest_wire_denied | A signed-out guest cannot wire | unit |
| delete_denied_for_customer | Only an admin may delete | unit |
| `test_router.py` | | |
| insufficient_funds_suggests_balance_view | A wire above the balance is denied and suggests the balance view | unit |
| confirm_then_settle_debits_once | Settlement debits once per transfer id | unit |
| confirm_refused_when_balance_drained | Confirmation is refused when funds are gone | unit |
| jev_none_stops_before_policy | A `none` action is denied before policy | unit |
| jev_wire_still_needs_confirmation | A wire chosen by the classifier still needs confirmation | unit |
| minilm_disagreement_does_not_override_jev | The explainer never replaces the classifier's choice | unit |
| classifier_failure_denies_without_guessing | A classifier that is down means deny, not a guess | unit |
| route_orders_money_movement_before_deletion | Close + wire: the wire goes first, close is listed | unit |
| route_denied_first_request_offers_no_follow_ups | A denied first request lists nothing | unit |
| route_one_request_uses_the_single_jev_rank | One request skips the split | unit |
| route_failed_split_denies_without_guessing | A failed split is denied | unit |
| route_failed_labeling_denies_without_guessing | A failed labeling is denied | unit |
| route_none_label_is_ordered_last | A `none` label goes last | unit |
| live_jev_abstains_on_unrelated_sentence | Real Jev returns `none` for "capital of Portugal" | integration |
| `test_jev.py` | | |
| unsure_action_and_count_become_none | Jev answers under 0.5 become `none` and an unknown count | unit |
| unsure_label_becomes_none_and_sure_label_is_kept | The same floor applies to split-request labels | unit |
| `test_minilm.py` | | |
| disk_cached_embed_does_not_load_model | A cached query vector skips the model | unit |
| disk_cached_actions_do_not_load_model | Cached action vectors skip the model | unit |
| minilm_agrees_with_wire_for_canonical_sentence | Real MiniLM agrees with the wire and shows no gap | integration |
| `test_intent_understanding.py` | | |
| prompt_appends_only_the_text | The prompt only appends the sentence | unit |
| parser_keeps_items_without_a_statement_field | The JSON parser keeps MedGemma's own fields | unit |
| parse_split_keeps_one_request_per_nonblank_line | The split parser keeps one request per line | unit |
| medgemma_reads_sentence (6 sentences) | Full reading, with loose keyword checks | integration |
| medgemma_splits_requests (4 sentences) | MedGemma splits in the user's words | integration |

Every unit test except `parser_keeps_items_without_a_statement_field` was checked by planting the bug it guards against and confirming that the test fails. `medgemma_reads_sentence` is weaker than it looks. Its keyword checks passed on a supplier reading that asserted a causal link the rules forbid.

## What is not solved

The router uses MedGemma only to split a sentence Jev has counted as several requests. The split is plain text, and Jev labels it. Amounts in words and "the remaining cash" are still not parsed.

The full sentence reader is still an experiment. It is not a front door for the router. MedGemma still adds plausible facts (a supplier "dependent" on the credit line, a subsidiary "capable" of production) and still sometimes misses the time link in "before I wire funds out". Status labels are not stable from one run to the next, and `ENTAILED` is sometimes spelled `ENTAILLED`.

Until those readings are trustworthy on mixed sentences, on questions, and on pronouns, they should not choose an action, fill an amount, or move the ledger.
