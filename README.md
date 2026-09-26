# State machine playground

This repository is a small experiment in routing a user's sentence to an action without letting the model authorize the action.

Two pieces of work sit side by side. They meet in one place: `route()` in `hybrid.py` uses the MedGemma splitter when Jev counts several requests.

- `src/hybrid.py` and `src/transfer.py` route a banking sentence. A classifier proposes an action. Rules about the session and the ledger decide whether that action may run. A wire then moves through a state machine and debits an in-memory balance once.
- `src/intent_understanding.py` is a separate reading of a sentence. It asks a local model what the text supports, and it does not know about the router.

`src/coffee.py` is only a short example of the `python-statemachine` library.

## Run

```bash
uv sync
uv run python src/hybrid.py
uv run python src/coffee.py
uv run pytest -m "not integration"
```

`hybrid.py` calls TypeSafe Jev when `TYPESAFE_API_KEY` is set in `.env`. That file is gitignored. MiniLM weights and embedding vectors are cached under `.cache/`, which is also gitignored. Hugging Face downloads use `HF_TOKEN` from the same `.env` file. Neither value is printed.

The sentence reader and the splitter call local Ollama. They expect `medgemma:27b`. The full reading does not use the router; `route()` uses only `split_requests()`.

```bash
uv run pytest tests/test_intent_understanding.py -m integration -s
```

## The router

`route()` in `src/hybrid.py` is the entry point.

1. Jev receives the sentence in one call with two closed questions: which action (the four catalog actions plus `none`), and how many requests the sentence makes (`none`, `one`, `several`). Identity, role, and balance are not sent.
2. If the count is not `several`, the sentence goes straight to `decide()` with that Jev answer. Most sentences stop here.
3. If the count is `several`, MedGemma splits the sentence into plain requests in the user's own words. It does not choose actions. Jev then labels every split request in one call. A label under 0.5 confidence counts as `none`.
4. A fixed rule, `ACTION_PRECEDENCE`, orders the requests: reads, then the wire, then deletion, then `none`. Only the first goes to `decide()`. The others are listed as follow-ups and are not evaluated. If the first is denied, nothing is listed. If the split or the labeling fails, the sentence is denied.

`decide()` then applies the rules.

1. If Jev returns `none`, or its confidence is under 0.5, the outcome is deny and the reason is "no matching action". Banking rules do not run.
2. If Jev commits to a catalog action, MiniLM scores the sentence against the written action descriptions. Those cosines are printed beside Jev's probabilities. They do not change the choice. A disagreement is labeled a description gap.
3. Policy then uses the session and the ledger. A wire needs authentication, an active account, a parsed amount, an allowlisted payee, and enough balance. A deletion needs an authenticated admin. A wire or a deletion stops at confirmation. `WireTransfer` in `src/transfer.py` moves drafted, awaiting confirmation, authorized, submitted, settled. The ledger debits on settle, once per transfer id.

If Jev cannot be called, MiniLM ranks the catalog alone and the request count is unknown, so the sentence is treated as one request. The fallback denies when the nearest cosine is under 0.45 or the top two are within 0.08.

```
                       user sentence
                             │
                             ▼
               ┌───────────────────────────┐
               │  route()                  │
               │  Jev: 1 call, 2 questions │
               │   • action (catalog+none) │
               │   • request_count         │
               │  (Jev down → MiniLM rank) │
               └─────────────┬─────────────┘
                             │
               request_count == "several"?
                 │                      │
                no                     yes
                 │                      ▼
                 │     ┌──────────────────────────────┐
                 │     │ MedGemma split_requests()    │
                 │     │ → plain request lines        │
                 │     └──────────────┬───────────────┘
                 │          fails or < 2 lines ──► DENY
                 │                    ▼
                 │     ┌──────────────────────────────┐
                 │     │ Jev label_with_jev()         │
                 │     │ 1 call, 1 question per line  │
                 │     └──────────────┬───────────────┘
                 │          fails ──────────────► DENY
                 │                    ▼
                 │     ┌──────────────────────────────┐
                 │     │ ACTION_PRECEDENCE (rule)     │
                 │     │ reads → wire → delete → none │
                 │     │ pick first; keep rest as     │
                 │     │ follow_ups                   │
                 │     └──────────────┬───────────────┘
                 │                    │ first request + its Jev rank
                 ▼                    ▼
               ┌───────────────────────────┐
               │  decide()                 │
               │  1. none / conf < 0.5     │──► DENY "no matching action"
               │  2. MiniLM cosines        │    (explanation only, no vote)
               │  3. parse_request()       │    ($ amount, allowlisted payee)
               │  4. evaluate_action()     │──► DENY (policy reason)
               │     session + ledger      │
               │  5. MiniLM-only gates     │──► DENY (score / margin)
               │  6. wire or delete?       │──► NEEDS_CONFIRMATION
               │  7. otherwise             │──► EXECUTE
               └─────────────┬─────────────┘
                             ▼
               Decision (+ follow_ups if not denied)
                             │
                             ▼
               print_decision()   (the __main__ demo)
                             │
            needs_confirmation + wire?
                             ▼
               ┌───────────────────────────┐
               │ WireTransfer (transfer.py)│
               │ drafted → awaiting_conf → │
               │ authorized → submitted →  │
               │ settled → Ledger.debit()  │
               │ (once per transfer id)    │
               └───────────────────────────┘
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

MiniLM makes the gap visible. Each cosine is closeness to a description we wrote. It is not a calibrated probability, and it must not cast a second vote. When Jev is down, that same nearest-neighbor step is the whole decision, which brings back the original problem: an absurd sentence still receives whichever action is least far away.

## The open task: read the sentence first

`src/intent_understanding.py` is the attempt to read a sentence before any router exists. `hybrid.py` imports only its `split_requests()`; the full reading is not used by the router.

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
*Status: solved.* MiniLM only explains a Jev choice and never overrides it. Its score and margin floors apply only when MiniLM ranks alone. Unit tests cover each rule.

**The amount in words and the amount in symbols diverged.** Jev treated "five thousand dollars" as a wire. The ledger rule never saw an amount, because the parser looked for `$5,000`.
*Status: not solved.* The parser still accepts only a `$` figure. "The remaining cash" is not resolved to a balance either.

**Rewriting a prompt that already worked made the readings worse.** The original instructions, sent as one message, already separated time from cause. Later drafts added a private JSON shape, moved the rules into a system message, and stacked extra bans ("never return an empty list", "do not attach a time to an event"). MedGemma followed the newest ban and dropped an earlier one. Questions came back empty, events were reduced to bare verbs, or the same events were copied into the causal section so the list would not be blank. A second bug hid good answers: the test kept only objects that contained a field named `statement`, so a correct payload in MedGemma's own shape was printed as empty.

*Status: a lesson, applied again.* The splitter has one narrow job. An attempt to make MedGemma also pick catalog actions made it copy the action descriptions in place of the user's words, and "$500" was lost. That job went back to Jev.

**"Omit anything uncertain" and "always return something" cannot both be the loudest rule.** Precision is what keeps a guessed cause out. Treated as the only rule, it also deletes a question, because a question states no fact. The fix used here was to stop editing the instructions and to accept the model's JSON.
*Status: open.* The full reading is unchanged and is not used for routing.

## Tests

`uv run pytest -m "not integration"` runs the unit tests. Integration tests call MiniLM, Jev, or MedGemma.

| Test | Goal | Type |
|---|---|---|
| `test_hybrid.py` | | |
| parse_amount_with_comma_and_dollar | "$5,000" parses to 5000 and the payee binds | unit |
| unknown_payee_does_not_bind | A payee outside the allowlist does not bind | unit |
| guest_wire_denied | A signed-out guest cannot wire | unit |
| delete_denied_for_customer | Only an admin may delete | unit |
| insufficient_funds_suggests_balance_view | A wire above the balance is denied and suggests the balance view | unit |
| ambiguous_margin_denied | MiniLM alone: top two within 0.08 is denied | unit |
| minilm_far_from_every_action_denied | MiniLM alone: nearest cosine under 0.45 is denied | unit |
| confirm_then_settle_debits_once | Settlement debits once per transfer id | unit |
| confirm_refused_when_balance_drained | Confirmation is refused when funds are gone | unit |
| disk_cached_embed_does_not_load_model | A cached query vector skips the model | unit |
| disk_cached_actions_do_not_load_model | Cached action vectors skip the model | unit |
| jev_none_stops_before_policy | Jev `none` is denied before policy | unit |
| jev_low_confidence_stops_before_policy | Jev under 0.5 is denied before policy | unit |
| jev_wire_still_needs_confirmation | A wire chosen by Jev still needs confirmation | unit |
| minilm_disagreement_does_not_override_jev | MiniLM never replaces Jev's choice | unit |
| jev_failure_falls_back_to_minilm | Jev down means MiniLM ranks | unit |
| route_orders_money_movement_before_deletion | Close + wire: the wire goes first, close is listed | unit |
| route_denied_first_request_offers_no_follow_ups | A denied first request lists nothing | unit |
| route_one_request_uses_the_single_jev_rank | One request skips the split | unit |
| route_failed_split_denies_without_guessing | A failed split is denied | unit |
| route_failed_labeling_denies_without_guessing | A failed labeling is denied | unit |
| route_low_confidence_label_is_ordered_last | A label under 0.5 counts as `none` and goes last | unit |
| minilm_ranks_wire_highest_for_canonical_sentence | Real MiniLM ranks the wire first | integration |
| live_jev_abstains_on_unrelated_sentence | Real Jev returns `none` for "capital of Portugal" | integration |
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
