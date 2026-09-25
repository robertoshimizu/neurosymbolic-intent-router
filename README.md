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

The sentence reader calls local Ollama. It expects `medgemma:27b` and does not use the router.

```bash
uv run pytest tests/test_intent_understanding.py -m integration -s
```

## The router

`decide()` in `src/hybrid.py` does four things.

1. Jev receives the sentence and one closed choice: the four catalog actions plus `none`. Identity, role, and balance are not sent.
2. If Jev returns `none`, or its confidence is under 0.5, the outcome is deny and the reason is "no matching action". Banking rules do not run.
3. If Jev commits to a catalog action, MiniLM scores the sentence against the written action descriptions. Those cosines are printed beside Jev's probabilities. They do not change the choice. A disagreement is labeled a description gap.
4. Policy then uses the session and the ledger. A wire needs authentication, an active account, a parsed amount, an allowlisted payee, and enough balance. A wire or a deletion stops at confirmation. `WireTransfer` in `src/transfer.py` moves drafted, awaiting confirmation, authorized, submitted, settled. The ledger debits on settle, once per transfer id.

If Jev cannot be called, MiniLM ranks the catalog alone. That fallback always names some action.

On a clear sentence, "I want to send $5,000 to my external bank account.", the three sessions behave as follows. A guest is denied because they are not signed in. A customer with no balance is denied for insufficient funds. A customer with $10,000 is asked to confirm, and settlement leaves $5,000.

## What the router does not understand

Jev assigns one label from a list. It does not unpack a sentence that contains two requests, and a confident label can be only one reading of a mixed sentence.

These showed up in live checks:

- "How much money do I have before I wire funds out?" Jev chose the balance view with very high confidence. MiniLM's nearest description was the wire. The balance view was executed.
- "Where can I read answers to common questions about transfers?" Jev chose the FAQ. MiniLM leaned toward the wire. The FAQ was executed.
- "Move five thousand dollars to my external bank account." Jev chose the wire, then policy denied it because the amount parser only accepts a `$` figure.
- "Close this account and send the remaining cash to my external bank account." Jev returned `none`.
- An unrelated sentence such as "What is the capital of Portugal?" correctly stopped at `none`.

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

**Similarity and probability are different scales.** MiniLM's 0.12 and Jev's 0.98 do not mean the same thing. Printing them in one column, or letting a cosine floor run before the label is known, produced denials that talked about banking rules for a sentence that was not a banking request.

**The amount in words and the amount in symbols diverged.** Jev treated "five thousand dollars" as a wire. The ledger rule never saw an amount, because the parser looked for `$5,000`.

**Rewriting a prompt that already worked made the readings worse.** The original instructions, sent as one message, already separated time from cause. Later drafts added a private JSON shape, moved the rules into a system message, and stacked extra bans ("never return an empty list", "do not attach a time to an event"). MedGemma followed the newest ban and dropped an earlier one. Questions came back empty, events were reduced to bare verbs, or the same events were copied into the causal section so the list would not be blank. A second bug hid good answers: the test kept only objects that contained a field named `statement`, so a correct payload in MedGemma's own shape was printed as empty.

**"Omit anything uncertain" and "always return something" cannot both be the loudest rule.** Precision is what keeps a guessed cause out. Treated as the only rule, it also deletes a question, because a question states no fact. The fix used here was to stop editing the instructions and to accept the model's JSON.

## What is not solved

The sentence reader is still an experiment. It is not a front door for the router. MedGemma still adds plausible facts (a supplier "dependent" on the credit line, a subsidiary "capable" of production) and still sometimes misses the time link in "before I wire funds out". Status labels are not stable from one run to the next, and `ENTAILED` is sometimes spelled `ENTAILLED`.

Until those readings are trustworthy on mixed sentences, on questions, and on pronouns, they should not choose an action, fill an amount, or move the ledger.
