# Experiment: MedGemma full-sentence reader

*Restored from `src/intent_understanding.py` (now `src/adapters/medgemma.py`) before `a16738d`, which removed it from routing. Nothing in `src/` uses it.*

**The question.** Can local MedGemma read a sentence into structured JSON (entities, relationships, events, causal and temporal relationships, concepts, implicit facts) without inventing facts, while keeping each request, question or prohibition and each prospective event as it was written?

**The method.** `reader.py` sends one fixed prompt (`ACTION_FOCUS` + `INSTRUCTIONS` + the text) to `medgemma:27b` on local Ollama, with JSON output, thinking off and temperature 0. `test_reader.py` reads six sentences. Four are general and two are about banking. It prints each reading next to draft criteria for a human reviewer. The only automated checks are keyword smoke checks.

**Why it left routing.** Routing now reads intent with Jev and uses MedGemma only to split requests. The reader's output was never consumed.

**Results.** None recorded.

```bash
uv run pytest experiments/medgemma_reader -m ollama -s
```
