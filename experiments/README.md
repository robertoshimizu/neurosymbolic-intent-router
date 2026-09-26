# Experiments

Isolated experiments. Nothing in `src/` imports from here, and `uv run pytest` does not collect them; run each one by path. Those that need extra packages use the `experiments` dependency group.

| Experiment | Question | Run |
|---|---|---|
| [Structured output](structured_output/README.md) | Native Ollama `format=<schema>` or Instructor, on MedGemma and Qwen? | `uv run --group experiments pytest experiments/structured_output -m ollama -s` |
| [GLiNER extraction](gliner_extraction/README.md) | Does GLiNER2.5 find the amount and payee where `parse_request` fails? | `uv run --group experiments pytest experiments/gliner_extraction -m gliner -s` |
| [MedGemma reader](medgemma_reader/README.md) | Can MedGemma read a sentence into JSON without inventing facts? | `uv run pytest experiments/medgemma_reader -m ollama -s` |
| Coffee statechart (`coffee/coffee.py`) | Toy python-statemachine example: how events, states and a final state behave. | `uv run python experiments/coffee/coffee.py` |
