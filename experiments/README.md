# Experiments

Isolated experiments. Nothing in `src/` imports from here, and `uv run pytest` does not collect them; run each one by path.

| Experiment | Question | Run |
|---|---|---|
| [Structured output](structured_output/README.md) | Native Ollama `format=<schema>` or Instructor, on MedGemma and Qwen? | `uv run pytest experiments/structured_output -m ollama -s` |
| [MedGemma reader](medgemma_reader/README.md) | Can MedGemma read a sentence into JSON without inventing facts? | `uv run pytest experiments/medgemma_reader -m ollama -s` |
| Coffee statechart (`coffee/coffee.py`) | Toy python-statemachine example: how events, states and a final state behave. | `uv run python experiments/coffee/coffee.py` |
