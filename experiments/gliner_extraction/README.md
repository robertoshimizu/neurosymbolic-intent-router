# Experiment: GLiNER2.5 amount and payee extraction

**Question.** Does [GLiNER2.5](https://fastino.ai/blog/gliner2-5-span-free-information-extraction) find the amount and payee in the 15 sentences of `sentences.py` where `parse_request` fails?

**Results** (`gliner2.5-base-v1`, one run, 2026-09-26)

| | GLiNER, amount + payee | GLiNER, + source_account | `parse_request` |
|---|---|---|---|
| Amount | 12/15 | 12/15 | 11/15, **2 wrong values** |
| Payee | 13/15 | 13/15 | 15/15 |
| Source account | — | 2/15 | — |

**Conclusions**
- GLiNER reads amounts better ("five hundred dollars", "$1.5k", "500 USD"), and it never returned a single wrong amount. `parse_request` read "$1.5k" as 1.5.
- It can't tell source from destination. Without `source_account` it gave both accounts as payee in "from … to …". With `source_account`, it labeled the destination as the source in 12 of the 15 sentences.
- The rules must deny unless there is exactly one amount and exactly one payee per request.

```bash
uv run --group experiments pytest experiments/gliner_extraction -m gliner -s
```
