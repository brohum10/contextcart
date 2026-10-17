# ContextCart

**Pack the most useful evidence into a limited AI context budget—without dropping required sources.**

[![Tests](https://github.com/brohum10/contextcart/actions/workflows/tests.yml/badge.svg)](https://github.com/brohum10/contextcart/actions/workflows/tests.yml)

ContextCart is an offline planner for retrieval-augmented AI systems. Given alternative versions of each source, integer costs, and utility scores, it selects an optimal combination under a fixed budget. It uses exact multiple-choice knapsack dynamic programming, not a greedy ranking heuristic.

Python 3.12+ · standard library only · no API keys · no network calls

## Run the example

Download the repository, open its folder, then run:

```sh
python demo.py
python contextcart.py example.json
python -m unittest -v
```

The example reserves 12 units from a 24-unit context window, leaving 12 for evidence. Required policy costs 2; sources A, B, C compete for the remaining 10.

| Selection | Evidence cost | Utility |
| --- | ---: | ---: |
| Policy + A full (greedy utility/cost) | 8 | 14 |
| Policy + B full + C full (optimal) | 12 | 20 |

**All demo costs and utilities are invented teaching units.** They are not token counts measured from the included text, model accuracy, or production savings. The example demonstrates a packing decision where greedy ranking loses.

## What it guarantees

- At most one variant per source; exactly one for each required source.
- No partial-text truncation: the selected variant text is returned verbatim.
- Maximum summed utility within the supplied additive cost budget.
- Stable ties: prefer less cost, then the lexicographically smallest sequence of `(source_id, variant_id)` pairs.
- Input order does not change the plan or normalized input fingerprint.
- Explicit failure when required evidence cannot fit, instead of silently omitting it.
- SHA-256 text fingerprints and source/variant IDs make the chosen material traceable.

These guarantees apply to **the supplied mathematical model**, not the quality or actual token length of an LLM prompt.

## Where it fits

```text
retrieval results → approved text variants + tokenizer costs + relevance scores
                                        │
                                  ContextCart
                                        │
                       selected evidence + provenance manifest
                                        │
               application renders and tokenizes the complete prompt
                                        │
                    final size check → provider call → answer evaluation
```

ContextCart handles only the selection step. Retrieval, summarization, scoring, tokenization, prompt assembly, and inference belong to the application. It does not treat source text as instructions, execute it, or claim to sanitize prompt injection.

## Input format

See [example.json](example.json) for a complete runnable input. The top-level object has exactly five fields:

- `context_window`: total positive integer capacity.
- `prompt_tokens`: reserved instructions, conversation, wrappers, and fixed overhead.
- `output_tokens`: capacity reserved for a response.
- `safety_tokens`: additional margin.
- `groups`: sources with unique `id`, boolean `required`, and 1–8 `variants` each.

Each variant has exactly `id`, `tokens` (positive integer cost), `utility` (nonnegative integer score), and `text` (nonempty). Variant IDs must be unique within their source. IDs use ASCII letters, digits, dots, underscores, or hyphens and are at most 80 characters.

For real use, obtain costs from the model-specific tokenizer and scores from a consistent retrieval/reranking policy. Account for separators, citation wrappers, message framing, and all fixed content. Independent token counts can be non-additive after concatenation, so **re-tokenize the complete rendered prompt before dispatch**, even if the planner says it fits. A safety margin is not a proof of fit.

Do not upload private documents to this public repository. The program itself runs locally, but its JSON output includes all selected text. Hashes are change detectors, not access control, authenticity certificates, or privacy protection.

## Library use

```python
from contextcart import load, optimize

report = optimize(load("example.json"))
for item in report["selected"]:
    print(item["source_id"], item["variant_id"], item["text_sha256"])
print(report["total_utility"], report["remaining_tokens"])
```

The report includes budget accounting, selected texts and hashes, omitted source IDs, and a normalized input hash. Omissions mean “not selected by the global optimizer”; they do not prove that a source was irrelevant. Output order is canonical source-ID order, not an optimized model attention order.

The CLI prints JSON and exits `0` on success, `3` for infeasible reservations/required sources, and `2` for malformed inputs or I/O errors. Errors go to stderr. Argument parser usage errors also exit `2`.

```sh
python contextcart.py example.json > report.json
```

## Algorithm and bounds

Process sources in sorted order. Each dynamic-programming state records the best utility and selection at one exact cost. For each source, transition once to a chosen variant, or skip it if optional. Keep a separate next-state table so a source cannot be selected twice. Final selection maximizes utility, then minimizes cost, then applies the deterministic tie rule.

Let B be the evidence budget, V the number of variant/skip options, and G the number of sources. There are O(B × V) candidate transitions. This readable implementation stores selection tuples, so tuple copying and tie comparison add an O(G) worst-case factor, with O(B × G) state storage. It is intended for bounded retrieval sets, not million-document optimization.

Limits: 2 MB input files, 40 sources, 8 variants/source, 50 KB UTF-8 text/variant, 20,000 evidence units, and a conservative bound of 2 million candidate transitions. These are validation limits, not guaranteed wall-clock or memory deadlines. The file-size limit applies to `load`; direct library callers are responsible for total object size. Oversized inputs fail explicitly rather than falling back to an approximate algorithm.

## Verification

**29 tests**, run on Python 3.12 and 3.14, include an independent exhaustive-search oracle covering **250 seeded randomized cases**. The oracle compares utility, total cost, and exact tie-broken selections, including infeasible cases.

Other checks cover required sources, alternate versions, no double-selection, zero budgets, strict fields, duplicate IDs/JSON keys, numeric types, resource bounds, ordering invariance, input immutability, Unicode text preservation, fingerprints, and CLI exit codes.

## Limitations and next steps

- Utility is additive and caller-supplied. The planner cannot infer truth, semantic relevance, source complementarity, or redundancy across different sources.
- A required source guarantees one of its variants, not preservation of every fact in its original document. Validate summaries upstream.
- There is no tokenizer/model adapter, live LLM benchmark, automatic summarizer, web UI, or cloud service.
- Token costs can be wrong. Final prompt tokenization and provider-specific limits must be checked externally.
- Dependencies between sources, cross-source diversity constraints, and answer quality optimization are not modeled.

Useful next work: connect a real tokenizer and reranker, evaluate downstream answer quality on held-out queries, then compare this additive optimizer against diversity-aware selection. No quality or savings claims are made before that experiment exists.

## License

MIT. See [LICENSE](LICENSE).
