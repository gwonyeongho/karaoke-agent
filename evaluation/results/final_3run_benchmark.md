# Karaoke Agent Benchmark

Generated: `2026-09-21T20:54:39.256662+00:00`

> These are measured local results, not estimates. Warm-ups and RAG setup/index time are excluded from per-case latency. Failed calls remain failures and are not imputed.

## Results

| Group | Calls | Primary accuracy | Avg (s) | Median (s) | StdDev (s) | P95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| `command/raw/qwen3:1.7b` | 27 | 81.5% (22/27) | 4.547268329630714 | 4.338308999998844 | 0.9121756028627224 | 6.965026799996849 |
| `command/raw/qwen3:8b` | 27 | 59.3% (16/27) | 12.064932522222753 | 10.27888580000581 | 4.663353322404055 | 17.477594800002407 |
| `command/structured/qwen3:1.7b` | 27 | 100.0% (27/27) | 4.830235966666131 | 4.636482400004752 | 0.9546635607012743 | 7.271753100001661 |
| `command/structured/qwen3:8b` | 27 | 100.0% (27/27) | 13.12556344444386 | 12.547181400004774 | 2.8575154049452176 | 19.189056100003654 |
| `qa/direct/qwen3:1.7b` | 27 | 14.3% (3/21) | 4.0315485407422615 | 3.7858969000080833 | 1.217476585069261 | 5.636301800012006 |
| `qa/direct/qwen3:8b` | 27 | 0.0% (0/21) | 10.908434996295707 | 10.257920900010504 | 3.627712774343143 | 17.152134999996633 |
| `qa/full_context/qwen3:1.7b` | 27 | 85.7% (18/21) | 2.611141970368745 | 2.5196391999925254 | 0.7764637841668113 | 4.150970999995479 |
| `qa/full_context/qwen3:8b` | 27 | 85.7% (18/21) | 7.93314710000144 | 8.191527599992696 | 1.8641642098485491 | 11.11432190000778 |
| `qa/rag/qwen3:1.7b` | 27 | 61.9% (13/21) | 1.268638662965343 | 1.5691383999946993 | 1.1622702360045063 | 2.9750325999921188 |
| `qa/rag/qwen3:8b` | 27 | 71.4% (15/21) | 4.1170776814828125 | 5.292514300002949 | 3.7618849592886687 | 8.97520740001346 |

## Detailed metrics

### `command/raw/qwen3:1.7b`
- call_success: 100.0% (27/27)
- json_validity: 100.0% (27/27)
- schema_validity: 81.5% (22/27)
- command_accuracy: 81.5% (22/27)
- no_unintended_mutation: 100.0% (22/22)

### `command/raw/qwen3:8b`
- call_success: 100.0% (27/27)
- json_validity: 100.0% (27/27)
- schema_validity: 59.3% (16/27)
- command_accuracy: 59.3% (16/27)
- no_unintended_mutation: 100.0% (16/16)

### `command/structured/qwen3:1.7b`
- call_success: 100.0% (27/27)
- json_validity: 100.0% (27/27)
- schema_validity: 100.0% (27/27)
- command_accuracy: 100.0% (27/27)
- no_unintended_mutation: 100.0% (27/27)

### `command/structured/qwen3:8b`
- call_success: 100.0% (27/27)
- json_validity: 100.0% (27/27)
- schema_validity: 100.0% (27/27)
- command_accuracy: 100.0% (27/27)
- no_unintended_mutation: 100.0% (27/27)

### `qa/direct/qwen3:1.7b`
- call_success: 100.0% (27/27)
- answer_keyword_accuracy: 14.3% (3/21)
- retrieval_success: n/a
- source_accuracy: n/a
- refusal_accuracy: n/a
- observed_refusal_rate: 0.0% (0/27)

### `qa/direct/qwen3:8b`
- call_success: 100.0% (27/27)
- answer_keyword_accuracy: 0.0% (0/21)
- retrieval_success: n/a
- source_accuracy: n/a
- refusal_accuracy: n/a
- observed_refusal_rate: 0.0% (0/27)

### `qa/full_context/qwen3:1.7b`
- call_success: 100.0% (27/27)
- answer_keyword_accuracy: 85.7% (18/21)
- retrieval_success: n/a
- source_accuracy: n/a
- refusal_accuracy: 100.0% (6/6)
- observed_refusal_rate: 33.3% (9/27)

### `qa/full_context/qwen3:8b`
- call_success: 100.0% (27/27)
- answer_keyword_accuracy: 85.7% (18/21)
- retrieval_success: n/a
- source_accuracy: n/a
- refusal_accuracy: 100.0% (6/6)
- observed_refusal_rate: 33.3% (9/27)

### `qa/rag/qwen3:1.7b`
- call_success: 100.0% (27/27)
- answer_keyword_accuracy: 61.9% (13/21)
- retrieval_success: 71.4% (15/21)
- source_accuracy: 71.4% (15/21)
- refusal_accuracy: 100.0% (6/6)
- observed_refusal_rate: 44.4% (12/27)

### `qa/rag/qwen3:8b`
- call_success: 100.0% (27/27)
- answer_keyword_accuracy: 71.4% (15/21)
- retrieval_success: 71.4% (15/21)
- source_accuracy: 71.4% (15/21)
- refusal_accuracy: 100.0% (6/6)
- observed_refusal_rate: 44.4% (12/27)

## Interpretation limits

- `raw` is Ollama free-form chat; `structured` is the full LangChain `with_structured_output` + Pydantic pipeline. This comparison changes output constraints/client processing and does **not** show that LangChain increases model intelligence.
- `direct` has no project documents. `full_context` receives every project document with the grounding prompt. `rag` receives only retrieved chunks through the production search/gating path. Use full_context versus rag to assess retrieval trade-offs; direct versus rag only demonstrates the value of external project knowledge.
- Keyword checks are deterministic proxies, not semantic grading. Source accuracy checks returned retrieval metadata, not whether the generated prose cites a source inline.
- Every case is repeated as recorded in metadata. This remains a small local benchmark rather than a publication-grade claim.
- P95 uses the nearest-rank definition over this small sample.

See the companion JSON for every prompt, output, error, source, state diff, and wall-clock measurement.


## Superseded long-generation records

The initial QA run had 3 records over 120 seconds. Their answer bodies are not retained in this final artifact; all QA conditions were rerun with the same token budget.

| Model | Pipeline | Case | Run | Previous latency (s) |
|---|---|---|---:|---:|
| qwen3:1.7b | direct | qa_out_of_scope_recipe | 1 | 627.989 |
| qwen3:1.7b | direct | qa_out_of_scope_recipe | 2 | 630.939 |
| qwen3:1.7b | direct | qa_out_of_scope_recipe | 3 | 682.692 |
