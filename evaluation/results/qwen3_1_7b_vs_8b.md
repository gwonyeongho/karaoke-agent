# Karaoke Agent Benchmark

Generated: `2026-09-21T07:30:25.863756+00:00`

> These are measured local results, not estimates. Warm-ups and RAG setup/index time are excluded from per-case latency. Failed calls remain failures and are not imputed.

## Results

| Group | Calls | Primary accuracy | Avg (s) | Median (s) | P95 (s) |
|---|---:|---:|---:|---:|---:|
| `command/raw/qwen3:1.7b` | 9 | 88.9% (8/9) | 4.40372852222289 | 4.400834199997917 | 5.634793600002013 |
| `command/raw/qwen3:8b` | 9 | 66.7% (6/9) | 16.505156633333375 | 12.78098559999853 | 35.46595690000322 |
| `command/structured/qwen3:1.7b` | 9 | 100.0% (9/9) | 4.660181511111457 | 4.7026209000032395 | 5.361668199999258 |
| `command/structured/qwen3:8b` | 9 | 100.0% (9/9) | 16.20538643333364 | 15.177270899999712 | 24.014904800002114 |
| `qa/direct/qwen3:1.7b` | 9 | 0.0% (0/7) | 66.27527308888926 | 3.576287100000627 | 566.3637627999997 |
| `qa/direct/qwen3:8b` | 9 | 0.0% (0/7) | 10.614646688890288 | 8.968155800001114 | 28.566731599999912 |
| `qa/rag/qwen3:1.7b` | 9 | 71.4% (5/7) | 1.6418974777779012 | 2.1615631000022404 | 2.4040653000010934 |
| `qa/rag/qwen3:8b` | 9 | 71.4% (5/7) | 3.3873115777775333 | 4.227288500002032 | 9.06040639999992 |

## Detailed metrics

### `command/raw/qwen3:1.7b`
- call_success: 100.0% (9/9)
- json_validity: 100.0% (9/9)
- schema_validity: 88.9% (8/9)
- command_accuracy: 88.9% (8/9)
- no_unintended_mutation: 100.0% (8/8)

### `command/raw/qwen3:8b`
- call_success: 100.0% (9/9)
- json_validity: 100.0% (9/9)
- schema_validity: 66.7% (6/9)
- command_accuracy: 66.7% (6/9)
- no_unintended_mutation: 100.0% (6/6)

### `command/structured/qwen3:1.7b`
- call_success: 100.0% (9/9)
- json_validity: 100.0% (9/9)
- schema_validity: 100.0% (9/9)
- command_accuracy: 100.0% (9/9)
- no_unintended_mutation: 100.0% (9/9)

### `command/structured/qwen3:8b`
- call_success: 100.0% (9/9)
- json_validity: 100.0% (9/9)
- schema_validity: 100.0% (9/9)
- command_accuracy: 100.0% (9/9)
- no_unintended_mutation: 100.0% (9/9)

### `qa/direct/qwen3:1.7b`
- call_success: 100.0% (9/9)
- answer_keyword_accuracy: 0.0% (0/7)
- retrieval_success: n/a
- source_accuracy: n/a
- refusal_accuracy: n/a
- observed_refusal_rate: 0.0% (0/9)

### `qa/direct/qwen3:8b`
- call_success: 100.0% (9/9)
- answer_keyword_accuracy: 0.0% (0/7)
- retrieval_success: n/a
- source_accuracy: n/a
- refusal_accuracy: n/a
- observed_refusal_rate: 0.0% (0/9)

### `qa/rag/qwen3:1.7b`
- call_success: 100.0% (9/9)
- answer_keyword_accuracy: 71.4% (5/7)
- retrieval_success: 71.4% (5/7)
- source_accuracy: 71.4% (5/7)
- refusal_accuracy: 100.0% (2/2)
- observed_refusal_rate: 44.4% (4/9)

### `qa/rag/qwen3:8b`
- call_success: 100.0% (9/9)
- answer_keyword_accuracy: 71.4% (5/7)
- retrieval_success: 71.4% (5/7)
- source_accuracy: 71.4% (5/7)
- refusal_accuracy: 100.0% (2/2)
- observed_refusal_rate: 44.4% (4/9)

## Interpretation limits

- `raw` is Ollama free-form chat; `structured` is the full LangChain `with_structured_output` + Pydantic pipeline. This comparison changes output constraints/client processing and does **not** show that LangChain increases model intelligence.
- `direct` is plain Qwen generation; `rag` adds Chroma retrieval, score gating, retrieved context, a grounding prompt, and source packaging. Differences cannot be attributed to retrieval alone.
- Keyword checks are deterministic proxies, not semantic grading. Source accuracy checks returned retrieval metadata, not whether the generated prose cites a source inline.
- One measured run per case is intentionally quick but gives noisy latency estimates; use repeated independent runs for publication-grade timing.
- P95 uses the nearest-rank definition over this small sample.

See the companion JSON for every prompt, output, error, source, state diff, and wall-clock measurement.
