# Karaoke Agent Benchmark

This directory contains a small, auditable benchmark for the current command agent and documentation RAG. It does not contain benchmark results; results are created only by real local model calls.

## What is compared

- `qwen3:1.7b` and `qwen3:8b` on the same nine command cases.
- Raw Ollama free-form chat versus the full LangChain `with_structured_output(KaraokeMachine)` + Pydantic pipeline.
- Plain Qwen informational answers versus the current Chroma/bge-m3/Qwen RAG service on nine questions.

The raw/structured comparison does **not** claim that LangChain improves model intelligence. It compares different output constraints and client-side processing. Likewise, direct/RAG changes retrieval, context, grounding instructions, score gating, and source packaging together.

## Requirements

Install the project dependencies and make these Ollama models available:

```bash
ollama pull qwen3:1.7b
ollama pull qwen3:8b
ollama pull bge-m3
python -m pip install -r requirements.txt
```

## Validate without Ollama

```bash
python evaluation/benchmark.py --help
python evaluation/benchmark.py --validate-only
python -m unittest evaluation.test_benchmark
```

These commands do not invoke a model. `--help` also does not import LangChain or the backend.

## Run

From the repository root:

```bash
python evaluation/benchmark.py
```

Default behavior is one excluded warm-up per model/pipeline followed by one measured run per case. Temperature is 0 and the Ollama seed is 42. Results are written to timestamped files under `evaluation/results/`:

- JSON: full configuration, environment metadata, per-case outputs/errors/timing/state diffs/sources, and aggregate metrics.
- Markdown: concise tables and interpretation limits.

Useful subsets:

```bash
# Commands only, one model
python evaluation/benchmark.py --category command --models qwen3:1.7b

# RAG and direct QA only
python evaluation/benchmark.py --category qa --qa-pipelines direct rag

# Explicit output paths: evaluation/results/my_run.json and .md
python evaluation/benchmark.py --output-prefix evaluation/results/my_run
```

Use `--ollama-host`, `--timeout`, `--command-pipelines`, `--qa-pipelines`, and `--no-warmup` for controlled variants. Keep the generated JSON with any reported numbers.

## Metrics

- **Command accuracy:** every expected dotted-path value matches and no other state leaf changed. A no-op case passes only with zero state mutation.
- **JSON validity:** a JSON object can be decoded from the model output.
- **Schema validity:** the output validates as the current `backend.main.KaraokeMachine` Pydantic model.
- **Answer accuracy:** all hand-labeled keyword groups occur; refusal cases are excluded.
- **Retrieval/source:** answerable RAG cases return grounded sources, and every gold source filename appears in retrieval metadata.
- **Refusal:** out-of-scope RAG cases return the exact project refusal sentence with `grounded=false` and no sources.
- **Latency:** per-case client wall time. Average, median, and nearest-rank p95 use successful measured calls only. Warm-ups and RAG construction/index setup are separately recorded and excluded.

Keyword matching is a deterministic proxy rather than semantic grading. Nine cases per category and one measured run keep runtime manageable but are not sufficient for publication-grade latency claims.
