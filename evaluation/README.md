# Karaoke Agent Benchmark

This directory contains a small, auditable benchmark for the current command agent and documentation RAG. It does not contain benchmark results; results are created only by real local model calls.

## What is compared

- `qwen3:1.7b` and `qwen3:8b` on the same nine command cases.
- Raw Ollama free-form chat versus the full LangChain `with_structured_output(KaraokeMachine)` + Pydantic pipeline.
- Plain Qwen without project documents, Qwen with the complete document corpus, and the current Chroma/bge-m3/Qwen RAG service on the same nine questions.

The raw/structured comparison measures the end-to-end effect of LangChain structured output and Pydantic validation on usable command accuracy and latency. For RAG, the primary retrieval comparison is full-context Qwen versus RAG because both receive project knowledge under the same grounding policy. Direct Qwen without documents is retained only as an external-knowledge baseline.

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

Default behavior is one excluded warm-up per model/pipeline followed by three measured runs per case. Temperature is 0, the Ollama seed is 42, and QA generation is capped at 512 predicted tokens to prevent runaway repetition. The HTTP timeout is 120 seconds, but it is not a total wall-clock deadline while tokens continue streaming. Results are written to timestamped files under `evaluation/results/`:

- JSON: full configuration, environment metadata, per-case outputs/errors/timing/state diffs/sources, and aggregate metrics.
- Markdown: concise tables and interpretation limits.

Useful subsets:

```bash
# Commands only, one model
python evaluation/benchmark.py --category command --models qwen3:1.7b

# No-document, full-context, and RAG QA
python evaluation/benchmark.py --category qa --qa-pipelines direct full_context rag

# Change the number of measured repetitions
python evaluation/benchmark.py --repetitions 5

# Change the QA output budget
python evaluation/benchmark.py --qa-num-predict 768

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
- **Latency:** per-case client wall time. Average, median, population standard deviation, and nearest-rank p95 use successful measured calls only. Warm-ups and RAG construction/index setup are separately recorded and excluded.

Keyword matching is a deterministic proxy rather than semantic grading. Nine cases per category with three measured runs reduce one-off timing noise but are not sufficient for publication-grade claims.
