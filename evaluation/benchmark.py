#!/usr/bin/env python
"""Reproducible local benchmark for the karaoke command agent and RAG.

No model calls happen on import, --help, or --validate-only. Actual benchmark
results are written only after real Ollama/LangChain calls; errors are recorded,
never replaced with synthetic values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("dataset.json")
NO_EVIDENCE_ANSWER = "제공된 문서에서 확인할 수 없습니다."


def nested_set(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = target
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = copy.deepcopy(value)


def nested_get(target: dict[str, Any], dotted_path: str) -> Any:
    current: Any = target
    for part in dotted_path.split("."):
        current = current[part]
    return current


def leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        leaves: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            leaves.update(leaf_values(child, path))
        return leaves
    return {prefix: value}


def changed_paths(origin: dict[str, Any], output: dict[str, Any]) -> list[str]:
    before = leaf_values(origin)
    after = leaf_values(output)
    return sorted(
        key
        for key in before.keys() | after.keys()
        if before.get(key, object()) != after.get(key, object())
    )


def score_command_state(
    origin: dict[str, Any],
    output: dict[str, Any] | None,
    expected_fields: dict[str, Any],
    *,
    schema_valid: bool,
) -> dict[str, Any]:
    if not schema_valid or output is None:
        return {
            "expected_fields_correct": False,
            "unexpected_mutations": [],
            "changed_paths": [],
            "command_correct": False,
        }
    expected_ok = all(
        nested_get(output, path) == expected for path, expected in expected_fields.items()
    )
    mutations = changed_paths(origin, output)
    unexpected = sorted(set(mutations) - set(expected_fields))
    expected_mutations = {
        path for path, expected in expected_fields.items() if nested_get(origin, path) != expected
    }
    missing_mutations = sorted(expected_mutations - set(mutations))
    return {
        "expected_fields_correct": expected_ok and not missing_mutations,
        "missing_expected_mutations": missing_mutations,
        "unexpected_mutations": unexpected,
        "changed_paths": mutations,
        "command_correct": expected_ok and not missing_mutations and not unexpected,
    }


def answer_has_gold_keywords(answer: str, groups: list[list[str]]) -> bool:
    folded = answer.casefold()
    return all(any(term.casefold() in folded for term in alternatives) for alternatives in groups)


def source_matches(actual: str, expected: str) -> bool:
    return Path(actual.replace("\\", "/")).name.casefold() == Path(expected).name.casefold()


def score_rag_case(
    case: dict[str, Any],
    *,
    answer: str,
    grounded: bool,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    refusal = NO_EVIDENCE_ANSWER in answer
    if case["expected_refusal"]:
        return {
            "answer_correct": None,
            "retrieval_success": None,
            "source_correct": None,
            "refusal_detected": refusal,
            "refusal_correct": refusal and not grounded and not sources,
        }
    actual_sources = [str(source.get("source", "")) for source in sources]
    expected_sources = case["gold_sources"]
    retrieval_success = bool(sources) and grounded
    source_correct = all(
        any(source_matches(actual, expected) for actual in actual_sources)
        for expected in expected_sources
    )
    return {
        "answer_correct": answer_has_gold_keywords(answer, case["gold_keyword_groups"]),
        "retrieval_success": retrieval_success,
        "source_correct": source_correct,
        "refusal_detected": refusal,
        "refusal_correct": None,
    }


def latency_summary(values: Iterable[float]) -> dict[str, Any]:
    samples = sorted(float(value) for value in values)
    if not samples:
        return {
            "count": 0,
            "average_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
        }
    p95_index = max(0, math.ceil(0.95 * len(samples)) - 1)
    return {
        "count": len(samples),
        "average_seconds": statistics.fmean(samples),
        "median_seconds": statistics.median(samples),
        "p95_seconds": samples[p95_index],
    }


def load_dataset(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "default_state", "command_cases", "qa_cases"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"dataset missing keys: {sorted(missing)}")
    ids = [case["id"] for key in ("command_cases", "qa_cases") for case in data[key]]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset case IDs must be unique")
    for case in data["command_cases"]:
        if not isinstance(case.get("expected_fields"), dict):
            raise ValueError(f"{case['id']}: expected_fields must be an object")
    for case in data["qa_cases"]:
        for key in ("expected_refusal", "gold_keyword_groups", "gold_sources"):
            if key not in case:
                raise ValueError(f"{case['id']}: missing {key}")
    return data


def case_origin(dataset: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    origin = copy.deepcopy(dataset["default_state"])
    for path, value in case.get("origin_overrides", {}).items():
        nested_set(origin, path, value)
    return origin


def command_messages(system_prompt: str, origin: dict[str, Any], command: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "현재 상태(JSON):\n"
            + json.dumps(origin, ensure_ascii=False, indent=2),
        },
        {"role": "user", "content": "사용자 명령:\n" + command},
        {"role": "user", "content": "위 규칙을 지켜서 '변경된 상태'만 출력해."},
    ]


def ollama_base_url(value: str) -> str:
    base = value.rstrip("/")
    if not re.match(r"^https?://", base):
        base = "http://" + base
    return base


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 300) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def raw_ollama_chat(base_url: str, model: str, messages: list[dict[str, str]], timeout: float) -> str:
    response = http_json(
        f"{base_url}/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
        },
        timeout,
    )
    return str(response.get("message", {}).get("content", ""))


def extract_json_object(text: str) -> tuple[bool, dict[str, Any] | None]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
        return isinstance(value, dict), value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for position, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[position:])
            if isinstance(value, dict):
                return True, value
        except json.JSONDecodeError:
            continue
    return False, None


def langchain_raw_text(raw: Any) -> str:
    content = getattr(raw, "content", "")
    if isinstance(content, str) and content.strip():
        return content
    for call in getattr(raw, "tool_calls", []) or []:
        args = call.get("args") if isinstance(call, dict) else None
        if isinstance(args, dict):
            return json.dumps(args, ensure_ascii=False)
        if isinstance(args, str):
            return args
    return ""


def import_command_components() -> tuple[Any, str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.main import KaraokeMachine, SYSTEM_PROMPT

    return KaraokeMachine, SYSTEM_PROMPT


def make_chat_model(model: str) -> Any:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0, seed=42)


def validate_state(schema: Any, value: dict[str, Any] | None) -> tuple[bool, dict[str, Any] | None, str | None]:
    if value is None:
        return False, None, "no JSON object"
    try:
        parsed = schema.model_validate(value)
        return True, parsed.model_dump(mode="json"), None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def run_command_pipeline(
    *,
    pipeline: str,
    model: str,
    cases: list[dict[str, Any]],
    dataset: dict[str, Any],
    base_url: str,
    timeout: float,
    warmup: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema, system_prompt = import_command_components()
    structured = None
    if pipeline == "structured":
        structured = make_chat_model(model).with_structured_output(schema, include_raw=True)

    def invoke(case: dict[str, Any]) -> tuple[str, dict[str, Any] | None, bool, bool, str | None]:
        origin = case_origin(dataset, case)
        messages = command_messages(system_prompt, origin, case["command"])
        if pipeline == "raw":
            text = raw_ollama_chat(base_url, model, messages, timeout)
            json_valid, value = extract_json_object(text)
            schema_valid, output, error = validate_state(schema, value)
            return text, output, json_valid, schema_valid, error
        result = structured.invoke(messages)
        parsed = result.get("parsed")
        error_obj = result.get("parsing_error")
        raw_text = langchain_raw_text(result.get("raw"))
        json_valid, raw_value = extract_json_object(raw_text)
        if parsed is not None:
            output = parsed.model_dump(mode="json")
            schema_valid = True
            error = None
        else:
            schema_valid, output, error = validate_state(schema, raw_value)
        if error_obj is not None:
            error = f"{type(error_obj).__name__}: {error_obj}"
        return raw_text, output, json_valid, schema_valid, error

    warmup_result: dict[str, Any] = {"performed": False}
    if warmup and cases:
        started = time.perf_counter()
        try:
            invoke(cases[0])
            warmup_result = {
                "performed": True,
                "success": True,
                "latency_seconds": time.perf_counter() - started,
                "excluded_from_metrics": True,
            }
        except Exception as exc:
            warmup_result = {
                "performed": True,
                "success": False,
                "latency_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "excluded_from_metrics": True,
            }

    records = []
    for case in cases:
        origin = case_origin(dataset, case)
        started = time.perf_counter()
        try:
            text, output, json_valid, schema_valid, validation_error = invoke(case)
            elapsed = time.perf_counter() - started
            score = score_command_state(
                origin, output, case["expected_fields"], schema_valid=schema_valid
            )
            records.append(
                {
                    "category": "command",
                    "pipeline": pipeline,
                    "model": model,
                    "case_id": case["id"],
                    "command": case["command"],
                    "origin": origin,
                    "expected_fields": case["expected_fields"],
                    "latency_seconds": elapsed,
                    "call_success": True,
                    "json_valid": json_valid,
                    "schema_valid": schema_valid,
                    "validation_error": validation_error,
                    "output_state": output,
                    "raw_output": text,
                    **score,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "category": "command",
                    "pipeline": pipeline,
                    "model": model,
                    "case_id": case["id"],
                    "command": case["command"],
                    "origin": origin,
                    "expected_fields": case["expected_fields"],
                    "latency_seconds": time.perf_counter() - started,
                    "call_success": False,
                    "json_valid": False,
                    "schema_valid": False,
                    "command_correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return records, warmup_result


def direct_qa(model: Any, question: str) -> str:
    result = model.invoke(
        [
            {
                "role": "system",
                "content": "질문에 짧고 명확한 한국어로 답하세요.",
            },
            {"role": "user", "content": question},
        ]
    )
    return str(getattr(result, "content", result)).strip()


def build_rag_for_model(model_name: str) -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.rag.runtime import build_service

    def chat_factory(**_: Any) -> Any:
        return make_chat_model(model_name)

    return build_service(chat_factory=chat_factory)


def run_qa_pipeline(
    *,
    pipeline: str,
    model_name: str,
    cases: list[dict[str, Any]],
    warmup: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    setup_started = time.perf_counter()
    runner = make_chat_model(model_name) if pipeline == "direct" else build_rag_for_model(model_name)
    setup_seconds = time.perf_counter() - setup_started

    def invoke(case: dict[str, Any]) -> tuple[str, bool | None, list[dict[str, Any]]]:
        if pipeline == "direct":
            return direct_qa(runner, case["question"]), None, []
        response = runner.answer(case["question"])
        return (
            response.answer,
            response.grounded,
            [source.model_dump(mode="json") for source in response.sources],
        )

    warmup_result: dict[str, Any] = {"performed": False}
    if warmup and cases:
        started = time.perf_counter()
        try:
            invoke(cases[0])
            warmup_result = {
                "performed": True,
                "success": True,
                "latency_seconds": time.perf_counter() - started,
                "excluded_from_metrics": True,
            }
        except Exception as exc:
            warmup_result = {
                "performed": True,
                "success": False,
                "latency_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "excluded_from_metrics": True,
            }

    records = []
    for case in cases:
        started = time.perf_counter()
        try:
            answer, grounded, sources = invoke(case)
            elapsed = time.perf_counter() - started
            if pipeline == "rag":
                scores = score_rag_case(
                    case, answer=answer, grounded=bool(grounded), sources=sources
                )
            else:
                expected_refusal = case["expected_refusal"]
                scores = {
                    "answer_correct": None
                    if expected_refusal
                    else answer_has_gold_keywords(answer, case["gold_keyword_groups"]),
                    "retrieval_success": None,
                    "source_correct": None,
                    "refusal_detected": NO_EVIDENCE_ANSWER in answer,
                    "refusal_correct": None,
                }
            records.append(
                {
                    "category": "qa",
                    "pipeline": pipeline,
                    "model": model_name,
                    "case_id": case["id"],
                    "question": case["question"],
                    "expected_refusal": case["expected_refusal"],
                    "gold_keyword_groups": case["gold_keyword_groups"],
                    "gold_sources": case["gold_sources"],
                    "latency_seconds": elapsed,
                    "call_success": True,
                    "answer": answer,
                    "grounded": grounded,
                    "sources": sources,
                    **scores,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "category": "qa",
                    "pipeline": pipeline,
                    "model": model_name,
                    "case_id": case["id"],
                    "question": case["question"],
                    "expected_refusal": case["expected_refusal"],
                    "latency_seconds": time.perf_counter() - started,
                    "call_success": False,
                    "answer_correct": False if not case["expected_refusal"] else None,
                    "refusal_correct": False if case["expected_refusal"] else None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return records, warmup_result, setup_seconds


def rate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [record[key] for record in records if record.get(key) is not None]
    passed = sum(value is True for value in values)
    return {
        "passed": passed,
        "applicable": len(values),
        "rate": passed / len(values) if values else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["category"], record["pipeline"], record["model"])].append(record)
    summaries: dict[str, Any] = {}
    for (category, pipeline, model), items in sorted(groups.items()):
        key = f"{category}/{pipeline}/{model}"
        successful_latencies = [
            item["latency_seconds"] for item in items if item.get("call_success")
        ]
        summary: dict[str, Any] = {
            "cases": len(items),
            "call_success": rate(items, "call_success"),
            "latency": latency_summary(successful_latencies),
        }
        if category == "command":
            summary.update(
                {
                    "json_validity": rate(items, "json_valid"),
                    "schema_validity": rate(items, "schema_valid"),
                    "command_accuracy": rate(items, "command_correct"),
                    "no_unintended_mutation": {
                        "passed": sum(
                            not item.get("unexpected_mutations")
                            for item in items
                            if item.get("schema_valid")
                        ),
                        "applicable": sum(bool(item.get("schema_valid")) for item in items),
                    },
                }
            )
            metric = summary["no_unintended_mutation"]
            metric["rate"] = (
                metric["passed"] / metric["applicable"] if metric["applicable"] else None
            )
        else:
            summary.update(
                {
                    "answer_keyword_accuracy": rate(items, "answer_correct"),
                    "retrieval_success": rate(items, "retrieval_success"),
                    "source_accuracy": rate(items, "source_correct"),
                    "refusal_accuracy": rate(items, "refusal_correct"),
                    "observed_refusal_rate": rate(items, "refusal_detected"),
                }
            )
        summaries[key] = summary

    overhead: dict[str, Any] = {}
    by_model_category: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for key, value in summaries.items():
        category, pipeline, model = key.split("/", 2)
        by_model_category[(category, model)][pipeline] = value
    for (category, model), pipelines in by_model_category.items():
        baseline, enhanced = ("raw", "structured") if category == "command" else ("direct", "rag")
        if baseline in pipelines and enhanced in pipelines:
            left = pipelines[baseline]["latency"]["average_seconds"]
            right = pipelines[enhanced]["latency"]["average_seconds"]
            overhead[f"{category}/{model}"] = {
                "baseline": baseline,
                "comparison": enhanced,
                "average_seconds_difference": None
                if left is None or right is None
                else right - left,
                "average_ratio": None
                if left in (None, 0) or right is None
                else right / left,
            }
    return {"groups": summaries, "latency_overhead": overhead}


def git_metadata() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def ollama_model_metadata(base_url: str, models: list[str], timeout: float) -> dict[str, Any]:
    try:
        tags = http_json(f"{base_url}/api/tags", timeout=min(timeout, 15))
        installed = tags.get("models", [])
        return {
            model: next(
                (item for item in installed if item.get("name") == model or item.get("model") == model),
                {"available": False},
            )
            for model in models
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def format_rate(metric: dict[str, Any]) -> str:
    if metric["rate"] is None:
        return "n/a"
    return f"{metric['rate'] * 100:.1f}% ({metric['passed']}/{metric['applicable']})"


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Karaoke Agent Benchmark",
        "",
        f"Generated: `{result['metadata']['finished_at_utc']}`",
        "",
        "> These are measured local results, not estimates. Warm-ups and RAG setup/index time are excluded from per-case latency. Failed calls remain failures and are not imputed.",
        "",
        "## Results",
        "",
        "| Group | Calls | Primary accuracy | Avg (s) | Median (s) | P95 (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, summary in result["summary"]["groups"].items():
        primary_key = "command_accuracy" if key.startswith("command/") else "answer_keyword_accuracy"
        latency = summary["latency"]
        lines.append(
            f"| `{key}` | {summary['cases']} | {format_rate(summary[primary_key])} | "
            f"{latency['average_seconds'] if latency['average_seconds'] is not None else 'n/a'} | "
            f"{latency['median_seconds'] if latency['median_seconds'] is not None else 'n/a'} | "
            f"{latency['p95_seconds'] if latency['p95_seconds'] is not None else 'n/a'} |"
        )
    lines.extend(["", "## Detailed metrics", ""])
    for key, summary in result["summary"]["groups"].items():
        lines.append(f"### `{key}`")
        for metric_name, metric in summary.items():
            if isinstance(metric, dict) and "rate" in metric:
                lines.append(f"- {metric_name}: {format_rate(metric)}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- `raw` is Ollama free-form chat; `structured` is the full LangChain `with_structured_output` + Pydantic pipeline. This comparison changes output constraints/client processing and does **not** show that LangChain increases model intelligence.",
            "- `direct` is plain Qwen generation; `rag` adds Chroma retrieval, score gating, retrieved context, a grounding prompt, and source packaging. Differences cannot be attributed to retrieval alone.",
            "- Keyword checks are deterministic proxies, not semantic grading. Source accuracy checks returned retrieval metadata, not whether the generated prose cites a source inline.",
            "- One measured run per case is intentionally quick but gives noisy latency estimates; use repeated independent runs for publication-grade timing.",
            "- P95 uses the nearest-rank definition over this small sample.",
            "",
            "See the companion JSON for every prompt, output, error, source, state diff, and wall-clock measurement.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models", nargs="+", default=["qwen3:1.7b", "qwen3:8b"])
    parser.add_argument(
        "--category", choices=["all", "command", "qa"], default="all"
    )
    parser.add_argument(
        "--command-pipelines",
        nargs="+",
        choices=["raw", "structured"],
        default=["raw", "structured"],
    )
    parser.add_argument(
        "--qa-pipelines",
        nargs="+",
        choices=["direct", "rag"],
        default=["direct", "rag"],
    )
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the dataset and exit without importing model libraries or calling Ollama",
    )
    return parser.parse_args(argv)


def setup_failure_records(
    *,
    category: str,
    pipeline: str,
    model: str,
    cases: list[dict[str, Any]],
    error: Exception,
) -> list[dict[str, Any]]:
    """Represent a pipeline setup failure without fabricating call timing/results."""
    message = f"{type(error).__name__}: {error}"
    return [
        {
            "category": category,
            "pipeline": pipeline,
            "model": model,
            "case_id": case["id"],
            "command" if category == "command" else "question": case[
                "command" if category == "command" else "question"
            ],
            "latency_seconds": None,
            "call_success": False,
            "setup_error": message,
            **(
                {
                    "json_valid": False,
                    "schema_valid": False,
                    "command_correct": False,
                }
                if category == "command"
                else {
                    "answer_correct": None
                    if case["expected_refusal"]
                    else False,
                    "refusal_correct": False
                    if case["expected_refusal"]
                    else None,
                }
            ),
        }
        for case in cases
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = args.dataset.resolve()
    dataset = load_dataset(dataset_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "dataset": str(dataset_path),
                    "command_cases": len(dataset["command_cases"]),
                    "qa_cases": len(dataset["qa_cases"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    base_url = ollama_base_url(args.ollama_host)
    started_at = datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    warmups: dict[str, Any] = {}
    setup_times: dict[str, float] = {}
    do_warmup = not args.no_warmup

    for model in args.models:
        if args.category in ("all", "command"):
            for pipeline in args.command_pipelines:
                key = f"command/{pipeline}/{model}"
                try:
                    pipeline_records, warmup_result = run_command_pipeline(
                        pipeline=pipeline,
                        model=model,
                        cases=dataset["command_cases"],
                        dataset=dataset,
                        base_url=base_url,
                        timeout=args.timeout,
                        warmup=do_warmup,
                    )
                except Exception as exc:
                    pipeline_records = setup_failure_records(
                        category="command",
                        pipeline=pipeline,
                        model=model,
                        cases=dataset["command_cases"],
                        error=exc,
                    )
                    warmup_result = {
                        "performed": False,
                        "setup_error": f"{type(exc).__name__}: {exc}",
                    }
                records.extend(pipeline_records)
                warmups[key] = warmup_result
        if args.category in ("all", "qa"):
            for pipeline in args.qa_pipelines:
                key = f"qa/{pipeline}/{model}"
                try:
                    pipeline_records, warmup_result, setup_seconds = run_qa_pipeline(
                        pipeline=pipeline,
                        model_name=model,
                        cases=dataset["qa_cases"],
                        warmup=do_warmup,
                    )
                    setup_times[key] = setup_seconds
                except Exception as exc:
                    pipeline_records = setup_failure_records(
                        category="qa",
                        pipeline=pipeline,
                        model=model,
                        cases=dataset["qa_cases"],
                        error=exc,
                    )
                    warmup_result = {
                        "performed": False,
                        "setup_error": f"{type(exc).__name__}: {exc}",
                    }
                records.extend(pipeline_records)
                warmups[key] = warmup_result

    finished_at = datetime.now(timezone.utc)
    result = {
        "benchmark_schema_version": 1,
        "metadata": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "total_wall_seconds": (finished_at - started_at).total_seconds(),
            "dataset": str(dataset_path),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "models": args.models,
            "temperature": 0,
            "seed": 42,
            "measured_runs_per_case": 1,
            "warmup_enabled": do_warmup,
            "warmups": warmups,
            "setup_seconds_excluded_from_case_latency": setup_times,
            "ollama_host": base_url,
            "ollama_models": ollama_model_metadata(base_url, args.models, args.timeout),
            "python": sys.version,
            "platform": platform.platform(),
            "git": git_metadata(),
        },
        "methodology": {
            "command_accuracy": "All expected dotted-path values match and no other state leaf differs from origin; invalid/no-op cases require zero mutations.",
            "json_validity": "A JSON object can be decoded from model output (including one fenced/object substring).",
            "schema_validity": "The decoded/structured object validates as backend.main.KaraokeMachine with Pydantic.",
            "answer_accuracy": "Every gold keyword group has at least one case-insensitive alternative in the answer; refusal cases are excluded.",
            "retrieval_success": "RAG returned grounded=true and at least one source; answerable cases only.",
            "source_accuracy": "Every gold source filename appears in returned retrieval metadata; answerable cases only.",
            "refusal_accuracy": f"Answer contains exact policy sentence '{NO_EVIDENCE_ANSWER}', grounded=false, and sources are empty; refusal cases only.",
            "latency": "Client wall time via time.perf_counter; warmups and service/index setup excluded; failures excluded from aggregate latency but retained per case.",
            "confounds": [
                "Raw Ollama free-form chat versus the full LangChain structured-output and Pydantic pipeline; this does not test whether LangChain increases intelligence.",
                "Direct QA versus RAG changes retrieval, context, prompting, score gating, and response packaging together.",
            ],
        },
        "summary": summarize(records),
        "records": records,
    }

    if args.output_prefix:
        prefix = args.output_prefix.resolve()
    else:
        stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        prefix = ROOT / "evaluation" / "results" / f"benchmark_{stamp}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # A model name can contain dots (for example `qwen3_1.7b`). Path.with_suffix
    # would treat `.7b_vs_8b` as an extension and silently truncate the name.
    json_path = Path(str(prefix) + ".json")
    markdown_path = Path(str(prefix) + ".md")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
