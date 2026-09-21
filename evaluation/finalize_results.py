"""Replace superseded QA records with capped retry results.

The initial run exposed runaway token repetition in one direct-QA case. This
script preserves command records, replaces every QA record with the fair
same-budget retry run, records only anomaly metadata (not the discarded
answers), and rebuilds aggregate metrics and Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark import markdown_report, summarize


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--qa-retry", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--wall-threshold", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load(args.base)
    retry = load(args.qa_retry)

    anomalies = [
        {
            "model": record["model"],
            "pipeline": record["pipeline"],
            "case_id": record["case_id"],
            "run_index": record.get("run_index"),
            "latency_seconds": record.get("latency_seconds"),
            "reason": "runaway repeated generation exceeded wall threshold",
        }
        for record in base["records"]
        if record["category"] == "qa"
        and (record.get("latency_seconds") or 0) > args.wall_threshold
    ]

    command_records = [r for r in base["records"] if r["category"] == "command"]
    qa_records = [r for r in retry["records"] if r["category"] == "qa"]
    records = command_records + qa_records

    metadata = dict(base["metadata"])
    metadata.update(
        {
            "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
            "finished_at_utc": retry["metadata"]["finished_at_utc"],
            "total_wall_seconds": None,
            "qa_results_replaced": True,
            "qa_retry_source": str(args.qa_retry),
            "qa_num_predict": retry["metadata"]["qa_num_predict"],
            "discarded_qa_record_count": sum(
                r["category"] == "qa" for r in base["records"]
            ),
            "replacement_qa_record_count": len(qa_records),
            "discarded_runaway_answers_retained": False,
            "retry_events": anomalies,
            "warmups": {
                **{
                    k: v
                    for k, v in base["metadata"]["warmups"].items()
                    if k.startswith("command/")
                },
                **retry["metadata"]["warmups"],
            },
            "setup_seconds_excluded_from_case_latency": {
                **{
                    k: v
                    for k, v in base["metadata"][
                        "setup_seconds_excluded_from_case_latency"
                    ].items()
                    if k.startswith("command/")
                },
                **retry["metadata"]["setup_seconds_excluded_from_case_latency"],
            },
            "component_runs": {
                "command": {
                    "source": str(args.base),
                    "started_at_utc": base["metadata"]["started_at_utc"],
                    "finished_at_utc": base["metadata"]["finished_at_utc"],
                },
                "qa_retry": {
                    "source": str(args.qa_retry),
                    "started_at_utc": retry["metadata"]["started_at_utc"],
                    "finished_at_utc": retry["metadata"]["finished_at_utc"],
                },
            },
        }
    )

    result = {
        "benchmark_schema_version": base["benchmark_schema_version"],
        "metadata": metadata,
        "methodology": retry["methodology"],
        "summary": summarize(records),
        "records": records,
    }

    prefix = args.output_prefix
    json_path = Path(str(prefix) + ".json")
    md_path = Path(str(prefix) + ".md")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    retry_lines = [
        "",
        "## Superseded long-generation records",
        "",
        f"The initial QA run had {len(anomalies)} records over "
        f"{args.wall_threshold:g} seconds. Their answer bodies are not retained in this "
        "final artifact; all QA conditions were rerun with the same token budget.",
        "",
        "| Model | Pipeline | Case | Run | Previous latency (s) |",
        "|---|---|---|---:|---:|",
    ]
    retry_lines.extend(
        f"| {x['model']} | {x['pipeline']} | {x['case_id']} | {x['run_index']} | "
        f"{x['latency_seconds']:.3f} |"
        for x in anomalies
    )
    md_path.write_text(
        markdown_report(result) + "\n" + "\n".join(retry_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
