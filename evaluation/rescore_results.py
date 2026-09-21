#!/usr/bin/env python
"""Re-score preserved benchmark outputs without making new model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmark import (
    markdown_report,
    score_rag_case,
    summarize,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset.json"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.input.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    qa_cases = {case["id"]: case for case in dataset["qa_cases"]}

    for record in result["records"]:
        if record.get("category") != "qa" or not record.get("call_success"):
            continue
        case = qa_cases[record["case_id"]]
        record["expected_refusal"] = case["expected_refusal"]
        record["gold_keyword_groups"] = case["gold_keyword_groups"]
        record["gold_sources"] = case["gold_sources"]
        if record["pipeline"] == "rag":
            record.update(
                score_rag_case(
                    case,
                    answer=record["answer"],
                    grounded=bool(record["grounded"]),
                    sources=record["sources"],
                )
            )
        else:
            from evaluation.benchmark import answer_has_gold_keywords, NO_EVIDENCE_ANSWER

            record["answer_correct"] = (
                None
                if case["expected_refusal"]
                else answer_has_gold_keywords(record["answer"], case["gold_keyword_groups"])
            )
            record["refusal_detected"] = NO_EVIDENCE_ANSWER in record["answer"]

    result["summary"] = summarize(result["records"])
    result["metadata"]["dataset"] = str(args.dataset.resolve())
    result["metadata"]["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    result["metadata"]["rescored_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["metadata"]["model_calls_reused"] = True
    result["metadata"]["rescore_source"] = str(args.input.resolve())
    result["metadata"]["rescore_note"] = (
        "No model calls were repeated. Preserved answers were re-scored after adding "
        "'-6에서' as an equivalent correct phrasing for the pitch range."
    )

    json_path = Path(str(args.output_prefix) + ".json")
    md_path = Path(str(args.output_prefix) + ".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
