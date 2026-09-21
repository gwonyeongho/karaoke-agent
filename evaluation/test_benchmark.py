import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.benchmark import (
    answer_has_gold_keywords,
    build_full_context,
    latency_summary,
    parse_args,
    repeated_cases,
    score_command_state,
    score_rag_case,
)


class BenchmarkScoringTests(unittest.TestCase):
    def setUp(self):
        self.origin = {
            "brand": "TJ",
            "model_name": "B80",
            "music": {"pitch": 0, "tempo": 0, "melody_volume": 50},
            "sound": {
                "echo_level": 50,
                "reverb_type": "Hall",
                "mic_volume": 70,
                "inst_volume": 60,
                "voice_cancel": False,
            },
            "queue": {"reserved_songs": [111, 222], "played_song": None},
            "is_playing": False,
            "remaining_time": 0,
            "remaining_coins": 0,
            "subtitle_lang": "KOR",
        }

    def test_command_requires_expected_fields_and_no_unintended_mutations(self):
        correct = {**self.origin, "music": {**self.origin["music"], "pitch": 1}}
        score = score_command_state(
            self.origin, correct, {"music.pitch": 1}, schema_valid=True
        )
        self.assertTrue(score["command_correct"])
        self.assertEqual(score["unexpected_mutations"], [])

        wrong = {**correct, "subtitle_lang": "ENG"}
        score = score_command_state(
            self.origin, wrong, {"music.pitch": 1}, schema_valid=True
        )
        self.assertFalse(score["command_correct"])
        self.assertEqual(score["unexpected_mutations"], ["subtitle_lang"])

    def test_noop_rejects_any_state_mutation(self):
        changed = {**self.origin, "remaining_coins": 1}
        score = score_command_state(self.origin, changed, {}, schema_valid=True)
        self.assertFalse(score["command_correct"])
        self.assertEqual(score["unexpected_mutations"], ["remaining_coins"])

    def test_keyword_groups_support_alternatives(self):
        groups = [["영어", "ENG"], ["자막"]]
        self.assertTrue(answer_has_gold_keywords("자막을 ENG로 설정합니다.", groups))
        self.assertFalse(answer_has_gold_keywords("영어로 설정합니다.", groups))

    def test_pitch_range_wording_is_accepted(self):
        groups = [["최소 -6", "-6부터", "-6에서"], ["최대 6", "6까지"]]
        self.assertTrue(answer_has_gold_keywords("음정은 -6에서 6까지 조절할 수 있습니다.", groups))

    def test_rag_scoring_separates_answer_retrieval_source_and_refusal(self):
        answerable = {
            "expected_refusal": False,
            "gold_keyword_groups": [["8000"], ["docs"]],
            "gold_sources": ["troubleshooting.md"],
        }
        result = score_rag_case(
            answerable,
            answer="127.0.0.1:8000/docs를 확인하세요.",
            grounded=True,
            sources=[{"source": "troubleshooting.md"}],
        )
        self.assertTrue(result["answer_correct"])
        self.assertTrue(result["retrieval_success"])
        self.assertTrue(result["source_correct"])
        self.assertIsNone(result["refusal_correct"])

        refusal = {
            "expected_refusal": True,
            "gold_keyword_groups": [],
            "gold_sources": [],
        }
        result = score_rag_case(
            refusal,
            answer="제공된 문서에서 확인할 수 없습니다.",
            grounded=False,
            sources=[],
        )
        self.assertTrue(result["refusal_correct"])
        self.assertIsNone(result["answer_correct"])

    def test_latency_summary_uses_nearest_rank_p95(self):
        summary = latency_summary([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["average_seconds"], 0.25)
        self.assertAlmostEqual(summary["median_seconds"], 0.25)
        self.assertAlmostEqual(summary["p95_seconds"], 0.4)
        self.assertAlmostEqual(summary["standard_deviation_seconds"], 0.11180339887498948)

    def test_repeated_cases_assigns_stable_run_numbers(self):
        cases = [{"id": "a"}, {"id": "b"}]
        observed = [(run, case["id"]) for run, case in repeated_cases(cases, 3)]
        self.assertEqual(
            observed,
            [(1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a"), (3, "b")],
        )

    def test_qa_generation_has_bounded_default(self):
        args = parse_args([])
        self.assertEqual(args.qa_num_predict, 512)

    def test_full_context_contains_all_markdown_documents_with_source_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.md").write_text("# A\n첫 문서", encoding="utf-8")
            (root / "b.md").write_text("# B\n둘째 문서", encoding="utf-8")
            context = build_full_context(root)
        self.assertIn("[문서: a.md]", context)
        self.assertIn("첫 문서", context)
        self.assertIn("[문서: b.md]", context)
        self.assertIn("둘째 문서", context)


if __name__ == "__main__":
    unittest.main()
