import unittest

from experiments.tasks.utils import process_results_v2_answer
from spatial.v2.evaluation_chat import encode_chat_input
from spatial.v2.study.analyze_randomized import (
    aggregate,
    answer_set,
    error_type,
    inspect_sample,
    normalized_footer,
    paired,
)


class ResearchDiagnosticsTests(unittest.TestCase):
    def test_answer_sets_require_canonical_order_and_no_duplicates(self):
        self.assertEqual(answer_set("reason\nAnswer: A, C"), frozenset("AC"))
        self.assertFalse(answer_set("Answer: C, A"))
        self.assertFalse(answer_set("Answer: A, A"))
        self.assertFalse(answer_set("Answer: A\nmore reasoning"))

    def test_bare_answer_only_in_stage_one_mode(self):
        self.assertFalse(answer_set("A, B"))
        self.assertEqual(answer_set("A, B", bare=True), frozenset("AB"))

    def test_normalized_footer_is_explicit_and_terminal(self):
        self.assertEqual(normalized_footer("reason\n**Answer: B**"), frozenset("B"))
        self.assertEqual(normalized_footer("Answer: B\n</think>"), frozenset("B"))
        self.assertFalse(normalized_footer("The answer should be B, but maybe C"))
        self.assertFalse(normalized_footer("Answer: B\nNow let me reconsider"))
        self.assertFalse(normalized_footer("B is west of A"))

    def test_error_categories_distinguish_missing_from_wrong_options(self):
        gold = frozenset("AC")
        cases = {
            "": "invalid", "AC": "exact", "A": "missing_only",
            "ABC": "extra_only", "B": "missing_and_extra",
        }
        for prediction, expected in cases.items():
            with self.subTest(prediction=prediction):
                self.assertEqual(error_type(frozenset(prediction), gold), expected)

    def test_pairing_rejects_mismatched_base_coverage(self):
        rows = [
            {"panel": "sft", "model": "untuned", "base_id": "one"},
            {"panel": "sft", "model": "multi-natural-seed-42", "base_id": "two"},
        ]
        with self.assertRaisesRegex(ValueError, "different base IDs"):
            paired(rows)

    def sample_fixture(self):
        source = {
            "messages": [{"role": "system", "content": "rules"},
                         {"role": "user", "content": "question"},
                         {"role": "assistant", "content": "secret target"}],
            "metadata": {"oracle_letters": ["A"]},
        }
        doc = {
            "oracle_option": "A", "trace_format": "natural",
            "metadata": {
                "base_id": "one", "query_kind": "count", "gold_size": 1,
                "benchmark_partition": "matched", "answer_mode": "single",
                "heldout_transitions": [], "difficulty": {
                    "proof_complexity": {"semantic_length": 2},
                    "num_distractor_premises": 0,
                },
            },
        }
        response = "<stage1_response>\nAnswer: A\n</stage1_response>\n\nAnswer: B"
        sample = {
            "doc": doc, "arguments": [[encode_chat_input(source["messages"][:2]),
                                      {"max_gen_toks": 4096}]],
            "filtered_resps": [response],
            **process_results_v2_answer(doc, [response]),
        }
        task = {"query_kind": "count", "gold_size": 1, "partition": "matched"}
        return sample, task, source

    def test_replay_detects_stage_two_regression_without_rewriting_scores(self):
        sample, task, source = self.sample_fixture()
        row = inspect_sample(sample, task, source, process_results_v2_answer)
        self.assertEqual(row["stage1_acc"], 1)
        self.assertEqual(row["strict_acc"], 0)
        self.assertEqual(row["prompt_matches_source"], 1)
        self.assertEqual(row["stage2_extra"], 1)

    def test_prompt_audit_flags_appended_target(self):
        sample, task, source = self.sample_fixture()
        sample["arguments"][0][0] += "secret target"
        row = inspect_sample(sample, task, source, process_results_v2_answer)
        self.assertEqual(row["prompt_matches_source"], 0)

    def test_replay_rejects_forged_metrics(self):
        sample, task, source = self.sample_fixture()
        sample["strict_acc"] = 1
        with self.assertRaisesRegex(ValueError, "Scoring replay mismatch"):
            inspect_sample(sample, task, source, process_results_v2_answer)

    def test_aggregate_preserves_zero_and_missing_token_observations(self):
        row = {
            "model": "test", "strict_acc": 0, "stage1_acc": 1,
            "normalized_stage1_acc": 1, "normalized_recovery": 0,
            "sensitive_prompt": 0, "prompt_matches_source": 1,
            "stage1_error": "exact", "stage1_closed_think": 0,
            "stage2_size": 1, "stage2_error": "missing_only",
            "near_budget_proxy": None,
        }
        result = aggregate([row], ("model",))[0]
        self.assertEqual(result["strict_acc"], 0)
        self.assertEqual(result["stage1_correct_to_wrong"], 1)
        self.assertEqual(result["token_proxy_available"], 0)
        self.assertEqual(result["stage2_missing_only"], 1)


if __name__ == "__main__":
    unittest.main()
