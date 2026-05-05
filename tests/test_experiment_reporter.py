import json
import os
import tempfile
import unittest

from common.utils.experiment_reporter import ExperimentReporter


class ExperimentReporterTests(unittest.TestCase):
    def test_reporter_writes_events_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ExperimentReporter(experiments_dir=tmpdir)
            run_id = reporter.start(
                goal_text="move forward",
                mode="dry-run",
                llm_provider="openai",
                llm_model="gpt-4o-mini",
                system_prompt="You are a careful hexapod planner.",
            )

            self.assertIsNotNone(run_id)
            run_dir = os.path.join(tmpdir, run_id)
            events_path = os.path.join(run_dir, "events.jsonl")
            summary_path = os.path.join(run_dir, "summary.json")

            reporter.log_event("session_start", session_id="abc")
            reporter.log_event("llm_message_sent", interaction_id="i1", llm_image_id="img1")
            reporter.log_event("llm_message_received", interaction_id="i1", llm_image_id="img1", latency_ms=120, llm_success=True)
            reporter.log_event("llm_iteration", iteration_index=1)
            reporter.log_event("action_rejected", safety_decision="rejected", safety_reason="obstacle")
            reporter.log_event("llm_invalid_response", error_type="InvalidJSON", error_message="bad")
            reporter.log_event("provider_throttle", error_type="RateLimit", error_message="429")
            reporter.log_event(
                "action_executed",
                iteration_index=1,
                interaction_id="i1",
                llm_image_id="img1",
                action_executed={"type": "walk", "params": {"y": 10}},
                execution_status="failure",
                execution_duration_ms=250,
                error_message="submit failed",
            )
            reporter.log_event("session_end", termination_reason="provider_throttle", success=False)
            reporter.finalize(success=False, termination_reason="provider_throttle")

            self.assertTrue(os.path.exists(events_path))
            self.assertTrue(os.path.exists(summary_path))

            with open(events_path, "r", encoding="utf-8") as events_file:
                lines = [line for line in events_file.read().strip().splitlines() if line]

            self.assertGreaterEqual(len(lines), 9)
            event = json.loads(lines[0])
            self.assertEqual(event["schema_version"], "1.0")
            self.assertEqual(event["run_id"], run_id)
            self.assertEqual(event["event_type"], "session_start")
            self.assertEqual(event["mode"], "dry-run")
            self.assertEqual(event["llm_provider"], "openai")
            self.assertEqual(event["llm_model"], "gpt-4o-mini")
            self.assertEqual(event["goal_text"], "move forward")
            self.assertEqual(event["system_prompt"], "You are a careful hexapod planner.")

            with open(summary_path, "r", encoding="utf-8") as summary_file:
                summary = json.load(summary_file)

            self.assertEqual(summary["schema_version"], "1.0")
            self.assertEqual(summary["run_id"], run_id)
            self.assertFalse(summary["success"])
            self.assertEqual(summary["termination_reason"], "provider_throttle")
            self.assertEqual(summary["iterations_total"], 1)
            self.assertEqual(summary["unsafe_action_count"], 1)
            self.assertEqual(summary["invalid_response_count"], 1)
            self.assertEqual(summary["action_execution_failures"], 1)
            self.assertEqual(summary["throttle_events"], 1)
            self.assertEqual(summary["llm_interaction_count"], 1)
            self.assertEqual(summary["llm_interaction_latency_total_ms"], 120)
            self.assertEqual(summary["llm_interaction_latency_avg_ms"], 120.0)
            self.assertEqual(summary["action_execution_count"], 1)
            self.assertEqual(summary["action_execution_latency_total_ms"], 250)
            self.assertEqual(summary["action_execution_latency_avg_ms"], 250.0)
            self.assertEqual(summary["errors_total"], 4)
            self.assertEqual(len(summary["action_history"]), 2)
            self.assertEqual(summary["action_history"][0]["execution_status"], "rejected")
            self.assertEqual(summary["action_history"][1]["execution_status"], "failure")
            self.assertEqual(summary["system_prompt"], "You are a careful hexapod planner.")
            self.assertGreaterEqual(summary["duration_ms"], 0)


if __name__ == "__main__":
    unittest.main()