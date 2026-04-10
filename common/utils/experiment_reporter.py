from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ExperimentReporter:
    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        experiments_dir: str = "experiments",
        schema_version: str = SCHEMA_VERSION,
        enable_mlflow_artifacts: bool = True,
    ):
        self.experiments_dir = experiments_dir
        self.schema_version = schema_version
        self.enable_mlflow_artifacts = enable_mlflow_artifacts

        self.run_id: str | None = None
        self.run_dir: str | None = None
        self.events_path: str | None = None
        self.summary_path: str | None = None
        self.started_at: str | None = None
        self._start_monotonic: float | None = None

        self._base_fields: dict[str, Any] = {}
        self._counters = {
            "iterations_total": 0,
            "unsafe_action_count": 0,
            "invalid_response_count": 0,
            "action_execution_failures": 0,
            "throttle_events": 0,
        }
        self._lock = threading.Lock()

    def start(self, goal_text: str, **context: Any) -> str | None:
        with self._lock:
            if self.run_id is not None:
                logger.warning("ExperimentReporter.start called more than once for run_id=%s", self.run_id)
                return self.run_id

            run_id = self._generate_run_id()
            run_dir = os.path.join(self.experiments_dir, run_id)
            events_path = os.path.join(run_dir, "events.jsonl")
            summary_path = os.path.join(run_dir, "summary.json")

            try:
                os.makedirs(run_dir, exist_ok=False)
            except Exception:
                logger.warning("Failed creating experiment run directory %s", run_dir, exc_info=True)
                return None

            self.run_id = run_id
            self.run_dir = run_dir
            self.events_path = events_path
            self.summary_path = summary_path
            self.started_at = self._now_iso()
            self._start_monotonic = time.monotonic()
            self._base_fields = {
                "mode": context.get("mode"),
                "llm_provider": context.get("llm_provider"),
                "llm_model": context.get("llm_model"),
                "goal_text": goal_text,
                "chat_id": context.get("chat_id"),
                "session_id": context.get("session_id"),
                "system_prompt": context.get("system_prompt"),
            }
            return self.run_id

    def log_event(self, event_type: str, **payload: Any) -> None:
        with self._lock:
            if self.run_id is None or self.events_path is None:
                logger.warning("ExperimentReporter.log_event called before start")
                return

            event = {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "timestamp": self._now_iso(),
                "event_type": event_type,
            }
            event.update(self._base_fields)
            event.update(payload)
            event = self._sanitize(event)

            try:
                with open(self.events_path, "a", encoding="utf-8") as events_file:
                    events_file.write(json.dumps(event, ensure_ascii=True) + "\n")
            except Exception:
                logger.warning("Failed writing experiment event run_id=%s event_type=%s", self.run_id, event_type, exc_info=True)
                return

            self._update_counters(event)

    def finalize(self, success: bool, termination_reason: str, **extra_summary: Any) -> None:
        with self._lock:
            if self.run_id is None or self.summary_path is None:
                logger.warning("ExperimentReporter.finalize called before start")
                return

            ended_at = self._now_iso()
            duration_ms = 0
            if self._start_monotonic is not None:
                duration_ms = max(0, int((time.monotonic() - self._start_monotonic) * 1000))

            summary = {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "started_at": self.started_at,
                "ended_at": ended_at,
                "duration_ms": duration_ms,
                "success": bool(success),
                "termination_reason": str(termination_reason),
                "iterations_total": int(self._counters["iterations_total"]),
                "unsafe_action_count": int(self._counters["unsafe_action_count"]),
                "invalid_response_count": int(self._counters["invalid_response_count"]),
                "action_execution_failures": int(self._counters["action_execution_failures"]),
                "throttle_events": int(self._counters["throttle_events"]),
                "mode": self._base_fields.get("mode"),
                "llm_provider": self._base_fields.get("llm_provider"),
                "llm_model": self._base_fields.get("llm_model"),
                "goal_text": self._base_fields.get("goal_text"),
                "chat_id": self._base_fields.get("chat_id"),
                "session_id": self._base_fields.get("session_id"),
                "system_prompt": self._base_fields.get("system_prompt"),
            }
            summary.update(extra_summary)
            summary = self._sanitize(summary)

            try:
                with open(self.summary_path, "w", encoding="utf-8") as summary_file:
                    json.dump(summary, summary_file, indent=2, ensure_ascii=True)
            except Exception:
                logger.warning("Failed writing experiment summary run_id=%s", self.run_id, exc_info=True)
                return

            self._log_mlflow_artifacts_best_effort()

    def _update_counters(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("event_type") or "")
        if event_type == "llm_iteration":
            self._counters["iterations_total"] += 1
        if event_type == "action_rejected":
            self._counters["unsafe_action_count"] += 1
        if event_type == "llm_invalid_response":
            self._counters["invalid_response_count"] += 1
        if event_type == "provider_throttle":
            self._counters["throttle_events"] += 1
        if event_type == "action_executed" and str(event.get("execution_status", "")).lower() != "success":
            self._counters["action_execution_failures"] += 1

    def _log_mlflow_artifacts_best_effort(self) -> None:
        if not self.enable_mlflow_artifacts:
            return
        if self.events_path is None or self.summary_path is None:
            return

        try:
            import mlflow  # type: ignore

            mlflow.log_artifact(self.events_path, artifact_path="experiment_reports")
            mlflow.log_artifact(self.summary_path, artifact_path="experiment_reports")
        except ImportError:
            return
        except Exception:
            logger.warning("Failed logging experiment artifacts to MLflow run_id=%s", self.run_id, exc_info=True)

    def _generate_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{uuid.uuid4().hex[:8]}"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _sanitize(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, str) and len(value) > 8000:
                return value[:8000] + "...[truncated]"
            return value

        if isinstance(value, Enum):
            return self._sanitize(value.value)

        if is_dataclass(value):
            return self._sanitize(asdict(value))

        if isinstance(value, bytes):
            return f"<bytes:{len(value)}>"

        if isinstance(value, (list, tuple)):
            return [self._sanitize(item) for item in value[:1000]]

        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                sanitized[str(key)] = self._sanitize(item)
            return sanitized

        if hasattr(value, "value") and isinstance(getattr(value, "value", None), str):
            return str(getattr(value, "value"))

        return str(value)