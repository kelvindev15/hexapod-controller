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
    ):
        self.experiments_dir = experiments_dir
        self.schema_version = schema_version

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
            "llm_interaction_count": 0,
            "llm_interaction_latency_total_ms": 0,
            "llm_interaction_latency_max_ms": 0,
            "action_execution_count": 0,
            "action_execution_latency_total_ms": 0,
            "action_execution_latency_max_ms": 0,
            "errors_total": 0,
        }
        self._action_history: list[dict[str, Any]] = []
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
                "llm_interaction_count": int(self._counters["llm_interaction_count"]),
                "llm_interaction_latency_total_ms": int(self._counters["llm_interaction_latency_total_ms"]),
                "llm_interaction_latency_avg_ms": self._safe_avg(
                    self._counters["llm_interaction_latency_total_ms"],
                    self._counters["llm_interaction_count"],
                ),
                "llm_interaction_latency_max_ms": int(self._counters["llm_interaction_latency_max_ms"]),
                "action_execution_count": int(self._counters["action_execution_count"]),
                "action_execution_latency_total_ms": int(self._counters["action_execution_latency_total_ms"]),
                "action_execution_latency_avg_ms": self._safe_avg(
                    self._counters["action_execution_latency_total_ms"],
                    self._counters["action_execution_count"],
                ),
                "action_execution_latency_max_ms": int(self._counters["action_execution_latency_max_ms"]),
                "errors_total": int(self._counters["errors_total"]),
                "action_history": self._action_history,
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
        if event_type == "llm_message_received":
            self._counters["llm_interaction_count"] += 1
            latency_ms = self._to_non_negative_int(event.get("latency_ms"))
            self._counters["llm_interaction_latency_total_ms"] += latency_ms
            self._counters["llm_interaction_latency_max_ms"] = max(
                self._counters["llm_interaction_latency_max_ms"],
                latency_ms,
            )
        if event_type == "action_executed":
            self._counters["action_execution_count"] += 1
            duration_ms = self._to_non_negative_int(event.get("execution_duration_ms"))
            self._counters["action_execution_latency_total_ms"] += duration_ms
            self._counters["action_execution_latency_max_ms"] = max(
                self._counters["action_execution_latency_max_ms"],
                duration_ms,
            )
            action_record = {
                "timestamp": event.get("timestamp"),
                "iteration_index": event.get("iteration_index"),
                "interaction_id": event.get("interaction_id"),
                "llm_image_id": event.get("llm_image_id"),
                "action": event.get("action_executed"),
                "execution_status": event.get("execution_status"),
                "execution_duration_ms": duration_ms,
                "error_message": event.get("error_message"),
            }
            self._action_history.append(self._sanitize(action_record))
        if event_type == "action_rejected":
            action_record = {
                "timestamp": event.get("timestamp"),
                "iteration_index": event.get("iteration_index"),
                "interaction_id": event.get("interaction_id"),
                "llm_image_id": event.get("llm_image_id"),
                "action": event.get("action_proposed"),
                "execution_status": "rejected",
                "execution_duration_ms": 0,
                "error_message": event.get("safety_reason"),
            }
            self._action_history.append(self._sanitize(action_record))
        if event_type == "action_executed" and str(event.get("execution_status", "")).lower() != "success":
            self._counters["action_execution_failures"] += 1
        if self._is_error_event(event_type, event):
            self._counters["errors_total"] += 1

    def _is_error_event(self, event_type: str, event: dict[str, Any]) -> bool:
        if event_type in {
            "session_error",
            "llm_invalid_response",
            "provider_throttle",
            "action_rejected",
        }:
            return True
        if event_type == "action_executed" and str(event.get("execution_status", "")).lower() != "success":
            return True
        if event.get("error_type") is not None or event.get("error_message") is not None:
            return True
        return False

    def _to_non_negative_int(self, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, parsed)

    def _safe_avg(self, total: int, count: int) -> float:
        if count <= 0:
            return 0.0
        return round(float(total) / float(count), 3)

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