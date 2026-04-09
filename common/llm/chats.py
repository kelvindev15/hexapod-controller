from abc import ABC
import asyncio
import threading
import logging
import os
import uuid
from langchain_core.messages.base import BaseMessage

try:
    from openai import RateLimitError
except ImportError:
    class RateLimitError(Exception):
        pass

from common.utils.llm import create_sys_message, geminiAPIKey, getOpenAIKey


logger = logging.getLogger(__name__)


class LLMRateLimitError(Exception):
    """Raised when the configured LLM backend is throttled after retries."""

    pass

class LLMChat(ABC):
    def __init__(self, max_history_turns: int | None = None):
        self.llm = None
        self.model_name = None
        self.system_instruction = None
        self.chat = []
        self.chat_id = None
        self.user_id = self._resolve_user_id()
        self.max_history_turns = self._normalize_max_history_turns(max_history_turns)
        self._send_lock = asyncio.Lock()
        self._chat_state_lock = threading.RLock()
        self.clear_chat()

    def _normalize_max_history_turns(self, max_history_turns: int | None) -> int | None:
        if max_history_turns is None:
            return None

        normalized = int(max_history_turns)
        if normalized < 0:
            raise ValueError("max_history_turns must be >= 0 or None")
        return normalized

    def set_max_history_turns(self, max_history_turns: int | None):
        with self._chat_state_lock:
            self.max_history_turns = self._normalize_max_history_turns(max_history_turns)
            self._prune_chat_history_locked()

    def _prune_chat_history_locked(self):
        if self.max_history_turns is None:
            return

        prefix_count = 1 if self.system_instruction is not None and self.chat else 0
        conversation = self.chat[prefix_count:]
        if not conversation:
            return

        pending_user_message = len(conversation) % 2 == 1
        max_messages = self.max_history_turns * 2 + (1 if pending_user_message else 0)
        if len(conversation) <= max_messages:
            return

        self.chat = self.chat[:prefix_count] + conversation[-max_messages:]

    def _resolve_user_id(self) -> str:
        user_id = os.getenv("HEXAPOD_MLFLOW_USER_ID") or os.getenv("USER") or os.getenv("USERNAME")
        if user_id and user_id.strip():
            return user_id.strip()
        return "unknown-user"

    def _ensure_chat_id(self) -> str:
        if not self.chat_id:
            self.chat_id = str(uuid.uuid4())
        return self.chat_id

    def _build_trace_config(self) -> dict:
        session_id = self._ensure_chat_id()
        user_id = self.user_id or self._resolve_user_id()
        metadata = {
            "session_id": session_id,
            "chat_id": session_id,
            "user_id": user_id,
            "model_name": self.model_name,
            # MLflow canonical metadata keys for user/session trace grouping.
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
            # Compatibility aliases for older/variant trace schema consumers.
            "mlflow.trace.user_id": user_id,
            "mlflow.trace.session_id": session_id,
        }
        experiment_name = os.getenv("MLFLOW_TRACKING_EXPERIMENT")
        if experiment_name:
            metadata["mlflow_experiment"] = experiment_name

        return {
            "metadata": metadata,
            "configurable": {"session_id": session_id, "user_id": user_id},
            "tags": [f"session:{session_id}", f"user:{user_id}"],
        }

    def _get_mlflow_module(self):
        try:
            import mlflow
            return mlflow
        except ImportError:
            return None

    def _trace_metadata(self, session_id: str) -> dict:
        user_id = self.user_id or self._resolve_user_id()
        return {
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
            "mlflow.trace.user_id": user_id,
            "mlflow.trace.session_id": session_id,
        }

    def _extract_message_text_and_image_flag(self, message: BaseMessage) -> tuple[str, bool]:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content, False

        if isinstance(content, list):
            text_parts: list[str] = []
            has_image = False
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "image_url":
                    has_image = True
            return "\n".join(part for part in text_parts if part).strip(), has_image

        return str(content), False

    def _build_trace_request_payload_from_messages(self, messages) -> dict:
        session_id = self._ensure_chat_id()
        user_id = self.user_id or self._resolve_user_id()
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "message_count": len(messages),
        }

        if not messages:
            payload["input_text"] = ""
            payload["has_image"] = False
            return payload

        last_message = messages[-1]
        text, has_image = self._extract_message_text_and_image_flag(last_message)
        payload["input_text"] = text[:8000]
        payload["has_image"] = has_image
        payload["input_text_truncated"] = len(text) > 8000
        return payload

    def _build_trace_request_payload_from_message(self, message: BaseMessage) -> dict:
        text, has_image = self._extract_message_text_and_image_flag(message)
        session_id = self._ensure_chat_id()
        user_id = self.user_id or self._resolve_user_id()
        return {
            "user_id": user_id,
            "session_id": session_id,
            "input_text": text[:8000],
            "has_image": has_image,
            "input_text_truncated": len(text) > 8000,
        }

    def _update_current_mlflow_trace(self, session_id: str, mlflow_module=None) -> None:
        """Best-effort enrichment for active MLflow traces."""
        mlflow_module = mlflow_module or self._get_mlflow_module()
        if mlflow_module is None:
            return

        update_current_trace = getattr(mlflow_module, "update_current_trace", None)
        if not callable(update_current_trace):
            return

        try:
            update_current_trace(metadata=self._trace_metadata(session_id))
        except Exception:
            logger.debug("Unable to update current MLflow trace metadata", exc_info=True)

    async def _ainvoke_llm_with_trace_context(self, messages, trace_config: dict, request_payload: dict):
        session_id = trace_config["metadata"]["mlflow.trace.session"]
        mlflow_module = self._get_mlflow_module()
        trace_decorator = getattr(mlflow_module, "trace", None) if mlflow_module else None

        if callable(trace_decorator):
            @trace_decorator(name="llm_chat_ainvoke")
            async def _invoke(trace_input: dict):
                _ = trace_input
                self._update_current_mlflow_trace(session_id=session_id, mlflow_module=mlflow_module)
                try:
                    return await self.llm.ainvoke(messages, config=trace_config)
                except TypeError:
                    logger.debug("LLM backend does not accept invoke config; calling ainvoke without trace config")
                    return await self.llm.ainvoke(messages)

            return await _invoke(request_payload)

        self._update_current_mlflow_trace(session_id=session_id, mlflow_module=mlflow_module)
        try:
            return await self.llm.ainvoke(messages, config=trace_config)
        except TypeError:
            logger.debug("LLM backend does not accept invoke config; calling ainvoke without trace config")
            return await self.llm.ainvoke(messages)

    def _invoke_llm_with_trace_context(self, message, trace_config: dict, request_payload: dict):
        session_id = trace_config["metadata"]["mlflow.trace.session"]
        mlflow_module = self._get_mlflow_module()
        trace_decorator = getattr(mlflow_module, "trace", None) if mlflow_module else None

        if callable(trace_decorator):
            @trace_decorator(name="llm_chat_invoke")
            def _invoke(trace_input: dict):
                _ = trace_input
                self._update_current_mlflow_trace(session_id=session_id, mlflow_module=mlflow_module)
                try:
                    return self.llm.invoke(message, config=trace_config)
                except TypeError:
                    logger.debug("LLM backend does not accept invoke config; calling invoke without trace config")
                    return self.llm.invoke(message)

            return _invoke(request_payload)

        self._update_current_mlflow_trace(session_id=session_id, mlflow_module=mlflow_module)
        try:
            return self.llm.invoke(message, config=trace_config)
        except TypeError:
            logger.debug("LLM backend does not accept invoke config; calling invoke without trace config")
            return self.llm.invoke(message)

    async def _ainvoke_with_trace_config(self, messages):
        trace_config = self._build_trace_config()
        request_payload = self._build_trace_request_payload_from_messages(messages)
        return await self._ainvoke_llm_with_trace_context(
            messages=messages,
            trace_config=trace_config,
            request_payload=request_payload,
        )

    def _is_rate_limit_error(self, error: Exception) -> bool:
        if isinstance(error, RateLimitError):
            return True

        error_name = type(error).__name__.lower()
        error_message = str(error).lower()
        return (
            "resourceexhausted" in error_name
            or "rate" in error_name and "limit" in error_name
            or "resource exhausted" in error_message
            or "rate limit" in error_message
            or "429" in error_message
        )

    def _invoke_with_trace_config(self, message):
        trace_config = self._build_trace_config()
        request_payload = self._build_trace_request_payload_from_message(message)
        return self._invoke_llm_with_trace_context(
            message=message,
            trace_config=trace_config,
            request_payload=request_payload,
        )

    async def send_message(self, message: BaseMessage):    
        self.__checkInitilization()
        async with self._send_lock:
            with self._chat_state_lock:
                self.chat.append(message)
                self._prune_chat_history_locked()
                messages = list(self.chat)
            tries = 0
            answer = None
            max_attempts = 3
            while tries < max_attempts:
                try:
                    answer = await self._ainvoke_with_trace_config(messages)
                    break
                except Exception as e:
                    if not self._is_rate_limit_error(e):
                        logger.exception(
                            "LLM invocation failed chat_id=%s model=%s error_type=%s",
                            self.chat_id,
                            self.model_name,
                            type(e).__name__,
                        )
                        raise e

                    tries += 1
                    wait_seconds = 2 ** tries
                    logger.warning(
                        "Rate limit exceeded chat_id=%s model=%s retry=%d/%d wait_seconds=%d error_type=%s",
                        self.chat_id,
                        self.model_name,
                        tries,
                        max_attempts,
                        wait_seconds,
                        type(e).__name__,
                    )
                    if tries >= max_attempts:
                        logger.warning(
                            "LLM provider throttled after retries chat_id=%s model=%s attempts=%d",
                            self.chat_id,
                            self.model_name,
                            max_attempts,
                        )
                        raise LLMRateLimitError(
                            f"LLM provider rate-limited for model={self.model_name}. Try again in a few seconds."
                        ) from e

                    await asyncio.sleep(wait_seconds)
            with self._chat_state_lock:
                if answer is None:
                    raise LLMRateLimitError(
                        "LLM response unavailable after retry attempts due to provider throttling"
                    )
                self.chat.append(answer)
                self._prune_chat_history_locked()
            return answer.content
                
    def generate(self, message):
        self.__checkInitilization()
        return self._invoke_with_trace_config(message).content

    def get_model_name(self):
        self.__checkInitilization()
        return self.model_name

    def set_system_instruction(self, system_instruction: str):
        with self._chat_state_lock:
            self.system_instruction = system_instruction
            self.clear_chat()

    def set_chat_id(self, chat_id: str):
        self.__checkInitilization()
        self.chat_id = chat_id

    def set_user_id(self, user_id: str | None):
        self.__checkInitilization()
        if user_id and user_id.strip():
            self.user_id = user_id.strip()
        else:
            self.user_id = self._resolve_user_id()

    def clear_system_instruction(self):
        self.__checkInitilization()
        with self._chat_state_lock:
            self.system_instruction = None
            self.clear_chat()
    
    def get_system_instruction(self):
        self.__checkInitilization()
        return self.system_instruction
    
    def clear_chat(self):
        with self._chat_state_lock:
            if self.system_instruction is not None:
                self.chat = [create_sys_message(self.system_instruction)]
            else:
                self.chat = []

    def __checkInitilization(self):
        if not self.llm:
            raise Exception("LLM not initialized")

class GeminiChat(LLMChat):
    def __init__(self, model_name="gemini-robotics-er-1.5-preview", max_history_turns: int | None = None):
        super().__init__(max_history_turns=max_history_turns)
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=geminiAPIKey(),
        )

class OllamaChat(LLMChat):
    def __init__(self, model_name="llava", max_history_turns: int | None = None):
        super().__init__(max_history_turns=max_history_turns)
        from langchain_ollama import ChatOllama

        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
        )       

class OpenAIChat(LLMChat):
    def __init__(self, model_name="gpt-4o-mini", max_history_turns: int | None = None):
        super().__init__(max_history_turns=max_history_turns)
        from langchain_openai import ChatOpenAI

        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=getOpenAIKey(),
        )
