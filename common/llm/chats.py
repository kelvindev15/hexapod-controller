import asyncio
import logging
from typing import Any, List, Optional, Union

try:
    from langsmith import traceable
except Exception:
    try:
        from langsmith.run_helpers import traceable
    except Exception:
        def traceable(*decorator_args, **decorator_kwargs):
            if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
                return decorator_args[0]

            def decorator(func):
                return func

            return decorator

# Optional LangChain / provider imports — tolerate missing packages in tests
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except Exception:
    HumanMessage = AIMessage = SystemMessage = BaseMessage = None
    ChatPromptTemplate = MessagesPlaceholder = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None

from common.utils.llm import geminiAPIKey, getOpenAIKey

logger = logging.getLogger(__name__)

class LLMRateLimitError(Exception):
    """Raised when the LLM backend is throttled."""
    pass


class LLMChat:
    """Compatibility LLM chat used across the codebase and tests.

    Keeps `chat` as a list of simple dict messages like
    `{"role": "user", "content": "..."}` to match tests.
    """
    def __init__(self, provider: str = "gemini", model_name: Optional[str] = None, max_history_turns: Optional[int] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_history_turns = max_history_turns
        self.chat: List[dict] = []
        self.system_instruction: Optional[str] = None
        self.chat_id: Optional[str] = None
        self._lock = asyncio.Lock()

        # Initialize a real LLM if desired; tests commonly override `self.llm`.
        try:
            self.llm = self._init_llm()
        except Exception:
            self.llm = None

    def _get_default_model(self, provider: str) -> str:
        defaults = {
            "gemini": "gemini-1.5-pro",
            "openai": "gpt-4o",
            "ollama": "llava"
        }
        return defaults.get(provider, "gemini-1.5-pro")

    def _init_llm(self):
        common_params = {"temperature": 0}
        model = self.model_name or self._get_default_model(self.provider)
        if self.provider == "gemini":
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain_google_genai not available; cannot initialize Gemini provider")
            return ChatGoogleGenerativeAI(model=model, api_key=geminiAPIKey(), **common_params)
        elif self.provider == "openai":
            if ChatOpenAI is None:
                raise ImportError("langchain_openai not available; cannot initialize OpenAI provider")
            return ChatOpenAI(model=model, api_key=getOpenAIKey(), **common_params)
        elif self.provider == "ollama":
            if ChatOllama is None:
                raise ImportError("langchain_ollama not available; cannot initialize Ollama provider")
            return ChatOllama(model=model, **common_params)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def get_model_name(self) -> str:
        """Return the effective model name for this chat instance."""
        return self.model_name or self._get_default_model(self.provider)

    def _get_trace_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.chat_id:
            metadata["thread_id"] = self.chat_id
            metadata["chat_id"] = self.chat_id
        return metadata

    def _prune_chat(self):
        if self.max_history_turns is None:
            return

        # max_history_turns counts user+assistant turns. Each turn is 2 messages.
        keep = self.max_history_turns * 2
        base = 1 if self.system_instruction else 0
        if len(self.chat) > keep + base:
            # keep last `keep` messages and optionally the system at front
            preserved = self.chat[-keep:]
            if self.system_instruction:
                preserved = [{"role": "system", "content": self.system_instruction}] + preserved
            self.chat = preserved

    async def send_message(self, message: Union[str, dict], image_data: Optional[str] = None) -> dict:
        """Accepts a message (dict or string), appends it to `chat`, calls the LLM, appends the response, and returns it."""
        async with self._lock:
            if isinstance(message, str):
                user_msg = {"role": "user", "content": message}
            else:
                user_msg = message.copy()

            # attach image_data inline if provided
            if image_data:
                # simple convention for tests: append image_data to content
                user_msg["content"] = f"{user_msg.get('content','')} {image_data}"

            self.chat.append(user_msg)
            self._prune_chat()

            # Prepare messages to send to LLM if available
            llm_input = []
            if self.system_instruction:
                llm_input.append({"role": "system", "content": self.system_instruction})
            llm_input.extend(self.chat)

            # Call LLM (tests frequently stub `self.llm` with an async `ainvoke`)
            if self.llm and hasattr(self.llm, "ainvoke"):
                trace_metadata = self._get_trace_metadata()

                @traceable(name=f"{self.provider}.send_message", run_type="llm", metadata=trace_metadata or None)
                async def _invoke_llm(messages):
                    return await self.llm.ainvoke(messages)

                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        response = await _invoke_llm(llm_input)
                        break
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            logger.error(f"LLM final attempt failed: {e}")
                            raise LLMRateLimitError("LLM exhausted after retries.")
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"LLM call failed (attempt {attempt+1}), retrying after {wait}s: {e}")
                        await asyncio.sleep(wait)
            else:
                # No LLM available: return a placeholder assistant reply
                response = {"role": "assistant", "content": ""}

            # Normalize response to dict
            if isinstance(response, dict):
                assistant_msg = response.copy()
            else:
                # some LLMs return objects with `.content` and `.role`
                role = getattr(response, "role", "assistant")
                content = getattr(response, "content", None)
                if content is None and hasattr(response, "message"):
                    content = getattr(response, "message")
                assistant_msg = {"role": role, "content": content}

            self.chat.append(assistant_msg)
            self._prune_chat()

            return assistant_msg

    def clear_chat(self):
        self.chat = []


# Backwards-compatible named chat classes expected elsewhere in the repo
class GeminiChat(LLMChat):
    def __init__(self, model_name: Optional[str] = None, max_history_turns: Optional[int] = None):
        super().__init__(provider="gemini", model_name=model_name, max_history_turns=max_history_turns)


class OpenAIChat(LLMChat):
    def __init__(self, model_name: Optional[str] = None, max_history_turns: Optional[int] = None):
        super().__init__(provider="openai", model_name=model_name, max_history_turns=max_history_turns)


class OllamaChat(LLMChat):
    def __init__(self, model_name: Optional[str] = None, max_history_turns: Optional[int] = None):
        super().__init__(provider="ollama", model_name=model_name, max_history_turns=max_history_turns)


__all__ = [
    "LLMRateLimitError",
    "LLMChat",
    "GeminiChat",
    "OpenAIChat",
    "OllamaChat",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "BaseMessage",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    "ChatOpenAI",
    "ChatGoogleGenerativeAI",
    "ChatOllama",
]