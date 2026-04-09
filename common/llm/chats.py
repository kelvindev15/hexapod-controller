from abc import ABC
import asyncio
import threading
import logging
from langchain_core.messages.base import BaseMessage

try:
    import langsmith as ls
except ImportError:
    ls = None

try:
    from openai import RateLimitError
except ImportError:
    class RateLimitError(Exception):
        pass

from common.utils.llm import create_sys_message, geminiAPIKey, getOpenAIKey


logger = logging.getLogger(__name__)


def traceable():
    if ls is None:
        def _identity(func):
            return func
        return _identity
    return ls.traceable()

class LLMChat(ABC):
    def __init__(self):
        self.llm = None
        self.model_name = None
        self.system_instruction = None
        self.chat = []
        self.chat_id = None
        self._send_lock = asyncio.Lock()
        self._chat_state_lock = threading.RLock()
        self.clear_chat()

    @traceable()
    async def send_message(self, message: BaseMessage):    
        self.__checkInitilization()
        async with self._send_lock:
            if ls is not None:
                rt = ls.get_current_run_tree()
                if rt is not None:
                    rt.metadata["experiment_id"] = self.chat_id
                    rt.metadata["session_id"] = self.chat_id
                    rt.tags.extend(["WEBOTS"])
            with self._chat_state_lock:
                self.chat.append(message)
                if len(self.chat) == 4:
                    del self.chat[1:3]  # remove 2nd and 3rd messages to keep context manageable
                messages = list(self.chat)
            tries = 0
            answer = None
            while tries < 3:
                try:
                    answer = await self.llm.ainvoke(messages)
                    break
                except RateLimitError as e:
                    tries += 1
                    logger.warning(
                        "Rate limit exceeded chat_id=%s model=%s retry=%d/3 wait_seconds=60",
                        self.chat_id,
                        self.model_name,
                        tries,
                    )
                    await asyncio.sleep(60)  # Wait for 60 seconds before retrying
                    if tries >= 3:
                        raise e
                except Exception as e:
                    logger.exception(
                        "LLM invocation failed chat_id=%s model=%s error_type=%s",
                        self.chat_id,
                        self.model_name,
                        type(e).__name__,
                    )
                    raise e
            with self._chat_state_lock:
                self.chat.append(answer)
            return answer.content
                
    def generate(self, message):
        self.__checkInitilization()
        return self.llm.invoke(message).content

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
    def __init__(self, model_name="gemini-robotics-er-1.5-preview"):
        super().__init__()
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
    def __init__(self, model_name="llava"):
        super().__init__()
        from langchain_ollama import ChatOllama

        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
        )       

class OpenAIChat(LLMChat):
    def __init__(self, model_name="gpt-4o-mini"):
        super().__init__()
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
