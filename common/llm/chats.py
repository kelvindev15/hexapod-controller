from abc import ABC
import asyncio
import threading
import logging
import langsmith as ls
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages.base import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import RateLimitError

from common.utils.llm import create_sys_message, geminiAPIKey, getOpenAIKey


logger = logging.getLogger(__name__)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.075,  # 6 seconds per request
    max_bucket_size=2,
    check_every_n_seconds=1
)

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

    @ls.traceable()
    async def send_message(self, message: BaseMessage):    
        self.__checkInitilization()
        async with self._send_lock:
            rt = ls.get_current_run_tree()
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
    def __init__(self, model_name="gemini-2.0-flash"):
        super().__init__()
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=geminiAPIKey(),
            rate_limiter=rate_limiter
        )

class OllamaChat(LLMChat):
    def __init__(self, model_name="llava"):
        super().__init__()
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
        )       

class OpenAIChat(LLMChat):
    def __init__(self, model_name="gpt-4o-mini"):
        super().__init__()
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=getOpenAIKey(),
        )
