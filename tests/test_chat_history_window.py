import unittest
import types
import sys

fake_lc_messages = types.ModuleType("langchain_core.messages")
fake_lc_messages_base = types.ModuleType("langchain_core.messages.base")


class BaseMessage:
    def __init__(self, content):
        self.content = content


class HumanMessage(BaseMessage):
    pass


class AIMessage(BaseMessage):
    pass


class SystemMessage(BaseMessage):
    pass


fake_lc_messages.HumanMessage = HumanMessage
fake_lc_messages.AIMessage = AIMessage
fake_lc_messages.SystemMessage = SystemMessage
fake_lc_messages_base.BaseMessage = BaseMessage
sys.modules.setdefault("langchain_core.messages", fake_lc_messages)
sys.modules.setdefault("langchain_core.messages.base", fake_lc_messages_base)

from common.llm.chats import LLMChat


class _StubLLM:
    async def ainvoke(self, _messages, config=None):
        _ = config
        return AIMessage(content='{"goal":"ok","scene_description":"ok","reasoning":"ok","action":{"command":"STOP"}}')


class _TestChat(LLMChat):
    def __init__(self, max_history_turns=None):
        super().__init__(max_history_turns=max_history_turns)
        self.model_name = "test"
        self.llm = _StubLLM()


class ChatHistoryWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_keeps_full_history(self):
        chat = _TestChat()

        await chat.send_message(HumanMessage(content="first"))
        await chat.send_message(HumanMessage(content="second"))
        await chat.send_message(HumanMessage(content="third"))

        self.assertEqual(len(chat.chat), 6)

    async def test_keeps_last_n_turns_without_system_instruction(self):
        chat = _TestChat(max_history_turns=1)

        await chat.send_message(HumanMessage(content="first"))
        await chat.send_message(HumanMessage(content="second"))

        self.assertEqual(len(chat.chat), 2)
        self.assertEqual(chat.chat[0].content, "second")
        self.assertIn('"command":"STOP"', chat.chat[1].content)

    async def test_keeps_last_n_turns_with_system_instruction(self):
        chat = _TestChat(max_history_turns=1)
        chat.set_system_instruction("system rules")

        await chat.send_message(HumanMessage(content="first"))
        await chat.send_message(HumanMessage(content="second"))

        self.assertEqual(len(chat.chat), 3)
        self.assertEqual(chat.chat[0].content, "system rules")
        self.assertEqual(chat.chat[1].content, "second")


if __name__ == "__main__":
    unittest.main()
