import unittest

from common.llm.chats import LLMChat


class _StubLLM:
    async def ainvoke(self, _messages, config=None):
        _ = config
        return {"role": "assistant", "content": '{"goal":"ok","scene_description":"ok","reasoning":"ok","action":{"command":"STOP"}}'}


class _TestChat(LLMChat):
    def __init__(self, max_history_turns=None):
        super().__init__(max_history_turns=max_history_turns)
        self.model_name = "test"
        self.llm = _StubLLM()


class ChatHistoryWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_id_is_exported_as_thread_metadata(self):
        chat = _TestChat()
        chat.chat_id = "chat-123"

        metadata = chat._get_trace_metadata()

        self.assertEqual(metadata["thread_id"], "chat-123")
        self.assertEqual(metadata["chat_id"], "chat-123")

    async def test_default_keeps_full_history(self):
        chat = _TestChat()

        await chat.send_message({"role": "user", "content": "first"})
        await chat.send_message({"role": "user", "content": "second"})
        await chat.send_message({"role": "user", "content": "third"})

        self.assertEqual(len(chat.chat), 6)

    async def test_keeps_last_n_turns_without_system_instruction(self):
        chat = _TestChat(max_history_turns=1)

        await chat.send_message({"role": "user", "content": "first"})
        await chat.send_message({"role": "user", "content": "second"})

        self.assertEqual(len(chat.chat), 2)
        self.assertEqual(chat.chat[0]["content"], "second")
        self.assertIn('"command":"STOP"', chat.chat[1]["content"])

    async def test_keeps_last_n_turns_with_system_instruction(self):
        chat = _TestChat(max_history_turns=1)
        chat.system_instruction = "system rules"

        await chat.send_message({"role": "user", "content": "first"})
        await chat.send_message({"role": "user", "content": "second"})

        self.assertEqual(len(chat.chat), 3)
        self.assertEqual(chat.chat[0]["content"], "system rules")
        self.assertEqual(chat.chat[1]["content"], "second")


if __name__ == "__main__":
    unittest.main()
