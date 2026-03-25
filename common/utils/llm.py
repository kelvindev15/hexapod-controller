from langchain_core.messages import HumanMessage, SystemMessage
import os


def geminiAPIKey():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return api_key


def getOpenAIKey():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key


def create_sys_message(text: str):
    return SystemMessage(content=text)


def create_message(text: str, image: str = None):
    message_content = [
        {
            "type": "text",
            "text": text
        }
    ]
    if image is not None:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })
    return HumanMessage(content=message_content)
