import os
from pathlib import Path


_ENV_LOADED = False


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export "):].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_wrapping_quotes(value.strip())

            if key:
                os.environ.setdefault(key, value)
    except Exception:
        pass


def geminiAPIKey():
    _load_env_file()
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Gemini API key not set (expected GEMINI_API_KEY or GOOGLE_API_KEY)")
    return api_key


def getOpenAIKey():
    _load_env_file()
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key


def create_sys_message(text: str):
    return {"role": "system", "content": text}


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
    return {"role": "user", "content": message_content}
