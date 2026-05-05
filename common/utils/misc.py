import json
import re
from typing import Any

def _normalize_llm_text_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload

    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")

    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item.get("text")))
                    continue
                if item.get("text") is not None:
                    parts.append(str(item.get("text")))
                    continue
                if item.get("content") is not None:
                    parts.append(_normalize_llm_text_payload(item.get("content")))
                    continue
            elif isinstance(item, str):
                parts.append(item)
                continue

            parts.append(str(item))

        return "\n".join(part for part in parts if part).strip()

    if isinstance(payload, dict):
        if payload.get("text") is not None:
            return str(payload.get("text"))
        if payload.get("content") is not None:
            return _normalize_llm_text_payload(payload.get("content"))
        return json.dumps(payload)

    return str(payload)


def extractJSON(text: Any):
    text = _normalize_llm_text_payload(text)

    # Prefer fenced JSON blocks first.
    fenced = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        json.loads(candidate)
        return candidate

    # Then try raw body as JSON.
    raw = text.strip()
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    # Fallback: find first balanced JSON object.
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:idx + 1]
                json.loads(candidate)
                return candidate

    raise ValueError("Unable to extract valid JSON object")
