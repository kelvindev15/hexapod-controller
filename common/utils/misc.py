import json
import re

def extractJSON(text: str):
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
