from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found")
