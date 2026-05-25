"""OpenAI call helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI


class OpenAIJsonError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON. Falls back to extracting the first JSON object from text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise OpenAIJsonError(f"No JSON object found in response: {text[:300]}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise OpenAIJsonError(f"Invalid JSON response: {text[:300]}") from exc


def call_openai_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> dict[str, Any]:
    """Call OpenAI Chat Completions and return parsed JSON."""
    if not api_key:
        raise ValueError("Missing OpenAI API key. Provide it in the sidebar, CLI argument, or .env file.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    )
    text = response.choices[0].message.content or "{}"
    return extract_json(text)
