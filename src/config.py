"""Configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    api_key: str | None
    generator_model: str
    validator_model: str
    temperature: float = 0.2
    max_tokens: int = 900


def get_config(
    api_key: str | None = None,
    generator_model: str | None = None,
    validator_model: str | None = None,
) -> AppConfig:
    """Create config. Explicit arguments override environment variables."""
    chosen_key = api_key or os.getenv("OPENAI_API_KEY")
    gen_model = generator_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    val_model = validator_model or os.getenv("OPENAI_VALIDATOR_MODEL", gen_model)
    return AppConfig(
        api_key=chosen_key,
        generator_model=gen_model,
        validator_model=val_model,
    )
