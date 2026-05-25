"""Generation functions."""

from __future__ import annotations

from .config import AppConfig
from .openai_utils import call_openai_json
from .prompts import (
    DIRECT_CONTEXT_PROMPT,
    GENERATOR_SYSTEM,
    MULTISTAGE_STEP1_PROMPT,
    MULTISTAGE_STEP2_PROMPT,
    SINGLE_STAGE_PROMPT,
)
from .schemas import GeneratedExample


VALID_MODES = {
    "direct": "direct_context_augmentation",
    "single_stage": "single_stage_new_pair",
    "multistage": "two_step_multistage",
}


def _normalize_mode(mode: str) -> str:
    mode = mode.strip().lower()
    aliases = {
        "direct context augmentation": "direct",
        "single-stage new pair": "single_stage",
        "single stage": "single_stage",
        "two-step multistage": "multistage",
        "two step multistage": "multistage",
    }
    return aliases.get(mode, mode)


def _to_example(seed_utterance: str, target_polarity: str, mode: str, payload: dict) -> GeneratedExample:
    return GeneratedExample(
        seed_utterance=seed_utterance,
        target_polarity=target_polarity,
        generation_mode=VALID_MODES[_normalize_mode(mode)],
        generated_utterance=str(payload.get("generated_utterance", "")).strip(),
        generated_context=str(payload.get("generated_context", "")).strip(),
        rationale=str(payload.get("rationale", "")).strip(),
    )


def generate_direct_context(seed_utterance: str, target_polarity: str, config: AppConfig) -> GeneratedExample:
    prompt = DIRECT_CONTEXT_PROMPT.format(seed_utterance=seed_utterance, target_polarity=target_polarity)
    payload = call_openai_json(
        api_key=config.api_key or "",
        model=config.generator_model,
        system_prompt=GENERATOR_SYSTEM,
        user_prompt=prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    # Enforce exact seed utterance for direct mode.
    payload["generated_utterance"] = seed_utterance
    return _to_example(seed_utterance, target_polarity, "direct", payload)


def generate_single_stage_pair(seed_utterance: str, target_polarity: str, config: AppConfig) -> GeneratedExample:
    prompt = SINGLE_STAGE_PROMPT.format(seed_utterance=seed_utterance, target_polarity=target_polarity)
    payload = call_openai_json(
        api_key=config.api_key or "",
        model=config.generator_model,
        system_prompt=GENERATOR_SYSTEM,
        user_prompt=prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return _to_example(seed_utterance, target_polarity, "single_stage", payload)


def generate_multistage_pair(seed_utterance: str, target_polarity: str, config: AppConfig) -> GeneratedExample:
    # Use the opposite polarity as an intermediate to encourage semantic diversity.
    intermediate_polarity = "benign" if target_polarity == "toxic" else "toxic"

    step1_prompt = MULTISTAGE_STEP1_PROMPT.format(
        seed_utterance=seed_utterance,
        intermediate_polarity=intermediate_polarity,
    )
    intermediate = call_openai_json(
        api_key=config.api_key or "",
        model=config.generator_model,
        system_prompt=GENERATOR_SYSTEM,
        user_prompt=step1_prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    step2_prompt = MULTISTAGE_STEP2_PROMPT.format(
        seed_utterance=seed_utterance,
        intermediate_utterance=intermediate.get("generated_utterance", seed_utterance),
        intermediate_context=intermediate.get("generated_context", ""),
        target_polarity=target_polarity,
    )
    payload = call_openai_json(
        api_key=config.api_key or "",
        model=config.generator_model,
        system_prompt=GENERATOR_SYSTEM,
        user_prompt=step2_prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return _to_example(seed_utterance, target_polarity, "multistage", payload)


def generate_example(seed_utterance: str, target_polarity: str, mode: str, config: AppConfig) -> GeneratedExample:
    mode = _normalize_mode(mode)
    if mode == "direct":
        return generate_direct_context(seed_utterance, target_polarity, config)
    if mode == "single_stage":
        return generate_single_stage_pair(seed_utterance, target_polarity, config)
    if mode == "multistage":
        return generate_multistage_pair(seed_utterance, target_polarity, config)
    raise ValueError(f"Unknown generation mode: {mode}. Valid modes: {list(VALID_MODES)}")
