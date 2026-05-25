"""Validation and repair."""

from __future__ import annotations

from .config import AppConfig
from .openai_utils import call_openai_json
from .prompts import REPAIR_PROMPT, VALIDATOR_PROMPT, VALIDATOR_SYSTEM, GENERATOR_SYSTEM
from .schemas import GeneratedExample, ValidationResult


def validate_example(example: GeneratedExample, config: AppConfig) -> ValidationResult:
    prompt = VALIDATOR_PROMPT.format(
        generated_context=example.generated_context,
        generated_utterance=example.generated_utterance,
        target_polarity=example.target_polarity,
    )
    payload = call_openai_json(
        api_key=config.api_key or "",
        model=config.validator_model,
        system_prompt=VALIDATOR_SYSTEM,
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=600,
    )
    return ValidationResult(
        predicted_polarity=str(payload.get("predicted_polarity", "ambiguous")).strip(),
        matches_target=bool(payload.get("matches_target", False)),
        confidence=float(payload.get("confidence", 0.0)),
        reason=str(payload.get("reason", "")).strip(),
        safety_ok=bool(payload.get("safety_ok", False)),
        context_dependent=bool(payload.get("context_dependent", False)),
        contains_slur=bool(payload.get("contains_slur", False)),
        too_explicit=bool(payload.get("too_explicit", False)),
    )


def repair_example(
    example: GeneratedExample,
    validation: ValidationResult,
    config: AppConfig,
) -> GeneratedExample:
    prompt = REPAIR_PROMPT.format(
        seed_utterance=example.seed_utterance,
        target_polarity=example.target_polarity,
        generated_utterance=example.generated_utterance,
        generated_context=example.generated_context,
        validator_reason=validation.reason,
    )
    payload = call_openai_json(
        api_key=config.api_key or "",
        model=config.generator_model,
        system_prompt=GENERATOR_SYSTEM,
        user_prompt=prompt,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return GeneratedExample(
        seed_utterance=example.seed_utterance,
        target_polarity=example.target_polarity,
        generation_mode=example.generation_mode + "_repaired",
        generated_utterance=str(payload.get("generated_utterance", "")).strip(),
        generated_context=str(payload.get("generated_context", "")).strip(),
        rationale=str(payload.get("rationale", "")).strip(),
    )
