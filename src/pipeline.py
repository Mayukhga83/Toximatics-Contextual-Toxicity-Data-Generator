"""Pipeline orchestration."""

from __future__ import annotations

import random
from typing import Iterable

from tqdm import tqdm

from .config import AppConfig
from .generator import generate_example
from .schemas import GeneratedExample, PipelineRow
from .validator import repair_example, validate_example


def _row_from_example(example: GeneratedExample, validation) -> dict:
    passed = (
        validation.matches_target
        and validation.safety_ok
        and validation.context_dependent
        and not validation.contains_slur
        and not validation.too_explicit
    )
    return PipelineRow(
        seed_utterance=example.seed_utterance,
        target_polarity=example.target_polarity,
        generation_mode=example.generation_mode,
        generated_utterance=example.generated_utterance,
        generated_context=example.generated_context,
        rationale=example.rationale,
        validator_prediction=validation.predicted_polarity,
        validator_confidence=validation.confidence,
        validation_passed=passed,
        validator_reason=validation.reason,
        safety_ok=validation.safety_ok,
        context_dependent=validation.context_dependent,
        contains_slur=validation.contains_slur,
        too_explicit=validation.too_explicit,
    ).model_dump()


def run_pipeline(
    *,
    utterances: list[str],
    target_polarity: str,
    generation_mode: str,
    num_examples: int,
    config: AppConfig,
    validate: bool = True,
    repair_failed: bool = False,
    shuffle: bool = True,
    progress: bool = True,
) -> list[dict]:
    """Generate and optionally validate examples."""
    if not utterances:
        raise ValueError("No utterances provided.")
    if target_polarity not in {"benign", "toxic", "neutral", "ambiguous"}:
        raise ValueError("target_polarity must be one of: benign, toxic, neutral, ambiguous")

    seeds = utterances.copy()
    if shuffle:
        random.shuffle(seeds)

    # Repeat seeds if the user requests more examples than available seeds.
    selected = [seeds[i % len(seeds)] for i in range(num_examples)]
    iterator = tqdm(selected, desc="Generating examples") if progress else selected

    rows: list[dict] = []
    for seed in iterator:
        example = generate_example(seed, target_polarity, generation_mode, config)

        if not validate:
            rows.append({
                "seed_utterance": example.seed_utterance,
                "target_polarity": example.target_polarity,
                "generation_mode": example.generation_mode,
                "generated_utterance": example.generated_utterance,
                "generated_context": example.generated_context,
                "rationale": example.rationale,
                "validator_prediction": "",
                "validator_confidence": None,
                "validation_passed": None,
                "validator_reason": "",
                "safety_ok": None,
                "context_dependent": None,
                "contains_slur": None,
                "too_explicit": None,
            })
            continue

        validation = validate_example(example, config)

        if repair_failed and not (
            validation.matches_target and validation.safety_ok and validation.context_dependent
        ):
            repaired = repair_example(example, validation, config)
            repaired_validation = validate_example(repaired, config)
            rows.append(_row_from_example(repaired, repaired_validation))
        else:
            rows.append(_row_from_example(example, validation))

    return rows
