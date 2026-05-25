"""Pydantic schemas used across the generator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedExample(BaseModel):
    seed_utterance: str
    target_polarity: str
    generation_mode: str
    generated_utterance: str
    generated_context: str
    rationale: str


class ValidationResult(BaseModel):
    predicted_polarity: str
    matches_target: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    safety_ok: bool
    context_dependent: bool
    contains_slur: bool
    too_explicit: bool


class PipelineRow(BaseModel):
    seed_utterance: str
    target_polarity: str
    generation_mode: str
    generated_utterance: str
    generated_context: str
    rationale: str
    validator_prediction: str
    validator_confidence: float
    validation_passed: bool
    validator_reason: str
    safety_ok: bool
    context_dependent: bool
    contains_slur: bool
    too_explicit: bool
