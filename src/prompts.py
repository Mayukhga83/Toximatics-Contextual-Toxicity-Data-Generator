"""Prompt templates for generation, validation, and repair."""

from __future__ import annotations


POLARITY_DEFINITIONS = """
Polarity definitions:
- toxic: In the given context, the utterance would likely be perceived as insulting, hostile, demeaning, dismissive, humiliating, exclusionary, or socially harmful.
- benign: In the given context, the utterance would likely be perceived as harmless, supportive, friendly, joking in a mutually accepted way, or socially appropriate.
- neutral: In the given context, the utterance is not clearly harmful or supportive.
- ambiguous: In the given context, the utterance could reasonably be interpreted in more than one way.
"""


GENERATOR_SYSTEM = f"""
You generate context-dependent toxicity data for NLP research demos.

The goal is to create realistic utterance-context pairs where polarity depends on the surrounding social situation.

{POLARITY_DEFINITIONS}

Strict requirements:
- Prefer subtle, pragmatic, context-dependent toxicity instead of obvious profanity.
- Do not use slurs.
- Do not target real private people.
- Do not generate threats, instructions for harm, or graphic abuse.
- The context should be realistic and concrete: include setting, participants, and why the utterance has the target polarity.
- Return valid JSON only.
"""


DIRECT_CONTEXT_PROMPT = """
Generation mode: direct_context_augmentation

Keep the seed utterance exactly the same as the generated utterance.
Generate a new realistic context in which the utterance has the target polarity.

Seed utterance: {seed_utterance}
Target polarity: {target_polarity}

Return JSON with exactly these keys:
{{
  "generated_utterance": "...",
  "generated_context": "...",
  "target_polarity": "{target_polarity}",
  "rationale": "..."
}}
"""


SINGLE_STAGE_PROMPT = """
Generation mode: single_stage_new_pair

Given the seed utterance, generate a completely new but semantically related utterance and a realistic context.
The new pair should preserve the broad pragmatic theme of the seed but should not simply copy it.
The generated utterance must have the target polarity only when interpreted inside the generated context.

Seed utterance: {seed_utterance}
Target polarity: {target_polarity}

Return JSON with exactly these keys:
{{
  "generated_utterance": "...",
  "generated_context": "...",
  "target_polarity": "{target_polarity}",
  "rationale": "..."
}}
"""


MULTISTAGE_STEP1_PROMPT = """
Generation mode: two_step_multistage, step 1

Create an intermediate context for the seed utterance. The intermediate context should make the seed utterance sound {intermediate_polarity}.
This is not the final output. It will be used to create a more diverse final example.

Seed utterance: {seed_utterance}
Intermediate polarity: {intermediate_polarity}

Return JSON with exactly these keys:
{{
  "generated_utterance": "{seed_utterance}",
  "generated_context": "...",
  "target_polarity": "{intermediate_polarity}",
  "rationale": "..."
}}
"""


MULTISTAGE_STEP2_PROMPT = """
Generation mode: two_step_multistage, step 2

Use the intermediate example to create a new final utterance-context pair.
The final pair should be inspired by the intermediate situation, but the generated utterance and generated context should be new.
The final utterance must have the final target polarity only in the generated context.

Original seed utterance: {seed_utterance}

Intermediate utterance:
{intermediate_utterance}

Intermediate context:
{intermediate_context}

Final target polarity: {target_polarity}

Return JSON with exactly these keys:
{{
  "generated_utterance": "...",
  "generated_context": "...",
  "target_polarity": "{target_polarity}",
  "rationale": "..."
}}
"""


VALIDATOR_SYSTEM = f"""
You validate context-dependent toxicity examples.

{POLARITY_DEFINITIONS}

Evaluate only the utterance in the supplied context.
Return valid JSON only.
"""


VALIDATOR_PROMPT = """
Context:
{generated_context}

Utterance:
{generated_utterance}

Target polarity:
{target_polarity}

Decide whether the utterance has the target polarity in the context.

Also check:
- safety_ok: false if the example contains slurs, direct threats, graphic abuse, personal data, or unsafe content.
- context_dependent: true if the target polarity depends meaningfully on the context rather than only on obvious lexical toxicity.
- contains_slur: true if a slur is present.
- too_explicit: true if the example relies on extreme profanity, threats, or graphic abuse.

Return JSON with exactly these keys:
{{
  "predicted_polarity": "benign|neutral|ambiguous|toxic",
  "matches_target": true,
  "confidence": 0.0,
  "reason": "...",
  "safety_ok": true,
  "context_dependent": true,
  "contains_slur": false,
  "too_explicit": false
}}
"""


REPAIR_PROMPT = """
The generated example failed validation.

Seed utterance:
{seed_utterance}

Target polarity:
{target_polarity}

Previous generated utterance:
{generated_utterance}

Previous context:
{generated_context}

Validator reason:
{validator_reason}

Revise the generated utterance and context so that:
- It clearly matches the target polarity.
- It remains realistic.
- It is context-dependent.
- It avoids slurs, threats, and graphic abuse.

Return JSON with exactly these keys:
{{
  "generated_utterance": "...",
  "generated_context": "...",
  "target_polarity": "{target_polarity}",
  "rationale": "..."
}}
"""
