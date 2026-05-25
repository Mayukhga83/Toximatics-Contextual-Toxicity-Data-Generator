from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_config
from src.io_utils import load_utterances_from_csv, rows_to_dataframe, save_dataframe
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate context-dependent toxicity utterance-context pairs."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Path to CSV containing seed utterances.")
    input_group.add_argument("--utterance", type=str, help="One manual seed utterance.")

    parser.add_argument(
        "--utterance-column",
        type=str,
        default="utterance",
        help="Column name containing seed utterances when using --input.",
    )
    parser.add_argument(
        "--target-polarity",
        type=str,
        required=True,
        choices=["toxic", "benign", "neutral", "ambiguous"],
    )
    parser.add_argument(
        "--generation-mode",
        type=str,
        default="direct",
        choices=["direct", "single_stage", "multistage"],
    )
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--output", type=str, default="data/outputs/generated_dataset.csv")
    parser.add_argument("--format", type=str, choices=["csv", "jsonl", "json"], default=None)

    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key. Overrides .env.")
    parser.add_argument("--generator-model", type=str, default=None)
    parser.add_argument("--validator-model", type=str, default=None)

    parser.add_argument("--no-validate", action="store_true", help="Skip validator model.")
    parser.add_argument("--repair-failed", action="store_true", help="Repair examples that fail validation.")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle CSV utterances.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input:
        utterances = load_utterances_from_csv(args.input, args.utterance_column)
    else:
        utterances = [args.utterance.strip()]

    config = get_config(
        api_key=args.api_key,
        generator_model=args.generator_model,
        validator_model=args.validator_model,
    )

    rows = run_pipeline(
        utterances=utterances,
        target_polarity=args.target_polarity,
        generation_mode=args.generation_mode,
        num_examples=args.num_examples,
        config=config,
        validate=not args.no_validate,
        repair_failed=args.repair_failed,
        shuffle=not args.no_shuffle,
        progress=True,
    )

    df = rows_to_dataframe(rows)
    save_dataframe(df, args.output, args.format)

    print(f"Saved {len(df)} examples to {args.output}")
    if "validation_passed" in df.columns:
        passed = int(df["validation_passed"].fillna(False).sum())
        print(f"Validation passed: {passed}/{len(df)}")


if __name__ == "__main__":
    main()
