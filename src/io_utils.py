"""Input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_utterances_from_csv(path: str | Path, utterance_column: str) -> list[str]:
    df = pd.read_csv(path)
    if utterance_column not in df.columns:
        raise ValueError(f"Column '{utterance_column}' not found. Available columns: {list(df.columns)}")
    utterances = (
        df[utterance_column]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    return [u for u in utterances if u]


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def dataframe_to_jsonl(df: pd.DataFrame) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in df.to_dict(orient="records")) + "\n"


def save_dataframe(df: pd.DataFrame, output_path: str | Path, fmt: str | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chosen_fmt = (fmt or output_path.suffix.replace(".", "") or "csv").lower()
    if chosen_fmt == "csv":
        df.to_csv(output_path, index=False)
    elif chosen_fmt == "jsonl":
        output_path.write_text(dataframe_to_jsonl(df), encoding="utf-8")
    elif chosen_fmt == "json":
        output_path.write_text(df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    else:
        raise ValueError("Unsupported format. Use csv, jsonl, or json.")
