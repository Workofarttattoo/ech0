#!/usr/bin/env python3
"""Convert legacy ech0_training_data JSON dumps into the new JSONL schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ech0_training import data_utils


def to_new_schema(record: dict, default_tags: List[str]) -> dict:
    prompt = record.get("instruction") or record.get("prompt") or ""
    input_text = record.get("input") or ""
    if input_text:
        prompt = f"{prompt}\n\nInput:\n{input_text}"
    completion = record.get("output") or record.get("completion") or ""
    tags = record.get("tags") or record.get("category") or record.get("categories") or default_tags
    if isinstance(tags, str):
        tags = [tags]
    if isinstance(tags, list):
        tags = default_tags + tags
    return {
        "prompt": prompt.strip(),
        "completion": completion.strip(),
        "tags": tags,
        "quality_score": float(record.get("quality_score", 0.7)),
        "metadata": {
            "legacy_domain": record.get("domain"),
            "source_file": record.get("source_file"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Import legacy dataset JSON files.")
    parser.add_argument("--input", required=True, help="Legacy JSON file (array of dicts).")
    parser.add_argument("--output", help="Target JSONL path under ech0_training/data/raw/")
    parser.add_argument("--tag", action="append", default=[], help="Extra tags to attach.")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"Input not found: {source}")

    raw = json.loads(source.read_text(encoding="utf-8"))
    rows = [to_new_schema(rec, args.tag) for rec in raw]

    target = (
        Path(args.output)
        if args.output
        else data_utils.RAW_DIR / (source.stem + ".jsonl")
    )
    data_utils.save_jsonl(rows, target)
    print(f"Wrote {len(rows)} rows → {target}")
    print("Next: python scripts/validate_dataset.py --input", target)


if __name__ == "__main__":
    main()

