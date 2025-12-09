#!/usr/bin/env python3
"""
Validate and stage datasets into the ech0_training data layout.

Usage:
    python scripts/validate_dataset.py --input ech0_training/data/raw/example.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ech0_training import data_utils


def _load_any(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        return data_utils.load_jsonl(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and stage ech0 datasets.")
    parser.add_argument("--input", required=True, help="Path to raw dataset (.jsonl or .json)")
    parser.add_argument("--output", help="Optional explicit output path (.jsonl)")
    parser.add_argument(
        "--stage",
        choices=["clean", "processed"],
        default="clean",
        help="Where to drop validated data.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="Token limit for prompt+completion.")
    parser.add_argument("--min-quality", type=float, default=0.25, help="Minimum allowed quality_score.")
    parser.add_argument(
        "--fail-on-secrets",
        action="store_true",
        help="Fail validation if potential secrets are detected.",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prompt+completion deduplication.",
    )
    args = parser.parse_args()

    raw_path = Path(args.input).expanduser()
    if not raw_path.exists():
        raise SystemExit(f"Input not found: {raw_path}")

    data_utils.ensure_data_dirs()
    target_dir = data_utils.CLEAN_DIR if args.stage == "clean" else data_utils.PROCESSED_DIR

    rows_raw = _load_any(raw_path)

    validated = []
    skipped = {"quality": 0, "tokens": 0}
    secret_hits = 0
    for raw in rows_raw:
        normalized, issues = data_utils.validate_row(
            raw, max_tokens=args.max_tokens, min_quality=args.min_quality, redact=True
        )
        if "token_overflow" in issues:
            skipped["tokens"] += 1
            continue
        if "quality_below_threshold" in issues:
            skipped["quality"] += 1
            continue
        if issues.get("secrets_redacted"):
            secret_hits += 1
            if args.fail_on_secrets:
                raise SystemExit("Aborting: potential secrets found.")
        validated.append(normalized)

    if args.dedupe:
        validated, dropped = data_utils.deduplicate(validated)
    else:
        dropped = 0

    stats = data_utils.collect_stats(validated)
    stats.update(
        {
            "skipped_quality": skipped["quality"],
            "skipped_tokens": skipped["tokens"],
            "secret_redactions": secret_hits,
            "deduplicated": dropped,
            "input_path": str(raw_path),
        }
    )

    if not validated:
        raise SystemExit("No valid rows after filtering.")

    out_path = Path(args.output) if args.output else target_dir / raw_path.name.replace(".json", ".jsonl")
    data_utils.save_jsonl([r.to_json() for r in validated], out_path)

    manifest_path = data_utils.write_manifest([out_path])

    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(out_path), "manifest": str(manifest_path), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()

