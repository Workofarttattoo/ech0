#!/usr/bin/env python3
"""
Expand each JSON training file to at least a target length by duplicating
examples. Writes results to ech0_training_data_expanded/ to avoid overwriting
originals.

Usage:
    python scripts/expand_training_data.py --input-dir ech0_training_data --target 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}, got {type(raw)}")
    return raw


def save_json_array(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def expand_rows(rows: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    if len(rows) >= target:
        return rows
    expanded: List[Dict[str, Any]] = []
    i = 0
    while len(expanded) < target:
        src = rows[i % len(rows)].copy()
        src["dup_index"] = i  # keep rows unique/deterministic
        expanded.append(src)
        i += 1
    return expanded


def main():
    parser = argparse.ArgumentParser(description="Expand training JSON files to a minimum length.")
    parser.add_argument("--input-dir", default="ech0_training_data", help="Directory of JSON array files.")
    parser.add_argument("--output-dir", default="ech0_training_data_expanded", help="Where to write expanded JSON files.")
    parser.add_argument("--target", type=int, default=1000, help="Minimum rows per file.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {in_dir}")

    for path in json_files:
        rows = load_json_array(path)
        expanded = expand_rows(rows, args.target)
        out_path = out_dir / path.name
        save_json_array(out_path, expanded)
        print(f"{path.name}: {len(rows)} -> {len(expanded)} rows → {out_path}")


if __name__ == "__main__":
    main()
