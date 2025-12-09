"""
Shared helpers for dataset IO, validation, and manifest generation.

All datasets are JSONL with the schema:
- prompt: str
- completion: str
- tags: list[str]
- quality_score: float (0-1)
- metadata: dict (optional)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DATA_ROOT = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_ROOT / "raw"
CLEAN_DIR = DATA_ROOT / "clean"
PROCESSED_DIR = DATA_ROOT / "processed"
MANIFEST_DIR = DATA_ROOT / "manifests"


SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"(?i)api[_-]?key[:=]\\s*[A-Za-z0-9\\-_=]{16,}"),
    re.compile(r"(?i)secret[:=]\\s*[A-Za-z0-9\\-_=]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # common OpenAI style
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub token
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+"),  # emails
]


@dataclass
class DatasetRow:
    prompt: str
    completion: str
    tags: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata or {}
        return payload


def ensure_data_dirs() -> None:
    """Create expected data directories."""
    for directory in (DATA_ROOT, RAW_DIR, CLEAN_DIR, PROCESSED_DIR, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    ensure_data_dirs()
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_data_dirs()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _estimate_tokens(text: str) -> int:
    # Rough heuristic; avoids extra deps while catching extreme lengths
    return max(1, len(text.split()) * 1)  # ~1 token per word for guardrails


def redact_secrets(text: str) -> Tuple[str, bool]:
    """Redact potential secrets and indicate if anything was changed."""
    replaced = False
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            text = pattern.sub("[REDACTED]", text)
            replaced = True
    return text, replaced


def validate_row(
    raw: Dict[str, Any],
    max_tokens: int,
    min_quality: float,
    redact: bool = True,
) -> Tuple[DatasetRow, Dict[str, Any]]:
    """Validate and normalize a raw row."""
    issues: Dict[str, Any] = {}

    prompt = str(raw.get("prompt", "")).strip()
    completion = str(raw.get("completion", "")).strip()
    tags = raw.get("tags") or []
    quality = float(raw.get("quality_score", 1.0))
    metadata = raw.get("metadata") or {}

    if isinstance(tags, str):
        tags = [tags]

    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(completion)

    if prompt_tokens + completion_tokens > max_tokens:
        issues["token_overflow"] = prompt_tokens + completion_tokens

    if quality < min_quality:
        issues["quality_below_threshold"] = quality

    redactions = False
    if redact:
        prompt, prompt_redacted = redact_secrets(prompt)
        completion, completion_redacted = redact_secrets(completion)
        redactions = prompt_redacted or completion_redacted
        if redactions:
            issues["secrets_redacted"] = True

    normalized = DatasetRow(
        prompt=prompt,
        completion=completion,
        tags=tags,
        quality_score=quality,
        metadata=metadata,
    )
    return normalized, issues


def deduplicate(rows: List[DatasetRow]) -> Tuple[List[DatasetRow], int]:
    """Drop duplicate prompt+completion pairs."""
    seen = set()
    unique: List[DatasetRow] = []
    drops = 0
    for row in rows:
        key = (row.prompt, row.completion)
        if key in seen:
            drops += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, drops


def manifest_entry(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_manifest(files: List[Path]) -> Path:
    ensure_data_dirs()
    MANIFEST_DIR.mkdir(exist_ok=True)
    manifest = {
        "files": [manifest_entry(p) for p in files],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    target = MANIFEST_DIR / "dataset_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def collect_stats(rows: List[DatasetRow]) -> Dict[str, Any]:
    token_counts = [
        _estimate_tokens(r.prompt) + _estimate_tokens(r.completion) for r in rows
    ]
    return {
        "examples": len(rows),
        "avg_tokens": (sum(token_counts) / len(token_counts)) if rows else 0,
        "max_tokens": max(token_counts) if rows else 0,
    }


def split_dataset(
    rows: List[DatasetRow], train_ratio: float = 0.9
) -> Dict[str, List[DatasetRow]]:
    cutoff = int(len(rows) * train_ratio)
    return {"train": rows[:cutoff], "eval": rows[cutoff:]}

