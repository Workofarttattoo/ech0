"""Lightweight model registry stored as JSON."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


@dataclass
class RunMetadata:
    run_id: str
    run_dir: str
    base_model: str
    config: str
    datasets: List[str]
    eval_report: Optional[str] = None
    score: Optional[float] = None
    stage: str = "staging"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def _load() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"runs": [], "active_run": None}


def _save(payload: Dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dump() -> Dict[str, Any]:
    """Return current registry payload."""
    return _load()


def register_run(meta: RunMetadata) -> None:
    registry = _load()
    registry.setdefault("runs", []).append(asdict(meta))
    _save(registry)


def promote(run_id: str, stage: str = "production") -> None:
    registry = _load()
    found = False
    for run in registry.get("runs", []):
        if run["run_id"] == run_id:
            run["stage"] = stage
            registry["active_run"] = run_id
            found = True
    if not found:
        raise ValueError(f"Run {run_id} not found in registry.")
    _save(registry)


def rollback() -> Optional[str]:
    registry = _load()
    runs = registry.get("runs", [])
    if len(runs) < 2:
        return None
    # Pick previous run as active
    previous = runs[-2]["run_id"]
    registry["active_run"] = previous
    _save(registry)
    return previous

