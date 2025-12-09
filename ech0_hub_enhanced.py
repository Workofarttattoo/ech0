#!/usr/bin/env python3
"""
Hub integration glue: spawn training/eval subprocesses and expose live status.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

from ech0_training import scheduler
from ech0_training.trainer import Config


def _resolve_run_dir(config_path: Path, run_id: str) -> Path:
    cfg = Config.from_yaml(config_path)
    return Path(cfg.output_root) / run_id


def launch_training(config: Path, run_id: Optional[str] = None, status_interval: float = 5.0) -> Iterator[Dict[str, object]]:
    """
    Start a training process and stream its stdout plus live status snapshots.

    Yields dictionaries tagged with "type" so the UI can distinguish between
    plain log lines and status updates.
    """
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = _resolve_run_dir(config, run_id)
    cmd = [sys.executable, "-m", "ech0_training.trainer", "--config", str(config), "--run-id", run_id]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout

    next_status_poll = time.time() + status_interval
    for line in proc.stdout:
        yield {"type": "log", "data": line.rstrip()}
        if time.time() >= next_status_poll:
            status_path = run_dir / "status.json"
            if status_path.exists():
                yield {"type": "status", "data": json.loads(status_path.read_text(encoding="utf-8"))}
            next_status_poll = time.time() + status_interval
    proc.wait()
    # Emit final status if available
    status_path = run_dir / "status.json"
    if status_path.exists():
        yield {"type": "status", "data": json.loads(status_path.read_text(encoding="utf-8"))}
    yield {"type": "summary", "data": run_dir.as_posix()}


def read_status(run_dir: str) -> dict:
    status_path = Path(run_dir) / "status.json"
    if status_path.exists():
        return json.loads(status_path.read_text(encoding="utf-8"))
    return {"status": "unknown"}


def run_and_report(config: Path, baseline: Optional[Path] = None, promote_on_pass: bool = False) -> dict:
    """Blocking helper that runs pipeline and returns summary for UI."""
    result = scheduler.run_pipeline(config, baseline, promote_on_pass)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hub-friendly training wrapper")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--baseline", help="Baseline JSONL for evaluation")
    parser.add_argument("--promote-on-pass", action="store_true", help="Promote run when all baselines pass")
    parser.add_argument("--run-id", help="Optional run id to make status path predictable")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "pipeline"],
        default="pipeline",
        help="Choose to train only, eval only (requires --model-dir), or run pipeline.",
    )
    parser.add_argument("--model-dir", help="Model directory for eval-only mode.")
    args = parser.parse_args()

    config_path = Path(args.config)

    if args.mode == "train":
        print(f"Starting training job (run_id={args.run_id or 'auto'})...")
        for event in launch_training(config_path, run_id=args.run_id):
            if event["type"] == "log":
                print(event["data"])
            elif event["type"] == "status":
                print(f"[status] {json.dumps(event['data'])}")
            elif event["type"] == "summary":
                print(f"Run artifacts at {event['data']}")
        return

    if args.mode == "eval":
        if not args.model_dir:
            raise SystemExit("--model-dir is required for eval mode")
        result = scheduler.run_eval(Path(args.model_dir), Path(args.baseline) if args.baseline else None)
        print(json.dumps(result, indent=2))
        return

    # pipeline mode (default)
    print("Running pipeline (train → eval)...")
    result = run_and_report(config_path, Path(args.baseline) if args.baseline else None, args.promote_on_pass)
    print(json.dumps(result, indent=2))
    latest_run_dir = result["train"]["run_dir"]
    print(f"Latest status: {read_status(str(latest_run_dir))}")


if __name__ == "__main__":
    main()

