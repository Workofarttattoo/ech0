#!/usr/bin/env python3
"""
Scheduler/entrypoint to run training + evaluation recipes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ech0_training.trainer import Ech0Trainer, Config
from ech0_training.eval.runner import EvaluationRunner
from ech0_training.models import registry


DEFAULT_BASELINE = Path("ech0_training/eval/baselines/regression.jsonl")


def run_training(config_path: Path, run_id: str | None = None) -> dict:
    trainer = Ech0Trainer(config_path, run_id=run_id)
    return trainer.train()


def run_eval(model_dir: Path, baseline: Path | None = None) -> dict:
    baseline_path = baseline or DEFAULT_BASELINE
    reports_dir = model_dir / "reports"
    runner = EvaluationRunner(model_dir, baseline_path, reports_dir)
    return runner.run_regression()


def run_pipeline(config_path: Path, baseline: Path | None, promote_on_pass: bool) -> dict:
    cfg = Config.from_yaml(config_path)
    summary = run_training(config_path)
    eval_summary = run_eval(Path(summary["run_dir"]), baseline)

    metadata = registry.RunMetadata(
        run_id=summary["run_id"],
        run_dir=summary["run_dir"],
        base_model=cfg.base_model,
        config=config_path.as_posix(),
        datasets=[cfg.dataset_path],
        eval_report=str(Path(summary["run_dir"]) / "reports" / "regression_report.json"),
        score=eval_summary.get("passed", 0) / max(eval_summary.get("total", 1), 1),
    )
    registry.register_run(metadata)

    if promote_on_pass and eval_summary["passed"] == eval_summary["total"]:
        registry.promote(summary["run_id"])
        eval_summary["promoted"] = True
    else:
        eval_summary["promoted"] = False

    return {"train": summary, "eval": eval_summary}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ech0 training/eval pipelines.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Run training only")
    train_p.add_argument("--config", required=True)
    train_p.add_argument("--run-id")

    eval_p = sub.add_parser("eval", help="Run evaluation only")
    eval_p.add_argument("--model-dir", required=True)
    eval_p.add_argument("--baseline", default=None)

    pipe_p = sub.add_parser("pipeline", help="Train then evaluate and optionally promote")
    pipe_p.add_argument("--config", required=True)
    pipe_p.add_argument("--baseline", default=None)
    pipe_p.add_argument("--promote-on-pass", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        result = run_training(Path(args.config), run_id=args.run_id)
    elif args.command == "eval":
        result = run_eval(Path(args.model_dir), Path(args.baseline) if args.baseline else None)
    else:
        result = run_pipeline(Path(args.config), Path(args.baseline) if args.baseline else None, args.promote_on_pass)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

