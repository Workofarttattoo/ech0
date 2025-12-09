#!/usr/bin/env python3
"""Helper CLI around the lightweight model registry."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ech0_training.models import registry


def cmd_register(args):
    meta = registry.RunMetadata(
        run_id=args.run_id,
        run_dir=args.run_dir,
        base_model=args.base_model,
        config=args.config,
        datasets=args.datasets,
        eval_report=args.eval_report,
        score=args.score,
    )
    registry.register_run(meta)
    print(f"Registered {args.run_id}")


def cmd_promote(args):
    registry.promote(args.run_id, stage=args.stage)
    print(f"Promoted {args.run_id} → {args.stage}")


def cmd_rollback(_args):
    previous = registry.rollback()
    print(f"Rolled back to {previous}")


def cmd_list(_args):
    payload = registry.dump()
    print(json.dumps(payload, indent=2))


def cmd_package(args):
    run_dir = Path(args.run_dir).expanduser()
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / run_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_dir, target)
    print(json.dumps({"copied_to": str(target)}, indent=2))


def parse():
    parser = argparse.ArgumentParser(description="Registry helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register")
    reg.add_argument("--run-id", required=True)
    reg.add_argument("--run-dir", required=True)
    reg.add_argument("--base-model", required=True)
    reg.add_argument("--config", required=True)
    reg.add_argument("--datasets", nargs="+", required=True)
    reg.add_argument("--eval-report")
    reg.add_argument("--score", type=float)
    reg.set_defaults(func=cmd_register)

    prom = sub.add_parser("promote")
    prom.add_argument("--run-id", required=True)
    prom.add_argument("--stage", default="production")
    prom.set_defaults(func=cmd_promote)

    roll = sub.add_parser("rollback")
    roll.set_defaults(func=cmd_rollback)

    list_cmd = sub.add_parser("list")
    list_cmd.set_defaults(func=cmd_list)

    pkg = sub.add_parser("package")
    pkg.add_argument("--run-dir", required=True)
    pkg.add_argument("--dest", default="~/models/ech0/")
    pkg.set_defaults(func=cmd_package)

    return parser.parse_args()


def main():
    args = parse()
    args.func(args)


if __name__ == "__main__":
    main()

