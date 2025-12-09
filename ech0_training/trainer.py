#!/usr/bin/env python3
"""
LoRA/standard fine-tuning entry point for ech0.

Examples:
    python -m ech0_training.trainer --config ech0_training/configs/conversation_lora.yml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainerState,
        TrainingArguments,
    )
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing training dependencies. Install requirements_finetuning.txt first."
    ) from exc

from ech0_training import data_utils


@dataclass
class Config:
    run_name: str
    base_model: str
    dataset_path: str
    prompt_column: str
    completion_column: str
    tags_column: str = "tags"
    quality_column: str = "quality_score"
    max_samples: Optional[int] = None
    eval_split: float = 0.1
    num_epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: Optional[int] = None
    weight_decay: float = 0.01
    warmup_steps: int = 0
    logging_steps: int = 10
    save_steps: int = 100
    fp16: bool = False
    bf16: bool = False
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    output_root: str = "ech0_training/models"
    logging_backend: str = "local"
    logging_project: str = "ech0-training"
    logging_group: str = "dev"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset_block = raw.get("dataset", {})
        training_block = raw.get("training", {})
        lora_block = raw.get("lora", {})
        output_block = raw.get("output", {})
        logging_block = raw.get("logging", {})
        return cls(
            run_name=raw.get("run_name", path.stem),
            base_model=raw["base_model"],
            dataset_path=dataset_block["path"],
            prompt_column=dataset_block.get("prompt_column", "prompt"),
            completion_column=dataset_block.get("completion_column", "completion"),
            tags_column=dataset_block.get("tags_column", "tags"),
            quality_column=dataset_block.get("quality_column", "quality_score"),
            max_samples=dataset_block.get("max_samples"),
            eval_split=dataset_block.get("eval_split", 0.1),
            num_epochs=training_block.get("num_epochs", 1),
            learning_rate=training_block.get("learning_rate", 2e-4),
            batch_size=training_block.get("batch_size", 2),
            gradient_accumulation_steps=training_block.get("gradient_accumulation_steps", 4),
            max_steps=training_block.get("max_steps"),
            weight_decay=training_block.get("weight_decay", 0.01),
            warmup_steps=training_block.get("warmup_steps", 0),
            logging_steps=training_block.get("logging_steps", 10),
            save_steps=training_block.get("save_steps", 100),
            fp16=training_block.get("fp16", False),
            bf16=training_block.get("bf16", False),
            lora_enabled=lora_block.get("enabled", True),
            lora_r=lora_block.get("r", 16),
            lora_alpha=lora_block.get("alpha", 32),
            lora_dropout=lora_block.get("dropout", 0.05),
            lora_target_modules=lora_block.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            output_root=output_block.get("root_dir", "ech0_training/models"),
            logging_backend=logging_block.get("backend", "local"),
            logging_project=logging_block.get("project", "ech0-training"),
            logging_group=logging_block.get("run_group", "dev"),
        )


class StatusWriterCallback(TrainerCallback):
    """Emit lightweight status for the hub UI to consume."""

    def __init__(self, status_path: Path):
        self.status_path = status_path

    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):  # type: ignore[override]
        payload = {
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": logs.get("loss") if logs else None,
            "learning_rate": logs.get("learning_rate") if logs else None,
            "timestamp": time.time(),
            "status": "running",
        }
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def on_train_end(self, args, state: TrainerState, control, **kwargs):  # type: ignore[override]
        payload = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": time.time(),
            "status": "completed",
        }
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class Ech0Trainer:
    def __init__(self, config_path: Path, run_id: Optional[str] = None):
        self.config_path = config_path
        self.config = Config.from_yaml(config_path)
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config.output_root) / self.run_id
        self.status_path = self.run_dir / "status.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        data_utils.ensure_data_dirs()

    def _load_dataset(self):
        ds = load_dataset(
            "json", data_files={"train": self.config.dataset_path}, split="train"
        )
        if self.config.max_samples:
            ds = ds.select(range(min(self.config.max_samples, len(ds))))
        eval_size = int(len(ds) * self.config.eval_split)
        train_ds = ds.select(range(len(ds) - eval_size))
        eval_ds = ds.select(range(len(ds) - eval_size, len(ds)))
        return train_ds, eval_ds

    def _format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example[self.config.prompt_column]
        completion = example[self.config.completion_column]
        text = f"<s>[INSTRUCTION]\n{prompt}\n[RESPONSE]\n{completion}</s>"
        return {"text": text}

    def _tokenize(self, tokenizer):
        def _inner(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=1024,
                padding="max_length",
            )

        return _inner

    def train(self) -> Dict[str, Any]:
        train_ds, eval_ds = self._load_dataset()
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = train_ds.map(self._format_example)
        eval_ds = eval_ds.map(self._format_example)

        train_ds = train_ds.map(self._tokenize(tokenizer), batched=True)
        eval_ds = eval_ds.map(self._tokenize(tokenizer), batched=True)
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        model = AutoModelForCausalLM.from_pretrained(self.config.base_model)

        if self.config.lora_enabled:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        args = TrainingArguments(
            output_dir=str(self.run_dir),
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            weight_decay=self.config.weight_decay,
            max_steps=self.config.max_steps or -1,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            evaluation_strategy="steps",
            eval_steps=self.config.logging_steps,
            report_to=[] if self.config.logging_backend == "local" else [self.config.logging_backend],
            run_name=self.config.run_name,
            logging_dir=str(self.run_dir / "logs"),
        )

        callbacks = [StatusWriterCallback(self.status_path)]

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        train_result = trainer.train()
        trainer.save_model(self.run_dir)
        tokenizer.save_pretrained(self.run_dir)
        metrics = train_result.metrics
        (self.run_dir / "train_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        # Copy config into run dir for reproducibility
        (self.run_dir / "config_used.yml").write_text(
            self.config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

        summary = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "metrics": metrics,
            "config": self.config_path.name,
            "lora_enabled": self.config.lora_enabled,
            "base_model": self.config.base_model,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="ech0 trainer")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--run-id", help="Optional run id override.")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Ech0Trainer(Path(args.config), run_id=args.run_id)
    summary = trainer.train()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

