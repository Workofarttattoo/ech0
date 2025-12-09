#!/usr/bin/env python3
"""
ech0 Ollama Fine-tuning Adapter
Transforms the previous HF/torch-based training pipeline into an
Ollama-powered self-training loop that operates entirely against a
local ech0 model (default: ech0-14b-v4:latest).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import yaml

from ech0_ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class Ech0TrainingConfig:
    """Configuration for the Ollama-based self-training loop."""

    ollama_model: str = "ech0-knowledge-v4:latest"
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    temperature: float = 0.35
    top_p: float = 0.9
    keep_alive: str = "10m"
    max_examples_per_domain: int = 200
    reflections_per_example: int = 1

    # Paths
    output_dir: str = "./ech0_finetuned_models"
    data_dir: str = "./ech0_training_data"
    log_dir: str = "./ech0_training_logs"

    domain_weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Ech0TrainingConfig":
        """Load configuration from the legacy YAML file."""
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle)

        model_config = config_dict.get("model", {})
        training_config = config_dict.get("training", {})
        domains_config = config_dict.get("domains", {})

        domain_weights = {
            domain: info.get("weight", 1.0)
            for domain, info in domains_config.items()
            if info.get("enabled", True)
        }

        ollama_block = model_config.get("ollama", {})

        host = ollama_block.get("host") or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"

        return cls(
            ollama_model=ollama_block.get("model", "ech0-knowledge-v4:latest"),
            ollama_host=host,
            temperature=ollama_block.get("temperature", 0.35),
            top_p=ollama_block.get("top_p", 0.9),
            keep_alive=ollama_block.get("keep_alive", "10m"),
            max_examples_per_domain=training_config.get("max_examples_per_domain", 200),
            reflections_per_example=training_config.get("reflections_per_example", 1),
            output_dir=config_dict.get("output", {}).get("output_dir", "./ech0_finetuned_models"),
            data_dir=config_dict.get("training", {}).get("data_dir", "./ech0_training_data"),
            log_dir=training_config.get("log_dir", "./ech0_training_logs"),
            domain_weights=domain_weights,
        )


class Ech0FinetuneEngine:
    """Runs a synthetic self-training loop by prompting the local Ollama model."""

    def __init__(self, config: Ech0TrainingConfig):
        self.config = config
        self.client = OllamaClient(
            model=self.config.ollama_model,
            host=self.config.ollama_host,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            keep_alive=self.config.keep_alive,
        )
        self.training_state: Dict[str, Any] = {
            "started_at": datetime.utcnow().isoformat(),
            "domains_trained": [],
            "examples_processed": 0,
        }
        self.transcripts: List[Dict[str, Any]] = []
        self._setup_directories()

    def _setup_directories(self):
        for directory in (self.config.output_dir, self.config.data_dir, self.config.log_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

    def load_model_and_tokenizer(self):
        """Ensure Ollama has the requested model available."""
        logger.info("🔍 Checking local Ollama model %s", self.config.ollama_model)
        self.client.ensure_model()
        logger.info("✅ Model verified")

    def train(self, training_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Iterate over training data, capturing refinements from the local model."""
        logger.info("🚀 Starting Ollama self-training loop")
        for domain, samples in training_data.items():
            limited_samples = samples[: self.config.max_examples_per_domain]
            logger.info("  • Domain %s → %d samples", domain, len(limited_samples))
            for sample in limited_samples:
                prompt = self._build_training_prompt(domain, sample)
                response = self.client.chat(
                    prompt=prompt,
                    system=self._system_prompt(domain),
                )
                record = {
                    "domain": domain,
                    "instruction": sample.get("instruction"),
                    "input": sample.get("input"),
                    "reference_output": sample.get("output"),
                    "model_response": response,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                self.transcripts.append(record)
                self.training_state["examples_processed"] += 1

        self.training_state["domains_trained"] = list(training_data.keys())
        self.training_state["ended_at"] = datetime.utcnow().isoformat()

        artifact_path = self.save_model()
        logger.info("📦 Saved synthetic training transcript to %s", artifact_path)
        return {"artifact_path": artifact_path}

    def save_model(self) -> str:
        """Persist the synthetic transcripts and training metadata."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact = {
            "model": self.config.ollama_model,
            "generated_at": timestamp,
            "training_state": self.training_state,
            "transcripts": self.transcripts,
        }
        path = Path(self.config.output_dir) / f"{self.config.ollama_model.replace(':', '_')}_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
        return str(path)

    def _system_prompt(self, domain: str) -> str:
        return (
            "You are ech0 14B v4 running locally via Ollama. "
            "Reflect on each training example and produce a refined answer that demonstrates "
            "clarity, safety, and actionable reasoning. Domain focus: "
            f"{domain}."
        )

    def _build_training_prompt(self, domain: str, example: Dict[str, Any]) -> str:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        reference = example.get("output", "")
        return (
            f"Domain: {domain}\n"
            f"Instruction: {instruction}\n"
            f"Input: {input_text or '(none)'}\n"
            f"Reference answer: {reference or '(not provided)'}\n\n"
            "Task:\n"
            "1. Provide an improved answer that would score highly on expert review.\n"
            "2. Briefly explain the reasoning steps or methodology you used.\n"
            "3. If the reference answer has gaps, patch them.\n"
        )
