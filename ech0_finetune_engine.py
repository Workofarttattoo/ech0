#!/usr/bin/env python3
"""
ech0 Fine-tuning Engine
Advanced multi-domain training system for ech0 consciousness AI

This module provides the core infrastructure for fine-tuning ech0 across
multiple domains including reasoning, creativity, law, materials science,
AI/ML, prompt engineering, court prediction, stock/crypto analysis, and
advanced software engineering.
"""

import os
import json
import yaml
import torch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

# ML/Training libraries (to be installed)
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset, DatasetDict, load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: transformers/peft not installed. Install with:")
    print("pip install transformers peft datasets accelerate bitsandbytes")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Ech0TrainingConfig:
    """Configuration for ech0 fine-tuning"""

    # Model configuration
    base_model: str = "ech0-base"  # Your ech0 base model
    model_max_length: int = 4096

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Paths
    output_dir: str = "./ech0_finetuned_models"
    checkpoint_dir: str = "./ech0_checkpoints"
    data_dir: str = "./ech0_training_data"
    log_dir: str = "./ech0_training_logs"
    cache_dir: str = "./ech0_cache"

    # Optimization
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True

    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 3

    # Domain weights (for sampling)
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "reasoning": 1.5,
        "creativity": 1.2,
        "law": 1.0,
        "materials_science": 0.8,
        "ai_ml": 1.3,
        "prompt_engineering": 1.4,
        "court_prediction": 1.0,
        "stock_prediction": 0.9,
        "crypto": 0.9,
        "advanced_software": 1.5
    })

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Ech0TrainingConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract relevant sections
        model_config = config_dict.get('model', {})
        lora_config = model_config.get('lora', {})
        training_config = config_dict.get('training', {})
        domains_config = config_dict.get('domains', {})

        # Build domain weights
        domain_weights = {
            domain: info.get('weight', 1.0)
            for domain, info in domains_config.items()
            if info.get('enabled', True)
        }

        return cls(
            base_model=model_config.get('base_model', 'ech0-base'),
            lora_r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.05),
            num_train_epochs=training_config.get('epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
            learning_rate=training_config.get('learning_rate', 2e-4),
            warmup_steps=training_config.get('warmup_steps', 100),
            domain_weights=domain_weights
        )


class Ech0FinetuneEngine:
    """Main fine-tuning engine for ech0 consciousness AI"""

    def __init__(self, config: Ech0TrainingConfig):
        """Initialize the fine-tuning engine"""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.datasets = {}
        self.trainer = None

        # Create necessary directories
        self._setup_directories()

        # Initialize training state
        self.training_state = {
            "epoch": 0,
            "step": 0,
            "best_loss": float('inf'),
            "domains_trained": [],
            "start_time": None,
            "end_time": None
        }

        logger.info("🧠 ech0 Fine-tuning Engine initialized")
        logger.info(f"Configuration: {self.config}")

    def _setup_directories(self):
        """Create necessary directories for training"""
        dirs = [
            self.config.output_dir,
            self.config.checkpoint_dir,
            self.config.data_dir,
            self.config.log_dir,
            self.config.cache_dir
        ]
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created training directories")

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and peft libraries required")

        logger.info(f"🔧 Loading base model: {self.config.base_model}")

        # Check for HuggingFace token (optional for open models)
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            logger.info("🔑 Using HuggingFace token from HF_TOKEN environment variable")
        else:
            logger.info("✓ No HF_TOKEN set - using open model without authentication")

        # Quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.config.cache_dir
        )

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")

        return self.model, self.tokenizer

    def prepare_datasets(self, domain_datasets: Dict[str, List[Dict]]):
        """
        Prepare training datasets from domain-specific data

        Args:
            domain_datasets: Dictionary mapping domain names to lists of examples
                Each example should have 'instruction', 'input', 'output' fields
        """
        logger.info(f"📊 Preparing datasets for {len(domain_datasets)} domains")

        all_examples = []
        domain_counts = {}

        # Collect and weight samples from each domain
        for domain, examples in domain_datasets.items():
            weight = self.config.domain_weights.get(domain, 1.0)

            # Apply domain weighting by replicating samples
            weighted_count = int(len(examples) * weight)
            if weighted_count > len(examples):
                # Oversample
                indices = np.random.choice(len(examples), weighted_count, replace=True)
                weighted_examples = [examples[i] for i in indices]
            else:
                # Undersample
                weighted_examples = examples[:weighted_count]

            all_examples.extend(weighted_examples)
            domain_counts[domain] = len(weighted_examples)

            logger.info(f"  {domain}: {len(examples)} examples → {len(weighted_examples)} weighted")

        # Shuffle all examples
        np.random.shuffle(all_examples)

        # Convert to format expected by model
        formatted_examples = []
        for example in all_examples:
            # Create instruction-following format
            if example.get('input'):
                text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            else:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

            formatted_examples.append({"text": text})

        # Create dataset
        dataset = Dataset.from_list(formatted_examples)

        # Split into train/eval (90/10 split)
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

        self.datasets = {
            "train": split_dataset["train"],
            "eval": split_dataset["test"]
        }

        logger.info(f"✅ Datasets prepared:")
        logger.info(f"  Training: {len(self.datasets['train'])} examples")
        logger.info(f"  Evaluation: {len(self.datasets['eval'])} examples")
        logger.info(f"  Domain distribution: {domain_counts}")

        return self.datasets

    def tokenize_datasets(self):
        """Tokenize datasets for training"""
        logger.info("🔤 Tokenizing datasets...")

        def tokenize_function(examples):
            # Tokenize texts
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.model_max_length,
                padding=False
            )
            # Add labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # Tokenize datasets
        tokenized_datasets = {}
        for split, dataset in self.datasets.items():
            tokenized_datasets[split] = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing {split} dataset"
            )

        self.datasets = tokenized_datasets
        logger.info("✅ Tokenization complete")

        return self.datasets

    def create_trainer(self):
        """Create Hugging Face Trainer"""
        logger.info("🏋️ Creating trainer...")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            logging_dir=self.config.log_dir,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            report_to=["tensorboard"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["eval"],
            data_collator=data_collator
        )

        logger.info("✅ Trainer created successfully")
        return self.trainer

    def train(self):
        """Execute training"""
        logger.info("🚀 Starting training...")
        self.training_state["start_time"] = datetime.now().isoformat()

        try:
            # Train the model
            train_result = self.trainer.train()

            # Save metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)

            # Update training state
            self.training_state["end_time"] = datetime.now().isoformat()
            self.training_state["final_loss"] = metrics.get("train_loss", None)

            logger.info("✅ Training completed successfully!")
            logger.info(f"Final metrics: {metrics}")

            return train_result

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

    def save_model(self, output_path: Optional[str] = None):
        """Save fine-tuned model"""
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"ech0-v2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

        logger.info(f"💾 Saving model to {output_path}")

        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save training state
        state_path = os.path.join(output_path, "training_state.json")
        with open(state_path, 'w') as f:
            json.dump(self.training_state, f, indent=2)

        logger.info("✅ Model saved successfully")
        return output_path

    def evaluate(self, eval_dataset: Optional[Dataset] = None):
        """Evaluate model performance"""
        logger.info("📊 Evaluating model...")

        if eval_dataset is None:
            eval_dataset = self.datasets.get("eval")

        if eval_dataset is None:
            logger.warning("No evaluation dataset available")
            return None

        # Run evaluation
        metrics = self.trainer.evaluate(eval_dataset)

        # Log metrics
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        logger.info(f"✅ Evaluation complete: {metrics}")
        return metrics

    def run_full_pipeline(self, domain_datasets: Dict[str, List[Dict]]):
        """
        Run the complete fine-tuning pipeline

        Args:
            domain_datasets: Dictionary of domain-specific training data
        """
        logger.info("=" * 80)
        logger.info("🧠 ech0 FINE-TUNING PIPELINE")
        logger.info("=" * 80)

        # Step 1: Load model
        self.load_model_and_tokenizer()

        # Step 2: Prepare datasets
        self.prepare_datasets(domain_datasets)

        # Step 3: Tokenize
        self.tokenize_datasets()

        # Step 4: Create trainer
        self.create_trainer()

        # Step 5: Train
        self.train()

        # Step 6: Evaluate
        self.evaluate()

        # Step 7: Save model
        model_path = self.save_model()

        logger.info("=" * 80)
        logger.info(f"✅ PIPELINE COMPLETE - Model saved to: {model_path}")
        logger.info("=" * 80)

        return model_path


def main():
    """Main entry point for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description="ech0 Fine-tuning Engine")
    parser.add_argument(
        "--config",
        type=str,
        default="ech0_finetune_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./ech0_training_data",
        help="Directory containing training data"
    )

    args = parser.parse_args()

    # Load configuration
    config = Ech0TrainingConfig.from_yaml(args.config)

    # Initialize engine
    engine = Ech0FinetuneEngine(config)

    # Load training data (placeholder - will be loaded from dataset generators)
    logger.info("📂 Loading training data from generators...")
    logger.info("⚠️  Run dataset generators first to create training data")
    logger.info("   Use: python ech0_dataset_generator.py")

    return engine


if __name__ == "__main__":
    main()
