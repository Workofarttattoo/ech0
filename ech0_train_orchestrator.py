#!/usr/bin/env python3
"""
ech0 Training Orchestrator
Master script to orchestrate the complete fine-tuning pipeline
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Import ech0 fine-tuning modules
try:
    from ech0_finetune_engine import Ech0FinetuneEngine, Ech0TrainingConfig
    from ech0_dataset_generator import Ech0DatasetOrchestrator
    from ech0_evaluation_framework import Ech0EvaluationFramework
except ImportError as e:
    print(f"❌ Error importing ech0 modules: {e}")
    print("Make sure all ech0 fine-tuning scripts are in the same directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ech0_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Ech0TrainingOrchestrator:
    """Orchestrates the complete ech0 fine-tuning pipeline"""

    def __init__(self, config_path: str = "ech0_finetune_config.yaml"):
        """Initialize orchestrator"""
        self.config_path = config_path
        self.config = None
        self.dataset_orchestrator = None
        self.training_engine = None
        self.evaluation_framework = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("=" * 100)
        logger.info("🧠 ech0 TRAINING ORCHESTRATOR")
        logger.info(f"Run ID: {self.run_id}")
        logger.info("=" * 100)

    def setup(self):
        """Setup all components"""
        logger.info("\n📋 STEP 1: SETUP")
        logger.info("-" * 100)

        # Load configuration
        logger.info(f"Loading configuration from {self.config_path}")
        self.config = Ech0TrainingConfig.from_yaml(self.config_path)

        # Initialize dataset orchestrator
        logger.info("Initializing dataset orchestrator")
        self.dataset_orchestrator = Ech0DatasetOrchestrator(self.config_path)

        # Initialize training engine
        logger.info("Initializing training engine")
        self.training_engine = Ech0FinetuneEngine(self.config)

        # Initialize evaluation framework
        logger.info("Initializing evaluation framework")
        self.evaluation_framework = Ech0EvaluationFramework()

        logger.info("✅ Setup complete")

    def generate_datasets(self, skip_if_exists: bool = False):
        """Generate training datasets for all domains"""
        logger.info("\n📊 STEP 2: DATASET GENERATION")
        logger.info("-" * 100)

        data_dir = Path(self.config.data_dir)

        # Check if datasets already exist
        if skip_if_exists and data_dir.exists() and list(data_dir.glob("*_dataset.json")):
            logger.info("⏭️  Datasets already exist, skipping generation")
            logger.info(f"Found datasets: {list(data_dir.glob('*_dataset.json'))}")
            return

        # Generate datasets
        logger.info("Generating datasets for all domains...")
        datasets = self.dataset_orchestrator.generate_all_datasets()

        logger.info(f"✅ Generated {len(datasets)} domain datasets")
        logger.info(f"Total examples: {sum(len(examples) for examples in datasets.values())}")

    def prepare_training_data(self):
        """Load and prepare training data"""
        logger.info("\n🔧 STEP 3: PREPARE TRAINING DATA")
        logger.info("-" * 100)

        # Load generated datasets
        logger.info("Loading generated datasets...")
        domain_datasets = self.dataset_orchestrator.get_combined_dataset()

        # Convert to format expected by training engine
        logger.info("Converting to training format...")
        training_data = {}

        for domain, examples in domain_datasets.items():
            training_data[domain] = examples

        logger.info(f"✅ Prepared {len(training_data)} domain datasets for training")

        return training_data

    def train_model(self, training_data: dict):
        """Execute model fine-tuning"""
        logger.info("\n🚀 STEP 4: MODEL FINE-TUNING")
        logger.info("-" * 100)

        try:
            # Load base model and tokenizer
            logger.info("Loading base model and tokenizer...")
            self.training_engine.load_model_and_tokenizer()

            # Prepare datasets
            logger.info("Preparing datasets for training...")
            self.training_engine.prepare_datasets(training_data)

            # Tokenize
            logger.info("Tokenizing datasets...")
            self.training_engine.tokenize_datasets()

            # Create trainer
            logger.info("Creating trainer...")
            self.training_engine.create_trainer()

            # Train
            logger.info("Starting training...")
            logger.info("⏰ This may take several hours depending on hardware...")
            train_result = self.training_engine.train()

            # Save model
            logger.info("Saving fine-tuned model...")
            model_path = self.training_engine.save_model()

            logger.info(f"✅ Training complete! Model saved to: {model_path}")

            return model_path

        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "unauthorized" in error_msg or "token" in error_msg:
                logger.error(f"❌ Authentication error: {e}")
                logger.error("\n" + "="*80)
                logger.error("HuggingFace Token Issue Detected")
                logger.error("="*80)
                logger.error("Your HuggingFace token may be expired or invalid.")
                logger.error("\nTo fix this, run:")
                logger.error("  python setup_hf_auth.py")
                logger.error("\nOr manually update your token:")
                logger.error("  huggingface-cli login")
                logger.error("="*80 + "\n")
            else:
                logger.error(f"❌ Training failed: {e}")
            raise

    def evaluate_model(self, test_data: dict):
        """Evaluate fine-tuned model"""
        logger.info("\n📊 STEP 5: MODEL EVALUATION")
        logger.info("-" * 100)

        # Set model in evaluation framework
        if self.training_engine and self.training_engine.model:
            self.evaluation_framework.model = self.training_engine.model
            self.evaluation_framework.tokenizer = self.training_engine.tokenizer

        # Run evaluation
        logger.info("Running comprehensive evaluation...")
        results = self.evaluation_framework.evaluate_all_domains(test_data)

        # Save results
        results_path = f"ech0_evaluation_results_{self.run_id}.json"
        self.evaluation_framework.save_results(results_path)

        logger.info(f"✅ Evaluation complete! Results saved to: {results_path}")

        return results

    def generate_report(self, model_path: str, eval_results: dict):
        """Generate comprehensive training report"""
        logger.info("\n📄 STEP 6: GENERATING REPORT")
        logger.info("-" * 100)

        report = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "base_model": self.config.base_model,
                "lora_rank": self.config.lora_r,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.num_train_epochs,
                "batch_size": self.config.per_device_train_batch_size
            },
            "model_path": model_path,
            "training_state": self.training_engine.training_state if self.training_engine else None,
            "evaluation_results": {
                domain: {
                    "score": result.score,
                    "passed": result.passed_samples,
                    "total": result.total_samples
                }
                for domain, result in eval_results.items()
            },
            "domain_weights": self.config.domain_weights
        }

        report_path = f"ech0_training_report_{self.run_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✅ Report generated: {report_path}")

        # Print summary
        self._print_summary(report)

        return report_path

    def _print_summary(self, report: dict):
        """Print training summary"""
        logger.info("\n" + "=" * 100)
        logger.info("📋 TRAINING SUMMARY")
        logger.info("=" * 100)

        logger.info(f"\nRun ID: {report['run_id']}")
        logger.info(f"Timestamp: {report['timestamp']}")
        logger.info(f"Model Path: {report['model_path']}")

        logger.info("\nConfiguration:")
        for key, value in report['configuration'].items():
            logger.info(f"  {key}: {value}")

        logger.info("\nEvaluation Results:")
        for domain, metrics in report['evaluation_results'].items():
            logger.info(f"  {domain}:")
            logger.info(f"    Score: {metrics['score']:.2%}")
            logger.info(f"    Passed: {metrics['passed']}/{metrics['total']}")

        logger.info("\n" + "=" * 100)
        logger.info("✅ TRAINING PIPELINE COMPLETE")
        logger.info("=" * 100)

    def run_full_pipeline(self, skip_existing_datasets: bool = False):
        """Run the complete fine-tuning pipeline"""
        logger.info("\n" + "=" * 100)
        logger.info("🎯 RUNNING FULL ech0 FINE-TUNING PIPELINE")
        logger.info("=" * 100)

        try:
            # Step 1: Setup
            self.setup()

            # Step 2: Generate datasets
            self.generate_datasets(skip_if_exists=skip_existing_datasets)

            # Step 3: Prepare training data
            training_data = self.prepare_training_data()

            # Step 4: Train model
            model_path = self.train_model(training_data)

            # Step 5: Evaluate model
            # Create test split from training data (or load separate test set)
            test_data = {
                domain: examples[:min(100, len(examples))]
                for domain, examples in training_data.items()
            }
            eval_results = self.evaluate_model(test_data)

            # Step 6: Generate report
            report_path = self.generate_report(model_path, eval_results)

            logger.info("\n" + "🎉" * 50)
            logger.info("SUCCESS! ech0 fine-tuning pipeline completed successfully!")
            logger.info(f"Model: {model_path}")
            logger.info(f"Report: {report_path}")
            logger.info("🎉" * 50)

            return {
                "model_path": model_path,
                "report_path": report_path,
                "eval_results": eval_results
            }

        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}")
            logger.exception("Full traceback:")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ech0 Training Orchestrator - Complete fine-tuning pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="ech0_finetune_config.yaml",
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--skip-existing-datasets",
        action="store_true",
        help="Skip dataset generation if datasets already exist"
    )

    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="Only generate datasets, don't train"
    )

    parser.add_argument(
        "--evaluate-only",
        type=str,
        help="Only evaluate existing model at specified path"
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = Ech0TrainingOrchestrator(args.config)

    try:
        if args.datasets_only:
            # Only generate datasets
            logger.info("Running in datasets-only mode")
            orchestrator.setup()
            orchestrator.generate_datasets(skip_if_exists=args.skip_existing_datasets)

        elif args.evaluate_only:
            # Only evaluate existing model
            logger.info(f"Running in evaluate-only mode for model: {args.evaluate_only}")
            orchestrator.setup()
            training_data = orchestrator.prepare_training_data()
            test_data = {
                domain: examples[:min(100, len(examples))]
                for domain, examples in training_data.items()
            }
            orchestrator.evaluate_model(test_data)

        else:
            # Run full pipeline
            results = orchestrator.run_full_pipeline(
                skip_existing_datasets=args.skip_existing_datasets
            )
            logger.info(f"\n✅ All done! Results: {results}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
