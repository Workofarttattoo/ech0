#!/usr/bin/env python3
"""
Generate 1 million training samples for ech0
Grounded in real, verified knowledge sources
"""

import sys
import argparse
from pathlib import Path
import logging

from ech0_dataset_generator import Ech0DatasetOrchestrator
from ech0_grounded_dataset_generator import generate_grounded_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for 1M grounded dataset generation"""
    parser = argparse.ArgumentParser(description="Generate 1M+ training dataset for ech0 (grounded in verified knowledge)")
    parser.add_argument(
        "--config",
        type=str,
        default="ech0_finetune_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ech0_training_data",
        help="Output directory for datasets"
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = Ech0DatasetOrchestrator(args.config)
    orchestrator.output_dir = Path(args.output_dir)
    orchestrator.output_dir.mkdir(exist_ok=True)

    # Generate grounded dataset (real, verified knowledge)
    datasets = generate_grounded_dataset(orchestrator)

    logger.info("\n")
    logger.info("🎉 1 Million grounded sample dataset generation complete!")
    logger.info("📊 Summary:")
    for domain, examples in sorted(datasets.items()):
        logger.info(f"  {domain}: {len(examples):,} samples (grounded in verified knowledge)")
    logger.info("\nNext step: Run 'python ech0_finetune_engine.py' to start training")


if __name__ == "__main__":
    main()
