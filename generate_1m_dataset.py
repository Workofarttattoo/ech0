#!/usr/bin/env python3
"""
Generate 1 million training samples for ech0
Grounded in real, verified knowledge sources
Automatically streams to external drive when connected
"""

import sys
import argparse
from pathlib import Path
import logging

from ech0_dataset_generator import Ech0DatasetOrchestrator
from ech0_grounded_dataset_generator import generate_grounded_dataset

try:
    from ech0_external_drive_manager import get_wisdom_storage_path, ExternalDriveManager
    EXTERNAL_DRIVE_AVAILABLE = True
except ImportError:
    logger.warning("External drive manager not available")
    EXTERNAL_DRIVE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for 1M grounded dataset generation"""
    parser = argparse.ArgumentParser(
        description="Generate 1M+ training dataset for ech0 (grounded in verified knowledge)\n"
                    "Automatically detects and uses external drives when connected"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ech0_finetune_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for datasets (overrides auto-detection)"
    )
    parser.add_argument(
        "--no-external-drive",
        action="store_true",
        help="Disable external drive detection, use local storage only"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        # User specified output directory
        output_dir = Path(args.output_dir)
        logger.info(f"📁 Using user-specified output directory: {output_dir}")
    elif args.no_external_drive or not EXTERNAL_DRIVE_AVAILABLE:
        # External drive disabled or not available
        output_dir = Path("./ech0_training_data")
        logger.info(f"📁 Using local storage: {output_dir}")
    else:
        # Auto-detect external drive
        logger.info("🔍 Checking for external drive...")
        drive_manager = ExternalDriveManager(preferred_label="ech0")
        drive_manager.monitor_and_report()
        output_dir = get_wisdom_storage_path(preferred_label="ech0")

    # Create orchestrator
    orchestrator = Ech0DatasetOrchestrator(args.config)
    orchestrator.output_dir = output_dir
    orchestrator.output_dir.mkdir(exist_ok=True, parents=True)

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
