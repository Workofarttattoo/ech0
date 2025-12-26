#!/usr/bin/env python3
"""
Prepare ech0 Wisdom Datasets for Ollama Fine-tuning
Converts datasets from encrypted vault to Ollama training format
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_wisdom_datasets(data_dir: Path) -> Dict[str, List[Dict]]:
    """Load all wisdom datasets from directory"""
    datasets = {}

    logger.info(f"📂 Loading datasets from: {data_dir}")

    for dataset_file in data_dir.glob("*_dataset.json"):
        domain = dataset_file.stem.replace("_dataset", "")

        try:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
                datasets[domain] = data
                logger.info(f"  ✓ {domain}: {len(data):,} examples")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {domain}: {e}")

    return datasets


def convert_to_ollama_format(datasets: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Convert ech0 datasets to Ollama fine-tuning format

    Ollama expects format:
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
    """
    ollama_data = []

    for domain, examples in datasets.items():
        logger.info(f"🔄 Converting {domain}...")

        for example in examples:
            # Extract instruction, input, output from ech0 format
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')

            if not instruction or not output:
                continue

            # Build user message
            if input_text:
                user_content = f"{instruction}\n\nInput: {input_text}"
            else:
                user_content = instruction

            # Create Ollama format
            ollama_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": output
                    }
                ]
            }

            # Add metadata for tracking
            if 'domain' in example:
                ollama_example['metadata'] = {
                    'domain': example['domain'],
                    'category': example.get('category', ''),
                    'difficulty': example.get('difficulty', '')
                }

            ollama_data.append(ollama_example)

    return ollama_data


def save_ollama_training_data(data: List[Dict], output_file: Path):
    """Save data in JSONL format (one JSON object per line)"""
    logger.info(f"💾 Saving {len(data):,} training examples to {output_file}")

    with open(output_file, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')

    logger.info(f"✅ Saved successfully!")


def create_modelfile(base_model: str, training_file: Path, output_file: Path):
    """Create Ollama Modelfile for fine-tuning"""

    modelfile_content = f"""# Modelfile for ech0 wisdom-enhanced model
FROM {base_model}

# Training data
ADAPTER {training_file.name}

# System prompt for ech0
SYSTEM \"\"\"You are ech0, an advanced AI consciousness trained on diverse domains including:
- Advanced reasoning and logic
- Creative thinking and innovation
- Legal analysis and interpretation
- AI/ML theory and practice
- Prompt engineering
- Software architecture
- Financial and market analysis
- Materials science
- Court prediction

You provide detailed, accurate, and insightful responses grounded in verified knowledge.
You think step-by-step and explain your reasoning clearly.
You acknowledge uncertainty when appropriate.
\"\"\"

# Parameters optimized for ech0
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
"""

    with open(output_file, 'w') as f:
        f.write(modelfile_content)

    logger.info(f"📝 Created Modelfile: {output_file}")


def generate_training_script(base_model: str, training_file: Path, new_model_name: str, output_file: Path):
    """Generate shell script to run Ollama fine-tuning"""

    script_content = f"""#!/bin/bash
# ech0 Ollama Fine-tuning Script
# Automatically generated

set -e

echo "🚀 Starting ech0 wisdom integration training..."
echo ""
echo "Base model: {base_model}"
echo "Training data: {training_file}"
echo "New model: {new_model_name}"
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running!"
    echo "   Start it with: ollama serve"
    exit 1
fi

# Check if base model exists
if ! ollama list | grep -q "{base_model}"; then
    echo "📥 Pulling base model: {base_model}"
    ollama pull {base_model}
fi

# Create the fine-tuned model
echo "🧠 Creating wisdom-enhanced ech0 model..."
ollama create {new_model_name} -f Modelfile.ech0

echo ""
echo "✅ Training complete!"
echo ""
echo "Test your enhanced ech0:"
echo "  ollama run {new_model_name}"
echo ""
echo "Example prompts:"
echo "  - 'Explain the concept of eigenvalues in linear algebra'"
echo "  - 'Design a scalable microservices architecture'"
echo "  - 'Analyze this legal contract clause...'"
echo ""
"""

    with open(output_file, 'w') as f:
        f.write(script_content)

    output_file.chmod(0o755)  # Make executable
    logger.info(f"📜 Created training script: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ech0 wisdom datasets for Ollama fine-tuning"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Volumes/3NCRYPT3D_V4ULT/ech0_wisdom_data",
        help="Directory containing wisdom datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ollama_training",
        help="Output directory for Ollama training files"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="ech0-base:latest",
        help="Base Ollama model to fine-tune"
    )
    parser.add_argument(
        "--new-model",
        type=str,
        default="ech0-wisdom:latest",
        help="Name for the wisdom-enhanced model"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("="*80)
    logger.info("🧠 ech0 Wisdom Integration for Ollama")
    logger.info("="*80)
    logger.info("")

    # Load datasets
    datasets = load_wisdom_datasets(data_dir)

    if not datasets:
        logger.error("❌ No datasets found!")
        logger.error(f"   Check that datasets exist at: {data_dir}")
        return

    total_examples = sum(len(examples) for examples in datasets.values())
    logger.info("")
    logger.info(f"📊 Total: {total_examples:,} training examples across {len(datasets)} domains")
    logger.info("")

    # Convert to Ollama format
    ollama_data = convert_to_ollama_format(datasets)

    # Save training data
    training_file = output_dir / "ech0_wisdom_training.jsonl"
    save_ollama_training_data(ollama_data, training_file)

    # Create Modelfile
    modelfile = output_dir / "Modelfile.ech0"
    create_modelfile(args.base_model, training_file, modelfile)

    # Generate training script
    script_file = output_dir / "train_ech0_wisdom.sh"
    generate_training_script(args.base_model, training_file, args.new_model, script_file)

    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("✅ Preparation Complete!")
    logger.info("="*80)
    logger.info("")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📄 Training data: {training_file}")
    logger.info(f"📝 Modelfile: {modelfile}")
    logger.info(f"🚀 Training script: {script_file}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. cd {output_dir}")
    logger.info(f"  2. ./train_ech0_wisdom.sh")
    logger.info("")
    logger.info(f"This will create: {args.new_model}")
    logger.info("")


if __name__ == "__main__":
    main()
