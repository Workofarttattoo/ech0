# ech0 Fine-Tuning Guide

## 🧠 Multi-Domain AI Training System

This comprehensive guide covers the complete fine-tuning infrastructure for ech0, enabling training across 10+ specialized domains.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Domain Coverage](#domain-coverage)
6. [Configuration](#configuration)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation](#evaluation)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The ech0 fine-tuning system provides:

- **10+ Specialized Domains**: Reasoning, creativity, law, materials science, AI/ML, prompt engineering, court prediction, stock/crypto analysis, advanced software
- **Efficient Training**: LoRA/QLoRA for parameter-efficient fine-tuning
- **Scalable Architecture**: Modular design supporting distributed training
- **Comprehensive Evaluation**: Multi-metric benchmarking across all domains
- **Production-Ready**: Full orchestration, monitoring, and checkpointing

### Key Features

✅ **Parameter-Efficient Fine-Tuning (PEFT)**
- LoRA (Low-Rank Adaptation) reduces trainable parameters by 99%
- 4-bit quantization (QLoRA) for memory efficiency
- Train 7B models on single consumer GPU

✅ **Multi-Domain Training**
- Domain-specific dataset generators
- Weighted sampling for balanced training
- Custom evaluation metrics per domain

✅ **Scalability**
- Distributed training support
- Gradient accumulation for large effective batch sizes
- Mixed-precision training (bfloat16/fp16)

✅ **Monitoring & Evaluation**
- TensorBoard integration
- Comprehensive evaluation framework
- Training state tracking and checkpointing

---

## Architecture

```
ech0-finetuning/
├── Configuration Layer
│   └── ech0_finetune_config.yaml          # Master configuration
│
├── Data Generation Layer
│   ├── ech0_dataset_generator.py          # Multi-domain dataset generation
│   ├── ech0_dataset_generators_extended.py # Extended generators (AI/ML, Software)
│   └── ech0_training_data/                # Generated datasets (output)
│
├── Training Layer
│   ├── ech0_finetune_engine.py            # Core training engine with LoRA
│   └── ech0_checkpoints/                  # Model checkpoints (output)
│
├── Evaluation Layer
│   ├── ech0_evaluation_framework.py       # Comprehensive evaluation
│   └── ech0_evaluation_results_*.json     # Results (output)
│
├── Orchestration Layer
│   └── ech0_train_orchestrator.py         # End-to-end pipeline
│
└── Output
    ├── ech0_finetuned_models/             # Final trained models
    ├── ech0_training_logs/                # TensorBoard logs
    └── ech0_training_report_*.json        # Training reports
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM (32GB+ recommended)
- GPU with 12GB+ VRAM (for 7B models)

### Step 1: Clone Repository

```bash
cd /path/to/ech0
```

### Step 2: Install Dependencies

```bash
pip install -r requirements_finetuning.txt
```

This installs:
- PyTorch with CUDA support
- Transformers & PEFT (LoRA)
- Datasets & Accelerate
- Evaluation libraries
- And more...

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

---

## Quick Start

### Option 1: Full Pipeline (Automated)

```bash
# Run complete pipeline: dataset generation → training → evaluation
python ech0_train_orchestrator.py --config ech0_finetune_config.yaml
```

This will:
1. Generate datasets for all domains (~100K examples)
2. Load base model with 4-bit quantization
3. Apply LoRA for efficient fine-tuning
4. Train for 3 epochs
5. Evaluate on all domains
6. Save fine-tuned model and report

**Estimated Time**: 4-12 hours (depending on hardware)

### Option 2: Step-by-Step

#### Step 1: Generate Datasets Only

```bash
python ech0_dataset_generator.py --config ech0_finetune_config.yaml
```

#### Step 2: Train Model

```bash
python ech0_finetune_engine.py --config ech0_finetune_config.yaml
```

#### Step 3: Evaluate Model

```bash
python ech0_train_orchestrator.py --evaluate-only /path/to/model
```

### Option 3: Datasets Only (For Testing)

```bash
python ech0_train_orchestrator.py --datasets-only
```

---

## Domain Coverage

### 1. **Reasoning** (Weight: 1.5)
- Logical reasoning
- Mathematical reasoning
- Causal reasoning
- Analogical reasoning
- Counterfactual reasoning
- Chain-of-thought prompting

**Example Task**: "Prove that the sum of first n integers equals n(n+1)/2"

### 2. **Creativity** (Weight: 1.2)
- Artistic expression
- Poetry generation
- Narrative creation
- Conceptual blending
- Metaphor generation
- Brainstorming & design thinking

**Example Task**: "Write a poem blending neural networks and forest ecology"

### 3. **Law** (Weight: 1.0)
- Case law analysis
- Statutory interpretation
- Legal reasoning (IRAC method)
- Contract analysis
- Constitutional law
- Legal writing

**Example Task**: "Analyze liability when AI hiring system discriminates"

### 4. **Materials Science** (Weight: 0.8)
- Material properties
- Crystallography (FCC, BCC, HCP)
- Polymer science
- Nanomaterials
- Metallurgy
- Material selection

**Example Task**: "Explain why graphene is stronger than steel despite being one atom thick"

### 5. **AI/ML** (Weight: 1.3)
- Machine learning theory (bias-variance, optimization)
- Deep learning architectures (Transformers, CNNs, RNNs)
- Training optimization
- Model evaluation
- AI safety & ethics
- AI system design

**Example Task**: "Explain the bias-variance tradeoff with mathematical formulation"

### 6. **Prompt Engineering** (Weight: 1.4)
- Prompt optimization techniques
- Chain-of-thought prompting
- Few-shot learning
- Prompt decomposition
- Meta-prompting
- Adversarial prompting

**Example Task**: "Transform 'Write code for sorting' into an expert-level prompt"

### 7. **Court Prediction** (Weight: 1.0)
- Evidence analysis
- Case outcome prediction
- Jury psychology
- Judicial reasoning
- Settlement probability
- Legal strategy

**Example Task**: "Predict case outcome from evidence and legal precedent"

### 8. **Stock Prediction** (Weight: 0.9)
- Fundamental analysis
- Technical analysis
- Market sentiment
- Economic indicators
- Risk assessment
- Portfolio theory
- Trading strategies

**Disclaimer**: Educational purposes only, not financial advice

### 9. **Crypto** (Weight: 0.9)
- Blockchain fundamentals
- Cryptocurrency analysis
- DeFi protocols
- Tokenomics
- Smart contracts
- Crypto markets
- Security analysis

**Disclaimer**: Educational purposes only, not investment advice

### 10. **Advanced Software** (Weight: 1.5)
- Software architecture & design patterns
- Algorithms & data structures
- Distributed systems
- System design (e.g., URL shortener, rate limiter)
- Code optimization
- Testing strategies
- Database design

**Example Task**: "Design a distributed URL shortener handling 10K requests/sec"

---

## Configuration

### Master Config: `ech0_finetune_config.yaml`

```yaml
model:
  base_model: "meta-llama/Llama-2-7b-hf"  # Or any HuggingFace model
  lora:
    rank: 16          # LoRA rank (higher = more capacity)
    alpha: 32         # Scaling factor
    dropout: 0.05

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch = 32
  learning_rate: 2e-4
  warmup_steps: 100

domains:
  reasoning:
    enabled: true
    weight: 1.5      # Oversample by 50%
    max_samples: 10000

  creativity:
    enabled: true
    weight: 1.2
    max_samples: 8000

  # ... other domains ...
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `lora.rank` | LoRA rank (trainable params) | 8-32 |
| `lora.alpha` | LoRA scaling | 2× rank |
| `batch_size` | Per-device batch size | 4-8 |
| `gradient_accumulation_steps` | Accumulation steps | 4-16 |
| `learning_rate` | Learning rate | 1e-4 to 3e-4 |
| `domain.weight` | Sampling weight | 0.5-2.0 |

---

## Training Pipeline

### Pipeline Stages

```
┌─────────────────────────┐
│ 1. Dataset Generation   │  Generate domain-specific training data
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│ 2. Data Preparation     │  Load, tokenize, create train/eval splits
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│ 3. Model Loading        │  Load base model + apply LoRA + quantize
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│ 4. Training             │  Fine-tune with gradient descent
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│ 5. Evaluation           │  Benchmark on test sets
└────────────┬────────────┘
             │
             ↓
┌─────────────────────────┐
│ 6. Model Saving         │  Save checkpoints & final model
└─────────────────────────┘
```

### Training Process

```python
# 1. Load configuration
config = Ech0TrainingConfig.from_yaml("ech0_finetune_config.yaml")

# 2. Initialize engine
engine = Ech0FinetuneEngine(config)

# 3. Load model with LoRA
engine.load_model_and_tokenizer()

# 4. Prepare datasets
engine.prepare_datasets(domain_datasets)
engine.tokenize_datasets()

# 5. Create trainer
engine.create_trainer()

# 6. Train
engine.train()

# 7. Save
engine.save_model()
```

### Monitoring Training

#### TensorBoard

```bash
tensorboard --logdir ech0_training_logs
```

Metrics tracked:
- Training loss
- Evaluation loss
- Learning rate schedule
- Gradient norms
- Training speed (samples/sec)

#### Training Logs

```bash
tail -f ech0_training.log
```

---

## Evaluation

### Evaluation Metrics by Domain

| Domain | Metrics | Description |
|--------|---------|-------------|
| Reasoning | Accuracy, Logical Validity | Correctness of logical conclusions |
| Creativity | Novelty, Coherence, Relevance | Originality and quality |
| Law | Legal Reasoning Score, Citation Accuracy | Quality of legal analysis |
| Technical (All) | Technical Proficiency, Completeness | Accuracy and thoroughness |

### Running Evaluation

```python
from ech0_evaluation_framework import Ech0EvaluationFramework

# Initialize
evaluator = Ech0EvaluationFramework(model, tokenizer)

# Evaluate all domains
results = evaluator.evaluate_all_domains(test_datasets)

# Save results
evaluator.save_results("evaluation_results.json")
```

### Interpreting Results

```json
{
  "reasoning": {
    "score": 0.87,       // 87% accuracy
    "passed": 870,
    "total": 1000
  },
  "creativity": {
    "score": 0.82,       // 82% creativity score
    "passed": 820,
    "total": 1000
  }
}
```

---

## Advanced Usage

### Custom Domain Addition

```python
# 1. Create custom generator
class CustomDomainGenerator(DomainDatasetGenerator):
    def __init__(self):
        super().__init__("custom_domain", ["category1", "category2"])

    def generate_examples(self, num_examples: int):
        # Your generation logic
        examples = []
        for i in range(num_examples):
            example = TrainingExample(
                instruction="...",
                input="...",
                output="...",
                domain="custom_domain",
                category="category1",
                difficulty="medium"
            )
            examples.append(example)
        return examples

# 2. Add to config
# ech0_finetune_config.yaml
domains:
  custom_domain:
    enabled: true
    weight: 1.0
    max_samples: 5000
```

### Distributed Training

```yaml
# ech0_finetune_config.yaml
infrastructure:
  distributed:
    enabled: true
    world_size: 4      # 4 GPUs
    backend: "nccl"
```

```bash
# Launch distributed training
torchrun --nproc_per_node=4 ech0_train_orchestrator.py
```

### Quantization Options

```yaml
model:
  quantization:
    enabled: true
    bits: 4           # 4-bit or 8-bit
    type: "nf4"       # or "fp4", "int8"
```

### Learning Rate Scheduling

```yaml
training:
  lr_scheduler_type: "cosine"  # or "linear", "polynomial"
  warmup_steps: 100
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solutions**:
```yaml
# Reduce batch size
training:
  batch_size: 2  # Was 4

# Enable gradient checkpointing (already enabled)
# Use smaller model
model:
  base_model: "meta-llama/Llama-2-7b-hf"  # Instead of 13B

# Increase quantization
model:
  quantization:
    bits: 4  # Instead of 8
```

#### 2. Slow Training

**Solutions**:
- Enable mixed precision (already enabled: `bf16: true`)
- Reduce gradient accumulation steps
- Use faster optimizer (`adamw_8bit`)
- Enable gradient checkpointing

#### 3. Poor Convergence

```yaml
# Adjust learning rate
training:
  learning_rate: 1e-4  # Try different values

# Increase warmup
training:
  warmup_steps: 500  # Was 100
```

#### 4. Installation Issues

```bash
# CUDA version mismatch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# bitsandbytes issues (Windows)
# Use WSL2 or Docker instead
```

---

## Performance Benchmarks

### Hardware Requirements

| Model Size | Min VRAM | Recommended VRAM | Training Time (3 epochs) |
|------------|----------|------------------|--------------------------|
| 7B (4-bit) | 12GB | 16GB | 6-8 hours |
| 7B (8-bit) | 16GB | 24GB | 8-12 hours |
| 13B (4-bit) | 24GB | 32GB | 12-18 hours |
| 13B (8-bit) | 40GB | 48GB | 18-24 hours |

### Optimization Tips

1. **Memory-Efficient Attention**:
```yaml
infrastructure:
  memory_efficient_attention: true
```

2. **Gradient Checkpointing**:
   - Already enabled by default
   - Trades compute for memory (30% slower, 50% less memory)

3. **CPU Offloading**:
```yaml
infrastructure:
  cpu_offload: true  # Offload optimizer states to CPU
```

---

## Integration with Existing ech0 Ecosystem

The fine-tuned model integrates seamlessly with existing ech0 tools:

```python
# In ech0_philosophy_engine.py
from transformers import AutoModelForCausalLM, AutoTokenizer

class Ech0PhilosophyEngine:
    def __init__(self, model_path="/path/to/ech0-v2-finetuned"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def contemplate(self, question):
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=500)
        return self.tokenizer.decode(outputs[0])
```

---

## Next Steps

1. **Run Quick Start**: Get familiar with the pipeline
2. **Customize Domains**: Adjust weights based on your priorities
3. **Experiment with Hyperparameters**: Find optimal settings
4. **Evaluate Thoroughly**: Use comprehensive test sets
5. **Deploy**: Integrate with ech0 tools

---

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review training logs: `ech0_training.log`
3. Open GitHub issue with logs and config

---

**Last Updated**: 2025-12-03
**Version**: 2.0.0
**Author**: ech0 Development Team
