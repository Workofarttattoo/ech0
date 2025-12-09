# ech0 Fine-Tuning (New Pipeline)

This guide captures the end-to-end workflow for the new training stack that
anchors data hygiene, LoRA training, evaluation gates, and registry promotion.

## Layout
- `ech0_training/data/{raw,clean,processed}/` — staged JSONL datasets
- `ech0_training/metadata/` — manifests, hashes, provenance notes
- `ech0_training/configs/` — training presets (LoRA/full)
- `ech0_training/trainer.py` — CLI entry for fine-tuning
- `ech0_training/eval/` — regression/eval harness + baselines
- `ech0_training/models/` — run artifacts + lightweight registry
- `scripts/validate_dataset.py` — validation/redaction/dedup
- `scripts/model_registry.py` — register/promote/rollback/package
- `ech0_hub_enhanced.py` — hub glue to stream real jobs and metrics

## Data Workflow
1) Drop raw dumps to `ech0_training/data/raw/` (or import legacy JSON arrays):
```
python scripts/import_legacy_data.py --input ech0_training_data/crypto_dataset.json --tag crypto
```
2) Validate + redact + dedupe into clean/processed stages:
```
python scripts/validate_dataset.py \
  --input ech0_training/data/raw/my_dump.jsonl \
  --stage clean --max-tokens 2048 --min-quality 0.25

python scripts/validate_dataset.py \
  --input ech0_training/data/clean/my_dump.jsonl \
  --stage processed --dedupe --fail-on-secrets
```
3) Manifests/hashes land in `ech0_training/metadata/` (auto-written).

Schema (JSONL rows):
```
{
  "prompt": "...",
  "completion": "...",
  "tags": ["domain", "topic"],
  "quality_score": 0.0-1.0,
  "metadata": {...}
}
```

## Training
LoRA smoke test (small run):
```
python -m ech0_training.trainer --config ech0_training/configs/conversation_lora.yml
```
Full fine-tune preset (longer):
```
python -m ech0_training.trainer --config ech0_training/configs/domain_full_finetune.yml
```
Artifacts land in `ech0_training/models/{run_id}/` (model, tokenizer, status).
Status updates stream to `status.json` for live dashboards.

## Evaluation & Gating
Run regression/eval on a checkpoint:
```
python -m ech0_training.eval.runner \
  --model-dir ech0_training/models/<run_id> \
  --baseline ech0_training/eval/baselines/regression.jsonl

python -m ech0_training.scheduler eval \
  --model-dir ech0_training/models/<run_id> \
  --baseline ech0_training/eval/baselines/regression.jsonl
```
Reports are written to `models/{run_id}/reports/`. Gate passes when all
regression cases meet thresholds and no safety flags are raised.

## Registry & Promotion
Register a run:
```
python scripts/model_registry.py register \
  --run-id <run_id> \
  --run-dir ech0_training/models/<run_id> \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --config ech0_training/configs/conversation_lora.yml \
  --datasets ech0_training/data/processed/sample_conversations.jsonl \
  --eval-report ech0_training/models/<run_id>/reports/regression_report.json \
  --score 0.9
```
Inspect registry contents:
```
python scripts/model_registry.py list
```
Promote to production (also updates `active_run`):
```
python scripts/model_registry.py promote --run-id <run_id> --stage production
```
Package/ship to hub load path:
```
python scripts/model_registry.py package \
  --run-dir ech0_training/models/<run_id> \
  --dest ~/models/ech0/
```
Rollback to previous active run:
```
python scripts/model_registry.py rollback
```

## Orchestration from Hub
Use `ech0_hub_enhanced.py` to call real trainers/evaluators with live metrics:
```
# Train only with status polling
python ech0_hub_enhanced.py --mode train --config ech0_training/configs/conversation_lora.yml

# Eval an existing model directory
python ech0_hub_enhanced.py --mode eval --model-dir ech0_training/models/<run_id>

# Pipeline: train → eval (optionally promote)
python ech0_hub_enhanced.py --mode pipeline --config ech0_training/configs/conversation_lora.yml --promote-on-pass
```
Hub UI can watch `status.json` under the run directory for live loss/step metrics.

## Remote/Compute Notes
- Configs are CPU-safe defaults; bump batch/precision for GPU boxes.
- Set `HF_HOME`/`TRANSFORMERS_CACHE` if training on remote nodes.
- For remote clusters, wrap scheduler invocations in your job runner (K8s/SLURM)
  and keep `ech0_training/models/` synchronized (rsync/DVC/git-lfs).***

