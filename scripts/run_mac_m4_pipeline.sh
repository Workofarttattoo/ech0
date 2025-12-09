#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# One-touch Mac M4 CPU pipeline for Ech0 fine-tuning + eval + registry.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "[1/6] Installing finetuning dependencies"
python3 -m pip install -r requirements_finetuning.txt

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

RAW_DIR="$ROOT/ech0_training/data/raw"
PROC_DIR="$ROOT/ech0_training/data/processed"
MANIFEST_DIR="$ROOT/ech0_training/data/manifests"
mkdir -p "$RAW_DIR" "$PROC_DIR" "$MANIFEST_DIR"

RAW_FILES=()
for src in "$ROOT"/ech0_training_data/*.json; do
  name="$(basename "${src%.json}")"
  out="$RAW_DIR/$name.jsonl"
  echo "[2/6] Importing $name"
  python scripts/import_legacy_data.py --input "$src" --output "$out" --tag "$name"
  RAW_FILES+=("$out")
done

if [ ${#RAW_FILES[@]} -eq 0 ]; then
  echo "No source JSON files found in ech0_training_data/"
  exit 1
fi

PROC_FILES=()
for raw in "${RAW_FILES[@]}"; do
  base="$(basename "$raw")"
  out="$PROC_DIR/$base"
  echo "[3/6] Validating $base"
  python scripts/validate_dataset.py \
    --input "$raw" \
    --output "$out" \
    --stage processed \
    --max-tokens 2048 \
    --min-quality 0.25 \
    --dedupe
  PROC_FILES+=("$out")
done

MERGED="$PROC_DIR/merged_full.jsonl"
echo "[4/6] Merging processed datasets → $(basename "$MERGED")"
python - "$MERGED" "${PROC_FILES[@]}" <<'PY'
import sys
from pathlib import Path

merged = Path(sys.argv[1])
inputs = [Path(p) for p in sys.argv[2:]]
with merged.open("w", encoding="utf-8") as out:
    for path in inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.write(line.rstrip() + "\n")
print(f"Wrote merged dataset with {len(inputs)} sources to {merged}")
PY

echo "[4.1/6] Writing manifest"
python - <<'PY'
from pathlib import Path
from ech0_training import data_utils

merged = Path("ech0_training/data/processed/merged_full.jsonl")
manifest = data_utils.write_manifest([merged])
print(f"Manifest → {manifest}")
PY

CONFIG="$ROOT/ech0_training/configs/conversation_lora_cpu.yml"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_cpu}"
RUN_DIR="$ROOT/ech0_training/models/$RUN_ID"

echo "[5/6] Training run_id=$RUN_ID"
python -m ech0_training.trainer --config "$CONFIG" --run-id "$RUN_ID"

REPORT_DIR="$RUN_DIR/reports"
REPORT_PATH="$REPORT_DIR/regression_report.json"
echo "[6/6] Running regression eval"
python -m ech0_training.eval.runner \
  --model-dir "$RUN_DIR" \
  --baseline "$ROOT/ech0_training/eval/baselines/regression.jsonl" \
  --output-dir "$REPORT_DIR"

echo "[6.1/6] Registering run"
python - "$CONFIG" "$RUN_ID" "$RUN_DIR" "$REPORT_PATH" "${PROC_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

import yaml
from ech0_training.models import registry

config_path, run_id, run_dir, report_path, *datasets = sys.argv[1:]
base_model = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["base_model"]
score = None

report_file = Path(report_path)
if report_file.exists():
    report = json.loads(report_file.read_text(encoding="utf-8"))
    total = report.get("total", 0) or 0
    passed = report.get("passed", 0) or 0
    score = (passed / total) if total else 0.0

meta = registry.RunMetadata(
    run_id=run_id,
    run_dir=run_dir,
    base_model=base_model,
    config=config_path,
    datasets=datasets,
    eval_report=str(report_file) if report_file.exists() else None,
    score=score,
)
registry.register_run(meta)

print(
    json.dumps(
        {
            "run_id": run_id,
            "run_dir": run_dir,
            "score": score,
            "datasets": datasets,
            "report": str(report_file),
        },
        indent=2,
    )
)
PY

echo ""
echo "Done."
echo "Run dir : $RUN_DIR"
echo "Report  : $REPORT_PATH"
echo "Manifest: $MANIFEST_DIR/dataset_manifest.json"
echo "Status  : $RUN_DIR/status.json"
