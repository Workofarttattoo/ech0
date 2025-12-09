# Vertex AI Staging & Dry-Run

Goal: prove Vertex AI permissions, bucket ACLs, and LoRA config before Week 9 full runs.

## 1. Reuse Service Account
- SA email lives in `~/.config/gcloud/application_default_credentials.json` (created during Week 1).
- Ensure it has `roles/aiplatform.user` and `roles/storage.objectAdmin`.
- Export for CLI:
  ```bash
  export PROJECT_ID="ech0-training-2025"
  export REGION="us-central1"
  export SA_EMAIL="ech0-training@${PROJECT_ID}.iam.gserviceaccount.com"
  export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/ech0-sa-key.json"
  ```

## 2. Create Mini Dataset for Dry-Run
```bash
source ~/.venvs/ech0/bin/activate
python ech0_dataset_generator.py --preset dryrun --rows 200 \
  --output ech0_training_data/dryrun/ech0_dryrun.jsonl
```
Add a stats file:
```bash
python scripts/dataset_stats.py ech0_training_data/dryrun/ech0_dryrun.jsonl \
  > ech0_training_logs/dryrun_stats.json
```
Upload to GCS:
```bash
DRYRUN_URI="gs://${GCS_BUCKET}/dryrun/ech0_dryrun.jsonl"
gsutil -m cp ech0_training_data/dryrun/ech0_dryrun.jsonl "$DRYRUN_URI"
```

## 3. Generate Config
`ech0_finetuning_config.json` example:
```json
{
  "model": "mistral-7b",
  "training_data": "gs://ech0-training-2025-training-data/dryrun/ech0_dryrun.jsonl",
  "output_model_display_name": "ech0-dryrun-v0",
  "hyperparameters": {
    "learning_rate_multiplier": 0.75,
    "num_train_epochs": 1,
    "batch_size": 4,
    "warmup_fraction": 0.05,
    "weight_decay": 0.01
  },
  "training_filter_split": "",
  "validation_split": 0.1,
  "test_split": 0.1
}
```
Save as `ech0/fine-tuning-kickoff/configs/ech0_vertex_dryrun.json` (create folder if needed).

## 4. Submit Job (CLI)
```bash
mkdir -p ech0/fine-tuning-kickoff/configs
cp ech0_finetuning_config.json ech0/fine-tuning-kickoff/configs/ech0_vertex_dryrun.json
sed -i '' "s/output_model_display_name.*/\"output_model_display_name\": \"ech0-dryrun-v0\",/" ech0/fine-tuning-kickoff/configs/ech0_vertex_dryrun.json

JOB_NAME="ech0-dryrun-$(date +%Y%m%d-%H%M)"
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name="$JOB_NAME" \
  --config=ech0/fine-tuning-kickoff/configs/ech0_vertex_dryrun.json \
  --service-account=$SA_EMAIL
```
Capture the job ID:
```bash
JOB_ID=$(gcloud ai custom-jobs list --region=$REGION --filter="displayName=$JOB_NAME" --format='value(name)')
echo $JOB_ID >> ech0_training_logs/vertex/dryrun_job_ids.txt
```

## 5. Monitor Logs
```bash
gcloud ai custom-jobs stream-logs "$JOB_ID" --region=$REGION
```
Look for:
- Data ingestion success
- Training steps >0
- Final metrics output

## 6. Download Artifacts
```bash
OUTPUT_BUCKET="gs://${PROJECT_ID}-vertex-outputs"
mkdir -p ech0_finetuned_models/dryrun

# Example path: gs://.../customJobs/123456789/outputs/model/
gsutil -m cp -r ${OUTPUT_BUCKET}/customJobs/*/model \
  ech0_finetuned_models/dryrun/
```
Record the model URI + eval metrics in `ech0_training_logs/vertex/dryrun_report.md`.

## 7. Acceptance Criteria
- [ ] Job status `SUCCEEDED`.
- [ ] Training + eval metrics persisted.
- [ ] Resulting adapter loads in `ech0_finetune_engine.py` without error (run `python ech0_finetune_engine.py --dryrun-load ech0_finetuned_models/dryrun/model`).
- [ ] IAM audit: service account displayed in `gcloud projects get-iam-policy` with limited scopes.

If any box fails, file ticket in `ech0_training_logs/incidents/vertex.md` and rerun after remediation.
