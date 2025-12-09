# ech0_training/data

Dataset staging areas for the training system. All JSONL rows follow the
schema:

- `prompt`: user/system prompt string
- `completion`: desired model answer string
- `tags`: list of domain or topic strings
- `quality_score`: float 0-1 describing curation quality
- `metadata`: optional dict (source, provenance, hash, split, etc.)

Directories:
- `raw/`: unvalidated dumps (encrypted or access-controlled; keep secrets here)
- `clean/`: validated/redacted outputs from the validator CLI
- `processed/`: train-ready JSONL files (deduplicated, split, token safe)
- `manifests/`: dataset manifests and hashes

Use `scripts/validate_dataset.py` to move data from `raw/` → `clean/` →
`processed/` with stats and secret checks.

Legacy `ech0_training_data/*.json` arrays can be converted with:
```
python scripts/import_legacy_data.py --input ech0_training_data/crypto_dataset.json --tag crypto
```
Then validate into `clean/` and `processed/` as needed.***

