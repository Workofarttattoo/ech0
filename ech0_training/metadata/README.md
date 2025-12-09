# ech0_training/metadata

Metadata, manifests, and reproducibility artifacts for datasets and runs.

- `dataset_manifest.json`: list of dataset files, sizes, hashes, and tags
- `provenance/`: provenance notes for synthetic/self-play/retrieval data
- `hashes/`: optional per-file SHA256 checksums

The validator CLI writes updated manifests after each run so training jobs can
record which exact artifacts were used. Keep raw-sensitive metadata (like
source URLs or access tokens) encrypted or excluded from version control.***

