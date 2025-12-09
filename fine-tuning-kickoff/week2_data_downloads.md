# Week 2 – Rolling Data Downloads

Objective: start continuous ingestion for all “free” data sources so we do not fall behind when licensed content lands in Weeks 3‑4.

## 0. External Drive Staging
We only write large dumps to the external disk to save the internal SSD. Before running any commands:

```bash
export DATA_ROOT=\"/Volumes/EXTERNAL/ech0_training_data\"

if [ ! -d \"$DATA_ROOT\" ]; then
  echo \"External drive not mounted at $DATA_ROOT\" >&2
  exit 1
fi

mkdir -p \"$DATA_ROOT\"/{technical,scientific,academic,business,arts,general,legal}
```

Quick mount check (macOS):

```bash
diskutil info /Volumes/EXTERNAL | grep -E \"(Device Identifier|Mounted|File System)\"
```

## 1. Daily Cadence
| Time (UTC) | Task | Owner | Notes |
| --- | --- | --- | --- |
| 02:00 | Run `scripts/api_healthcheck.py` | Ops | Ensures tokens still valid |
| 03:00‑06:00 | Technical block (arXiv, GitHub) | Data Eng | Use `aria2` with 20 connections |
| 07:00‑10:00 | Scientific block (PubMed, NIH) | Data Eng | Respect NCBI 10 req/sec |
| 11:00‑14:00 | Knowledge block (Wikipedia, StackOverflow) | Data Eng | Keep under 1TB/day cap |
| 15:00‑17:00 | General crawl (Common Crawl subset) | Infra | Stage to `general/` |
| 18:00‑20:00 | Validation + hash logs | QA | Dedup + token counts |

## 2. Command Templates
### 2.1 arXiv
```bash
source ~/.venvs/ech0/bin/activate
python ech0_dataset_generators_extended.py --source arxiv \
  --categories cs.AI cs.LG cs.SE \
  --years 2018 2019 2020 2021 2022 2023 2024 \
  --output "$DATA_ROOT/technical/arxiv_$(date +%Y%m%d).jsonl"
```

### 2.2 GitHub Documentation
```bash
python ech0_dataset_generators_extended.py --source github \
  --token "$GITHUB_TOKEN" \
  --repo-list configs/github_repos.txt \
  --output "$DATA_ROOT/technical/github_docs_$(date +%Y%m%d).jsonl"
```
`configs/github_repos.txt` should contain the curated list from the master plan.

### 2.3 PubMed / PMC
```bash
python ech0_dataset_generators_extended.py --source pubmed \
  --email "$NCBI_EMAIL" --api-key "$NCBI_API_KEY" \
  --query-file configs/pubmed_queries.txt \
  --output "$DATA_ROOT/scientific/pubmed_$(date +%Y%m%d).jsonl"
```

### 2.4 Wikipedia Snapshot
```bash
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz -O /tmp/enwiki-abstract.gz
python tools/wikipedia_extract.py /tmp/enwiki-abstract.gz \
  > "$DATA_ROOT/general/wikipedia_$(date +%Y%m%d).jsonl"
```

### 2.5 Stack Overflow Dump
```bash
aria2c -x 16 -s 16 https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z -d /tmp/so_dump
python tools/stack_overflow_transform.py /tmp/so_dump/stackoverflow.com-Posts.7z \
  > "$DATA_ROOT/technical/stackoverflow_$(date +%Y%m%d).jsonl"
```

## 3. Rolling Uploads
Every night after validation (drive must still be mounted):
```bash
find "$DATA_ROOT" -type f -mtime -1 -name '*.jsonl' \
  -print -exec gsutil -m cp {} "gs://${GCS_BUCKET}/week2/" \;
```
Maintain manifest `ech0_training_logs/week2_ingest_manifest.csv` with columns `timestamp,domain,file,size_bytes,gcs_uri,md5`.

## 4. Validation Checklist
- [ ] SHA256 + MD5 recorded for each file (use `shasum -a 256 file`).
- [ ] Spot-check 100 records per file for encoding/UTF‑8.
- [ ] Append summary to `ech0_training_logs/week2_validation.md`:
  ```
  ## 2025‑12‑07
  - arxiv_20251207.jsonl: 48,220 rows, 1.82 GB, checksum OK.
  - pubmed_20251207.jsonl: 62,101 rows, 2.15 GB, checksum OK.
  - Issues: None.
  ```
- [ ] Update total GB counter (target 100 GB by end of Week 2).

## 5. Escalation
Any API failures >3 retries must be logged in `ech0_training_logs/incidents/week2.md` with time, request, HTTP code, owner, and mitigation (e.g., paused for 1 hr, rotated key).
