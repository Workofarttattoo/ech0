# Week 1 – Infrastructure & API Connectivity

Reference: `ECH0_MULTI_DOMAIN_TRAINING_MASTER_PLAN.md` lines 799‑821.

## 1. Project & Storage Skeleton
- [ ] Confirm budget approval + project code.
- [ ] Export canonical variables (add to shell profile):
  ```bash
  export PROJECT_ID="ech0-training-2025"
  export REGION="us-central1"
  export GCS_BUCKET="${PROJECT_ID}-training-data"
  export DATA_ROOT="$HOME/ech0_training_data"
  ```
- [ ] Create the bucket if it does not exist:
  ```bash
  gcloud config set project "$PROJECT_ID"
  gsutil mb -l "$REGION" "gs://${GCS_BUCKET}/"
  ```
- [ ] Create local mirrors to stage downloads:
  ```bash
  mkdir -p "$DATA_ROOT"/{technical,scientific,academic,business,arts,general,legal}
  ```

## 2. Dependency & Tooling Install
- [ ] Install/upgrade Python toolchain (3.10+ recommended) and Poetry/Pipenv if used.
- [ ] `pip install -r requirements_finetuning.txt` from repo root (ensures PEFT, transformers, datasets, arxiv, biopython, praw, wikipedia, etc.).
- [ ] Install system deps:
  - `brew install jq parallel aria2 git-lfs`
  - `sudo apt-get install poppler-utils tesseract-ocr` (if on Linux) for PDF extraction.
- [ ] Initialize virtualenv for ingestion scripts: `python -m venv ~/.venvs/ech0 && source ~/.venvs/ech0/bin/activate`.

## 3. API Credentials
| API | Action | Notes |
| --- | --- | --- |
| arXiv | none | Rate-limited; use `arxiv` Python client |
| PubMed / PMC | Obtain NCBI API key | https://www.ncbi.nlm.nih.gov/account/ |
| Semantic Scholar | Generate key | https://www.semanticscholar.org/product/api |
| GitHub | PAT with `repo` scope | Needed for documentation scraping |
| Stack Exchange | Register app for Stack Overflow dump | store key in `.env` |
| JSTOR / HBS | Start legal/licensing paperwork | tracked separately |
| Google Cloud | Service account with `roles/aiplatform.admin` & `storage.admin` | reused for Vertex |

Store all secrets in 1Password/`secrets.ech0.env` and never commit.

## 4. Connectivity Smoke Tests
Create `scripts/api_healthcheck.py` with the snippet below and run daily until Week 2 to ensure endpoints are reachable.

```python
#!/usr/bin/env python3
import os, requests, arxiv, datetime, json
from Bio import Entrez

LOG = os.path.expanduser("~/ech0_training_logs/infra/api_health.jsonl")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

status = {
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "arxiv": None,
    "pubmed": None,
    "semantic_scholar": None,
    "github": None
}

try:
    result = next(arxiv.Client().results(arxiv.Search(query="cat:cs.AI", max_results=1)))
    status["arxiv"] = {"ok": True, "id": result.get_short_id()}
except Exception as exc:
    status["arxiv"] = {"ok": False, "error": str(exc)}

try:
    Entrez.email = os.environ.get("NCBI_EMAIL", "ops@corpoflight.ai")
    Entrez.api_key = os.environ.get("NCBI_API_KEY")
    handle = Entrez.esearch(db="pubmed", term="cancer", retmax=1)
    status["pubmed"] = {"ok": True, "count": int(Entrez.read(handle)['Count'])}
except Exception as exc:
    status["pubmed"] = {"ok": False, "error": str(exc)}

try:
    key = os.environ["SEMANTIC_SCHOLAR_KEY"]
    resp = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params={
        "query": "quantum",
        "limit": 1
    }, headers={"x-api-key": key}, timeout=20)
    status["semantic_scholar"] = {"ok": resp.ok, "status": resp.status_code}
except Exception as exc:
    status["semantic_scholar"] = {"ok": False, "error": str(exc)}

try:
    token = os.environ["GITHUB_TOKEN"]
    resp = requests.get("https://api.github.com/rate_limit", headers={"Authorization": f"token {token}"}, timeout=20)
    status["github"] = {"ok": resp.ok, "remaining": resp.json()["resources"]["core"]["remaining"]}
except Exception as exc:
    status["github"] = {"ok": False, "error": str(exc)}

with open(LOG, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(status) + "\n")

print(json.dumps(status, indent=2))
```

Usage:
```bash
chmod +x scripts/api_healthcheck.py
./scripts/api_healthcheck.py
```
Review log entries; any `ok: false` needs remediation before Week 2.

## 5. Reporting
- Drop screenshots/logs into `ech0_training_logs/infra/`.
- Update the overall tracker with checkmarks for each Week 1 bullet.
