#!/usr/bin/env python3
"""
Generate fresh (non-duplicated) synthetic training data with chain-of-thought
reasoning plus concise answers. Produces JSON array files that mirror the
existing schema in ech0_training_data/*.json without overwriting them.

Usage:
    python scripts/generate_synthetic_training_data.py --output-dir ech0_training_data_generated --count 1000
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ISO = lambda: datetime.now(timezone.utc).isoformat()


def make_generator(seed: int = 42) -> random.Random:
    return random.Random(seed)


def synth_example(rng: random.Random, cfg: Dict[str, Any], idx: int) -> Dict[str, Any]:
    topic = rng.choice(cfg["topics"])
    task = rng.choice(cfg["tasks"])
    constraint = rng.choice(cfg["constraints"])
    style = rng.choice(cfg["styles"])
    difficulty = rng.choice(cfg["difficulties"])

    instruction = f"{task} (focus: {topic})"
    input_text = (
        f"Context: {topic}. Constraint: {constraint}. Style: {style}. "
        f"Provide depth then finish with a concise, actionable answer."
    )

    reasoning_points = rng.sample(cfg["reasoning"], k=min(3, len(cfg["reasoning"])))
    steps = "\n".join(f"- {p}" for p in reasoning_points)

    concise = rng.choice(cfg["concise_answers"])
    output = (
        f"Reasoning:\n{steps}\n\n"
        f"Concise answer: {concise}"
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "domain": cfg["domain"],
        "category": cfg.get("category", cfg["domain"]),
        "difficulty": difficulty,
        "metadata": {
            "generated_at": ISO(),
            "source": "synthetic_v1",
            "style": style,
            "topic": topic,
            "constraint": constraint,
            "idx": idx,
        },
        "quality_score": 0.92,
    }


DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "advanced_software_dataset.json": {
        "domain": "advanced_software",
        "category": "systems_architecture",
        "topics": [
            "distributed tracing design",
            "zero-downtime database migrations",
            "backpressure in message brokers",
            "circuit breakers and retries",
            "edge caching for APIs",
            "SLO error budgets",
            "multi-tenant isolation",
            "GPU scheduling for inference",
        ],
        "tasks": [
            "Draft a remediation plan",
            "Propose an architecture change",
            "Compare two implementation strategies",
            "Design an experiment to validate the approach",
            "Summarize tradeoffs and recommend a path",
        ],
        "constraints": [
            "latency < 100ms P95",
            "budget cap at $5k/month",
            "must be deployable incrementally",
            "requires observability hooks",
            "must tolerate regional failover",
        ],
        "styles": ["blunt and concise", "calm and precise", "operator-ready"],
        "difficulties": ["hard", "expert"],
        "reasoning": [
            "identify failure modes",
            "bound latency and throughput",
            "validate rollout safety",
            "ensure debuggability",
            "consider cost vs resilience",
        ],
        "concise_answers": [
            "Use staged rollout with feature flags; add tracing; cap concurrency.",
            "Choose broker backpressure + idempotent consumers; add DLQ metrics.",
            "Prefer online migrations with shadow writes; gate with canaries.",
        ],
    },
    "ai_ml_dataset.json": {
        "domain": "ai_ml",
        "category": "modeling",
        "topics": [
            "mixture-of-experts routing",
            "retrieval-augmented generation",
            "alignment with small preference sets",
            "tokenization impacts on latency",
            "quantization tradeoffs",
            "long-context window training",
        ],
        "tasks": [
            "Outline a training recipe",
            "Diagnose quality regressions",
            "Design an eval slice",
            "Propose a data curation rule",
            "Summarize deployment risks",
        ],
        "constraints": [
            "target latency < 400ms P95",
            "must run on CPU-only nodes",
            "keep perplexity within 5% of baseline",
            "no new dependencies",
            "prefer interpretable signals",
        ],
        "styles": ["lab-notes", "runbook", "executive-brief"],
        "difficulties": ["hard", "expert"],
        "reasoning": [
            "baseline then isolate change",
            "cover eval on style + safety + latency",
            "watch routing entropy",
            "bound context growth",
            "quantify quantization loss",
        ],
        "concise_answers": [
            "Start with CPU quantization + RAG; measure latency + BLEU on eval slice.",
            "Use small DPO pass; hold out style eval; pin tokenizer version.",
            "Route 10% traffic to MoE canary; log routing stats; rollback on entropy drift.",
        ],
    },
    "court_prediction_dataset.json": {
        "domain": "legal_court",
        "category": "legal_reasoning",
        "topics": [
            "4th amendment search scenarios",
            "contract breach remedies",
            "tort negligence elements",
            "summary judgment likelihood",
            "standing and jurisdiction checks",
            "evidence admissibility under FRE",
        ],
        "tasks": [
            "Assess likely outcome",
            "Map elements to facts",
            "Spot dispositive issues",
            "Draft a concise argument",
            "List key precedent factors",
        ],
        "constraints": [
            "stay within stated facts",
            "cite factors, not new law",
            "avoid overclaiming probability",
            "keep to brief style",
            "note burdens of proof",
        ],
        "styles": ["IRAC", "checklist", "mini-brief"],
        "difficulties": ["hard", "expert"],
        "reasoning": [
            "identify governing rule",
            "map facts to elements",
            "evaluate defenses",
            "weigh procedural posture",
            "note burden and standard",
        ],
        "concise_answers": [
            "Likely summary judgment for defendant; element 2 not met; burden unmet.",
            "Plaintiff shows duty/breach; causation weak; expect denial of SJ.",
            "Exclusion likely; FRE 403 prejudice outweighs probative value.",
        ],
    },
    "creativity_dataset.json": {
        "domain": "creativity",
        "category": "ideation",
        "topics": [
            "XR learning modules",
            "climate-positive product ideas",
            "community resilience programs",
            "calm technology interfaces",
            "bio-inspired design patterns",
        ],
        "tasks": [
            "Generate 3 contrasting ideas",
            "Refine a concept with constraints",
            "Draft a user journey",
            "Create a naming set",
            "Outline a pilot plan",
        ],
        "constraints": [
            "budget under $50k",
            "prototype in 4 weeks",
            "accessibility first",
            "low-carbon materials",
            "evidence-based outcomes",
        ],
        "styles": ["concise", "uplifting", "practical"],
        "difficulties": ["medium", "hard"],
        "reasoning": [
            "anchor on user need",
            "balance feasibility and novelty",
            "measure impact early",
            "keep scope narrow",
        ],
        "concise_answers": [
            "Pilot XR micro-lessons with haptics; measure retention; ship in 4 weeks.",
            "Choose low-carbon kit; run 2-week usability test; iterate on accessibility.",
            "Name options: LumenStep, EchoField, ClarityLoop.",
        ],
    },
    "crypto_dataset.json": {
        "domain": "crypto",
        "category": "risk_analysis",
        "topics": [
            "MEV mitigation in DEXs",
            "L2 data availability risks",
            "bridge security patterns",
            "staking economics stress",
            "wallet key management UX",
            "zk proof system tradeoffs",
        ],
        "tasks": [
            "Assess risk and mitigations",
            "Compare protocol designs",
            "Draft incident playbook steps",
            "Summarize economic incentives",
            "Propose observability signals",
        ],
        "constraints": [
            "minimize trust assumptions",
            "must be backward-compatible",
            "gas budget limited",
            "threat model includes MEV",
            "avoid single points of failure",
        ],
        "styles": ["threat-model", "concise", "operator-brief"],
        "difficulties": ["hard", "expert"],
        "reasoning": [
            "identify attack surface",
            "map mitigations to vectors",
            "check economic incentives",
            "consider upgrade path",
            "plan monitoring and alerting",
        ],
        "concise_answers": [
            "Adopt batch auctions; add on-chain alerts; stage rollout with circuit breakers.",
            "Use zk rollup with DA committee; monitor DA liveness; keep exit hatch.",
            "Bridge: favor light-client proofs; cap TVL per route; add rate limits.",
        ],
    },
    "law_dataset.json": {
        "domain": "law",
        "category": "analysis",
        "topics": [
            "privacy compliance for wearables",
            "cross-border data transfer",
            "consumer protection disclosures",
            "employment classification tests",
            "IP ownership in collaborations",
        ],
        "tasks": [
            "Issue-spot the scenario",
            "Draft a short compliance plan",
            "Compare jurisdictional rules",
            "List required disclosures",
            "Summarize risk posture",
        ],
        "constraints": [
            "plain-English summary",
            "note jurisdictional variance",
            "focus on likelihood not certainty",
            "keep under 200 words",
            "avoid creating attorney-client relationship",
        ],
        "styles": ["plain", "checklist", "brief"],
        "difficulties": ["medium", "hard"],
        "reasoning": [
            "identify applicable regimes",
            "map obligations to actions",
            "flag high-risk areas",
            "note open questions",
        ],
        "concise_answers": [
            "Add clear consent + DSR flow; vendor DPAs; SCCs for EU data.",
            "Classify carefully under ABC test; document control factors.",
            "Provide upfront pricing, refund terms, data use; keep under 200 words.",
        ],
    },
    "materials_science_dataset.json": {
        "domain": "materials_science",
        "category": "research",
        "topics": [
            "NV-center diamond sensing",
            "perovskite solar stability",
            "solid-state battery electrolytes",
            "graphene composite strength",
            "quantum dots for displays",
        ],
        "tasks": [
            "Design an experiment",
            "Summarize key mechanisms",
            "Compare material choices",
            "Outline failure modes",
            "Propose optimization steps",
        ],
        "constraints": [
            "lab-feasible",
            "focus on reproducibility",
            "budget-aware",
            "report key metrics",
            "short timeline",
        ],
        "styles": ["lab-notes", "concise", "checklist"],
        "difficulties": ["hard", "expert"],
        "reasoning": [
            "state hypothesis",
            "define controls and metrics",
            "anticipate degradation paths",
            "plan replication",
        ],
        "concise_answers": [
            "Run 3-arm test with controls; measure stability weekly; log humidity/heat.",
            "Favor solid electrolyte X; test cycling at 1C; track impedance drift.",
            "Use NV center array; compare sensitivity vs temperature; calibrate weekly.",
        ],
    },
    "prompt_engineering_dataset.json": {
        "domain": "prompt_engineering",
        "category": "prompting",
        "topics": [
            "few-shot selection",
            "tool-call prompting",
            "guarding verbosity",
            "structured JSON outputs",
            "long-context compression",
        ],
        "tasks": [
            "Write a prompt template",
            "Diagnose failure modes",
            "Design eval for prompts",
            "Suggest mitigation for drift",
            "Summarize best practices",
        ],
        "constraints": [
            "must return valid JSON",
            "keep answers concise",
            "no hidden assumptions",
            "include self-checks",
            "compatible with chat templates",
        ],
        "styles": ["runbook", "concise", "checklist"],
        "difficulties": ["medium", "hard"],
        "reasoning": [
            "specify role and format",
            "bound response length",
            "add exemplars",
            "add validation hints",
        ],
        "concise_answers": [
            "Use role+format header; 3 few-shots; add JSON schema + length guard.",
            "Add self-check step; clip to N tokens; include failure fallback text.",
        ],
    },
    "reasoning_dataset.json": {
        "domain": "reasoning",
        "category": "analysis",
        "topics": [
            "chain-of-thought for planning",
            "counterfactual reasoning",
            "multi-step tool use",
            "error analysis loops",
            "evidence ranking",
        ],
        "tasks": [
            "Solve a scenario with steps",
            "Design a reasoning checklist",
            "Find minimal info needed",
            "Explain a failure trace",
            "Propose a verification loop",
        ],
        "constraints": [
            "keep steps explicit",
            "limit to essential facts",
            "avoid speculation beyond data",
            "finish with brief answer",
            "prefer deterministic steps",
        ],
        "styles": ["thinking", "concise", "audit"],
        "difficulties": ["medium", "hard"],
        "reasoning": [
            "list knowns/unknowns",
            "test simplest hypothesis first",
            "check for hidden assumptions",
            "verify with alternate path",
        ],
        "concise_answers": [
            "Answer: choose option A; it satisfies constraints with least risk.",
            "Minimal info needed: X, Y; collect then recompute decision.",
            "Fix: step 3 assumption wrong; retry with corrected input; re-eval.",
        ],
    },
    "stock_prediction_dataset.json": {
        "domain": "stock_prediction",
        "category": "finance",
        "topics": [
            "earnings surprise analysis",
            "macro sensitivity for tech",
            "credit risk signals",
            "supply chain disruptions",
            "factor rotation timing",
        ],
        "tasks": [
            "Build a thesis quickly",
            "List key signals to watch",
            "Draft a risk case",
            "Compare two scenarios",
            "Summarize catalysts and timing",
        ],
        "constraints": [
            "no price targets",
            "focus on process not advice",
            "note uncertainty clearly",
            "brief and structured",
            "cite leading indicators",
        ],
        "styles": ["concise", "analyst-note", "checklist"],
        "difficulties": ["medium", "hard"],
        "reasoning": [
            "state base case",
            "identify upside/downside drivers",
            "flag leading indicators",
            "note risk controls",
        ],
        "concise_answers": [
            "Base: steady growth; watch gross margin + bookings; risk: supply chain.",
            "Catalyst: earnings beat; indicator: channel checks; keep position small.",
            "No price target; focus on process and signals listed.",
        ],
    },
}


def generate_file(cfg_name: str, cfg: Dict[str, Any], count: int, rng: random.Random, out_dir: Path) -> None:
    rows = [synth_example(rng, cfg, idx) for idx in range(count)]
    target = out_dir / cfg_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{cfg_name}: wrote {count} rows -> {target}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data with reasoning + concise answers.")
    parser.add_argument("--output-dir", default="ech0_training_data_generated", help="Where to write JSON files.")
    parser.add_argument("--count", type=int, default=1000, help="Rows per file.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    args = parser.parse_args()

    rng = make_generator(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in DOMAIN_CONFIGS.items():
        generate_file(name, cfg, args.count, rng, out_dir)


if __name__ == "__main__":
    main()
