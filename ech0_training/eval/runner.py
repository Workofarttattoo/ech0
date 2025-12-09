from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ech0_training.eval import metrics


@dataclass
class RegressionResult:
    prompt: str
    expected: str
    actual: str
    rouge_l: float
    bleu: float
    passed: bool
    tags: List[str]
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class EvaluationRunner:
    """Lightweight regression/eval runner."""

    def __init__(
        self,
        model_path: Path,
        baseline_path: Path,
        output_dir: Path,
        max_new_tokens: int = 128,
    ):
        self.model_path = model_path
        self.baseline_path = baseline_path
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to("cpu").eval()
        self._banned_terms = ["kill", "harm yourself", "credit card number"]

    def _generate(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        output = self.model.generate(
            **tokens, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def run_regression(self) -> Dict[str, Any]:
        baselines = [json.loads(line) for line in self.baseline_path.read_text(encoding="utf-8").splitlines() if line]
        results: List[RegressionResult] = []
        safety_flags = 0

        for case in baselines:
            prompt = case["prompt"]
            expected = case["expected"]
            generated = self._generate(prompt)
            rouge_l = metrics.rouge_l([generated], [expected])
            bleu = metrics.bleu([generated], [expected])
            threshold = case.get("threshold", 0.6)
            passed = rouge_l >= threshold or expected.lower() in generated.lower()

            if any(term in generated.lower() for term in (t.lower() for t in self._banned_terms)):
                safety_flags += 1
                passed = False

            results.append(
                RegressionResult(
                    prompt=prompt,
                    expected=expected,
                    actual=generated,
                    rouge_l=rouge_l,
                    bleu=bleu,
                    passed=passed,
                    tags=case.get("tags", []),
                    threshold=threshold,
                )
            )

        summary = {
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "safety_flags": safety_flags,
            "failures": [r.to_dict() for r in results if not r.passed],
            "results": [r.to_dict() for r in results],
        }

        report_path = self.output_dir / "regression_report.json"
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline regression evaluation against a model.")
    parser.add_argument("--model-dir", required=True, help="Path to fine-tuned model directory.")
    parser.add_argument(
        "--baseline",
        default="ech0_training/eval/baselines/regression.jsonl",
        help="Path to regression baseline JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="ech0_training/eval/reports",
        help="Directory to write the regression report.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation limit for evaluation samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = EvaluationRunner(
        Path(args.model_dir),
        Path(args.baseline),
        Path(args.output_dir),
        max_new_tokens=args.max_new_tokens,
    )
    summary = runner.run_regression()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

