"""Metric helpers for quick regression/eval runs."""

from __future__ import annotations

from typing import List

from rouge_score import rouge_scorer
import sacrebleu


def bleu(preds: List[str], refs: List[str]) -> float:
    return float(sacrebleu.corpus_bleu(preds, [refs]).score) / 100.0


def rouge_l(preds: List[str], refs: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(preds, refs)]
    return float(sum(scores) / len(scores)) if scores else 0.0

