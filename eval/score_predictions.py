"""Aggregate per-benchmark and equal-weight five-benchmark accuracy from predictions."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from eval.scoring import score_prediction

BENCHMARK_ORDER = ("CHOICE", "GEOBench-VLM", "OmniEarth", "HRVQA", "LHRS-Bench")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_predictions(manifest_rows: list[dict], prediction_rows: list[dict]) -> dict:
    lookup = {row["uid"]: row for row in manifest_rows}
    by_dataset: dict[str, list[dict]] = collections.defaultdict(list)

    for pred in prediction_rows:
        uid = pred["uid"]
        if uid not in lookup:
            continue
        ref = lookup[uid]
        answer_text = pred.get("prediction", pred.get("answer", ""))
        outcome = score_prediction(ref, answer_text)
        by_dataset[ref["dataset"]].append(outcome)

    dataset_summary = {}
    ordered_names = [d for d in BENCHMARK_ORDER if d in by_dataset] + sorted(
        d for d in by_dataset if d not in BENCHMARK_ORDER
    )
    for name in ordered_names:
        items = by_dataset[name]
        acc = 100.0 * sum(item["score"] for item in items) / len(items)
        dataset_summary[name] = {
            "count": len(items),
            "accuracy": round(acc, 2),
            "needs_review": sum(1 for item in items if item["needs_review"]),
        }

    mean_score = (
        round(sum(v["accuracy"] for v in dataset_summary.values()) / len(dataset_summary), 2)
        if dataset_summary
        else None
    )
    return {
        "datasets": dataset_summary,
        "equal_weight_average": mean_score,
        "total_scored": sum(v["count"] for v in dataset_summary.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions against a GeOPD benchmark manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to benchmark manifest JSONL.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSONL.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save summary JSON.")
    args = parser.parse_args()

    summary = evaluate_predictions(load_jsonl(args.manifest), load_jsonl(args.predictions))
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
