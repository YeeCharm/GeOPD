"""Validate benchmark manifest schemas and local image paths before inference."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

EXPECTED_COUNTS = {
    "CHOICE": 9407,
    "GEOBench-VLM": 3211,
    "OmniEarth": 5656,
    "HRVQA": 2000,
    "LHRS-Bench": 690,
}

REQUIRED_FIELDS = ("uid", "dataset", "task", "format", "images", "question", "answer")


def validate_manifest(
    manifest_path: Path,
    image_root: Path | None = None,
    *,
    skip_images: bool = False,
    strict_counts: bool = False,
) -> dict:
    """Check required fields, duplicate IDs, expected counts, and image readability."""
    seen_uids: set[str] = set()
    dataset_counts: collections.Counter[str] = collections.Counter()
    format_counts: collections.Counter[str] = collections.Counter()
    missing_fields = 0
    duplicate_uids = 0
    missing_images = 0

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if any(field not in row for field in REQUIRED_FIELDS):
                missing_fields += 1
                continue

            uid = row["uid"]
            if uid in seen_uids:
                duplicate_uids += 1
            seen_uids.add(uid)

            dataset_counts[row["dataset"]] += 1
            format_counts[row["format"]] += 1

            if not skip_images:
                for img_path in row["images"]:
                    candidate = Path(img_path)
                    if not candidate.is_absolute() and image_root is not None:
                        candidate = image_root / candidate
                    if not candidate.exists():
                        missing_images += 1

    count_mismatches = {}
    if strict_counts:
        for name, expected in EXPECTED_COUNTS.items():
            actual = dataset_counts.get(name, 0)
            if actual != expected:
                count_mismatches[name] = {"expected": expected, "actual": actual}

    ok = (
        missing_fields == 0
        and duplicate_uids == 0
        and missing_images == 0
        and not count_mismatches
    )
    return {
        "ok": ok,
        "manifest": str(manifest_path),
        "total_records": sum(dataset_counts.values()),
        "datasets": dict(sorted(dataset_counts.items())),
        "formats": dict(sorted(format_counts.items())),
        "missing_fields": missing_fields,
        "duplicate_uids": duplicate_uids,
        "missing_images": missing_images,
        "count_mismatches": count_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GeOPD evaluation manifests.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to a benchmark JSONL manifest.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root prepended to relative image paths.")
    parser.add_argument("--skip-images", action="store_true", help="Skip checking image files on disk.")
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help="Verify exact five-benchmark counts (20,964 total).",
    )
    args = parser.parse_args()

    report = validate_manifest(
        args.manifest,
        args.image_root,
        skip_images=args.skip_images,
        strict_counts=args.strict_counts,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
