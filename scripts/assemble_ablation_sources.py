"""Assemble per-level source CSVs for information ablation from existing data.

One-off script that builds all 4 per-level source CSVs without making API calls,
using extraction data we already have:
- problem/environment: from natural source (v8_opus_natural)
- test: from TEST override extraction output
- solution: from v7_opus_solution extraction output

This script reuses assemble_per_level_source() from the permanent extraction script
to avoid code duplication.

Usage:
    python scripts/assemble_ablation_sources.py
    python scripts/assemble_ablation_sources.py --datasets swebench_verified gso
"""

import argparse
import shutil
from pathlib import Path

from llm_judge_feature_extraction.extract_ablation_overrides import (
    ALL_DATASETS,
    EXTRACTION_OUTPUT_BASE,
    NATURAL_SOURCE_BASE,
    PER_LEVEL_SOURCE_BASE,
    ROOT,
    assemble_per_level_source,
)

# v7_opus_solution CSVs: all 20 non-ENV features extracted at SOLUTION level
V7_SOLUTION_BASE = ROOT / "output" / "llm_judge_features" / "v7_opus_solution"


def assemble_for_dataset(dataset: str) -> None:
    """Assemble all 4 per-level source CSVs for a dataset."""
    natural_source = NATURAL_SOURCE_BASE / f"{dataset}.csv"
    if not natural_source.exists():
        raise FileNotFoundError(f"Natural source not found: {natural_source}")

    output_dir = PER_LEVEL_SOURCE_BASE / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Assembling per-level source CSVs for {dataset}")
    print(f"{'='*60}")

    # problem.csv and environment.csv: copy natural source
    # (PROBLEM features at their natural level, plus ENV features from auditor)
    for level in ("problem", "environment"):
        output_path = output_dir / f"{level}.csv"
        shutil.copy2(natural_source, output_path)
        print(f"  Copied natural source → {output_path.relative_to(ROOT)}")

    # test.csv: PROBLEM features from TEST override + ENV from natural + TEST from natural
    test_extraction_csv = EXTRACTION_OUTPUT_BASE / dataset / "test_override" / "llm_judge_features.csv"
    if not test_extraction_csv.exists():
        raise FileNotFoundError(
            f"TEST override extraction not found: {test_extraction_csv}. "
            f"Run: python -m llm_judge_feature_extraction.extract_ablation_overrides "
            f"--info-level test --datasets {dataset} --parallel --concurrency 30"
        )
    assemble_per_level_source(
        dataset, "test", test_extraction_csv, natural_source,
        output_dir / "test.csv",
    )

    # solution.csv: all non-ENV features from v7_opus_solution + ENV from natural
    v7_csv = V7_SOLUTION_BASE / dataset / "llm_judge_features.csv"
    if not v7_csv.exists():
        raise FileNotFoundError(
            f"v7_opus_solution CSV not found: {v7_csv}. "
            f"Expected pre-existing SOLUTION override extraction."
        )
    assemble_per_level_source(
        dataset, "solution", v7_csv, natural_source,
        output_dir / "solution.csv",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Assemble per-level source CSVs from existing extraction data",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS, help="Datasets to process (default: all)",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        assemble_for_dataset(dataset)

    print(f"\nDone. Per-level source CSVs written to {PER_LEVEL_SOURCE_BASE.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
