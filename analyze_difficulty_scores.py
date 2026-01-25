#!/usr/bin/env python3
"""
Analyze and plot IRT difficulty ("b") distributions across benchmarks.

Reads:
- SWE-bench Verified: items_verified.csv
- SWE-bench Pro:      items_pro.csv
- Terminal-Bench:     items_terminal_bench.csv

Each CSV is expected to contain a 'b' column (difficulty) and optionally 'b_std'.
We compute mean and variance of 'b' per benchmark and save an overlapping histogram.

Example:
  python fulcrum/fellowship/analyze_difficulty_scores.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(f"Missing dependency {pkg!r}. Original error: {e}") from e


_require("numpy")
_require("matplotlib")

import numpy as np  # noqa: E402

# Use a non-interactive backend by default (safe on clusters).
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class BenchmarkScores:
    name: str
    path: Path
    scores: np.ndarray


def _guess_id_field(fieldnames: Sequence[str]) -> Optional[str]:
    """
    Items CSVs sometimes use an empty first header for the item id (",b,b_std").
    """
    fns = [str(x) for x in fieldnames if x is not None]
    candidates = ["", "item_id", "id", "instance_id", "task_id", "name"]
    for c in candidates:
        if c in fns:
            return c
    # Fallback: choose the first non-b column if present.
    for fn in fns:
        if fn not in ("b", "b_std") and fn.strip() != "":
            return fn
    return None


def load_b_scores(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    scores: List[float] = []
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        fns = list(r.fieldnames or [])
        if "b" not in set(fns):
            raise ValueError(f"Missing required column 'b' in {csv_path}; got columns={fns}")
        _ = _guess_id_field(fns)  # not required, but validates expected format
        for row in r:
            s = str(row.get("b", "") or "").strip()
            if not s:
                continue
            try:
                scores.append(float(s))
            except ValueError:
                continue
    if not scores:
        raise ValueError(f"No valid difficulty scores found in {csv_path} (column 'b').")
    return np.asarray(scores, dtype=float)


def mean_and_variance(x: np.ndarray) -> Tuple[float, float]:
    """
    Returns (mean, variance) with population variance (ddof=0).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.mean(x)), float(np.var(x, ddof=0)))


def compute_bin_edges(all_scores: np.ndarray, bins: str) -> np.ndarray:
    # numpy accepts 'auto', 'fd', 'sturges', etc.
    edges = np.histogram_bin_edges(all_scores, bins=bins)
    if edges.size < 2:
        # Extremely degenerate case; fall back.
        edges = np.histogram_bin_edges(all_scores, bins=30)
    return edges


def plot_overlapping_histograms(
    *,
    benchmarks: Sequence[BenchmarkScores],
    out_path: Path,
    bins: str,
    title: str,
    alpha: float,
    normalize: bool,
) -> None:
    all_scores = np.concatenate([b.scores for b in benchmarks], axis=0)
    edges = compute_bin_edges(all_scores, bins=bins)

    plt.figure(figsize=(10, 6))
    for b in benchmarks:
        weights = None
        if normalize:
            # Each benchmark's histogram sums to 1, so bar heights represent the
            # fraction of tasks in that benchmark falling into the bin.
            weights = np.ones_like(b.scores, dtype=float) / float(b.scores.size)
        plt.hist(
            b.scores,
            bins=edges,
            alpha=alpha,
            label=f"{b.name} (n={b.scores.size})",
            edgecolor="none",
            weights=weights,
        )
        mu, _ = mean_and_variance(b.scores)
        plt.axvline(mu, linewidth=1.5, linestyle="--")

    plt.title(title)
    plt.xlabel("IRT difficulty score (b)")
    plt.ylabel("Fraction of tasks" if normalize else "Count")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--items_verified_csv",
        type=str,
        default="out/multi_benchmark_ood/irt_model_scaffold_1pl/items_verified.csv",
        help="Path to SWE-bench Verified items CSV (must contain column 'b').",
    )
    p.add_argument(
        "--items_pro_csv",
        type=str,
        default="out/multi_benchmark_ood/irt_model_scaffold_1pl/items_pro.csv",
        help="Path to SWE-bench Pro items CSV (must contain column 'b').",
    )
    p.add_argument(
        "--items_terminal_bench_csv",
        type=str,
        default="out/multi_benchmark_ood/irt_model_scaffold_1pl/items_terminal_bench.csv",
        help="Path to Terminal-Bench items CSV (must contain column 'b').",
    )
    p.add_argument(
        "--out_plot",
        type=str,
        default="out/difficulty_score_histograms.png",
        help="Where to write the histogram PNG.",
    )
    p.add_argument(
        "--bins",
        type=str,
        default="fd",
        help="Histogram binning strategy (numpy bins: e.g. 'fd', 'auto', 'sturges', or an int as string).",
    )
    p.add_argument("--alpha", type=float, default=0.45, help="Histogram transparency.")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize each benchmark's histogram so bars show fractions (sum to 1 per benchmark).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="IRT difficulty score distributions",
        help="Plot title.",
    )
    args = p.parse_args(argv)

    # Resolve relative paths from repo/workspace root (current working directory).
    verified_path = Path(args.items_verified_csv)
    pro_path = Path(args.items_pro_csv)
    tb_path = Path(args.items_terminal_bench_csv)

    benchmarks = [
        BenchmarkScores("SWE-bench Verified", verified_path, load_b_scores(verified_path)),
        BenchmarkScores("SWE-bench Pro", pro_path, load_b_scores(pro_path)),
        BenchmarkScores("Terminal-Bench", tb_path, load_b_scores(tb_path)),
    ]

    print("Benchmark difficulty statistics (b):")
    for b in benchmarks:
        mu, var = mean_and_variance(b.scores)
        print(f"- {b.name}: n={b.scores.size}, mean={mu:.6f}, var={var:.6f}")

    # bins can be an int or a named strategy; accept either.
    bins: str
    try:
        bins = str(int(str(args.bins).strip()))
    except Exception:
        bins = str(args.bins).strip()
        if not bins:
            bins = "fd"

    plot_overlapping_histograms(
        benchmarks=benchmarks,
        out_path=Path(args.out_plot),
        bins=bins,
        title=str(args.title),
        alpha=float(args.alpha),
        normalize=bool(args.normalize),
    )
    print(f"Wrote plot: {args.out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

