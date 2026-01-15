#!/usr/bin/env python3
"""
Compute Spearman's rho between predicted difficulty scores and HF difficulty categories.

We compute rho separately for:
- train split (from predictions.csv split column)
- test split  (from predictions.csv split column)
- zero-success tasks (from predictions.csv split column)

HF dataset: defaults to princeton-nlp/SWE-bench_Verified test split, which includes a
`difficulty` categorical column (e.g. "<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours").

Example:
  /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/python \\
    /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/compute_spearman_difficulty_alignment.py \\
    --predictions_csv /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt_qwen3vl8b_linear/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(f"Missing dependency {pkg!r}. Original error: {e}") from e


_require("numpy")
_require("datasets")

import numpy as np  # noqa: E402
from datasets import load_dataset  # type: ignore  # noqa: E402


_V_SUFFIX_RE = re.compile(r"-v.*$")


def normalize_swebench_item_id(raw_item_id: str) -> str:
    s = str(raw_item_id or "").strip()
    if s.startswith("instance_"):
        s = s[len("instance_") :]
    s = _V_SUFFIX_RE.sub("", s)
    return s.strip()


def parse_difficulty_order(order: str) -> List[str]:
    xs = [x.strip() for x in str(order).split(",") if x.strip()]
    if not xs:
        raise ValueError("difficulty_order must contain at least one comma-separated label")
    return xs


def load_hf_difficulty_map(*, dataset_name: str, split: str, difficulty_col: str) -> Dict[str, str]:
    ds = load_dataset(str(dataset_name), split=str(split))
    out: Dict[str, str] = {}
    for row in ds:
        iid = normalize_swebench_item_id(str(row.get("instance_id", "") or ""))
        if not iid:
            continue
        d = row.get(str(difficulty_col), None)
        if d is None:
            continue
        s = str(d).strip()
        if not s:
            continue
        out[iid] = s
    return out


def load_predictions_csv(path: str) -> List[Tuple[str, float, str]]:
    """
    Returns list of (item_id, diff_pred, split).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"predictions_csv not found: {path}")
    out: List[Tuple[str, float, str]] = []
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        fns = set(r.fieldnames or [])
        need = {"item_id", "diff_pred", "split"}
        missing = sorted(need - fns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}; got columns={sorted(fns)}")
        for row in r:
            iid = normalize_swebench_item_id(str(row.get("item_id", "") or ""))
            split = str(row.get("split", "") or "").strip().lower()
            s = str(row.get("diff_pred", "") or "").strip()
            if not iid or split not in ("train", "test", "zero_success") or not s:
                continue
            out.append((iid, float(s), split))
    return out


@dataclass(frozen=True)
class GroupResult:
    name: str
    n: int
    rho: float
    p_value: float
    n_missing_hf_difficulty: int


def scipy_spearmanr_with_p(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Spearman rho + two-sided p-value computed by SciPy (tie-aware).
    """
    try:
        from scipy.stats import spearmanr as _spearmanr  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'scipy' required to compute Spearman rho and p-value. "
            "Install scipy and re-run."
        ) from e

    res = _spearmanr(x, y)
    rho = float(getattr(res, "correlation", res[0]))
    p = float(getattr(res, "pvalue", res[1]))
    return rho, p


def compute_group_rho(
    *,
    name: str,
    items: Sequence[Tuple[str, float]],
    hf_difficulty_by_id: Dict[str, str],
    difficulty_rank_by_label: Dict[str, int],
) -> GroupResult:
    xs: List[float] = []
    ys: List[float] = []
    missing = 0
    for iid, pred in items:
        lab = hf_difficulty_by_id.get(iid)
        if lab is None:
            missing += 1
            continue
        if lab not in difficulty_rank_by_label:
            missing += 1
            continue
        xs.append(float(pred))
        ys.append(float(difficulty_rank_by_label[lab]))
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = int(x.size)
    rho, p = scipy_spearmanr_with_p(x, y)

    return GroupResult(name=name, n=n, rho=float(rho), p_value=float(p), n_missing_hf_difficulty=int(missing))


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to predictions.csv (expects item_id,diff_pred,split including split=zero_success).",
    )

    ap.add_argument("--hf_dataset", type=str, default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--hf_split", type=str, default="test")
    ap.add_argument("--difficulty_col", type=str, default="difficulty")
    ap.add_argument(
        "--difficulty_order",
        type=str,
        default="<15 min fix,15 min - 1 hour,1-4 hours,>4 hours",
        help="Comma-separated difficulty labels from easiest to hardest (used to assign ordinal ranks).",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    order = parse_difficulty_order(str(args.difficulty_order))
    difficulty_rank_by_label = {lab: i for i, lab in enumerate(order)}

    hf_map = load_hf_difficulty_map(dataset_name=str(args.hf_dataset), split=str(args.hf_split), difficulty_col=str(args.difficulty_col))
    preds = load_predictions_csv(str(args.predictions_csv))

    train_items: List[Tuple[str, float]] = [(iid, pred) for iid, pred, split in preds if split == "train"]
    test_items: List[Tuple[str, float]] = [(iid, pred) for iid, pred, split in preds if split == "test"]
    zero_items_from_csv: List[Tuple[str, float]] = [(iid, pred) for iid, pred, split in preds if split == "zero_success"]
    if not zero_items_from_csv:
        raise ValueError(
            "predictions_csv must include rows with split=zero_success. "
            "No fallbacks are supported; add the split to the CSV and re-run."
        )

    results: List[GroupResult] = []
    results.append(
        compute_group_rho(
            name="train",
            items=train_items,
            hf_difficulty_by_id=hf_map,
            difficulty_rank_by_label=difficulty_rank_by_label,
        )
    )
    results.append(
        compute_group_rho(
            name="test",
            items=test_items,
            hf_difficulty_by_id=hf_map,
            difficulty_rank_by_label=difficulty_rank_by_label,
        )
    )
    results.append(
        compute_group_rho(
            name="zero_success",
            items=zero_items_from_csv,
            hf_difficulty_by_id=hf_map,
            difficulty_rank_by_label=difficulty_rank_by_label,
        )
    )

    print(f"predictions_csv: {args.predictions_csv}")
    print(f"hf_dataset: {args.hf_dataset}  split: {args.hf_split}  difficulty_col: {args.difficulty_col}")
    print(f"difficulty_order (easy->hard): {order}")
    for r in results:
        print(
            f"Spearman rho ({r.name}): {r.rho}  "
            f"(p={r.p_value}, n={r.n}, missing_hf_difficulty={r.n_missing_hf_difficulty})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

