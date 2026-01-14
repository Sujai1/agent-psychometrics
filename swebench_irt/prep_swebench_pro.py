"""
Prepare agent-runs CSV into JSONL for IRT training.

This mirrors the response-matrix JSONL format used by `swebench_irt/prep_swebench.py`:

  {"subject_id": "<model_or_agent_name>", "responses": {"<task_id>": 0/1, ...}}

Expected CSV columns (as exported from the agent runs table):
  - metadata.model_name
  - metadata.instance_id
  - metadata.resolved

We also "stabilize" the task ids by default to avoid version/hash suffix churn:
  - strip leading "instance_"
  - strip trailing "-v..." (covers "-vc<hash>" and "-vnan")
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]

_V_SUFFIX_RE = re.compile(r"-v.*$")  # strip "-v..." (includes "-vc<hash>" and "-vnan")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def clean_instance_id(
    raw_instance_id: str,
    *,
    strip_instance_prefix: bool = True,
    strip_version_suffix: bool = True,
) -> str:
    s = (raw_instance_id or "").strip()
    if strip_instance_prefix and s.startswith("instance_"):
        s = s[len("instance_") :]
    if strip_version_suffix:
        s = _V_SUFFIX_RE.sub("", s)
    return s


def _parse_bool(raw: str) -> int:
    s = (raw or "").strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return 1
    if s in {"false", "0", "no", "n", "f"}:
        return 0
    # Be strict to avoid silently corrupting labels.
    raise ValueError(f"Unrecognized boolean value: {raw!r}")


def _iter_rows(csv_path: Path) -> Iterable[Dict[str, str]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def collect_from_csv(
    csv_path: Path,
    *,
    subject_col: str,
    item_col: str,
    outcome_col: str,
    strip_instance_prefix: bool,
    strip_version_suffix: bool,
    dedupe: str,
) -> Tuple[Dict[str, Dict[str, int]], set[str], int]:
    """
    Returns:
      - subject_id -> {item_id -> 0/1}
      - all_items (set)
      - duplicate (subject,item) count encountered
    """
    responses: Dict[str, Dict[str, int]] = defaultdict(dict)
    all_items: set[str] = set()
    dupes = 0

    if dedupe not in {"last", "max"}:
        raise ValueError(f"Unsupported --dedupe {dedupe!r} (expected 'last' or 'max')")

    for row in _iter_rows(csv_path):
        if subject_col not in row or item_col not in row or outcome_col not in row:
            missing = [c for c in (subject_col, item_col, outcome_col) if c not in row]
            raise KeyError(f"CSV missing required columns: {missing}. Found: {sorted(row.keys())}")

        subject_id = (row[subject_col] or "").strip()
        raw_item = row[item_col] or ""
        item_id = clean_instance_id(
            raw_item,
            strip_instance_prefix=strip_instance_prefix,
            strip_version_suffix=strip_version_suffix,
        )
        if not subject_id or not item_id:
            # Skip malformed rows rather than creating empty ids.
            continue

        val = _parse_bool(row[outcome_col])

        existing = responses[subject_id].get(item_id)
        if existing is not None:
            dupes += 1
            if dedupe == "max":
                val = max(existing, val)
        responses[subject_id][item_id] = val
        all_items.add(item_id)

    return dict(responses), all_items, dupes


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare agent-runs CSV response JSONL")
    p.add_argument(
        "--csv_path",
        type=str,
        default="agent-runs-032fb63d-4992-4bfc-911d-3b7dafcb931f.csv",
        help="Path to agent-runs CSV",
    )
    p.add_argument(
        "--subject_col",
        type=str,
        default="metadata.model_name",
        help="CSV column containing subject_id (default: metadata.model_name)",
    )
    p.add_argument(
        "--item_col",
        type=str,
        default="metadata.instance_id",
        help="CSV column containing item_id (default: metadata.instance_id)",
    )
    p.add_argument(
        "--outcome_col",
        type=str,
        default="metadata.resolved",
        help="CSV column containing binary outcome (default: metadata.resolved)",
    )
    p.add_argument(
        "--model_regex",
        type=str,
        default=None,
        help="Only include subjects whose subject_id matches this regex (applied after parsing).",
    )
    p.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Limit number of subjects after sorting by subject_id",
    )
    p.add_argument(
        "--no_complete_matrix",
        action="store_true",
        help="Don't fill missing tasks (sparse matrix). Default is to fill with 0.",
    )
    p.add_argument(
        "--dedupe",
        type=str,
        default="max",
        choices=["max", "last"],
        help="How to combine duplicate (subject_id,item_id) rows (default: max)",
    )
    p.add_argument(
        "--no_strip_instance_prefix",
        action="store_true",
        help="Keep leading 'instance_' in metadata.instance_id",
    )
    p.add_argument(
        "--no_strip_version_suffix",
        action="store_true",
        help="Keep trailing '-v...' suffix in metadata.instance_id",
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="clean_data/agent_runs/agent_runs.jsonl",
        help="Output JSONL path",
    )
    args = p.parse_args()

    csv_path = resolve_path(args.csv_path)
    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subject_responses, all_items, dupes = collect_from_csv(
        csv_path,
        subject_col=args.subject_col,
        item_col=args.item_col,
        outcome_col=args.outcome_col,
        strip_instance_prefix=not args.no_strip_instance_prefix,
        strip_version_suffix=not args.no_strip_version_suffix,
        dedupe=args.dedupe,
    )

    if not subject_responses:
        raise ValueError(f"No responses found in {csv_path}")

    selected_subjects = sorted(subject_responses.keys())
    if args.model_regex is not None:
        rx = re.compile(args.model_regex)
        selected_subjects = [s for s in selected_subjects if rx.search(s)]

    if args.max_subjects is not None:
        selected_subjects = selected_subjects[: args.max_subjects]

    items_list = sorted(all_items)

    records = []
    summary = []
    for subject_id in selected_subjects:
        responses = subject_responses.get(subject_id, {})
        if not responses and args.no_complete_matrix:
            summary.append((subject_id, 0, 0))
            continue
        if not args.no_complete_matrix:
            responses = {iid: int(responses.get(iid, 0)) for iid in items_list}
        resolved_ct = sum(responses.values())
        summary.append((subject_id, len(responses), resolved_ct))
        records.append({"subject_id": subject_id, "responses": responses})

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")

    print(f"Wrote {len(records)} subjects to {output_path}")
    print(f"Unique tasks observed: {len(all_items)}")
    obs_total = sum(len(r["responses"]) for r in records)
    if records:
        counts = [len(r["responses"]) for r in records]
        print(f"Total observations: {obs_total}")
        print(f"Tasks per subject: min={min(counts)} max={max(counts)} mean={obs_total / len(counts):.2f}")
        print(f"Complete matrix: {'no (sparse)' if args.no_complete_matrix else 'yes'}")
    if dupes:
        print(f"Duplicate (subject_id,item_id) rows encountered: {dupes} (dedupe={args.dedupe})")
    if summary:
        empty = sum(1 for _name, count, _res in summary if count == 0)
        print(f"Subjects with no responses: {empty}")


if __name__ == "__main__":
    main()

