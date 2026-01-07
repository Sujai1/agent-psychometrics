#!/usr/bin/env python3
"""
Embed SWE-agent trajectories with Qwen/Qwen2.5-Coder-14B and fit linear regression to
predict per-question difficulty.

Embedding definition (per trajectory):
- Run the trajectory text through the model
- Take the last token embedding from the last hidden state (i.e., pooled from
  `out.last_hidden_state` using attention_mask lengths).

Aggregation (per question / task_id):
- trajectories.jsonl contains multiple trajectories per task_id. We aggregate
  trajectory embeddings by task_id (default: mean).

Outputs:
- embeddings cache (.npz) with per-task embeddings
- regression predictions CSV + metrics JSON

Example:
  /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/python \
    /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/predict_question_difficulty.py \
    --trajectories /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/trajectory_data/trajectories.jsonl \
    --difficulties /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/question_difficulties.csv \
    --backbone Qwen/Qwen2.5-Coder-14B \
    --max_length 1024 \
    --max_chars 12000 \
    --text_sampling tail \
    --batch_size 1 \
    --device_map auto \
    --out_dir /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/qwen25coder14b_lr
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Set


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Please install requirements (see "
            f"`fulcrum/fellowship/trajectory_embedding_requirements.txt`). Original error: {e}"
        ) from e


_require("numpy")
_require("torch")
_require("transformers")
_require("tqdm")
_require("sklearn")

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def sample_trajectory_text(
    trajectory: str,
    *,
    max_chars: int,
    strategy: str,
    rng: random.Random,
) -> str:
    """
    Reduce an extremely long trajectory string to a smaller, representative slice.
    """
    if max_chars <= 0 or len(trajectory) <= max_chars:
        return trajectory

    if strategy == "head":
        return trajectory[:max_chars]
    if strategy == "tail":
        return trajectory[-max_chars:]
    if strategy == "random_span":
        start = rng.randint(0, max(0, len(trajectory) - max_chars))
        return trajectory[start : start + max_chars]
    if strategy == "headtail":
        h = max_chars // 2
        t = max_chars - h
        return trajectory[:h] + "\n\n[...TRUNCATED...]\n\n" + trajectory[-t:]
    if strategy == "headtail_random":
        h = max_chars // 3
        t = max_chars // 3
        m = max_chars - h - t
        mid_start = rng.randint(0, max(0, len(trajectory) - m))
        mid = trajectory[mid_start : mid_start + m]
        return (
            trajectory[:h]
            + "\n\n[...MID-SAMPLE...]\n\n"
            + mid
            + "\n\n[...TRUNCATED...]\n\n"
            + trajectory[-t:]
        )

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    lengths = attention_mask.sum(dim=1).clamp(min=1)  # [B]
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
    return last_hidden_state.gather(dim=1, index=idx).squeeze(1)  # [B, H]


def load_difficulties_csv(path: str) -> Dict[str, float]:
    diffs: Dict[str, float] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if "item_id" not in (r.fieldnames or []) or "diff" not in (r.fieldnames or []):
            raise ValueError(f"Expected columns item_id,diff in {path}; got {r.fieldnames}")
        for row in r:
            diffs[row["item_id"]] = float(row["diff"])
    return diffs


@dataclass
class TrajRecord:
    task_id: str
    text: str


def iter_trajectories_jsonl(
    path: str,
    *,
    max_chars: int,
    text_sampling: str,
    seed: int,
    include_metadata_prefix: bool,
    max_examples: Optional[int],
) -> Iterator[TrajRecord]:
    """
    Stream trajectories from a JSONL file.

    Notes on robustness:
    - In practice, some trajectory builders/logs occasionally emit a line with
      literal control characters (ASCII < 0x20) that make the JSON invalid and
      can crash a long embedding job. We defensively attempt to sanitize such
      lines; if still unparsable, we skip them (with a brief warning).
    """
    def _sanitize_jsonl_line(s: str) -> str:
        # Replace any literal ASCII control characters (0x00-0x1F) with spaces.
        # JSON strings must not contain these characters unescaped; if they show
        # up literally, `json.loads` will raise JSONDecodeError.
        #
        # We keep this intentionally simple: for embedding, losing a few
        # characters is preferable to crashing the full run.
        return "".join((" " if (ord(ch) < 32) else ch) for ch in s)

    rng = random.Random(int(seed))
    yielded = 0
    skipped_bad_json = 0
    sanitized_lines = 0
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if max_examples is not None and yielded >= max_examples:
                return
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                # Try a quick sanitize pass for literal control characters.
                fixed = _sanitize_jsonl_line(line)
                if fixed != line:
                    try:
                        obj = json.loads(fixed)
                        sanitized_lines += 1
                    except json.JSONDecodeError:
                        skipped_bad_json += 1
                        if skipped_bad_json <= 10:
                            print(
                                f"[warn] Skipping invalid JSONL line {line_num} in {path}: {e}",
                                file=sys.stderr,
                            )
                        continue
                else:
                    skipped_bad_json += 1
                    if skipped_bad_json <= 10:
                        print(
                            f"[warn] Skipping invalid JSONL line {line_num} in {path}: {e}",
                            file=sys.stderr,
                        )
                    continue
            task_id = str(obj.get("task_id", ""))
            agent = str(obj.get("agent", ""))
            success = bool(obj.get("success"))
            traj = str(obj.get("trajectory", "") or "")
            text = sample_trajectory_text(traj, max_chars=max_chars, strategy=text_sampling, rng=rng)
            # Some sources may produce empty/whitespace trajectories; skip these to avoid
            # tokenizers producing 0-length sequences (which can crash some models).
            if not text.strip():
                continue
            if include_metadata_prefix:
                text = f"[task_id={task_id}] [agent={agent}] [success={success}]\n\n" + text
            yielded += 1
            yield TrajRecord(task_id=task_id, text=text)

    # Only emit a summary if we encountered issues; keeps normal runs quiet.
    if skipped_bad_json or sanitized_lines:
        print(
            f"[info] iter_trajectories_jsonl: sanitized_lines={sanitized_lines} skipped_bad_json={skipped_bad_json} path={path}",
            file=sys.stderr,
        )


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Backwards compatible with older scikit-learn versions (no `squared=` kwarg).
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def stable_split_ids(ids: Sequence[str], test_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """
    Deterministic split by hashing ids. Returns train indices, test indices.
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1")
    xs: List[Tuple[float, int]] = []
    for i, s in enumerate(ids):
        h = hashlib.md5((str(s) + f"::{seed}").encode("utf-8")).hexdigest()
        x = int(h[:8], 16) / float(16**8)
        xs.append((x, i))
    xs.sort()
    n_test = int(round(len(ids) * float(test_fraction)))
    test = [i for _, i in xs[:n_test]]
    train = [i for _, i in xs[n_test:]]
    return train, test


def embed_trajectories(
    *,
    trajectories_path: str,
    backbone: str,
    trust_remote_code: bool,
    max_length: int,
    max_chars: int,
    text_sampling: str,
    include_metadata_prefix: bool,
    seed: int,
    batch_size: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    max_examples: Optional[int],
    only_task_ids: Optional[Set[str]] = None,
) -> Dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch_dtype
    if dtype == "auto":
        dtype_arg = "auto"
    elif dtype in ("float16", "fp16"):
        dtype_arg = torch.float16
    elif dtype in ("bfloat16", "bf16"):
        dtype_arg = torch.bfloat16
    elif dtype in ("float32", "fp32"):
        dtype_arg = torch.float32
    else:
        raise ValueError(f"Unknown torch_dtype: {torch_dtype}")

    # transformers>=4.57 prefers `dtype=`; `torch_dtype=` is deprecated.
    model_kwargs = {"trust_remote_code": trust_remote_code, "dtype": dtype_arg}
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    if attn_implementation and attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModel.from_pretrained(backbone, **model_kwargs)
    model.eval()
    # If not using device_map, move to a single device.
    if device_map in ("", "none", None):
        model.to(device)
    # Determine where token indices must live (embedding lookup requires same device).
    # This is critical when using `device_map=auto`, where weights are on CUDA but
    # inputs default to CPU unless moved.
    try:
        embed_device = model.get_input_embeddings().weight.device
    except Exception:
        # Fallback: use "device" for single-device models.
        embed_device = device

    # Collect per-task embeddings across trajectories
    per_task: Dict[str, List[np.ndarray]] = {}

    batch_task_ids: List[str] = []
    batch_texts: List[str] = []

    def flush_batch() -> None:
        nonlocal batch_task_ids, batch_texts, per_task
        if not batch_texts:
            return
        # Filter out empty/whitespace samples (defensive).
        pairs = [(tid, txt) for tid, txt in zip(batch_task_ids, batch_texts) if txt.strip()]
        if not pairs:
            batch_task_ids = []
            batch_texts = []
            return
        batch_task_ids = [tid for tid, _ in pairs]
        batch_texts = [txt for _, txt in pairs]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # If the tokenizer produced an empty sequence (rare but possible), skip batch.
        if int(input_ids.shape[1]) == 0:
            batch_task_ids = []
            batch_texts = []
            return
        # Always move inputs to the device where the token embedding matrix lives.
        # (Needed for `device_map=auto` and generally safe otherwise.)
        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = last_token_pool(out.last_hidden_state, attention_mask)
            pooled = pooled.detach().float().cpu().numpy()  # [B, H]

        for task_id, vec in zip(batch_task_ids, pooled):
            per_task.setdefault(task_id, []).append(vec.astype(np.float32, copy=False))

        batch_task_ids = []
        batch_texts = []

    iterator = iter_trajectories_jsonl(
        trajectories_path,
        max_chars=int(max_chars),
        text_sampling=text_sampling,
        seed=int(seed),
        include_metadata_prefix=bool(include_metadata_prefix),
        max_examples=max_examples,
    )

    for rec in tqdm(iterator, desc="embed_trajectories"):
        if only_task_ids is not None and rec.task_id not in only_task_ids:
            continue
        batch_task_ids.append(rec.task_id)
        batch_texts.append(rec.text)
        if len(batch_texts) >= int(batch_size):
            flush_batch()
    flush_batch()

    # Aggregate across trajectories per task_id: mean
    agg: Dict[str, np.ndarray] = {}
    for task_id, vecs in per_task.items():
        m = np.mean(np.stack(vecs, axis=0), axis=0)
        agg[task_id] = m.astype(np.float32, copy=False)
    return agg


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trajectories",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/trajectory_data/trajectories.jsonl",
    )
    p.add_argument(
        "--difficulties",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/question_difficulties.csv",
    )
    p.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-Coder-14B")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_chars", type=int, default=12000)
    p.add_argument(
        "--text_sampling",
        type=str,
        default="tail",
        choices=["head", "tail", "random_span", "headtail", "headtail_random"],
    )
    p.add_argument("--include_metadata_prefix", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_examples", type=int, default=0, help="If >0, cap trajectories for debugging.")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HF device_map (e.g. auto). Use 'none' to force single-device .to(device).",
    )
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="auto", help="e.g. auto, flash_attention_2")

    p.add_argument(
        "--out_dir",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/qwen25coder14b_lr",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--resume_cache",
        action="store_true",
        help=(
            "If set and an embeddings cache exists, reuse cached task embeddings and compute ONLY the missing task_ids, "
            "then rewrite the cache. (Default behavior without this flag is all-or-nothing: if the cache exists, "
            "no new embeddings are computed.)"
        ),
    )

    p.add_argument("--test_fraction", type=float, default=0.2)

    args = p.parse_args(argv)
    ensure_dir(args.out_dir)

    max_examples = None if int(args.max_examples) <= 0 else int(args.max_examples)

    # Cache path is derived from key settings to prevent accidental mismatch reuse.
    safe_backbone = args.backbone.replace("/", "__")
    meta_flag = "meta1" if bool(args.include_metadata_prefix) else "meta0"
    emb_cache = os.path.join(
        args.out_dir,
        f"embeddings__{safe_backbone}__pool-lasttoken__maxlen{int(args.max_length)}__maxchars{int(args.max_chars)}__{args.text_sampling}__{meta_flag}__seed{int(args.seed)}.npz",
    )

    def _scan_trajectory_counts() -> Tuple[Set[str], Dict[str, int], int]:
        """
        Returns:
          - set of task_ids present
          - per-task count of trajectories (JSONL rows) contributing to that task_id
          - total number of yielded trajectory rows
        """
        task_ids_set: Set[str] = set()
        counts: Dict[str, int] = {}
        total = 0
        for rec in iter_trajectories_jsonl(
            args.trajectories,
            max_chars=int(args.max_chars),
            text_sampling=str(args.text_sampling),
            seed=int(args.seed),
            include_metadata_prefix=False,
            max_examples=max_examples,
        ):
            if not rec.task_id:
                continue
            task_ids_set.add(rec.task_id)
            counts[rec.task_id] = counts.get(rec.task_id, 0) + 1
            total += 1
        return task_ids_set, counts, total

    def _load_embeddings_cache(path: str) -> Tuple[List[str], Dict[str, np.ndarray], Optional[Dict[str, int]]]:
        data = np.load(path, allow_pickle=True)
        task_ids_local = list(data["task_ids"].tolist())
        X_local = data["X"].astype(np.float32)
        emb = {tid: X_local[i] for i, tid in enumerate(task_ids_local)}
        if "counts" in data:
            counts_arr = data["counts"].astype(np.int64)
            counts = {tid: int(counts_arr[i]) for i, tid in enumerate(task_ids_local)}
        else:
            counts = None
        return task_ids_local, emb, counts

    if os.path.exists(emb_cache) and not args.overwrite and not args.resume_cache:
        data = np.load(emb_cache, allow_pickle=True)
        task_ids = list(data["task_ids"].tolist())
        X = data["X"].astype(np.float32)
        embeddings = {tid: X[i] for i, tid in enumerate(task_ids)}
        print(f"Loaded embeddings cache: {emb_cache} ({len(embeddings)} tasks)")
    else:
        cached_task_ids: List[str] = []
        cached: Dict[str, np.ndarray] = {}
        cached_counts: Optional[Dict[str, int]] = None
        if os.path.exists(emb_cache) and not args.overwrite and args.resume_cache:
            cached_task_ids, cached, cached_counts = _load_embeddings_cache(emb_cache)
            print(f"Loaded embeddings cache (resume enabled): {emb_cache} ({len(cached)} tasks)")

        # Scan trajectories to detect:
        # - missing task_ids (cache is partial)
        # - changed per-task trajectory counts (cache is stale, e.g. trajectories JSONL was regenerated)
        traj_task_ids: Set[str] = set()
        traj_counts: Dict[str, int] = {}
        traj_total = 0
        try:
            traj_task_ids, traj_counts, traj_total = _scan_trajectory_counts()
        except Exception as e:
            print(f"[warn] Failed to scan trajectories for task_ids/counts ({args.trajectories}): {e}", file=sys.stderr)
            traj_task_ids, traj_counts, traj_total = set(), {}, 0

        to_compute: Optional[Set[str]] = None
        if cached and traj_task_ids and args.resume_cache:
            if cached_counts is None:
                print(
                    "[warn] Existing embeddings cache has no per-task counts metadata; treating as stale and recomputing.",
                    file=sys.stderr,
                )
                to_compute = set(traj_task_ids)
            else:
                changed_or_missing = set()
                for tid in traj_task_ids:
                    if tid not in cached:
                        changed_or_missing.add(tid)
                        continue
                    if cached_counts.get(tid) != traj_counts.get(tid):
                        changed_or_missing.add(tid)
                to_compute = changed_or_missing
        elif cached and args.resume_cache:
            # Couldn't scan; safest is to recompute everything.
            to_compute = None

        if cached and args.resume_cache and to_compute is not None and len(to_compute) == 0 and traj_task_ids:
            # Cache is up-to-date for the scanned trajectories (same task set + per-task counts).
            # Restrict to the current scanned task ids (avoid carrying over old/extra entries).
            embeddings = {tid: cached[tid] for tid in sorted(traj_task_ids) if tid in cached}
            task_ids = sorted(embeddings.keys())
            X = np.stack([embeddings[t] for t in task_ids], axis=0).astype(np.float32)
            print(f"Embeddings cache up-to-date for scanned trajectories: {emb_cache} ({len(task_ids)} tasks)")
        else:
            # Compute embeddings:
            # - if resume_cache: compute only `to_compute` (or all if scan failed)
            # - else: compute all (overwrite or cache missing)
            embeddings_new = embed_trajectories(
                trajectories_path=args.trajectories,
                backbone=args.backbone,
                trust_remote_code=bool(args.trust_remote_code),
                max_length=int(args.max_length),
                max_chars=int(args.max_chars),
                text_sampling=str(args.text_sampling),
                include_metadata_prefix=bool(args.include_metadata_prefix),
                seed=int(args.seed),
                batch_size=int(args.batch_size),
                device_map=str(args.device_map),
                torch_dtype=str(args.torch_dtype),
                attn_implementation=str(args.attn_implementation),
                max_examples=max_examples,
                only_task_ids=(to_compute if args.resume_cache else None),
            )
            embeddings = dict(cached) if (cached and args.resume_cache) else {}
            embeddings.update(embeddings_new)

            # If we successfully scanned, restrict cache to exactly the current tasks.
            if traj_task_ids:
                embeddings = {tid: embeddings[tid] for tid in traj_task_ids if tid in embeddings}
                task_ids = sorted(embeddings.keys())
                counts_arr = np.array([int(traj_counts.get(tid, 0)) for tid in task_ids], dtype=np.int64)
            else:
                task_ids = sorted(embeddings.keys())
                counts_arr = np.array([], dtype=np.int64)

            X = np.stack([embeddings[t] for t in task_ids], axis=0).astype(np.float32)
            meta_path = np.array([str(args.trajectories)], dtype=object)
            meta_total = np.array([int(traj_total)], dtype=np.int64)
            np.savez_compressed(
                emb_cache,
                task_ids=np.array(task_ids, dtype=object),
                X=X,
                counts=counts_arr,
                trajectories_path=meta_path,
                trajectories_total_rows=meta_total,
            )
            print(f"Wrote embeddings cache: {emb_cache} ({len(task_ids)} tasks, dim={X.shape[1]})")

    diffs = load_difficulties_csv(args.difficulties)

    # Align X with y by item_id / task_id
    common = [tid for tid in task_ids if tid in diffs]
    missing_diff = [tid for tid in task_ids if tid not in diffs]
    if missing_diff:
        print(f"WARNING: {len(missing_diff)} task_ids missing difficulty; ignoring (e.g. {missing_diff[:3]})")

    Xy = np.stack([embeddings[tid] for tid in common], axis=0).astype(np.float32)
    y = np.array([diffs[tid] for tid in common], dtype=np.float32)

    # Deterministic split
    train_idx, test_idx = stable_split_ids(common, test_fraction=float(args.test_fraction), seed=int(args.seed))
    X_train, y_train = Xy[train_idx], y[train_idx]
    X_test, y_test = Xy[test_idx], y[test_idx]

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    yhat_train = lr.predict(X_train).astype(np.float64)
    yhat_test = lr.predict(X_test).astype(np.float64)
    yhat_all = lr.predict(Xy).astype(np.float64)

    metrics = {
        "n_tasks_total": int(len(task_ids)),
        "n_tasks_with_difficulty": int(len(common)),
        "embedding_dim": int(Xy.shape[1]),
        "train_fraction": float(1.0 - args.test_fraction),
        "test_fraction": float(args.test_fraction),
        "seed": int(args.seed),
        "train_r2": float(r2_score(y_train, yhat_train)),
        "test_r2": float(r2_score(y_test, yhat_test)),
        "train_rmse": float(_rmse(y_train, yhat_train)),
        "test_rmse": float(_rmse(y_test, yhat_test)),
        "train_pearson": float(_pearsonr(y_train, yhat_train)),
        "test_pearson": float(_pearsonr(y_test, yhat_test)),
        "backbone": str(args.backbone),
        "pooling": "last_token_of_last_hidden_state",
        "max_length": int(args.max_length),
        "max_chars": int(args.max_chars),
        "text_sampling": str(args.text_sampling),
        "include_metadata_prefix": bool(args.include_metadata_prefix),
        "batch_size": int(args.batch_size),
        "device_map": str(args.device_map),
        "torch_dtype": str(args.torch_dtype),
        "attn_implementation": str(args.attn_implementation),
        "embeddings_cache": emb_cache,
    }

    save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

    # Write per-item predictions
    split_set = set(test_idx)
    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_true", "diff_pred", "split"])
        w.writeheader()
        for i, tid in enumerate(common):
            w.writerow(
                {
                    "item_id": tid,
                    "diff_true": float(y[i]),
                    "diff_pred": float(yhat_all[i]),
                    "split": "test" if i in split_set else "train",
                }
            )

    print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
    print(f"Wrote predictions: {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

