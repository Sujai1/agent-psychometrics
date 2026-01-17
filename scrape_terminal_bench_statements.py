#!/usr/bin/env python3
"""
Extract Terminal-Bench 2.0 task statements ("Instruction") from the public registry
and write a SWE-bench-like JSONL for embedding scripts.

Output JSONL schema per line:
  {"task_id": "...", "problem_statement": "...", "patch": ""}

Source pages:
  - https://www.tbench.ai/registry/terminal-bench/2.0
  - https://www.tbench.ai/registry/terminal-bench/2.0/<task_id>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import List, Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup  # type: ignore[import-not-found]
import requests  # type: ignore[import-not-found]


BASE = "https://www.tbench.ai"
INDEX_PATH = "/registry/terminal-bench/2.0"
# Task ids are mostly kebab-case, but a few include dots (e.g. "install-windows-3.11").
TASK_URL_RE = re.compile(r"^/registry/terminal-bench/2\.0/([a-z0-9][a-z0-9\-.]*)/?$", re.I)


def fetch(url: str, *, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "tb2-statement-extractor/1.0"})
    r.raise_for_status()
    return r.text


def discover_task_ids(index_html: str) -> List[str]:
    # The index contains many links; task links look like:
    #   /registry/terminal-bench/2.0/<task-id>
    # We extract all hrefs and filter by pattern.
    hrefs = re.findall(r'href="([^"]+)"', index_html)
    out = []
    seen = set()
    for h in hrefs:
        m = TASK_URL_RE.match(h.strip())
        if not m:
            continue
        tid = m.group(1)
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def strip_tags(s: str) -> str:
    # very small HTML-to-text helper (good enough for these pages)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    # unescape common entities
    s = (
        s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return s


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Trim trailing whitespace per-line and collapse excessive blank lines.
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def extract_instruction(task_html: str) -> Optional[str]:
    """
    Extract text after "### Instruction" up to the next "### " section.
    The registry pages render headings like "### Instruction" and "### Tags".
    """
    # Newer registry pages render something like:
    #   <h3 ...>Instruction</h3>
    #   <p ...>...</p>
    # rather than literal "### Instruction" markdown in the HTML source.
    # Prefer a structured parse first.
    soup = BeautifulSoup(task_html, "html.parser")
    heading = soup.find(
        lambda t: getattr(t, "name", None) in {"h1", "h2", "h3", "h4"}
        and t.get_text(strip=True).lower() == "instruction"
    )
    if heading:
        chunks: List[str] = []
        # Usually the instruction body is in the heading's sibling <p>/<div> nodes.
        for sib in heading.find_next_siblings():
            if getattr(sib, "name", None) in {"h1", "h2", "h3", "h4"}:
                break
            text = sib.get_text("\n", strip=False)
            text = normalize_text(text)
            if text:
                chunks.append(text)
        if chunks:
            return normalize_text("\n\n".join(chunks))

    # Try a few patterns because the page is simple but could vary slightly.
    patterns: List[re.Pattern] = [
        re.compile(r"###\s*Instruction\s*(.*?)\s*###\s*Tags", re.S | re.I),
        re.compile(r"###\s*Instruction\s*(.*?)\s*Created by", re.S | re.I),
    ]
    for pat in patterns:
        m = pat.search(task_html)
        if m:
            chunk = m.group(1)
            text = strip_tags(chunk).strip()
            return normalize_text(text) if text else None

    # Fallback: sometimes the content is already mostly text in the HTML;
    # try searching in the tag-stripped full page.
    full = strip_tags(task_html)
    m2 = re.search(r"###\s*Instruction\s*(.*?)\s*###\s*Tags", full, flags=re.S | re.I)
    if m2:
        t = m2.group(1).strip()
        return normalize_text(t) if t else None

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write JSONL, e.g. terminal_bench_2_statements.jsonl")
    ap.add_argument("--sleep", type=float, default=0.10, help="Sleep between task page fetches (seconds)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only extract this many tasks (debugging)")
    args = ap.parse_args()

    index_url = urljoin(BASE, INDEX_PATH)
    index_html = fetch(index_url)
    task_ids = discover_task_ids(index_html)

    if not task_ids:
        raise RuntimeError(f"Found 0 task ids on {index_url}. The page structure may have changed.")

    if args.limit and args.limit > 0:
        task_ids = task_ids[: int(args.limit)]

    records = []
    missing: List[str] = []

    for i, tid in enumerate(task_ids, start=1):
        task_url = urljoin(BASE, f"{INDEX_PATH}/{tid}")
        html = fetch(task_url)
        instr = extract_instruction(html)
        if not instr:
            missing.append(tid)
            continue
        records.append({"task_id": tid, "problem_statement": instr, "patch": ""})
        if args.sleep > 0:
            time.sleep(float(args.sleep))

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} tasks to {args.out}")
    if missing:
        print(f"WARNING: failed to extract instruction for {len(missing)} tasks, e.g. {missing[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
