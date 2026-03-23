#!/usr/bin/env python3
"""Sample tier records from complexity label JSONL.

Default behavior:
- Read: Kickstarter_Clean/complexity_labels.jsonl
- Sample: 10 records for L1 and 10 records for L3
- Write: Kickstarter_Clean/complexity_labels_sample_L1_L3.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_record(line: str) -> Dict[str, object]:
    outer = json.loads(line)
    output_raw = outer.get("output", "{}")
    output_obj = output_raw if isinstance(output_raw, dict) else json.loads(str(output_raw))
    outer["output"] = output_obj
    return outer


def collect_by_tier(input_path: Path) -> Dict[str, List[Dict[str, object]]]:
    buckets: Dict[str, List[Dict[str, object]]] = {"L1": [], "L3": []}
    with input_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = parse_record(line)
            except Exception:
                continue

            output_obj = record.get("output", {})
            if not isinstance(output_obj, dict):
                continue

            tier = str(output_obj.get("tier", "")).strip().upper()
            if tier in buckets:
                buckets[tier].append(record)

    return buckets


def sample_tiers(input_path: Path, output_path: Path, per_tier: int, seed: int | None = None) -> Dict[str, int]:
    if seed is not None:
        random.seed(seed)

    buckets = collect_by_tier(input_path)

    selected: List[Dict[str, object]] = []
    stats: Dict[str, int] = {}
    for tier in ("L1", "L3"):
        pool = buckets[tier]
        take_n = min(per_tier, len(pool))
        stats[tier] = take_n
        if take_n > 0:
            selected.extend(random.sample(pool, take_n))

    random.shuffle(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats["total"] = len(selected)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample L1 and L3 tier records from complexity labels JSONL.")
    parser.add_argument(
        "--input",
        default="Kickstarter_Clean/complexity_labels.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="Kickstarter_Clean/complexity_labels_sample_L1_L3.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--per-tier",
        type=int,
        default=10,
        help="Number of records to sample for each tier (L1 and L3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    stats = sample_tiers(
        input_path=input_path,
        output_path=output_path,
        per_tier=max(1, args.per_tier),
        seed=args.seed,
    )
    print(f"[done] input={input_path}")
    print(f"[done] output={output_path}")
    print(f"[done] sampled L1={stats.get('L1', 0)} L3={stats.get('L3', 0)} total={stats.get('total', 0)}")


if __name__ == "__main__":
    main()
