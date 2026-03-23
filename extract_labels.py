#!/usr/bin/env python3
"""Extract training labels for complexity classifier from existing project assets.

Output format (JSONL):
{"input": "...", "output": "{...}"}

The output JSON string contains:
- complexity_score (0-10)
- tier (L1/L2/L3)
- reasoning
- suggested_agents
- risk_flags
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY_GLOB = str(ROOT / "Kickstarter_Clean" / "full_prediction_summary_*.csv")
DEFAULT_INTENT_FILE = ROOT / "config" / "m7_intent_library.json"
DEFAULT_OUTPUT_FILE = ROOT / "Kickstarter_Clean" / "complexity_labels.jsonl"


def pick_latest(pattern: str) -> Path:
    candidates = [Path(p) for p in glob.glob(pattern)]
    if not candidates:
        raise FileNotFoundError(f"No files match: {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_intent_library(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def parse_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float((row.get(key) or "").strip())
    except Exception:
        return default


def parse_missing_fields(raw: str) -> List[str]:
    txt = (raw or "").strip()
    if not txt:
        return []
    if txt in {"none", "null", "na", "n/a", "无缺失"}:
        return []
    if "|" in txt:
        return [x.strip() for x in txt.split("|") if x.strip()]
    if "," in txt:
        return [x.strip() for x in txt.split(",") if x.strip()]
    return [txt]


def risk_to_complexity_score(goal_ratio: float, time_penalty: float, category_risk: float, combined_risk: float, missing_count: int) -> float:
    # Rule-inspired score using existing engineered risk features.
    score = 2.0
    score += min(3.0, goal_ratio / 2.0)
    score += min(2.0, time_penalty / 2.0)
    score += min(2.0, category_risk * 2.5)
    score += min(2.0, combined_risk / 5.0)
    score += min(1.0, 0.4 * missing_count)
    return round(max(0.0, min(10.0, score)), 2)


def score_to_tier(score: float) -> str:
    if score <= 3.5:
        return "L1"
    if score <= 6.5:
        return "L2"
    return "L3"


def base_agents_for_tier(tier: str) -> List[str]:
    if tier == "L1":
        return ["general_agent"]
    if tier == "L2":
        return ["legal_agent", "finance_agent", "tech_agent"]
    return ["legal_agent", "finance_agent", "tech_agent", "general_agent"]


def apply_intent_keyword_boost(project_text: str, intent_library: Sequence[Dict[str, object]], agents: List[str]) -> List[str]:
    boosted = list(agents)
    lower_text = (project_text or "").lower()

    expert_map = {
        "risk_guardian": "legal_agent",
        "finance_advisor": "finance_agent",
        "ops_executor": "general_agent",
        "growth_strategist": "general_agent",
    }

    for item in intent_library:
        keywords = item.get("keywords", [])
        required = item.get("default_required_experts", [])
        if not isinstance(keywords, list) or not isinstance(required, list):
            continue

        if any(str(k).lower() in lower_text for k in keywords):
            for expert in required:
                mapped = expert_map.get(str(expert))
                if mapped and mapped not in boosted:
                    boosted.append(mapped)

    return boosted


def build_risk_flags(goal_ratio: float, time_penalty: float, category_risk: float, country_factor: float, urgency_score: float, missing_fields: List[str]) -> List[str]:
    flags: List[str] = []

    if goal_ratio > 1.8:
        flags.append("high_goal_ratio")
    if time_penalty > 2.4:
        flags.append("long_duration_penalty")
    if category_risk > 0.32:
        flags.append("high_category_risk")
    if country_factor > 0.26:
        flags.append("country_risk")
    if urgency_score > 0.10:
        flags.append("urgency_risk")
    if missing_fields:
        flags.append("missing_fields")

    return flags


def build_reasoning(row: Dict[str, str], score: float, tier: str, flags: List[str]) -> str:
    project_name = row.get("Project Name", "")
    parts = [
        f"project={project_name}",
        f"score={score}",
        f"tier={tier}",
        f"flags={','.join(flags) if flags else 'none'}",
    ]
    return "; ".join(parts)


def row_to_record(row: Dict[str, str], intent_library: Sequence[Dict[str, object]]) -> Dict[str, str]:
    name = (row.get("Project Name") or "").strip()
    category = (row.get("Main Category") or "").strip()
    country = (row.get("Country") or "").strip()
    goal = parse_float(row, "Funding Goal (USD)")
    duration = parse_float(row, "Funding Duration (days)")

    goal_ratio = parse_float(row, "goal_ratio")
    time_penalty = parse_float(row, "time_penalty")
    category_risk = parse_float(row, "category_risk")
    combined_risk = parse_float(row, "combined_risk")
    country_factor = parse_float(row, "country_factor")
    urgency_score = parse_float(row, "urgency_score")
    missing_fields = parse_missing_fields(row.get("Missing Fields", ""))

    score = risk_to_complexity_score(
        goal_ratio=goal_ratio,
        time_penalty=time_penalty,
        category_risk=category_risk,
        combined_risk=combined_risk,
        missing_count=len(missing_fields),
    )
    tier = score_to_tier(score)

    risk_flags = build_risk_flags(
        goal_ratio=goal_ratio,
        time_penalty=time_penalty,
        category_risk=category_risk,
        country_factor=country_factor,
        urgency_score=urgency_score,
        missing_fields=missing_fields,
    )

    agents = base_agents_for_tier(tier)
    keyword_text = f"{name} {category}"
    agents = apply_intent_keyword_boost(keyword_text, intent_library, agents)

    input_text = (
        f"Project: {name}. Category: {category}. Country: {country}. "
        f"GoalUSD: {goal:.2f}. DurationDays: {duration:.0f}."
    )

    output_payload = {
        "complexity_score": score,
        "tier": tier,
        "reasoning": build_reasoning(row, score, tier, risk_flags),
        "suggested_agents": agents,
        "risk_flags": risk_flags,
    }

    return {
        "input": input_text,
        "output": json.dumps(output_payload, ensure_ascii=False),
    }


def convert(summary_csv: Path, intent_file: Path, output_jsonl: Path, limit: int = 0) -> int:
    intent_library = load_intent_library(intent_file)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with summary_csv.open("r", encoding="utf-8", newline="") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        for row in reader:
            record = row_to_record(row, intent_library)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if limit > 0 and count >= limit:
                break

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract complexity labels from existing risk/routing assets.")
    parser.add_argument("--summary", default="", help="Path to full_prediction_summary CSV. Default: latest file")
    parser.add_argument("--intent", default=str(DEFAULT_INTENT_FILE), help="Path to m7_intent_library.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE), help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Optional max records for quick run")
    args = parser.parse_args()

    summary_csv = Path(args.summary) if args.summary else pick_latest(DEFAULT_SUMMARY_GLOB)
    intent_file = Path(args.intent)
    output_jsonl = Path(args.output)

    count = convert(summary_csv=summary_csv, intent_file=intent_file, output_jsonl=output_jsonl, limit=args.limit)
    print(f"[done] summary={summary_csv}")
    print(f"[done] output={output_jsonl}")
    print(f"[done] records={count}")


if __name__ == "__main__":
    main()
