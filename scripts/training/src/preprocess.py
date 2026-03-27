import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

from utils import load_yaml, resolve_existing_path


REQUIRED_OUTPUT_KEYS = {
    "tier",
    "sub_type",
    "suggested_agents",
    "parallelism",
    "confidence_score",
}


def _extract_sub_type(user_input: str) -> str:
    match = re.search(r"Category:\s*([^\.]+)\.", user_input)
    if match:
        return match.group(1).strip().lower()
    return "unknown"


def _compute_parallelism(tier: str, risk_flags: List[str]) -> int:
    tier_parallelism = {"L1": 1, "L2": 2, "L3": 3}
    value = tier_parallelism.get(tier, 1)
    if len(risk_flags) > 2:
        value += 1
    return value


def _build_structured_output(parsed_output: Dict[str, Any], sub_type: str) -> Dict[str, Any]:
    tier = parsed_output.get("tier", "L1")
    risk_flags = parsed_output.get("risk_flags", []) or []
    complexity_score = float(parsed_output.get("complexity_score", 5.0))

    return {
        "tier": tier,
        "sub_type": sub_type,
        "suggested_agents": parsed_output.get("suggested_agents", ["general_agent"]),
        "parallelism": _compute_parallelism(tier, risk_flags),
        "confidence_score": round(max(0.0, min(1.0, complexity_score / 10.0)), 3),
    }


def _validate_output_format(output_obj: Dict[str, Any]) -> bool:
    if not REQUIRED_OUTPUT_KEYS.issubset(set(output_obj.keys())):
        return False

    if output_obj["tier"] not in {"L1", "L2", "L3"}:
        return False

    if not isinstance(output_obj["suggested_agents"], list):
        return False

    if not isinstance(output_obj["parallelism"], int) or output_obj["parallelism"] < 1:
        return False

    if not isinstance(output_obj["confidence_score"], (float, int)):
        return False

    # Ensure tokenizer-friendly JSON serialization and parsing.
    serialized = json.dumps(output_obj, ensure_ascii=False)
    try:
        json.loads(serialized)
    except json.JSONDecodeError:
        return False

    return True


def _is_within_length_limit(tokenizer, user_input: str, output_json_str: str, max_seq_length: int) -> bool:
    prompt = (
        "<|im_start|>system\n路由决策器<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{output_json_str}<|im_end|>"
    )
    tokenized = tokenizer(prompt, add_special_tokens=False)
    return len(tokenized["input_ids"]) <= max_seq_length


def _write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    input_path = resolve_existing_path(cfg["data"]["input_labels"], anchor_file=__file__)
    output_dir = Path(cfg["data"]["processed_dir"])
    seed = int(cfg["data"].get("split_seed", 42))
    max_seq_length = int(cfg["training"]["max_seq_length"])

    model_path = resolve_existing_path(cfg["model"]["model_name_or_path"], anchor_file=__file__)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cleaned_rows: List[Dict[str, str]] = []
    skipped_format = 0
    skipped_length = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                user_input = record["input"]
                parsed_output = json.loads(record["output"])
            except Exception:
                skipped_format += 1
                continue

            sub_type = _extract_sub_type(user_input)
            structured_output = _build_structured_output(parsed_output, sub_type)

            if not _validate_output_format(structured_output):
                skipped_format += 1
                continue

            output_json_str = json.dumps(structured_output, ensure_ascii=False)
            if not _is_within_length_limit(tokenizer, user_input, output_json_str, max_seq_length):
                skipped_length += 1
                continue

            cleaned_rows.append({"input": user_input, "output": output_json_str})

    random.seed(seed)
    random.shuffle(cleaned_rows)

    total = len(cleaned_rows)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_rows = cleaned_rows[:train_end]
    val_rows = cleaned_rows[train_end:val_end]
    test_rows = cleaned_rows[val_end:]

    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "val.jsonl", val_rows)
    _write_jsonl(output_dir / "test.jsonl", test_rows)

    print(f"Total kept: {total}")
    print(f"Skipped(format): {skipped_format}")
    print(f"Skipped(length>{max_seq_length}): {skipped_length}")
    print(
        "Split sizes -> "
        f"train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}"
    )


if __name__ == "__main__":
    main()
