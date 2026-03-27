import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover
    repair_json = None


def load_yaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_existing_path(relative_or_absolute_path: str, anchor_file: Optional[str] = None) -> Path:
    """Resolve a path robustly across script cwd differences while keeping relative paths."""
    raw = Path(relative_or_absolute_path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [Path.cwd() / raw]
    if anchor_file:
        anchor = Path(anchor_file).resolve().parent
        candidates.extend(
            [
                anchor / raw,
                anchor.parent / raw,
                anchor.parent.parent / raw,
                anchor.parent.parent.parent / raw,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve path '{relative_or_absolute_path}'. Tried: {', '.join(str(c) for c in candidates)}"
    )


def extract_json_candidate(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        return match.group(0)
    return text.strip()


def repair_json_output(raw_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    candidate = extract_json_candidate(raw_text)

    try:
        return json.loads(candidate), candidate
    except Exception:
        pass

    if repair_json is not None:
        try:
            repaired = repair_json(candidate)
            return json.loads(repaired), repaired
        except Exception:
            pass

    # Last fallback: trim to outermost braces and retry.
    left = candidate.find("{")
    right = candidate.rfind("}")
    if left != -1 and right != -1 and right > left:
        sliced = candidate[left : right + 1]
        try:
            return json.loads(sliced), sliced
        except Exception:
            return None, sliced

    return None, candidate


def load_qwen3_with_lora(base_model_path: str, lora_dir: str, anchor_file: Optional[str] = None):
    model_path = resolve_existing_path(base_model_path, anchor_file=anchor_file)
    lora_path = resolve_existing_path(lora_dir, anchor_file=anchor_file)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    merged_model = PeftModel.from_pretrained(base_model, str(lora_path))
    merged_model.eval()
    return tokenizer, merged_model
