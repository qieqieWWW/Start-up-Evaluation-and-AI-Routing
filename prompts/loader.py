"""
Centralized prompt loader.

All LLM prompts are stored as JSON files under this directory.
Code should never hardcode prompt text — always load via this module.

Usage:
    from prompts.loader import load_prompt, load_prompt_template, load_prompt_dict
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

_PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=64)
def _read_json(rel_path: str) -> Any:
    """Read a JSON prompt file with caching."""
    path = _PROMPTS_DIR / rel_path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(rel_path: str) -> str:
    """Load a prompt stored as a plain string or a dict with a 'content' key.

    Usage:
        prompt = load_prompt("m7/blender.json")          # plain string
        prompt = load_prompt("m7/some_block.json")       # {"content": "..."}
    """
    data = _read_json(rel_path)
    if isinstance(data, str):
        return data
    if isinstance(data, dict) and "content" in data:
        return data["content"]
    if isinstance(data, dict) and "system_prompt" in data:
        return data["system_prompt"]
    raise ValueError(
        f"Prompt file {rel_path} has unexpected structure. "
        f"Expected a string or dict with 'content'/'system_prompt' key."
    )


def load_prompt_dict(rel_path: str) -> dict:
    """Load a prompt file that contains structured data (few-shot, schema, metadata).

    Usage:
        experts = load_prompt_dict("experts/risk_guardian.json")
        hints   = load_prompt_dict("m7/search_hints.json")["hints"]
    """
    data = _read_json(rel_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {rel_path}, got {type(data).__name__}")
    return data


def load_prompt_template(rel_path: str) -> str:
    """Load a template prompt string that will be .format()'ed at runtime.

    The template is stored in the 'template' field of the JSON object.

    Usage:
        tpl = load_prompt_template("m7/system_prompt.json")
        result = tpl.format(expert_role="...", expert_name="...", ...)
    """
    data = _read_json(rel_path)
    template: Optional[str] = None
    if isinstance(data, dict):
        template = data.get("template")
    if not template:
        raise ValueError(f"Prompt file {rel_path} missing 'template' field")
    return template


def load_prompt_metadata(rel_path: str) -> Optional[dict]:
    """Load only the metadata block from a prompt file."""
    data = _read_json(rel_path)
    if isinstance(data, dict):
        return data.get("metadata")
    return None


def load_prompts_from_dir(rel_dir: str, key_by: str = "stem") -> Dict[str, dict]:
    """Load all JSON files in a directory, keyed by filename stem or 'name' field.

    Usage:
        experts = load_prompts_from_dir("experts")
        # Returns {"risk_guardian": {...}, "finance_advisor": {...}, ...}
    """
    results: Dict[str, dict] = {}
    dir_path = _PROMPTS_DIR / rel_dir
    if not dir_path.is_dir():
        return results
    for fpath in sorted(dir_path.glob("*.json")):
        with fpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        if key_by == "name":
            key = data.get("metadata", {}).get("name", fpath.stem)
        else:
            key = fpath.stem
        results[key] = data
    return results


def invalidate_cache():
    """Clear the LRU cache (useful in development / hot-reload scenarios)."""
    _read_json.cache_clear()
