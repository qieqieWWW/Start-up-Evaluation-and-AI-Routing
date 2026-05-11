from typing import Dict, List, Optional
import json
import os
from pathlib import Path

from prompts.loader import load_prompts_from_dir


# Default expert pool loaded from prompts/experts/ directory
_DEFAULT_EXPERT_POOL: Optional[List[Dict[str, str]]] = None


def _get_default_pool() -> List[Dict[str, str]]:
    """Lazy-load expert pool from prompts/experts/ directory."""
    global _DEFAULT_EXPERT_POOL
    if _DEFAULT_EXPERT_POOL is None:
        pool: List[Dict[str, str]] = []
        by_name = load_prompts_from_dir("experts", key_by="stem")
        for name in sorted(by_name.keys()):
            data = by_name[name]
            meta = data.get("metadata", {})
            pool.append({
                "name": meta.get("name", name),
                "role": meta.get("role", "专家"),
                "risk_focus": meta.get("risk_focus", ""),
                "system_prompt": data.get("system_prompt", ""),
            })
        _DEFAULT_EXPERT_POOL = pool
    return _DEFAULT_EXPERT_POOL


# Module-level cache for loaded experts
_CACHED_EXPERTS: Optional[List[Dict[str, str]]] = None


def _default_config_paths() -> list:
    """Return list of config paths in priority order."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return [
        project_root / "config" / "m7_experts.json",   # legacy path (fallback)
    ]


def load_expert_pool(config_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load expert pool from prompts/experts/ or JSON config file.

    Priority:
        1. prompts/experts/ directory (new, primary)
        2. explicit config_path arg (legacy)
        3. M7_EXPERTS_PATH env var (legacy)
        4. config/m7_experts.json (legacy fallback)
        5. built-in default from prompts/experts/
    """
    # 1. Try prompts/experts/ directory first (new approach)
    try:
        from prompts.loader import load_prompts_from_dir
        by_name = load_prompts_from_dir("experts", key_by="stem")
        if by_name:
            validated: List[Dict[str, str]] = []
            for name in sorted(by_name.keys()):
                data = by_name[name]
                meta = data.get("metadata", {})
                sp = data.get("system_prompt", "")
                if name and sp:
                    validated.append({
                        "name": meta.get("name", name),
                        "role": meta.get("role", "专家"),
                        "risk_focus": meta.get("risk_focus", ""),
                        "system_prompt": sp,
                    })
            if validated:
                return validated
    except Exception:
        pass

    # 2-4. Legacy paths (config file)
    paths: list = []
    if config_path:
        paths.append(Path(config_path))
    elif os.getenv("M7_EXPERTS_PATH"):
        paths.append(Path(os.getenv("M7_EXPERTS_PATH")))
    else:
        paths = _default_config_paths()

    for path in paths:
        if path and path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    validated = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name")
                        role = item.get("role")
                        system_prompt = item.get("system_prompt")
                        risk_focus = item.get("risk_focus", "")
                        if name and role and system_prompt:
                            validated.append({
                                "name": name,
                                "role": role,
                                "system_prompt": system_prompt,
                                "risk_focus": risk_focus,
                            })
                    if validated:
                        return validated
            except Exception:
                pass

    # 5. Ultimate fallback
    return _get_default_pool()


def get_expert_map(config_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
	"""Return a map of expert_name -> expert dict.

	If a config_path is provided, reload from that path. Otherwise use cached value if available.
	"""
	global _CACHED_EXPERTS
	if config_path:
		experts = load_expert_pool(config_path)
		_CACHED_EXPERTS = experts
		return {e["name"]: e for e in experts}

	if _CACHED_EXPERTS is None:
		_CACHED_EXPERTS = load_expert_pool()

	return {e["name"]: e for e in _CACHED_EXPERTS}


def reload_expert_pool(config_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
	"""Force reload expert pool from disk (or given path) and update cache."""
	return get_expert_map(config_path=config_path)


