from typing import Dict, List, Optional
import json
import os
from pathlib import Path


DEFAULT_EXPERT_POOL: List[Dict[str, str]] = [
	{
		"name": "risk_guardian",
		"role": "风控专家",
		"risk_focus": "extreme_high,high,medium",
		"system_prompt": "你是风控专家。请识别核心风险、按严重级别排序，并给出可执行缓释动作。",
	},
	{
		"name": "finance_advisor",
		"role": "财务专家",
		"risk_focus": "high,medium,low",
		"system_prompt": "你是财务专家。请评估预算可行性、融资节奏与现金流安全边界。",
	},
	{
		"name": "growth_strategist",
		"role": "增长专家",
		"risk_focus": "medium,low",
		"system_prompt": "你是增长专家。请给出转化率提升、渠道优先级与实验计划。",
	},
	{
		"name": "ops_executor",
		"role": "运营专家",
		"risk_focus": "medium,low",
		"system_prompt": "你是运营专家。请给出落地排期、资源分配与里程碑检查点。",
	},
]


# Module-level cache for loaded experts
_CACHED_EXPERTS: Optional[List[Dict[str, str]]] = None


def _default_config_path() -> Path:
	# project root is parent.parent.parent of this file (scripts/m7/...)
	project_root = Path(__file__).resolve().parent.parent.parent
	return project_root / "config" / "m7_experts.json"


def load_expert_pool(config_path: Optional[str] = None) -> List[Dict[str, str]]:
	"""Load expert pool from JSON config file.

	Priority: explicit config_path arg > M7_EXPERTS_PATH env var > config/m7_experts.json > built-in default
	"""
	# Resolve candidate path
	path = None
	if config_path:
		path = Path(config_path)
	elif os.getenv("M7_EXPERTS_PATH"):
		path = Path(os.getenv("M7_EXPERTS_PATH"))
	else:
		path = _default_config_path()

	if path and path.exists():
		try:
			with path.open("r", encoding="utf-8") as fh:
				data = json.load(fh)
			if isinstance(data, list):
				# basic validation
				validated: List[Dict[str, str]] = []
				for item in data:
					if not isinstance(item, dict):
						continue
					name = item.get("name")
					role = item.get("role")
					system_prompt = item.get("system_prompt")
					risk_focus = item.get("risk_focus", "")
					if name and role and system_prompt:
						validated.append(
							{
								"name": name,
								"role": role,
								"system_prompt": system_prompt,
								"risk_focus": risk_focus,
							}
						)
				if validated:
					return validated
		except Exception:
			# fall through to default
			pass

	return DEFAULT_EXPERT_POOL


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


