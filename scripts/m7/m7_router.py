from typing import Dict, List, Tuple

from m7_expert_pool import get_expert_map


def _normalize_risk_level(risk_level: str) -> str:
	text = (risk_level or "").strip()
	if "极高" in text:
		return "extreme_high"
	if "高" in text:
		return "high"
	if "中" in text:
		return "medium"
	return "low"


def route_experts(
	risk_level: str,
	intermediate: Dict[str, float],
	project_data: Dict[str, float],
) -> Dict[str, object]:
	del intermediate
	del project_data

	normalized = _normalize_risk_level(risk_level)
	expert_map = get_expert_map()

	policy: Dict[str, Tuple[List[str], str, float]] = {
		"extreme_high": (
			["risk_guardian", "finance_advisor"],
			"极高风险优先控制下行与现金流安全。",
			0.9,
		),
		"high": (
			["risk_guardian", "finance_advisor"],
			"高风险优先风控与财务兜底。",
			0.82,
		),
		"medium": (
			["finance_advisor", "ops_executor"],
			"中风险采取财务稳态加运营修复。",
			0.76,
		),
		"low": (
			["growth_strategist", "ops_executor"],
			"低风险转向增长与运营放大。",
			0.72,
		),
	}

	selected_names, reason, confidence = policy[normalized]
	selected_experts = [expert_map[name] for name in selected_names if name in expert_map]

	return {
		"input_risk_level": risk_level,
		"normalized_risk_level": normalized,
		"selected_experts": selected_experts,
		"route_reason": reason,
		"confidence": confidence,
	}

