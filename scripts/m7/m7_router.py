from typing import Any, Dict, List, Optional, Tuple

from m7_context_analyzer import build_layer3_context, build_layer4_context
from m7_expert_pool import get_expert_map
from m7_intent_engine import recognize_intent
from m7_trajectory_manager import build_trajectory_context


def _normalize_risk_level(risk_level: str) -> str:
	text = (risk_level or "").strip()
	if "极高" in text:
		return "extreme_high"
	if "高" in text:
		return "high"
	if "中" in text:
		return "medium"
	return "low"


def _base_expert_scores(normalized: str) -> Dict[str, float]:
	if normalized == "extreme_high":
		return {
			"risk_guardian": 1.0,
			"finance_advisor": 0.9,
			"ops_executor": 0.35,
			"growth_strategist": 0.2,
		}
	if normalized == "high":
		return {
			"risk_guardian": 0.95,
			"finance_advisor": 0.85,
			"ops_executor": 0.45,
			"growth_strategist": 0.3,
		}
	if normalized == "medium":
		return {
			"risk_guardian": 0.6,
			"finance_advisor": 0.8,
			"ops_executor": 0.75,
			"growth_strategist": 0.55,
		}
	return {
		"risk_guardian": 0.35,
		"finance_advisor": 0.55,
		"ops_executor": 0.85,
		"growth_strategist": 0.9,
	}


def _apply_layer3_bias(scores: Dict[str, float], layer3_context: Dict[str, Any]) -> None:
	profile = layer3_context.get("profile_summary", {})
	if not isinstance(profile, dict):
		return

	risk_appetite = str(profile.get("dominant_risk_appetite", "")).lower()
	prefs = " ".join(str(item) for item in profile.get("top_preferences", []))

	if risk_appetite in {"high", "medium_high"}:
		scores["growth_strategist"] += 0.12
		scores["risk_guardian"] -= 0.05

	if risk_appetite in {"low", "medium_low"}:
		scores["risk_guardian"] += 0.1
		scores["finance_advisor"] += 0.08

	if "高风险高回报" in prefs or "激进" in prefs:
		scores["growth_strategist"] += 0.08

	if "稳健" in prefs or "成本控制" in prefs:
		scores["finance_advisor"] += 0.08
		scores["ops_executor"] += 0.05


def _apply_layer4_bias(scores: Dict[str, float], layer4_context: Dict[str, Any]) -> None:
	kb_summary = layer4_context.get("kb_summary", {})
	if not isinstance(kb_summary, dict):
		return

	categories = [str(item) for item in kb_summary.get("top_categories", [])]
	guidelines = " ".join(str(item) for item in kb_summary.get("key_guidelines", []))

	if "risk_model" in categories:
		scores["risk_guardian"] += 0.08
		scores["finance_advisor"] += 0.08

	if "industry_standard" in categories:
		scores["ops_executor"] += 0.06

	if "success_case" in categories:
		scores["growth_strategist"] += 0.06

	if "留存" in guidelines:
		scores["growth_strategist"] += 0.06
		scores["ops_executor"] += 0.04

	if "现金流" in guidelines or "生存线" in guidelines:
		scores["finance_advisor"] += 0.08
		scores["risk_guardian"] += 0.04

	if "渠道冗余" in guidelines or "单点依赖" in guidelines:
		scores["risk_guardian"] += 0.06
		scores["ops_executor"] += 0.04


def _apply_intent_bias(scores: Dict[str, float], intent_result: Dict[str, Any]) -> None:
	required_experts = intent_result.get("required_experts", [])
	if isinstance(required_experts, list):
		for name in required_experts:
			key = str(name)
			if key in scores:
				scores[key] += 0.2

	urgency = str(intent_result.get("urgency", "")).lower()
	if urgency == "high":
		scores["risk_guardian"] += 0.08
		scores["finance_advisor"] += 0.05


def _apply_trajectory_bias(scores: Dict[str, float], trajectory_context: Dict[str, Any]) -> None:
	short_term = trajectory_context.get("short_term_memory", {})
	if isinstance(short_term, dict):
		state_machine = short_term.get("state_machine", {})
		if isinstance(state_machine, dict):
			state = str(state_machine.get("current_state", ""))
			if state == "Analyzing_Financials":
				scores["finance_advisor"] += 0.12
			if state == "Analyzing_IP":
				scores["risk_guardian"] += 0.12

	long_term = trajectory_context.get("long_term_memory", {})
	if isinstance(long_term, dict):
		forced_nodes = long_term.get("forced_nodes", [])
		if isinstance(forced_nodes, list) and "team_risk_strong_alert" in forced_nodes:
			scores["risk_guardian"] += 0.1
			scores["ops_executor"] += 0.08


def route_experts(
	risk_level: str,
	intermediate: Dict[str, float],
	project_data: Dict[str, float],
	user_id: str = "",
	user_input: str = "",
	profile_db_path: Optional[str] = None,
	profile_top_k: int = 5,
	global_kb_path: Optional[str] = None,
	global_kb_top_k: int = 5,
	conversation_turns: Optional[List[Dict[str, str]]] = None,
	previous_state: str = "",
	intent_library_path: Optional[str] = None,
	intent_model: str = "deepseek-chat",
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

	base_names, base_reason, base_confidence = policy[normalized]
	layer3_context = build_layer3_context(
		user_id=user_id,
		current_query=user_input,
		top_k=profile_top_k,
		profile_db_path=profile_db_path,
	)
	layer4_context = build_layer4_context(
		current_query=user_input,
		top_k=global_kb_top_k,
		kb_path=global_kb_path,
	)
	intent_result = recognize_intent(
		user_input=user_input,
		library_path=intent_library_path,
		model=intent_model,
	)
	trajectory_context = build_trajectory_context(
		current_input=user_input,
		conversation_turns=conversation_turns,
		previous_state=previous_state,
		user_id=user_id,
		profile_db_path=profile_db_path,
	)

	scores = _base_expert_scores(normalized)
	_apply_layer3_bias(scores, layer3_context)
	_apply_layer4_bias(scores, layer4_context)
	_apply_intent_bias(scores, intent_result)
	_apply_trajectory_bias(scores, trajectory_context)

	ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
	selected_names = [name for name, _ in ranked[:2] if name in expert_map]
	if not selected_names:
		selected_names = [name for name in base_names if name in expert_map]

	reason_parts = [base_reason]
	profile_summary = layer3_context.get("profile_summary", {})
	if isinstance(profile_summary, dict):
		risk_appetite = str(profile_summary.get("dominant_risk_appetite", "")).strip()
		if risk_appetite:
			reason_parts.append(f"结合用户历史偏好（risk_appetite={risk_appetite}）进行个性化偏置。")

	kb_summary = layer4_context.get("kb_summary", {})
	if isinstance(kb_summary, dict) and kb_summary.get("top_categories"):
		reason_parts.append("结合全局知识库中的行业基准与风险模型进行Grounding修正。")

	if intent_result:
		reason_parts.append(
			f"意图识别结果：{intent_result.get('primary_intent', 'unknown')} / {intent_result.get('sub_intent', 'unknown')}。"
		)

	short_term = trajectory_context.get("short_term_memory", {})
	if isinstance(short_term, dict):
		state_machine = short_term.get("state_machine", {})
		if isinstance(state_machine, dict):
			current_state = str(state_machine.get("current_state", ""))
			if current_state:
				reason_parts.append(f"当前会话状态：{current_state}。")

	confidence = min(0.95, base_confidence + (0.02 if user_id else 0.0) + (0.02 if user_input else 0.0))
	selected_experts = [expert_map[name] for name in selected_names]

	return {
		"input_risk_level": risk_level,
		"normalized_risk_level": normalized,
		"selected_experts": selected_experts,
		"route_reason": " ".join(reason_parts),
		"confidence": confidence,
		"routing_scores": {name: round(score, 4) for name, score in ranked if name in expert_map},
		"intent_result": intent_result,
		"trajectory_context": trajectory_context,
	}

