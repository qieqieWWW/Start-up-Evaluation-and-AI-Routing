from typing import Dict, List


def build_expert_pool() -> List[Dict[str, str]]:
	return [
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


def get_expert_map() -> Dict[str, Dict[str, str]]:
	experts = build_expert_pool()
	return {expert["name"]: expert for expert in experts}

