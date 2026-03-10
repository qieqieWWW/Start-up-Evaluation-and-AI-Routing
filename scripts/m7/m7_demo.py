import os
import sys
import json
from pathlib import Path

from m7_router import route_experts
from m7_inference_runner import run_expert_llm_inference

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from m8_rule_adapter import judge_project_risk_m8


def _pretty_json(data: object) -> str:
	return json.dumps(data, ensure_ascii=False, indent=2)


def _print_structured_result(parsed: dict) -> None:
	risk_summary = parsed.get("risk_summary")
	actions = parsed.get("actions", [])
	alerts = parsed.get("alerts", [])

	if risk_summary:
		print("  风险总结:")
		print(f"    {risk_summary}")

	if isinstance(actions, list) and actions:
		print("  建议动作:")
		for action in actions:
			priority = action.get("priority", "-")
			title = action.get("title", "未命名动作")
			owner = action.get("owner", "未指定")
			due_days = action.get("due_days", "-")
			impact = action.get("expected_impact", "")
			print(f"    [{priority}] {title}")
			print(f"      owner: {owner} | due_days: {due_days}")
			if impact:
				print(f"      impact: {impact}")

	if isinstance(alerts, list) and alerts:
		print("  风险提醒:")
		for idx, alert in enumerate(alerts, 1):
			print(f"    {idx}. {alert}")


def run_m8_to_m7_demo() -> None:
	project_data = {
		"goal_usd": 15000,
		"duration_days": 60,
		"main_category": "Technology",
		"country": "US",
		"country_factor": 0.3,
		"actual_funding_usd": 5000,
		"planned_duration_days": 45,
	}

	risk_level, reasons, intermediate = judge_project_risk_m8(project_data, verbose=False)
	route_result = route_experts(risk_level, intermediate, project_data)
	llm_outputs = []
	if os.getenv("DEEPSEEK_API_KEY"):
		try:
			llm_outputs = run_expert_llm_inference(
				risk_level=risk_level,
				reasons=reasons,
				intermediate=intermediate,
				project_data=project_data,
				route_result=route_result,
				top_k=1,
				model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
			)
		except Exception as exc:
			print(f"⚠️ DeepSeek 调用失败，已跳过 LLM 推理: {exc}")
	else:
		print("ℹ️ 未设置 DEEPSEEK_API_KEY，跳过真实 LLM 推理")
	visual_paths = None
	try:
		from m7_visualization import save_m7_visualizations
		visual_paths = save_m7_visualizations(
			route_result,
			str(PROJECT_ROOT / "Kickstarter_Clean" / "m7_visualization"),
		)
	except ModuleNotFoundError as exc:
		print(f"⚠️ 跳过M7可视化（缺少依赖）: {exc}")

	print("=" * 70)
	print("M8 → M7 Demo")
	print("=" * 70)
	print(f"M8 风险等级: {risk_level}")
	print("M8 风险原因:")
	for idx, reason in enumerate(reasons, 1):
		print(f"  {idx}. {reason}")

	print("\nM7 路由结果:")
	print(f"  归一化风险: {route_result['normalized_risk_level']}")
	print(f"  路由理由: {route_result['route_reason']}")
	print(f"  置信度: {route_result['confidence']}")
	print("  选中专家:")
	for expert in route_result["selected_experts"]:
		print(f"    - {expert['role']} ({expert['name']})")
		print(f"      prompt: {expert['system_prompt']}")

	print("\nM7 可视化输出:")
	if visual_paths:
		print(f"  专家选择图: {visual_paths['expert_selection_chart']}")
		print(f"  路由流程图: {visual_paths['route_flow_chart']}")
	else:
		print("  未生成（可安装 matplotlib 后重试）")

	print("\nM7 LLM 专家输出:")
	if llm_outputs:
		for item in llm_outputs:
			expert = item.get("expert", {})
			print(f"  专家: {expert.get('role', '未知')} ({expert.get('name', 'unknown')})")
			print(f"  模型: {item.get('model', '')}")
			print("  用量:")
			print(_pretty_json(item.get("usage", {})))

			parsed = item.get("parsed", {})
			if isinstance(parsed, dict):
				_print_structured_result(parsed)
				print("  原始结构化JSON:")
				print(_pretty_json(parsed))
			else:
				print("  结构化结果:")
				print(_pretty_json(parsed))

			print("-" * 70)
	else:
		print("  未生成（可设置 DEEPSEEK_API_KEY 后重试）")


if __name__ == "__main__":
	os.chdir(PROJECT_ROOT)
	run_m8_to_m7_demo()

