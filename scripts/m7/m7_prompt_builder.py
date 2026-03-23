import json
from typing import Any, Dict, List, Optional

from m7_context_analyzer import (
    render_layer1_for_prompt,
    render_layer2_for_prompt,
    render_layer3_for_prompt,
    render_layer4_for_prompt,
)


def build_system_prompt(expert: Dict[str, str]) -> str:
    expert_role = expert.get("role", "专家")
    expert_name = expert.get("name", "expert")
    expert_instruction = expert.get("system_prompt", "请基于输入数据给出可执行建议。")

    return (
        f"你是{expert_role}（{expert_name}）。"
        f"核心职责：{expert_instruction}"
        "\n\n"
        "你必须遵守以下规则："
        "\n1) 只基于输入数据作答，不要臆造未给出的事实；"
        "\n2) 结论要可执行，禁止空泛表述；"
        "\n3) 风险判断必须与给定 risk_level、risk_reasons 保持一致；"
        "\n4) 输出必须是严格 JSON（UTF-8），不要输出 markdown、代码块、额外解释；"
        "\n5) JSON 键名必须与要求完全一致。"
    )


def build_user_prompt(
    risk_level: str,
    reasons: List[str],
    intermediate: Dict[str, float],
    project_data: Dict[str, object],
    route_reason: str,
    layer1_context: Optional[Dict[str, Any]] = None,
    layer2_context: Optional[Dict[str, Any]] = None,
    layer3_context: Optional[Dict[str, Any]] = None,
    layer4_context: Optional[Dict[str, Any]] = None,
) -> str:
    schema_example = {
        "expert": "string",
        "risk_summary": "string",
        "actions": [
            {
                "priority": 1,
                "title": "string",
                "owner": "string",
                "due_days": 7,
                "expected_impact": "string",
            }
        ],
        "alerts": ["string"],
    }

    action_quality_criteria = [
        "title 必须是可执行动作，不得是抽象口号",
        "owner 必须是可落地责任方（例如：增长运营团队、风控与合规团队）",
        "due_days 取 1~90 的整数",
        "expected_impact 需要明确业务影响（如转化率、融资进度、合规风险）",
    ]

    payload = {
        "risk_level": risk_level,
        "route_reason": route_reason,
        "risk_reasons": reasons,
        "intermediate": intermediate,
        "project_data": project_data,
    }

    prompt = (
        "任务：你将作为被路由到的专家，基于输入项目输出结构化决策建议。\n"
        "\n输出要求（必须同时满足）：\n"
        "1) 仅输出 1 个 JSON 对象，不得出现其他文本；\n"
        "2) 字段必须包含：expert, risk_summary, actions, alerts；\n"
        "3) actions 至少 3 条，按 priority 从小到大排序且不重复；\n"
        "4) 内容必须使用中文；\n"
        "5) risk_summary 需要点出核心风险 + 当前目标差距；\n"
        "6) alerts 至少 1 条，必须是监控/预警项。\n"
        "\n动作质量标准：\n"
        + "\n".join(f"- {item}" for item in action_quality_criteria)
        + "\n\n"
        f"JSON schema 示例：{json.dumps(schema_example, ensure_ascii=False)}\n\n"
        f"输入数据：{json.dumps(payload, ensure_ascii=False)}"
    )

    if layer1_context:
        prompt += render_layer1_for_prompt(layer1_context)

    if layer2_context:
        prompt += render_layer2_for_prompt(layer2_context)

    if layer3_context:
        prompt += render_layer3_for_prompt(layer3_context)

    if layer4_context:
        prompt += render_layer4_for_prompt(layer4_context)

    return prompt
