import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 动态添加 m7 目录到 sys.path
_m7_dir = Path(__file__).parent
if str(_m7_dir) not in sys.path:
    sys.path.insert(0, str(_m7_dir))

from m7_context_analyzer import (
    render_layer1_for_prompt,
    render_layer2_for_prompt,
    render_layer3_for_prompt,
    render_layer4_for_prompt,
)
from prompts.loader import load_prompt_template, load_prompt_dict


def build_system_prompt(expert: Dict[str, str]) -> str:
    expert_role = expert.get("role", "专家")
    expert_name = expert.get("name", "expert")
    expert_instruction = expert.get("system_prompt", "请基于输入数据给出可执行建议。")

    template = load_prompt_template("m7/system_prompt.json")
    return template.format(
        expert_role=expert_role,
        expert_name=expert_name,
        expert_instruction=expert_instruction,
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
    user_cfg = load_prompt_dict("m7/user_prompt.json")
    template = user_cfg["template"]
    schema_example = user_cfg["schema_example"]
    action_criteria = user_cfg["action_criteria"]

    payload = {
        "risk_level": risk_level,
        "route_reason": route_reason,
        "risk_reasons": reasons,
        "intermediate": intermediate,
        "project_data": project_data,
    }

    action_criteria_str = "\n".join(f"- {item}" for item in action_criteria)
    schema_example_str = json.dumps(schema_example, ensure_ascii=False)
    payload_str = json.dumps(payload, ensure_ascii=False)

    prompt = template.format(
        action_criteria=action_criteria_str,
        schema_example=schema_example_str,
        payload=payload_str,
    )

    if layer1_context:
        prompt += render_layer1_for_prompt(layer1_context)
        prompt += render_layer1_for_prompt(layer1_context)

    if layer2_context:
        prompt += render_layer2_for_prompt(layer2_context)

    if layer3_context:
        prompt += render_layer3_for_prompt(layer3_context)

    if layer4_context:
        prompt += render_layer4_for_prompt(layer4_context)

    return prompt
