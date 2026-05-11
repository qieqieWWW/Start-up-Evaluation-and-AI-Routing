"""
M11 路由 prompt 构建器。

从 prompts/m11/routing.json 加载模板，注入运行时变量，
输出为 Qwen3 兼容的 <|im_start|> 格式。
"""

import json
import logging
from typing import Any, Dict, List

from prompts.loader import load_prompt_dict

logger = logging.getLogger("m11_prompt")


def build_routing_prompt(context: Dict[str, Any]) -> str:
    """构造 M11 路由 prompt。

    Args:
        context: 包含 user_input, risk_level, normalized_risk, expert_pool_summary 等字段

    Returns:
        格式化为 <|im_start|> 格式的完整 prompt 字符串
    """
    try:
        data = load_prompt_dict("m11/routing.json")
        template = data["template"]

        # 填充模板变量
        prompt = template.format(
            user_input=context.get("user_input", ""),
            risk_level=context.get("risk_level", "未知"),
            normalized_risk=context.get("normalized_risk", "unknown"),
            expert_pool_summary=context.get("expert_pool_summary", ""),
        )
        return prompt

    except Exception as e:
        logger.warning(f"加载 prompts/m11/routing.json 失败: {e}，使用内联 prompt")
        # 终极 fallback
        user_input = context.get("user_input", "")
        return (
            "<|im_start|>system\n"
            "你是一个创业项目路由决策助手。分析用户输入和风险信号，"
            "选择最合适的专家。输出JSON，字段：selected_experts, "
            "expert_assignments, routing_rationale, confidence。\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def get_few_shot_examples() -> List[Dict]:
    """获取 few-shot 示例列表（用于单元测试验证）。"""
    data = load_prompt_dict("m11/routing.json")
    return data.get("few_shot_examples", [])
