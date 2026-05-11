"""
M11Router — 小模型动态路由核心。

流程：
    1. 从 Blackboard 读取 M8 风险信号 + 用户输入
    2. 用小模型做意图分析 + 专家选择
    3. 路由决策写回 Blackboard
    4. 返回与 m7_router 兼容的路由结果
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from prompts.loader import load_prompt_template

logger = logging.getLogger("m11_core")


# ════════════════════════════════════════════
# 异常定义
# ════════════════════════════════════════════

class M11Error(Exception):
    """M11 模块基础异常。"""
    pass


class SmallModelNotConfigured(M11Error):
    """小模型未配置（USE_REAL_SMALL_MODEL != true）。"""
    pass


class SmallModelLoadFailed(M11Error):
    """小模型加载失败（路径不存在 / 库缺失 / 内存不足）。"""
    pass


class SmallModelInferenceFailed(M11Error):
    """小模型推理异常（OOM / 推理过程异常）。"""
    pass


class RoutingOutputParseError(M11Error):
    """小模型输出 JSON 解析失败。"""
    pass


# ════════════════════════════════════════════
# 风险等级归一化（与 m7_router 保持一致）
# ════════════════════════════════════════════

_RISK_LEVEL_MAP = {
    "风险很高": "extreme_high",
    "风险较高": "high",
    "风险中等": "medium",
    "风险较低": "low",
    "风险很低": "low",
}


def _normalize_risk_level(risk_level: str) -> str:
    """将中文风险等级映射为归一化枚举。"""
    for cn, en in _RISK_LEVEL_MAP.items():
        if cn in risk_level:
            return en
    return "medium"


# ════════════════════════════════════════════
# M11Router
# ════════════════════════════════════════════

class M11Router:
    """小模型路由核心。

    使用 Qwen3-1.7B 分析用户输入和风险信号，选择最合适的专家。
    模型在 __init__ 时立即加载；加载失败直接抛出异常。

    Args:
        blackboard: 可选的 SharedBlackboard 实例，用于读写上下文
    """

    def __init__(self, blackboard: Optional[Any] = None):
        self.blackboard = blackboard
        self.model = None
        self.tokenizer = None

        # 立即加载模型
        self._initialize_model()

    def _initialize_model(self):
        """加载小模型（从单例获取或首次加载）。"""
        # 优先从 m11_model 单例获取
        try:
            from .m11_model import get_small_model
            result = get_small_model()
            if result is not None:
                self.model, self.tokenizer = result
                return
        except M11Error:
            raise
        except Exception as e:
            raise SmallModelLoadFailed(f"模型获取失败: {e}") from e

        # 单例为空（首次调用），直接加载
        try:
            from .m11_model import _load_small_model
            self.model, self.tokenizer = _load_small_model()
        except M11Error:
            raise
        except Exception as e:
            raise SmallModelLoadFailed(f"模型加载失败: {e}") from e

    # ── 公开接口 ──

    def route(
        self,
        user_input: str,
        risk_level: str = "",
        intermediate: Optional[Dict] = None,
        project_data: Optional[Dict] = None,
        user_id: str = "web_user",
        **kwargs,
    ) -> Dict[str, Any]:
        """主要路由入口。

        Args:
            user_input: 用户原始输入文本
            risk_level: M8 风险等级（如"风险较高"）
            intermediate: M8 中间特征值字典
            project_data: 原始项目数据
            user_id: 用户标识

        Returns:
            路由结果字典（与 m7_router.route_experts() 格式兼容）

        Raises:
            SmallModelInferenceFailed: 推理异常
            RoutingOutputParseError: JSON 解析失败
        """
        # 1. 从 Blackboard 读取上下文
        m8_signal = self._read_m8_from_blackboard()
        if m8_signal:
            risk_level = m8_signal.get("risk_level", risk_level)
            intermediate = m8_signal.get("intermediate", intermediate or {})

        # 2. 构建推理上下文
        context = self._build_context(
            user_input=user_input,
            risk_level=risk_level,
            intermediate=intermediate or {},
        )

        # 3. 构建 prompt + 推理
        prompt = self._build_m11_prompt(context)
        raw_output = self._predict(prompt)

        # 4. 解析输出
        decision = self._parse_output(raw_output)
        if decision is None:
            raise RoutingOutputParseError(
                f"路由输出解析失败，原始输出前200字: {raw_output[:200]}"
            )

        # 5. 写回 Blackboard
        self._write_to_blackboard(decision, risk_level)

        # 6. 格式化为兼容输出
        return self._format_output(decision, risk_level)

    # ── 内部方法 ──

    def _read_m8_from_blackboard(self) -> Optional[Dict]:
        """从 Blackboard 的 m8_risk zone 读取 M8 信号。"""
        if not self.blackboard:
            return None
        try:
            entries = self.blackboard.read(zone="m8_risk", agent_id="M11_router")
            if entries:
                return entries[-1].content  # 取最新一条
        except Exception:
            logger.warning("从 Blackboard 读取 m8_risk 失败", exc_info=True)
        return None

    def _build_context(
        self,
        user_input: str,
        risk_level: str,
        intermediate: Dict,
    ) -> Dict:
        """拼装推理上下文。"""
        normalized = _normalize_risk_level(risk_level)

        # 获取专家池信息
        expert_summary = self._get_expert_pool_summary()

        return {
            "user_input": user_input,
            "risk_level": risk_level,
            "normalized_risk": normalized,
            "combined_risk": intermediate.get("combined_risk", 0),
            "goal_ratio": intermediate.get("goal_ratio", 0),
            "expert_pool_summary": expert_summary,
        }

    def _get_expert_pool_summary(self) -> str:
        """获取可选专家池的文字摘要。"""
        try:
            from scripts.m7.m7_expert_pool import load_expert_pool
            experts = load_expert_pool()
            lines = []
            for e in experts:
                lines.append(f"{e['name']}({e['role']})")
            return "，".join(lines)
        except Exception:
            return "risk_guardian(风控专家), finance_advisor(财务顾问), "
            "ops_executor(运营执行), growth_strategist(增长策略师)"

    def _build_m11_prompt(self, context: Dict) -> str:
        """构建小模型输入 prompt。"""
        try:
            from .m11_prompt import build_routing_prompt
            return build_routing_prompt(context)
        except ImportError:
            # fallback: 直接加载模板
            pass

        # 直接加载模板（当 m11_prompt 尚未实现时）
        try:
            template = load_prompt_template("m11/routing.json")
        except Exception:
            # 终极 fallback：内联 prompt
            template = (
                "<|im_start|>system\n"
                "你是一个创业项目路由决策助手。分析用户输入和风险信号，"
                "选择最合适的专家。输出JSON，字段：selected_experts, "
                "expert_assignments, routing_rationale, confidence。\n"
                "<|im_end|>\n"
                "<|im_start|>user\n{user_input}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        return template.format(
            user_input=context["user_input"],
            risk_level=context["risk_level"],
            normalized_risk=context["normalized_risk"],
            expert_pool_summary=context["expert_pool_summary"],
        )

    def _predict(self, prompt: str) -> str:
        """调用小模型推理。

        Returns:
            模型输出的原始文本
        """
        if self.model is None or self.tokenizer is None:
            raise SmallModelNotConfigured("小模型未就绪，无法推理")

        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            # 将输入移到模型所在设备
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # 只取新生成的部分（去掉输入部分）
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            result = self.tokenizer.decode(generated, skip_special_tokens=True)
            return result.strip()

        except torch.cuda.OutOfMemoryError as e:
            raise SmallModelInferenceFailed(f"GPU 显存不足: {e}") from e
        except Exception as e:
            raise SmallModelInferenceFailed(f"推理异常: {e}") from e

    def _parse_output(self, raw: str) -> Optional[Dict]:
        """从小模型输出中提取 JSON。"""
        # 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块（处理模型输出包含前缀文本的情况）
        for delimiter in ["```json", "```", "\n"]:
            parts = raw.split(delimiter)
            for part in parts:
                part = part.strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

        return None

    def _write_to_blackboard(self, decision: Dict, risk_level: str):
        """将路由决策写入 Blackboard 的 m11_routing zone。"""
        if not self.blackboard:
            return
        try:
            self.blackboard.write(
                zone="m11_routing",
                content={
                    "selected_experts": decision.get("selected_experts", []),
                    "expert_assignments": decision.get("expert_assignments", {}),
                    "parallelism": decision.get("parallelism", 1),
                    "routing_rationale": decision.get("routing_rationale", ""),
                    "input_risk_level": risk_level,
                    "confidence": decision.get("confidence", 0.5),
                    "routing_scores": decision.get("routing_scores", {}),
                    "intent_analysis": decision.get("intent_analysis", {}),
                    "routing_path": "m11_model",
                },
                tags=["m11", "routing"],
                agent_id="M11_router",
            )
        except Exception:
            logger.warning("写入 Blackboard m11_routing 失败", exc_info=True)

    def _format_output(self, decision: Dict, risk_level: str) -> Dict:
        """格式化为与 m7_router.route_experts() 兼容的返回字典。"""
        selected_names = decision.get("selected_experts", [])

        # 从专家池获取完整专家信息
        selected_experts = self._resolve_experts(selected_names)

        normalized = _normalize_risk_level(risk_level)

        # 构建路由分数
        routing_scores = decision.get("routing_scores", {})
        if not routing_scores:
            # 没分数就自动分配
            all_names = ["risk_guardian", "finance_advisor", "ops_executor", "growth_strategist"]
            for i, name in enumerate(all_names):
                routing_scores[name] = 0.9 - i * 0.15 if name in selected_names else 0.3

        return {
            "input_risk_level": risk_level,
            "normalized_risk_level": normalized,
            "selected_experts": selected_experts,
            "expert_assignments": decision.get("expert_assignments", {}),
            "parallelism": decision.get("parallelism", 1),
            "route_reason": decision.get("routing_rationale", "M11 路由决策"),
            "confidence": decision.get("confidence", 0.5),
            "routing_scores": routing_scores,
            "routing_rationale": decision.get("routing_rationale", ""),
            "intent_result": decision.get("intent_analysis", {}),
            "_path": "m11_model",
        }

    def _resolve_experts(self, names: List[str]) -> List[Dict]:
        """根据专家名称列表，返回完整的专家信息 dict 列表。"""
        try:
            from scripts.m7.m7_expert_pool import get_expert_map
            expert_map = get_expert_map()
            result = []
            for name in names:
                if name in expert_map:
                    result.append(expert_map[name])
            return result
        except Exception:
            return [{"name": n, "role": n, "system_prompt": ""} for n in names]
