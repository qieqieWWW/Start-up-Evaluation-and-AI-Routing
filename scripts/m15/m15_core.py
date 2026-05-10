"""
M15 核心：数据类定义与消融维度枚举

包含：
- AblationDimension: 消融维度枚举
- AblationCondition: 单条消融条件配置
- AblationConfig: 一次消融实验的完整配置
- SingleConditionResult: 单条条件的运行结果
- AblationReport: 完整的消融实验报告
- _build_default_config(): 默认消融条件工厂
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AblationDimension(str, Enum):
    """消融维度枚举"""

    ROUTING_STRATEGY = "routing_strategy"  # 路由策略消融
    LLM_REASONING = "llm_reasoning"  # LLM 推理消融
    COMPONENT_INTEGRITY = "component_integrity"  # 组件完整度消融
    # 以下为后续扩展
    CONTEXT_LAYERS = "context_layers"  # 上下文窗口消融
    M8_THRESHOLDS = "m8_thresholds"  # M8 风险阈值消融
    EXPERT_COUNT = "expert_count"  # 专家数量消融

    @classmethod
    def core_dimensions(cls) -> list["AblationDimension"]:
        """首期实现的 3 个核心维度"""
        return [
            cls.ROUTING_STRATEGY,
            cls.LLM_REASONING,
            cls.COMPONENT_INTEGRITY,
        ]


@dataclass
class AblationCondition:
    """单条消融条件配置"""

    label: str  # 条件标签，如 "无LLM推理"
    params: Dict[str, Any]  # 覆写参数键值对
    description: str = ""  # 条件描述


@dataclass
class AblationConfig:
    """一次消融实验的完整配置"""

    dimension: AblationDimension  # 消融维度
    conditions: List[AblationCondition]  # 对比条件列表（至少 2 个）
    project_data: Dict[str, Any]  # 项目数据
    user_query: str = ""  # 用户原始查询
    user_id: str = "m15_ablation_user"
    api_key: str = ""  # LLM API key
    enable_blender: bool = True  # 是否启用 Blender
    m8_result: Optional[Dict[str, Any]] = None  # 已有 M8 结果（避免重复计算）
    m7_result: Optional[Dict[str, Any]] = None  # 已有 M7 结果


@dataclass
class SingleConditionResult:
    """单条消融条件的运行结果"""

    label: str
    params: Dict[str, Any]
    risk_level: str = ""
    confidence: float = 0.0
    route_reason: str = ""
    selected_experts: List[str] = field(default_factory=list)
    routing_scores: Dict[str, float] = field(default_factory=dict)
    expert_count: int = 0
    execution_time_ms: float = 0.0
    # LLM 结果
    llm_output_size: int = 0
    fused_summary: str = ""
    action_count: int = 0
    alert_count: int = 0
    # 组件完整度的逐层指标
    component_level: str = ""
    m8_risk_score: float = 0.0
    # 错误
    error: str = ""


@dataclass
class AblationReport:
    """完整的消融实验报告"""

    config: AblationConfig
    results: List[SingleConditionResult]
    timestamp: str = ""
    comparison_summary: Dict[str, Any] = field(default_factory=dict)
    dimension_label: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.dimension_label:
            self.dimension_label = self.config.dimension.value

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于存储/传输）"""
        return {
            "timestamp": self.timestamp,
            "dimension": self.config.dimension.value,
            "dimension_label": self.dimension_label,
            "conditions": [
                {"label": c.label, "params": c.params, "description": c.description}
                for c in self.config.conditions
            ],
            "results": [asdict(r) for r in self.results],
            "comparison_summary": self.comparison_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ==========================================
# 默认消融条件工厂
# ==========================================

def _build_default_conditions(dimension: AblationDimension) -> List[AblationCondition]:
    """为指定维度构建默认的消融条件列表"""

    if dimension == AblationDimension.ROUTING_STRATEGY:
        return [
            AblationCondition(
                label="规则路由 (Baseline)",
                params={"route_mode": "default"},
                description="只用 M8 风险等级决定专家，不加任何偏置",
            ),
            AblationCondition(
                label="画像偏置",
                params={"route_mode": "with_profile"},
                description="加入 L3 用户画像偏置",
            ),
            AblationCondition(
                label="知识库偏置",
                params={"route_mode": "with_kb"},
                description="加入 L4 全局知识偏置",
            ),
            AblationCondition(
                label="全量路由",
                params={"route_mode": "full"},
                description="五层偏置全部启用",
            ),
        ]

    elif dimension == AblationDimension.LLM_REASONING:
        return [
            AblationCondition(
                label="纯规则模式",
                params={"enable_blender": False, "enable_llm": False},
                description="无 LLM 调用，仅规则路由",
            ),
            AblationCondition(
                label="LLM 推理",
                params={"enable_blender": False, "enable_llm": True},
                description="调用 LLM 但不做多专家融合",
            ),
            AblationCondition(
                label="LLM + Blender",
                params={"enable_blender": True, "enable_llm": True},
                description="调用 LLM + PairRank + GenFuser 融合",
            ),
        ]

    elif dimension == AblationDimension.COMPONENT_INTEGRITY:
        return [
            AblationCondition(
                label="M8 only",
                params={"component_level": "m8_only"},
                description="只执行 M8 风险判定",
            ),
            AblationCondition(
                label="M8 → M7",
                params={"component_level": "m8_m7"},
                description="风险判定 + 专家路由（不含 LLM）",
            ),
            AblationCondition(
                label="M8 → M7 → LLM",
                params={"component_level": "m8_m7_llm", "enable_blender": False},
                description="风险判定 + 路由 + LLM 推理",
            ),
            AblationCondition(
                label="M8 → M7 → Blender",
                params={"component_level": "m8_m7_blender", "enable_blender": True},
                description="风险判定 + 路由 + LLM + Blender 融合",
            ),
            AblationCondition(
                label="全流水线",
                params={"component_level": "full_pipeline"},
                description="M8 → M7 → LLM → Blender → M9 完整链路",
            ),
        ]

    return []


def build_default_config(
    dimension: AblationDimension,
    project_data: Dict[str, Any],
    user_query: str = "",
    api_key: str = "",
    user_id: str = "m15_ablation_user",
    enable_blender: bool = True,
    m8_result: Optional[Dict[str, Any]] = None,
    m7_result: Optional[Dict[str, Any]] = None,
) -> AblationConfig:
    """为指定维度构建默认的 AblationConfig"""
    return AblationConfig(
        dimension=dimension,
        conditions=_build_default_conditions(dimension),
        project_data=project_data,
        user_query=user_query,
        api_key=api_key,
        user_id=user_id,
        enable_blender=enable_blender,
        m8_result=m8_result,
        m7_result=m7_result,
    )
