"""
M11 — 小模型动态路由核心

使用 Qwen3-1.7B 小模型替代 m7_router.py 做路由决策。
从 Blackboard 读取输入，做意图分析 + 专家选择，写回 Blackboard。
"""

from .m11_core import M11Router, M11Error, SmallModelNotConfigured, SmallModelLoadFailed, SmallModelInferenceFailed, RoutingOutputParseError

__all__ = [
    "M11Router",
    "M11Error",
    "SmallModelNotConfigured",
    "SmallModelLoadFailed",
    "SmallModelInferenceFailed",
    "RoutingOutputParseError",
]
