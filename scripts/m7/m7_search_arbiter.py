#!/usr/bin/env python
# coding: utf-8

"""
M7 SearchArbiter — 意图感知搜索决策器
========================================

核心职责：在意图识别之后、LLM 推理之前，根据三个维度决定：
    1. 是否需要引导 LLM 联网搜索（L1: agent_search_hint）
  2. 是否需要额外的本地搜索做交叉验证（L2: need_local_evidence）
  3. 搜索的深度和优先级

关键设计原则（v2 双层检索架构）:
    - L1 为主力：LLM 自身具备"可联网搜索"模型能力，
    我们通过 Prompt 注入 agent_search_hint 来引导它主动联网。
  - L2 为补充：可选的外部搜索引擎（Serper/Bing），
    仅在任务型高可信需求时启用。

嵌入点：被 m7_inference_runner.run_expert_llm_inference() 直接调用，
       是主流程中不可跳过的环节。
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from m7_freshness_detector import FreshnessDetector, FreshnessResult
from prompts.loader import load_prompt_dict

logger = logging.getLogger("m7_search_arbiter")


class SearchMode(Enum):
    DEEP = "deep"     # 深度搜索: 全面联网 + L2 本地补搜 + 多轮验证
    QUICK = "quick"   # 快速搜索: 引导Agent联网（不启用 L2）
    SKIP = "skip"     # 跳过搜索


@dataclass
class SearchDecision:
    """搜索决策结果 — 这个对象会在整个调用链中传递"""
    should_search: bool
    search_mode: SearchMode

    # ── L1: 给 LLM 的联网提示文本（注入 System Prompt） ──
    agent_search_hint: Optional[str] = None

    # ── L2: 是否需要本地 WebRetriever 补充证据 ──
    need_local_evidence: bool = False

    # ── 生成的优化搜索查询（用于 L2 WebRetriever） ──
    local_search_queries: List[str] = field(default_factory=list)

    # ── 元信息 ──
    reason: str = ""
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_search": self.should_search,
            "search_mode": self.search_mode.value,
            "agent_search_hint": self.agent_search_hint,
            "need_local_evidence": self.need_local_evidence,
            "local_search_queries": self.local_search_queries,
            "reason": self.reason,
            "scores": self.scores,
        }


# ─────────── 权重配置 ───────────

INTENT_WEIGHTS = {
    "task": 1.0,
    "assessment": 1.0,
    "analysis": 1.0,
    "risk_assessment": 1.0,
    "mixed": 0.6,
    "chat": 0.3,
    "chitchat": 0.25,
    "greeting": 0.05,
    "unknown": 0.5,
}

# 可验证性领域关键词（命中则提升可验证性分数）
VERIFIABLE_DOMAIN_KEYWORDS = [
    "风险", "市场", "行业", "政策", "法律", "法规", "财务", "数据",
    "价格", "成本", "营收", "估值", "融资", "竞争", "竞品", "技术趋势",
]


class SearchArbiter:
    """
    三维搜索决策器
    
    用法（在 m7_inference_runner 中集成）:
    
        arbiter = SearchArbiter()
        decision = arbiter.decide(
            user_input=user_query,
            intent_type=intent_result.get("primary_intent", "unknown"),
            domain_context=project_data.get("main_category", ""),
        )
        
        if decision.agent_search_hint:
            # 注入到 system_prompt 或 user_prompt 中
            enhanced_system_prompt += build_search_hints_block(decision.agent_search_hint)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        freshness_detector: Optional[FreshnessDetector] = None,
        task_threshold: float = 0.45,
        chat_threshold: float = 0.70,
        mixed_threshold: float = 0.55,
    ):
        self.freshness_detector = freshness_detector or FreshnessDetector(config_path=config_path)
        self.task_threshold = task_threshold
        self.chat_threshold = chat_threshold
        self.mixed_threshold = mixed_threshold

    def decide(
        self,
        user_input: str,
        intent_type: str = "task",
        domain_context: str = "",
        project_data: Optional[Dict[str, Any]] = None,
    ) -> SearchDecision:
        """
        三维决策:
          维度1: 意图类型权重 (intent_weight)
          维度2: 时效性信号 (temporal_score from FreshnessDetector)
          维度3: 领域可验证性 (verifiability_score)
        """
        # ── 维度2: 时效性检测 ──
        freshness: FreshnessResult = self.freshness_detector.detect(user_input)
        temporal_score = freshness.temporal_score

        # ── 维度3: 可验证性检测 ──
        verifiability_score = self._detect_verifiability(user_input, domain_context, project_data)

        # ── 维度1: 意图类型归一化 ──
        normalized_intent = intent_type.lower().strip()
        intent_weight = INTENT_WEIGHTS.get(normalized_intent, 0.5)

        # ── 综合倾向分数 ──
        search_propensity = (
            0.35 * intent_weight +
            0.40 * temporal_score +
            0.25 * verifiability_score
        )

        scores = {
            "intent_weight": round(intent_weight, 4),
            "temporal_score": round(temporal_score, 4),
            "verifiability_score": round(verifiability_score, 4),
            "search_propensity": round(search_propensity, 4),
        }

        logger.debug(
            f"[SearchArbiter] intent={intent_type} temporal={temporal_score:.2f} "
            f"verifiable={verifiability_score:.2f} propensity={search_propensity:.2f}"
        )

        # ── 分流决策 ──
        if normalized_intent in ("task", "assessment", "analysis", "risk_assessment"):
            decision = self._decide_task(
                search_propensity, temporal_score, user_input,
                domain_context, project_data, freshness, scores,
            )
        elif normalized_intent in ("chat", "chitchat", "greeting"):
            decision = self._decide_chat(
                search_propensity, temporal_score, user_input, freshness, scores,
            )
        else:
            # mixed / unknown
            decision = self._decide_mixed(
                search_propensity, temporal_score, user_input,
                domain_context, project_data, freshness, scores,
            )

        return decision

    # ════════════ 任务型决策路径 ════════════

    def _decide_task(
        self,
        propensity: float,
        temporal_score: float,
        user_input: str,
        domain_context: str,
        project_data: Optional[Dict],
        freshness: FreshnessResult,
        scores: Dict,
    ) -> SearchDecision:
        """任务型: 低门槛，倾向于无条件搜索"""
        if propensity >= self.task_threshold:
            hint = self._build_deep_task_hint(user_input, domain_context, project_data, freshness)
            queries = self._generate_l2_queries(user_input, domain_context, freshness)
            return SearchDecision(
                should_search=True,
                search_mode=SearchMode.DEEP,
                agent_search_hint=hint,
                need_local_evidence=True,
                local_search_queries=queries,
                reason=f"任务型+高时效需求(propensity={propensity:.2f}) → DEEP",
                scores=scores,
            )
        
        # 即使低倾向，任务型也至少给一个基本搜索指引
        hint = self._build_basic_task_hint(domain_context, freshness)
        return SearchDecision(
            should_search=True,
            search_mode=SearchMode.QUICK,
            agent_search_hint=hint,
            need_local_evidence=False,
            reason=f"任务型标准搜索(propensity={propensity:.2f}) → QUICK",
            scores=scores,
        )

    # ════════════ 闲聊型决策路径 ════════════

    def _decide_chat(
        self,
        propensity: float,
        temporal_score: float,
        user_input: str,
        freshness: FreshnessResult,
        scores: Dict,
    ) -> SearchDecision:
        """闲聊型: 高门槛，仅在有明确时效信号时触发"""
        if temporal_score >= self.chat_threshold:
            hint = self._build_factual_chat_hint(user_input, freshness)
            return SearchDecision(
                should_search=True,
                search_mode=SearchMode.QUICK,
                agent_search_hint=hint,
                need_local_evidence=False,
                reason=f"闲聊+强时效信号(temporal={temporal_score:.2f}) → QUICK",
                scores=scores,
            )

        if temporal_score >= 0.45 and freshness.suggested_time_range == "past_day":
            # 实时类问题（天气、营业状态等）
            hint = self._build_factual_chat_hint(user_input, freshness)
            return SearchDecision(
                should_search=True,
                search_mode=SearchMode.QUICK,
                agent_search_hint=hint,
                need_local_evidence=False,
                reason=f"闲聊+实时信息需求 → QUICK",
                scores=scores,
            )

        return SearchDecision(
            should_search=False,
            search_mode=SearchMode.SKIP,
            reason=f"闲聊无时效需求(temporal={temporal_score:.2f}) → SKIP",
            scores=scores,
        )

    # ════════════ 混合型决策路径 ════════════

    def _decide_mixed(
        self,
        propensity: float,
        temporal_score: float,
        user_input: str,
        domain_context: str,
        project_data: Optional[Dict],
        freshness: FreshnessResult,
        scores: Dict,
    ) -> SearchDecision:
        """混合型: 中等门槛"""
        if propensity >= self.mixed_threshold:
            hint = self._build_mixed_hint(user_input, domain_context, freshness)
            return SearchDecision(
                should_search=True,
                search_mode=SearchMode.QUICK,
                agent_search_hint=hint,
                need_local_evidence=(propensity >= 0.65),
                reason=f"混合型+中等需求(propensity={propensity:.2f}) → QUICK",
                scores=scores,
            )

        if temporal_score >= 0.6:
            hint = self._build_factual_chat_hint(user_input, freshness)
            return SearchDecision(
                should_search=True,
                search_mode=SearchMode.QUICK,
                agent_search_hint=hint,
                need_local_evidence=False,
                reason=f"混合型+时效信号(temporal={temporal_score:.2f}) → QUICK",
                scores=scores,
            )

        return SearchDecision(
            should_search=False,
            search_mode=SearchMode.SKIP,
            reason=f"混合型无显著搜索需求(propensity={propensity:.2f}) → SKIP",
            scores=scores,
        )

    # ════════════ Hint 生成策略（核心：Prompt 即控制） ════════════

    _SEARCH_HINTS_CACHE = None

    @classmethod
    def _get_hints(cls, hint_type: str) -> dict:
        """Lazy-load search hints from JSON file."""
        if cls._SEARCH_HINTS_CACHE is None:
            cls._SEARCH_HINTS_CACHE = load_prompt_dict("m7/search_hints.json")["hints"]
        return cls._SEARCH_HINTS_CACHE.get(hint_type, {})

    @staticmethod
    def _build_deep_task_hint(
        user_input: str,
        domain_context: str,
        project_data: Optional[Dict],
        freshness: FreshnessResult,
    ) -> str:
        """生成深度任务型搜索引导 — 注入 LLM 的 System Prompt"""
        hints = SearchArbiter._get_hints("deep_task")
        parts: List[str] = []

        # 1. 核心领域搜索指引
        if domain_context:
            parts.append(hints["domain_fragment"].format(domain_context=domain_context))
        else:
            parts.append(hints["domain_fallback"])

        # 2. 时效性实体特别提醒
        if freshness.extracted_entities:
            entity_list = "、".join(freshness.extracted_entities[:4])
            parts.append(hints["entity_fragment"].format(entity_list=entity_list))

        # 3. 时间范围指引
        time_range = freshness.suggested_time_range
        if time_range and time_range != "past_2years":
            range_map = {
                "past_day": "最近24小时内",
                "past_week": "最近一周内",
                "past_month": "最近一月内",
                "past_year": "最近一年内",
            }
            range_desc = range_map.get(time_range, "近期")
            parts.append(hints["time_range_fragment"].format(time_range_desc=range_desc))

        # 4. 来源可信度要求
        parts.append(hints["source_credibility"])

        # 5. 输出格式要求
        parts.append(hints["format_requirement"])
        parts.append(hints["conflict_rule"])

        return "\n".join(parts)

    @staticmethod
    def _build_basic_task_hint(domain_context: str, freshness: FreshnessResult) -> str:
        """生成基础任务型搜索指引"""
        hints = SearchArbiter._get_hints("basic_task")
        if domain_context:
            parts = [hints["domain_template"].format(domain_context=domain_context)]
        else:
            parts = [hints["fallback"]]
        if freshness.extracted_entities:
            entity_list = "、".join(freshness.extracted_entities[:3])
            parts.append(hints["entity_fragment"].format(entity_list=entity_list))
        parts.append(hints["source_tagging"])
        return "\n".join(parts)

    @staticmethod
    def _build_factual_chat_hint(user_input: str, freshness: FreshnessResult) -> str:
        """生成事实型闲聊搜索指引"""
        hints = SearchArbiter._get_hints("factual_chat")
        parts = [hints["opener"]]
        if freshness.extracted_entities:
            entity_list = "、".join(freshness.extracted_entities[:3])
            parts.append(hints["entity_fragment"].format(entity_list=entity_list))
        if freshness.suggested_time_range:
            range_map = {"past_day": "最新", "past_week": "近期", "past_month": "近一月"}
            range_desc = range_map.get(freshness.suggested_time_range, "最新")
            parts.append(hints["time_range_fragment"].format(time_range_desc=range_desc))
        parts.append(hints["source_tagging"])
        return "\n".join(parts)

    @staticmethod
    def _build_mixed_hint(user_input: str, domain_context: str, freshness: FreshnessResult) -> str:
        """生成混合型搜索指引"""
        hints = SearchArbiter._get_hints("mixed")
        task_parts: List[str] = []
        if domain_context:
            task_parts.append(hints["domain_template"].format(domain_context=domain_context))
        else:
            task_parts.append(hints["fallback"])

        if freshness.temporal_score > 0.5:
            task_parts.append(hints["temporal_fragment"])
        task_parts.append(hints["source_tagging"])
        task_parts.append("网络信息请标注来源。")
        return "\n".join(task_parts)

    @staticmethod
    def _generate_l2_queries(
        user_input: str,
        domain_context: str,
        freshness: FreshnessResult,
    ) -> List[str]:
        """为 L2 WebRetriever 生成优化的搜索查询词"""
        queries: List[str] = []

        # 主查询: 领域 + 核心关键词
        if domain_context:
            queries.append(f"{domain_context} 最新 2025 2026 数据 政策")

        # 如果有时效性实体，追加精确查询
        for entity in freshness.extracted_entities[:2]:
            # 清理 entity 前缀（如 "product_version:iPhone 17" → "iPhone 17"）
            clean_entity = entity.split(":")[-1] if ":" in entity else entity
            queries.append(clean_entity)

        return list(dict.fromkeys(queries))  # 去重保序

    @staticmethod
    def _detect_verifiability(
        user_input: str,
        domain_context: str,
        project_data: Optional[Dict[str, Any]],
    ) -> float:
        """检测问题的可验证性（是否需要现实世界的事实来佐证）"""
        score = 0.5  # 默认中等
        
        text_lower = (user_input + " " + domain_context).lower()

        # 高可验证性关键词命中
        hits = sum(1 for kw in VERIFIABLE_DOMAIN_KEYWORDS if kw in text_lower)
        if hits > 0:
            score = min(1.0, 0.6 + 0.08 * hits)

        # 项目数据中有具体数值 → 更需要验证
        if isinstance(project_data, dict):
            numeric_fields = ["goal_usd", "duration_days", "category_risk", "urgency_score"]
            has_numeric = any(str(project_data.get(f, "")) for f in numeric_fields if project_data.get(f))
            if has_numeric:
                score = max(score, 0.75)

        # 疑问/否定/比较句式 → 需要验证
        question_markers = ["吗", "?", "？", "是不是", "对不对", "多少", "怎么样", "如何"]
        if any(m in user_input for m in question_markers):
            score = max(score, 0.8)

        # 纯主观表达 → 降低可验证性
        subjective = ["我觉得", "感觉", "认为", "希望", "想问问", "帮我看看"]
        if any(s in user_input for s in subjective):
            score = min(score, 0.5)

        return round(score, 4)


def decide_search(
    user_input: str,
    intent_type: str = "task",
    domain_context: str = "",
    **kwargs,
) -> SearchDecision:
    """
    快捷函数: 执行一次搜索决策
    
    供 m7_inference_runner 等调用方直接使用，
    无需手动创建 SearchArbiter 实例。
    """
    arbiter = SearchArbiter(**{k: v for k, v in kwargs.items() if k in ("config_path", "task_threshold", "chat_threshold")})
    return arbiter.decide(
        user_input=user_input,
        intent_type=intent_type,
        domain_context=domain_context,
    )
