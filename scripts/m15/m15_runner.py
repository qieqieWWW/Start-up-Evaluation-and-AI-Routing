"""
M15 消融执行器

核心函数 run_ablation_experiments() 直接调用 M8/M7 模块，
对每个消融维度的每条条件执行一遍分析并收集结果。

兼容 main_ui.py 的 run_analysis_pipeline() 调用模式。
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .m15_core import (
    AblationDimension,
    AblationCondition,
    AblationConfig,
    AblationReport,
    SingleConditionResult,
    build_default_config,
)

logger = logging.getLogger("m15_runner")

# ==========================================
# 路径与延迟导入
# ==========================================

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_M7_DIR = _SCRIPTS_DIR / "m7"
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _ensure_sys_path():
    for p in [str(_SCRIPTS_DIR), str(_M7_DIR), str(_PROJECT_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _import_m8():
    _ensure_sys_path()
    from m8_rule_adapter import judge_project_risk_m8, analyze_text_risk
    return judge_project_risk_m8, analyze_text_risk


def _import_m7_router():
    _ensure_sys_path()
    from m7_router import route_experts
    return route_experts


def _import_m7_inference():
    _ensure_sys_path()
    from m7_inference_runner import (
        run_expert_llm_inference,
        run_expert_llm_inference_with_blender,
    )
    return run_expert_llm_inference, run_expert_llm_inference_with_blender


def _import_m9():
    _ensure_sys_path()
    sys.path.insert(0, str(_SCRIPTS_DIR))
    from m9 import M9UltimateRiskEngine
    return M9UltimateRiskEngine


# ==========================================
# 单条件执行
# ==========================================

def _run_single_condition(
    condition: AblationCondition,
    config: AblationConfig,
) -> SingleConditionResult:
    """对一条消融条件执行分析，返回 SingleConditionResult"""
    params = condition.params
    project_data = config.project_data
    user_query = config.user_query
    api_key = config.api_key

    start = time.time()
    result = SingleConditionResult(
        label=condition.label,
        params=params,
    )

    try:
        # ----- 导入模块 -----
        judge_project_risk_m8, analyze_text_risk = _import_m8()
        route_experts = _import_m7_router()

        project_data_with_text = {**project_data, "user_input": user_query}

        # ----- Step 1: 始终执行 M8 风险判定 -----
        risk_level, risk_reasons, intermediate = judge_project_risk_m8(
            project_data_with_text, verbose=False
        )
        result.m8_risk_score = intermediate.get("combined_risk", 0.0)

        # ----- 组件完整度消融的特殊处理：决定跑多远 -----
        component_level = params.get("component_level", "full_pipeline")

        # 如果只需要 M8 only，在此返回
        if component_level == "m8_only":
            result.risk_level = risk_level
            result.component_level = "m8_only"
            elapsed = (time.time() - start) * 1000
            result.execution_time_ms = elapsed
            return result

        # ----- Step 2: 构建路由参数 -----
        route_mode = params.get("route_mode", "default")
        route_kwargs = {
            "risk_level": risk_level,
            "intermediate": intermediate,
            "project_data": project_data,
            "user_id": config.user_id,
            "user_input": user_query,
        }

        if route_mode == "default":
            route_kwargs["profile_db_path"] = None
            route_kwargs["global_kb_path"] = None
        elif route_mode == "with_profile":
            profile_path = _CONFIG_DIR / "m7_user_profile_records.json"
            route_kwargs["profile_db_path"] = str(profile_path) if profile_path.exists() else None
            route_kwargs["global_kb_path"] = None
        elif route_mode == "with_kb":
            kb_path = _CONFIG_DIR / "m7_global_knowledge_base.json"
            route_kwargs["profile_db_path"] = None
            route_kwargs["global_kb_path"] = str(kb_path) if kb_path.exists() else None
        else:  # full
            profile_path = _CONFIG_DIR / "m7_user_profile_records.json"
            kb_path = _CONFIG_DIR / "m7_global_knowledge_base.json"
            route_kwargs["profile_db_path"] = str(profile_path) if profile_path.exists() else None
            route_kwargs["global_kb_path"] = str(kb_path) if kb_path.exists() else None

        # 执行路由
        route_result = route_experts(**route_kwargs)
        selected_experts = route_result.get("selected_experts", [])
        result.selected_experts = [
            e.get("name", "") for e in selected_experts
        ]
        result.confidence = route_result.get("confidence", 0.0)
        result.route_reason = route_result.get("route_reason", "")
        result.routing_scores = route_result.get("routing_scores", {})
        result.expert_count = len(selected_experts)

        # 如果只需要 M8 → M7，在此返回
        if component_level == "m8_m7":
            result.risk_level = risk_level
            result.component_level = "m8_m7"
            elapsed = (time.time() - start) * 1000
            result.execution_time_ms = elapsed
            return result

        # ----- Step 3: LLM 推理（专家绑定的固定环节） -----
        dimension = config.dimension
        need_llm = True  # 默认始终调 LLM（路由策略消融必须走完整链路）

        # LLM_REASONING 维度：根据条件决定是否跳过 LLM
        if dimension == AblationDimension.LLM_REASONING:
            enable_llm = params.get("enable_llm", True)
            need_llm = api_key and enable_llm

        # COMPONENT_INTEGRITY 维度：渐进式，levels 决定是否到 LLM
        if dimension == AblationDimension.COMPONENT_INTEGRITY:
            need_llm = api_key and component_level in ("m8_m7_llm", "m8_m7_blender", "full_pipeline")

        if not need_llm:
            result.risk_level = risk_level
            result.component_level = component_level if "component" in params else "rule_only"
            selected_expert_names = [e.get("name", "") for e in selected_experts]
            result.fused_summary = (
                f"路由完成：选中 {', '.join(selected_expert_names)}，置信度 {result.confidence}"
            )
            elapsed = (time.time() - start) * 1000
            result.execution_time_ms = elapsed
            return result

        # 需要 LLM，设置 API key
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key

        run_expert_llm_inference, run_expert_llm_inference_with_blender = _import_m7_inference()

        # 是否启用 Blender 融合（由条件参数或全局配置决定）
        enable_blender = params.get("enable_blender", config.enable_blender)

        common_llm_kwargs = {
            "risk_level": risk_level,
            "reasons": risk_reasons,
            "intermediate": intermediate,
            "project_data": project_data,
            "route_result": route_result,
            "user_input": user_query,
            "user_id": config.user_id,
        }

        if enable_blender:
            llm_result = run_expert_llm_inference_with_blender(
                **common_llm_kwargs,
                use_llm_fuser=True,
            )
            fused = llm_result.get("fused_result", {})
            if isinstance(fused, dict):
                result.fused_summary = fused.get("fused_risk_summary", "")
                actions = fused.get("fused_actions", [])
                alerts = fused.get("fused_alerts", [])
                result.action_count = len(actions)
                result.alert_count = len(alerts)
                # 大致估算输出量
                import json
                try:
                    result.llm_output_size = len(json.dumps(llm_result))
                except Exception:
                    result.llm_output_size = 0
        else:
            candidates = run_expert_llm_inference(**common_llm_kwargs)
            result.action_count = len(candidates)
            result.fused_summary = str(candidates[:1]) if candidates else ""
            import json
            try:
                result.llm_output_size = len(json.dumps(candidates))
            except Exception:
                result.llm_output_size = 0

        result.risk_level = risk_level
        result.component_level = component_level

        # 如果只需要到 Blender，在此返回
        if component_level in ("m8_m7_blender", "m8_m7_llm"):
            elapsed = (time.time() - start) * 1000
            result.execution_time_ms = elapsed
            return result

        # ----- Step 4: M9 全流水线（仅 full_pipeline 模式） -----
        if component_level == "full_pipeline":
            try:
                M9UltimateRiskEngine = _import_m9()
                engine = M9UltimateRiskEngine(
                    api_key=api_key if api_key else None,
                    verbose=False,
                )
                m9_result = engine.run_full_decision(
                    user_query=user_query,
                    project_data=project_data,
                    user_id=config.user_id,
                    enable_ood_test=False,
                    enable_perf_monitor=False,
                )
                if isinstance(m9_result, dict) and m9_result.get("code") == 0:
                    final = m9_result.get("final", {})
                    if isinstance(final, dict):
                        if not result.fused_summary and final.get("fused_risk_summary"):
                            result.fused_summary = final["fused_risk_summary"]
            except Exception as e:
                logger.warning(f"M9 执行异常（full_pipeline）: {e}")

        elapsed = (time.time() - start) * 1000
        result.execution_time_ms = elapsed
        return result

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        result.execution_time_ms = elapsed
        result.error = str(e)
        logger.error(f"消融条件 '{condition.label}' 执行异常: {e}")
        return result


# ==========================================
# 消融实验主入口
# ==========================================

def run_ablation_experiments(
    project_data: Dict[str, Any],
    user_query: str = "",
    api_key: str = "",
    enable_blender: bool = True,
    user_id: str = "m15_ablation_user",
    m8_result: Optional[Dict[str, Any]] = None,
    m7_result: Optional[Dict[str, Any]] = None,
    selected_dimensions: Optional[List[AblationDimension]] = None,
) -> Dict[str, Any]:
    """
    消融实验主入口。

    对 selected_dimensions 中的每个维度，用默认消融条件运行对比实验。
    返回 { dimension_label: report_dict } 的字典，直接用于 results["modules"]["M15_消融实验"]。

    参数:
        project_data: 项目特征数据
        user_query: 用户查询文本
        api_key: DeepSeek API key
        enable_blender: 是否启用 Blender
        user_id: 用户标识
        m8_result: 主流水线的 M8 结果（可选，用于避免重复计算）
        m7_result: 主流水线的 M7 结果（可选）
        selected_dimensions: 要运行的消融维度列表，默认 3 个核心维度
    """
    if selected_dimensions is None:
        selected_dimensions = AblationDimension.core_dimensions()

    reports: List[AblationReport] = []

    for dimension in selected_dimensions:
        config = build_default_config(
            dimension=dimension,
            project_data=project_data,
            user_query=user_query,
            api_key=api_key,
            user_id=user_id,
            enable_blender=enable_blender,
            m8_result=m8_result,
            m7_result=m7_result,
        )

        report = run_ablation_experiment(config)
        reports.append(report)

    # 组装返回结果
    result: Dict[str, Any] = {
        "status": "completed",
        "dimension_count": len(reports),
        "reports": {},
    }

    for report in reports:
        dim_key = report.config.dimension.value
        result["reports"][dim_key] = report.to_dict()

    # 添加简要的对比摘要文本
    result["summary_text"] = _build_summary_text(reports)

    return result


def run_ablation_experiment(
    config: AblationConfig,
) -> AblationReport:
    """
    执行一次指定维度的消融实验。

    对 config.conditions 中的每条条件调用 _run_single_condition()，
    收集结果并生成对比摘要。
    """
    results: List[SingleConditionResult] = []

    for condition in config.conditions:
        cond_result = _run_single_condition(condition, config)
        results.append(cond_result)

    # 生成对比摘要
    from .m15_report import build_comparison_summary
    comparison_summary = build_comparison_summary(results, config.dimension)

    return AblationReport(
        config=config,
        results=results,
        comparison_summary=comparison_summary,
    )


def _build_summary_text(reports: List[AblationReport]) -> str:
    """生成简短的对比摘要文本"""
    lines = [f"运行了 {len(reports)} 个消融维度："]

    for report in reports:
        dim_label = report.dimension_label
        dim_cn = {
            "routing_strategy": "路由策略",
            "llm_reasoning": "LLM 推理",
            "component_integrity": "组件完整度",
        }.get(dim_label, dim_label)

        cs = report.comparison_summary
        max_diff = cs.get("max_difference", "N/A")
        baseline = cs.get("baseline_label", "—")
        best = cs.get("best_label", "—")

        lines.append(f"  • {dim_cn}: Baseline={baseline}, Best={best}, 最大差异={max_diff}")

    return "\n".join(lines)
