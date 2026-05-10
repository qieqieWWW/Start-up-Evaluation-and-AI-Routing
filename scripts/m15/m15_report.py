"""
M15 报告生成：对比摘要 + 图表数据 + JSON 持久化

build_comparison_summary() 被 m15_runner.py 调用，
对每个维度的消融结果计算对比指标。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .m15_core import (
    AblationDimension,
    AblationReport,
    SingleConditionResult,
)


def build_comparison_summary(
    results: List[SingleConditionResult],
    dimension: AblationDimension,
) -> Dict[str, Any]:
    """
    对比摘要生成。

    对比基线（第一条条件）与其它条件的差异，
    自动识别最大差异、最优指标。
    """
    if not results:
        return {"error": "no_results", "max_difference": 0}

    baseline = results[0]

    # 对比指标
    comparisons = []
    max_conf_diff = 0.0
    max_time_diff = 0.0
    best_confidence = baseline.confidence
    best_label = baseline.label

    for r in results:
        conf_diff = r.confidence - baseline.confidence
        time_diff = r.execution_time_ms - baseline.execution_time_ms

        if abs(conf_diff) > abs(max_conf_diff):
            max_conf_diff = conf_diff
        if abs(time_diff) > abs(max_time_diff):
            max_time_diff = time_diff

        if r.confidence > best_confidence:
            best_confidence = r.confidence
            best_label = r.label

        comparison = {
            "label": r.label,
            "confidence": r.confidence,
            "confidence_diff": round(conf_diff, 3),
            "execution_time_ms": round(r.execution_time_ms, 2),
            "time_diff_ms": round(time_diff, 2),
            "expert_count": r.expert_count,
            "selected_experts": r.selected_experts,
            "action_count": r.action_count,
            "alert_count": r.alert_count,
            "llm_output_size": r.llm_output_size,
            "error": r.error,
        }

        if dimension == AblationDimension.COMPONENT_INTEGRITY:
            comparison["m8_risk_score"] = round(r.m8_risk_score, 4)
            comparison["component_level"] = r.component_level

        comparisons.append(comparison)

    # 构建 summary
    summary: Dict[str, Any] = {
        "dimension": dimension.value,
        "baseline_label": baseline.label,
        "best_label": best_label,
        "max_difference": {
            "confidence": round(max_conf_diff, 3),
            "execution_time_ms": round(max_time_diff, 2),
        },
        "condition_count": len(results),
        "comparisons": comparisons,
    }

    # 维度特定的关键发现
    if dimension == AblationDimension.ROUTING_STRATEGY:
        summary["key_finding"] = _summarize_routing(results, baseline)
    elif dimension == AblationDimension.LLM_REASONING:
        summary["key_finding"] = _summarize_llm(results, baseline)
    elif dimension == AblationDimension.COMPONENT_INTEGRITY:
        summary["key_finding"] = _summarize_integrity(results)

    return summary


def _summarize_routing(
    results: List[SingleConditionResult],
    baseline: SingleConditionResult,
) -> str:
    """路由策略消融的关键发现"""
    findings = []
    for r in results[1:]:
        conf_diff = r.confidence - baseline.confidence
        expert_shift = set(r.selected_experts) - set(baseline.selected_experts)
        if conf_diff > 0.05:
            findings.append(
                f"「{r.label}」置信度提升 +{conf_diff:.1%}"
            )
        if expert_shift:
            findings.append(
                f"「{r.label}」专家选择偏移: {', '.join(expert_shift)}"
            )
    return "；".join(findings) if findings else "各条件差异不明显"


def _summarize_llm(
    results: List[SingleConditionResult],
    baseline: SingleConditionResult,
) -> str:
    """LLM 推理消融的关键发现"""
    findings = []
    for r in results[1:]:
        time_cost = r.execution_time_ms - baseline.execution_time_ms
        findings.append(
            f"「{r.label}」执行时间 {r.execution_time_ms:.0f}ms "
            f"(较基线 +{time_cost:.0f}ms), "
            f"Action数={r.action_count}, 输出大小={r.llm_output_size}B"
        )
    return "；".join(findings) if findings else "无显著差异"


def _summarize_integrity(
    results: List[SingleConditionResult],
) -> str:
    """组件完整度消融的关键发现"""
    findings = []
    prev_result = None
    for r in results:
        if prev_result:
            time_delta = r.execution_time_ms - prev_result.execution_time_ms
            findings.append(
                f"「{r.label}」耗时 {r.execution_time_ms:.0f}ms "
                f"(较上一级 +{time_delta:.0f}ms), "
                f"Action={r.action_count}"
            )
        else:
            findings.append(
                f"「{r.label}」耗时 {r.execution_time_ms:.0f}ms, "
                f"风险分值={r.m8_risk_score:.2f}"
            )
        prev_result = r
    return " → ".join(findings)


def prepare_chart_data(
    reports: List[AblationReport],
) -> Dict[str, Any]:
    """
    准备图表数据（用于 UI 渲染 bar_chart 等）。

    返回结构：
    {
        "dimension_key": {
            "labels": [...],
            "confidence": [...],
            "execution_time_ms": [...],
            "action_count": [...],
        },
        ...
    }
    """
    chart_data: Dict[str, Any] = {}

    for report in reports:
        dim_key = report.config.dimension.value
        labels = []
        confidence = []
        exec_times = []
        action_counts = []

        for r in report.results:
            labels.append(r.label)
            confidence.append(r.confidence)
            exec_times.append(r.execution_time_ms)
            action_counts.append(r.action_count)

        chart_data[dim_key] = {
            "labels": labels,
            "confidence": confidence,
            "execution_time_ms": exec_times,
            "action_count": action_counts,
        }

    # 增加汇总
    if len(reports) > 1:
        chart_data["_comparison"] = _build_multi_dimension_comparison(reports)

    return chart_data


def _build_multi_dimension_comparison(
    reports: List[AblationReport],
) -> Dict[str, Any]:
    """多维度汇总数据"""
    dim_labels = []
    baseline_confidences = []
    best_confidences = []

    for report in reports:
        dim_cn = {
            "routing_strategy": "路由策略",
            "llm_reasoning": "LLM 推理",
            "component_integrity": "组件完整度",
        }.get(report.dimension_label, report.dimension_label)

        dim_labels.append(dim_cn)
        results = report.results
        baseline_confidences.append(results[0].confidence if results else 0)
        max_conf = max(r.confidence for r in results) if results else 0
        best_confidences.append(max_conf)

    return {
        "dimension_labels": dim_labels,
        "baseline_confidences": baseline_confidences,
        "best_confidences": best_confidences,
    }


# ==========================================
# 持久化
# ==========================================

_ABLATION_REPORT_DIR = Path(__file__).resolve().parents[2] / "Kickstarter_Clean" / "m15_ablation_reports"


def save_report_to_json(
    report: AblationReport,
    output_dir: Optional[str] = None,
) -> str:
    """
    将消融报告保存为 JSON 文件。

    返回保存的文件路径。
    """
    dir_path = Path(output_dir) if output_dir else _ABLATION_REPORT_DIR
    dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"m15_ablation_{report.dimension_label}_{timestamp}.json"
    filepath = dir_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report.to_json())

    return str(filepath)
