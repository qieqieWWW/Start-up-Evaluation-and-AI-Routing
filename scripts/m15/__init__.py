"""
M15 应用封装与重构 — 消融实验模块

提供消融实验核心功能，嵌入到 main_ui.py 的 run_analysis_pipeline() 中。

导出:
    run_ablation_experiments: 消融实验主入口
    AblationDimension: 消融维度枚举
    AblationReport: 消融报告数据类
    build_default_config: 默认配置工厂
    save_report_to_json: 报告持久化
"""

from .m15_core import (
    AblationDimension,
    AblationCondition,
    AblationConfig,
    AblationReport,
    SingleConditionResult,
    build_default_config,
)
from .m15_runner import run_ablation_experiments, run_ablation_experiment
from .m15_report import (
    build_comparison_summary,
    prepare_chart_data,
    save_report_to_json,
)

__all__ = [
    "run_ablation_experiments",
    "run_ablation_experiment",
    "AblationDimension",
    "AblationCondition",
    "AblationConfig",
    "AblationReport",
    "SingleConditionResult",
    "build_default_config",
    "build_comparison_summary",
    "prepare_chart_data",
    "save_report_to_json",
]
