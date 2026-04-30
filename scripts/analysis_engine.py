#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
M13 模块：实验分析与开源工程化
================================================================================

功能描述：
    - 自动处理实验数据
    - 进行统计分析与假设验证
    - 生成 analysis_report.ipynb (Notebook)
    - 整理最终实验结论
    - 确保代码已清理，准备发布到 GitHub 或 arXiv

设计原则：
    - 每个逻辑模块对应一个独立文件
    - Clean Code 规范
    - 可复现的研究流程

接口依赖：
    - 从 data_processor.py 导入 ProcessedExperiment
    - 从 experiment_runner.py 导入 ExperimentSummary

前置条件：
    - 需要安装 jupyter: pip install jupyter nbformat nbconvert
    - 需要安装 scipy: pip install scipy

作者：阶段4开发团队
日期：2026-04-09
================================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import subprocess

# 数据处理
import numpy as np
import pandas as pd

# 统计检验
try:
    from scipy import stats
    from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy 未安装，统计检验功能将受限。请运行 'pip install scipy' 安装。")

# Notebook 生成
try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False
    print("警告: nbformat 未安装，Notebook 生成功能将受限。请运行 'pip install nbformat' 安装。")


# ================================================================================
# 配置常量
# ================================================================================

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 默认路径
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
DEFAULT_EXPERIMENT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_results")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis_reports")


# ================================================================================
# 数据类定义
# ================================================================================

@dataclass
class AnalysisConfig:
    """分析配置"""
    data_dir: str = DEFAULT_DATA_DIR
    results_dir: str = DEFAULT_EXPERIMENT_RESULTS_DIR
    output_dir: str = DEFAULT_OUTPUT_DIR
    significance_level: float = 0.05
    confidence_level: float = 0.95
    random_seed: int = 42
    verbose: bool = True


@dataclass
class HypothesisResult:
    """假设检验结果"""
    hypothesis_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    significance_level: float
    is_significant: bool
    conclusion: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hypothesis_name": self.hypothesis_name,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "test_statistic": float(self.test_statistic),
            "p_value": float(self.p_value),
            "significance_level": float(self.significance_level),
            "is_significant": bool(self.is_significant),
            "conclusion": str(self.conclusion),
            "effect_size": float(self.effect_size) if self.effect_size is not None else None,
            "confidence_interval": self.confidence_interval
        }


@dataclass
class AnalysisReport:
    """分析报告"""
    title: str
    timestamp: str
    data_summary: Dict[str, Any]
    descriptive_stats: Dict[str, Dict[str, float]]
    hypothesis_tests: List[HypothesisResult]
    correlation_analysis: Dict[str, float]
    key_findings: List[str]
    conclusions: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "timestamp": self.timestamp,
            "data_summary": self.data_summary,
            "descriptive_stats": self.descriptive_stats,
            "hypothesis_tests": [h.to_dict() for h in self.hypothesis_tests],
            "correlation_analysis": self.correlation_analysis,
            "key_findings": self.key_findings,
            "conclusions": self.conclusions,
            "recommendations": self.recommendations
        }


# ================================================================================
# 数据加载器
# ================================================================================

class ExperimentDataLoader:
    """
    实验数据加载器

    功能：
        - 从多个来源加载实验数据
        - 数据验证与清洗
    """

    def __init__(self, data_dir: str, results_dir: str):
        """
        初始化数据加载器

        Args:
            data_dir: 处理后数据目录
            results_dir: 实验结果目录
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self._experiments: List[Dict[str, Any]] = []
        self._df: Optional[pd.DataFrame] = None

    def load_all(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        加载所有实验数据

        Returns:
            (原始数据列表, DataFrame)
        """
        all_experiments = []

        # 从JSONL加载
        jsonl_files = list(self.data_dir.glob("experiments_*.jsonl"))
        for file_path in jsonl_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_experiments.append(json.loads(line))

        # 从结果文件加载
        result_files = list(self.results_dir.glob("experiment_results_*.json"))
        for file_path in result_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_experiments.extend(data.get("experiments", []))

        # 去重
        seen = set()
        unique_experiments = []
        for exp in all_experiments:
            exp_id = exp.get("experiment_id", "")
            if exp_id not in seen:
                seen.add(exp_id)
                unique_experiments.append(exp)

        self._experiments = unique_experiments
        self._df = self._to_dataframe(unique_experiments)

        return unique_experiments, self._df

    def _to_dataframe(self, experiments: List[Dict[str, Any]]) -> pd.DataFrame:
        """转换为DataFrame"""
        if not experiments:
            return pd.DataFrame()

        rows = []
        for exp in experiments:
            row = {
                "experiment_id": exp.get("experiment_id", ""),
                "timestamp": exp.get("timestamp", ""),
                "duration_ms": exp.get("duration_ms", 0),
                "status": exp.get("status", "unknown"),
            }

            # 指标
            metrics = exp.get("metrics", {})
            for k, v in metrics.items():
                row[f"metric_{k}"] = v

            # 初始状态
            initial = exp.get("initial_state", {})
            for k, v in initial.items():
                row[f"init_{k}"] = v

            # 最终状态
            final = exp.get("final_state", {})
            for k, v in final.items():
                row[f"final_{k}"] = v

            rows.append(row)

        return pd.DataFrame(rows)

    @property
    def experiments(self) -> List[Dict[str, Any]]:
        """获取实验数据"""
        return self._experiments

    @property
    def dataframe(self) -> pd.DataFrame:
        """获取DataFrame"""
        return self._df


# ================================================================================
# 统计分析引擎
# ================================================================================

class StatisticalAnalyzer:
    """
    统计分析引擎

    功能：
        - 描述性统计
        - 假设检验
        - 相关性分析
    """

    def __init__(self, df: pd.DataFrame, significance_level: float = 0.05):
        """
        初始化分析器

        Args:
            df: 实验数据DataFrame
            significance_level: 显著性水平
        """
        self.df = df
        self.significance_level = significance_level
        self._results: List[HypothesisResult] = []

    def compute_descriptive_stats(
        self,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        计算描述性统计

        Args:
            columns: 要分析的列，None则分析所有数值列

        Returns:
            描述性统计字典
        """
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            columns = numeric_cols.tolist()

        results = {}
        for col in columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                results[col] = {
                    "count": len(data),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "Q25": float(data.quantile(0.25)),
                    "median": float(data.median()),
                    "Q75": float(data.quantile(0.75)),
                    "max": float(data.max()),
                    "skewness": float(stats.skew(data)) if SCIPY_AVAILABLE else 0.0,
                    "kurtosis": float(stats.kurtosis(data)) if SCIPY_AVAILABLE else 0.0
                }

        return results

    def test_normality(
        self,
        column: str,
        method: str = "shapiro"
    ) -> Dict[str, Any]:
        """
        正态性检验

        Args:
            column: 列名
            method: 检验方法 ("shapiro", "ks")

        Returns:
            检验结果
        """
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}

        data = self.df[column].dropna()
        if len(data) < 3:
            return {"error": "数据点不足"}

        if method == "shapiro":
            statistic, p_value = stats.shapiro(data)
        else:  # kolmogorov-smirnov
            statistic, p_value = stats.kstest(data, 'norm')

        return {
            "column": column,
            "test": method,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > self.significance_level
        }

    def compare_groups(
        self,
        group_col: str,
        value_col: str,
        test_type: str = "ttest"
    ) -> HypothesisResult:
        """
        组间比较检验

        Args:
            group_col: 分组列
            value_col: 数值列
            test_type: 检验类型 ("ttest", "mannwhitney", "anova")

        Returns:
            假设检验结果
        """
        if not SCIPY_AVAILABLE:
            return HypothesisResult(
                hypothesis_name="Group Comparison",
                null_hypothesis="两组均值相等",
                alternative_hypothesis="两组均值不相等",
                test_statistic=0,
                p_value=1,
                significance_level=self.significance_level,
                is_significant=False,
                conclusion="scipy not available"
            )

        groups = self.df[group_col].unique()
        if len(groups) < 2:
            return HypothesisResult(
                hypothesis_name="Group Comparison",
                null_hypothesis="两组均值相等",
                alternative_hypothesis="两组均值不相等",
                test_statistic=0,
                p_value=1,
                significance_level=self.significance_level,
                is_significant=False,
                conclusion="只有一组数据"
            )

        group_data = [
            self.df[self.df[group_col] == g][value_col].dropna()
            for g in groups
        ]

        # 过滤空组
        group_data = [g for g in group_data if len(g) > 0]

        if len(group_data) < 2:
            return HypothesisResult(
                hypothesis_name="Group Comparison",
                null_hypothesis="两组均值相等",
                alternative_hypothesis="两组均值不相等",
                test_statistic=0,
                p_value=1,
                significance_level=self.significance_level,
                is_significant=False,
                conclusion="有效组数不足"
            )

        # 执行检验
        if test_type == "ttest" and len(group_data) == 2:
            stat, p = ttest_ind(group_data[0], group_data[1])
        elif test_type == "mannwhitney":
            stat, p = mannwhitneyu(*group_data)
        else:
            # Kruskal-Wallis 作为非参数替代
            stat, p = stats.kruskal(*group_data)

        is_sig = p < self.significance_level

        return HypothesisResult(
            hypothesis_name=f"{group_col} 对 {value_col} 的影响",
            null_hypothesis=f"不同 {group_col} 组之间 {value_col} 无显著差异",
            alternative_hypothesis=f"不同 {group_col} 组之间 {value_col} 存在显著差异",
            test_statistic=float(stat),
            p_value=float(p),
            significance_level=self.significance_level,
            is_significant=is_sig,
            conclusion=f"拒绝原假设" if is_sig else "无法拒绝原假设"
        )

    def test_correlation(
        self,
        col1: str,
        col2: str,
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        相关性检验

        Args:
            col1: 第一列
            col2: 第二列
            method: 相关方法 ("pearson", "spearman", "kendall")

        Returns:
            相关性结果
        """
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available"}

        data = self.df[[col1, col2]].dropna()

        if len(data) < 3:
            return {"error": "数据点不足"}

        if method == "pearson":
            corr, p = stats.pearsonr(data[col1], data[col2])
        elif method == "spearman":
            corr, p = stats.spearmanr(data[col1], data[col2])
        else:
            corr, p = stats.kendalltau(data[col1], data[col2])

        return {
            "column1": col1,
            "column2": col2,
            "method": method,
            "correlation": float(corr),
            "p_value": float(p),
            "is_significant": p < self.significance_level
        }

    def correlation_matrix(
        self,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        计算相关矩阵

        Args:
            columns: 要分析的列

        Returns:
            相关矩阵DataFrame
        """
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            columns = numeric_cols.tolist()

        subset = self.df[columns].dropna()
        return subset.corr()


# ================================================================================
# 假设验证器
# ================================================================================

class HypothesisValidator:
    """
    假设验证器

    功能：
        - 定义和验证研究假设
        - 生成结构化结论
    """

    def __init__(self, analyzer: StatisticalAnalyzer):
        """
        初始化验证器

        Args:
            analyzer: 统计分析器
        """
        self.analyzer = analyzer
        self.hypotheses: List[HypothesisResult] = []

    def validate_success_rate_hypothesis(
        self,
        expected_rate: float = 0.5
    ) -> HypothesisResult:
        """
        验证成功率假设

        Args:
            expected_rate: 期望成功率

        Returns:
            假设检验结果
        """
        df = self.analyzer.df
        success_count = (df["status"] == "success").sum()
        total_count = len(df)
        observed_rate = success_count / total_count if total_count > 0 else 0

        # 二项检验
        if not SCIPY_AVAILABLE:
            return HypothesisResult(
                hypothesis_name="成功率假设",
                null_hypothesis=f"成功率 = {expected_rate}",
                alternative_hypothesis=f"成功率 ≠ {expected_rate}",
                test_statistic=0,
                p_value=1,
                significance_level=self.analyzer.significance_level,
                is_significant=False,
                conclusion=f"观察成功率: {observed_rate:.2%}"
            )

        # 使用 scipy.stats.binomtest (scipy >= 1.7)
        try:
            from scipy.stats import binomtest
            p_value = binomtest(success_count, total_count, expected_rate).pvalue
        except ImportError:
            # 备用方法：正态近似
            stat = (observed_rate - expected_rate) / np.sqrt(expected_rate * (1 - expected_rate) / total_count)
            p_value = 2 * (1 - stats.norm.cdf(abs(stat)))

        is_sig = p_value < self.analyzer.significance_level

        return HypothesisResult(
            hypothesis_name="成功率假设",
            null_hypothesis=f"成功率 = {expected_rate}",
            alternative_hypothesis=f"成功率 ≠ {expected_rate}",
            test_statistic=observed_rate,
            p_value=float(p_value),
            significance_level=self.analyzer.significance_level,
            is_significant=is_sig,
            conclusion=f"观察成功率 {observed_rate:.2%} {'显著不同于' if is_sig else '与'}期望 {expected_rate:.2%}"
        )

    def validate_performance_improvement(
        self,
        metric_col: str,
        threshold: float = 0.0
    ) -> HypothesisResult:
        """
        验证性能改进假设

        Args:
            metric_col: 指标列名
            threshold: 期望阈值

        Returns:
            假设检验结果
        """
        df = self.analyzer.df
        data = df[metric_col].dropna()

        if len(data) == 0 or not SCIPY_AVAILABLE:
            return HypothesisResult(
                hypothesis_name=f"{metric_col} 达标假设",
                null_hypothesis=f"{metric_col} 均值 ≤ {threshold}",
                alternative_hypothesis=f"{metric_col} 均值 > {threshold}",
                test_statistic=0,
                p_value=1,
                significance_level=self.analyzer.significance_level,
                is_significant=False,
                conclusion="无数据"
            )

        # 单样本t检验
        from scipy.stats import ttest_1samp
        stat, p = ttest_1samp(data, threshold)

        return HypothesisResult(
            hypothesis_name=f"{metric_col} 达标假设",
            null_hypothesis=f"{metric_col} 均值 ≤ {threshold}",
            alternative_hypothesis=f"{metric_col} 均值 > {threshold}",
            test_statistic=float(stat),
            p_value=float(p / 2) if stat > 0 else float(1 - p / 2),  # 单尾
            significance_level=self.analyzer.significance_level,
            is_significant=p < self.analyzer.significance_level and stat > 0,
            conclusion=f"{metric_col} 均值 = {data.mean():.4f}"
        )

    def run_all_validations(self) -> List[HypothesisResult]:
        """
        运行所有预设验证

        Returns:
            验证结果列表
        """
        results = []

        # 成功率假设
        results.append(self.validate_success_rate_hypothesis())

        # 性能指标假设
        numeric_cols = self.analyzer.df.select_dtypes(include=[np.number]).columns
        for col in ["metric_total_reward", "metric_avg_reward"]:
            if col in numeric_cols:
                results.append(self.validate_performance_improvement(col))

        self.hypotheses = results
        return results


# ================================================================================
# 报告生成器
# ================================================================================

class ReportGenerator:
    """
    报告生成器

    功能：
        - 生成结构化分析报告
        - 生成Jupyter Notebook
        - 生成Markdown文档
    """

    def __init__(self, config: AnalysisConfig):
        """
        初始化报告生成器

        Args:
            config: 分析配置
        """
        self.config = config
        self._report: Optional[AnalysisReport] = None

    def generate_report(
        self,
        df: pd.DataFrame,
        hypotheses: List[HypothesisResult],
        key_findings: List[str]
    ) -> AnalysisReport:
        """
        生成分析报告

        Args:
            df: 实验数据DataFrame
            hypotheses: 假设检验结果
            key_findings: 关键发现

        Returns:
            分析报告
        """
        # 描述性统计
        analyzer = StatisticalAnalyzer(df, self.config.significance_level)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        desc_stats = analyzer.compute_descriptive_stats(numeric_cols)

        # 相关性分析
        corr_matrix = analyzer.correlation_matrix(numeric_cols)
        top_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # 只记录中等以上相关
                    top_correlations.append({
                        "pair": f"{col1} vs {col2}",
                        "correlation": float(corr_val)
                    })

        # 生成结论
        conclusions = self._generate_conclusions(df, hypotheses, key_findings)

        # 生成建议
        recommendations = self._generate_recommendations(df, hypotheses)

        self._report = AnalysisReport(
            title="Kickstarter 仿真实验分析报告",
            timestamp=datetime.now().isoformat(),
            data_summary={
                "total_experiments": len(df),
                "success_rate": float((df["status"] == "success").mean()),
                "avg_duration_ms": float(df["duration_ms"].mean()),
                "feature_count": len(numeric_cols)
            },
            descriptive_stats=desc_stats,
            hypothesis_tests=hypotheses,
            correlation_analysis={
                c["pair"]: c["correlation"] for c in top_correlations
            },
            key_findings=key_findings,
            conclusions=conclusions,
            recommendations=recommendations
        )

        return self._report

    def _generate_conclusions(
        self,
        df: pd.DataFrame,
        hypotheses: List[HypothesisResult],
        key_findings: List[str]
    ) -> List[str]:
        """生成结论"""
        conclusions = []

        # 基于假设检验
        sig_tests = [h for h in hypotheses if h.is_significant]
        if sig_tests:
            conclusions.append(
                f"在 {self.config.significance_level} 显著性水平下，"
                f"共发现 {len(sig_tests)} 个统计显著差异。"
            )
        else:
            conclusions.append(
                f"在 {self.config.significance_level} 显著性水平下，"
                f"未发现统计显著差异。"
            )

        # 基于成功率
        success_rate = (df["status"] == "success").mean()
        if success_rate > 0.7:
            conclusions.append(f"系统表现优异，成功率达到 {success_rate:.1%}。")
        elif success_rate > 0.5:
            conclusions.append(f"系统表现中等，成功率为 {success_rate:.1%}。")
        else:
            conclusions.append(f"系统表现有待改进，成功率仅为 {success_rate:.1%}。")

        # 添加关键发现
        conclusions.extend(key_findings[:3])

        return conclusions

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        hypotheses: List[HypothesisResult]
    ) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于显著性检验的建议
        for hyp in hypotheses:
            if hyp.is_significant:
                if "成功率" in hyp.hypothesis_name:
                    recommendations.append("建议进一步分析影响成功率的关键因素。")
                elif "reward" in hyp.hypothesis_name.lower():
                    recommendations.append("建议优化奖励机制以提升系统性能。")

        # 基于相关性的建议
        analyzer = StatisticalAnalyzer(df)
        corr = analyzer.correlation_matrix()
        for col in corr.columns:
            if "risk" in col.lower() and "reward" in corr.columns:
                corr_val = corr.loc[col, "reward"] if col in corr.index and "reward" in corr.columns else 0
                if abs(corr_val) > 0.5:
                    recommendations.append(
                        "发现风险与奖励之间存在强相关，建议优化风险控制策略。"
                    )

        # 通用建议
        if not recommendations:
            recommendations.append("继续监控关键指标，收集更多数据以支持后续分析。")

        return recommendations

    def save_json_report(self, report: AnalysisReport, output_path: Optional[str] = None) -> str:
        """
        保存JSON格式报告

        Args:
            report: 分析报告
            output_path: 输出路径

        Returns:
            保存的文件路径
        """
        output_path = Path(output_path or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"analysis_report_{timestamp}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        return str(file_path)

    def generate_jupyter_notebook(
        self,
        report: AnalysisReport,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成Jupyter Notebook

        Args:
            report: 分析报告
            df: 实验数据
            output_path: 输出路径

        Returns:
            保存的文件路径
        """
        if not NBFORMAT_AVAILABLE:
            return "nbformat not available"

        output_path = Path(output_path or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"analysis_report_{timestamp}.ipynb"

        nb = new_notebook()
        nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        }

        cells = []

        # 标题
        cells.append(new_markdown_cell(f"# {report.title}"))
        cells.append(new_markdown_cell(f"**生成时间**: {report.timestamp}"))

        # 摘要
        cells.append(new_markdown_cell("## 摘要"))
        cells.append(new_markdown_cell(f"""
| 指标 | 值 |
|------|-----|
| 总实验数 | {report.data_summary['total_experiments']} |
| 成功率 | {report.data_summary['success_rate']:.2%} |
| 平均耗时 | {report.data_summary['avg_duration_ms']:.2f} ms |
| 特征数 | {report.data_summary['feature_count']} |
"""))

        # 关键发现
        cells.append(new_markdown_cell("## 关键发现"))
        for finding in report.key_findings:
            cells.append(new_markdown_cell(f"- {finding}"))

        # 假设检验结果
        cells.append(new_markdown_cell("## 假设检验结果"))
        for hyp in report.hypothesis_tests:
            status = "✅ 显著" if hyp.is_significant else "❌ 不显著"
            cells.append(new_markdown_cell(f"""
### {hyp.hypothesis_name}
- **原假设**: {hyp.null_hypothesis}
- **备择假设**: {hyp.alternative_hypothesis}
- **检验统计量**: {hyp.test_statistic:.4f}
- **p值**: {hyp.p_value:.4f}
- **显著性**: {status}
- **结论**: {hyp.conclusion}
"""))

        # 相关性分析
        cells.append(new_markdown_cell("## 相关性分析"))
        if report.correlation_analysis:
            corr_table = "| 变量对 | 相关系数 |\n|--------|----------|\n"
            for pair, corr in report.correlation_analysis.items():
                corr_table += f"| {pair} | {corr:.4f} |\n"
            cells.append(new_markdown_cell(corr_table))
        else:
            cells.append(new_markdown_cell("未发现显著相关性。"))

        # 结论
        cells.append(new_markdown_cell("## 结论"))
        for conclusion in report.conclusions:
            cells.append(new_markdown_cell(f"- {conclusion}"))

        # 建议
        cells.append(new_markdown_cell("## 建议"))
        for rec in report.recommendations:
            cells.append(new_markdown_cell(f"- {rec}"))

        # 数据探索代码
        cells.append(new_markdown_cell("## 数据探索"))
        cells.append(new_code_cell("""
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('experiment_data.csv')

# 基本信息
print(f"数据形状: {df.shape}")
print(f"\\n数据类型:\\n{df.dtypes}")
print(f"\\n缺失值:\\n{df.isnull().sum()}")
"""))

        nb.cells = cells

        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        return str(file_path)

    def save_markdown_report(
        self,
        report: AnalysisReport,
        output_path: Optional[str] = None
    ) -> str:
        """
        保存Markdown格式报告

        Args:
            report: 分析报告
            output_path: 输出路径

        Returns:
            保存的文件路径
        """
        output_path = Path(output_path or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"analysis_report_{timestamp}.md"

        md_content = f"""# {report.title}

**生成时间**: {report.timestamp}

---

## 摘要

| 指标 | 值 |
|------|-----|
| 总实验数 | {report.data_summary['total_experiments']} |
| 成功率 | {report.data_summary['success_rate']:.2%} |
| 平均耗时 | {report.data_summary['avg_duration_ms']:.2f} ms |
| 特征数 | {report.data_summary['feature_count']} |

---

## 关键发现

"""
        for finding in report.key_findings:
            md_content += f"- {finding}\n"

        md_content += "\n## 假设检验结果\n\n"
        for hyp in report.hypothesis_tests:
            status = "✅ 显著" if hyp.is_significant else "❌ 不显著"
            md_content += f"""
### {hyp.hypothesis_name}

- **原假设**: {hyp.null_hypothesis}
- **备择假设**: {hyp.alternative_hypothesis}
- **检验统计量**: {hyp.test_statistic:.4f}
- **p值**: {hyp.p_value:.4f}
- **显著性**: {status}
- **结论**: {hyp.conclusion}

"""

        md_content += "## 相关性分析\n\n"
        if report.correlation_analysis:
            md_content += "| 变量对 | 相关系数 |\n|--------|----------|\n"
            for pair, corr in report.correlation_analysis.items():
                md_content += f"| {pair} | {corr:.4f} |\n"
        else:
            md_content += "未发现显著相关性。\n"

        md_content += "\n## 结论\n\n"
        for conclusion in report.conclusions:
            md_content += f"- {conclusion}\n"

        md_content += "\n## 建议\n\n"
        for rec in report.recommendations:
            md_content += f"- {rec}\n"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return str(file_path)


# ================================================================================
# 主分析流程
# ================================================================================

class AnalysisEngine:
    """
    分析引擎

    功能：
        - 整合所有分析组件
        - 执行完整分析流程
        - 生成多种格式报告
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        初始化分析引擎

        Args:
            config: 分析配置
        """
        self.config = config or AnalysisConfig()
        self._loader: Optional[ExperimentDataLoader] = None
        self._analyzer: Optional[StatisticalAnalyzer] = None
        self._report: Optional[AnalysisReport] = None

    def run(self) -> AnalysisReport:
        """
        执行完整分析流程

        Returns:
            分析报告
        """
        # 1. 加载数据
        self._loader = ExperimentDataLoader(
            self.config.data_dir,
            self.config.results_dir
        )
        experiments, df = self._loader.load_all()

        if df.empty:
            raise ValueError("没有可分析的数据")

        # 2. 统计分析
        self._analyzer = StatisticalAnalyzer(df, self.config.significance_level)

        # 3. 假设验证
        validator = HypothesisValidator(self._analyzer)
        hypotheses = validator.run_all_validations()

        # 4. 提取关键发现
        key_findings = self._extract_key_findings(df, hypotheses)

        # 5. 生成报告
        generator = ReportGenerator(self.config)
        self._report = generator.generate_report(df, hypotheses, key_findings)

        # 6. 保存报告
        generator.save_json_report(self._report)
        generator.save_markdown_report(self._report)

        if NBFORMAT_AVAILABLE:
            generator.generate_jupyter_notebook(self._report, df)

        return self._report

    def _extract_key_findings(
        self,
        df: pd.DataFrame,
        hypotheses: List[HypothesisResult]
    ) -> List[str]:
        """
        提取关键发现

        Args:
            df: 实验数据
            hypotheses: 假设检验结果

        Returns:
            关键发现列表
        """
        findings = []

        # 成功率
        success_rate = (df["status"] == "success").mean()
        findings.append(f"实验总体成功率为 {success_rate:.1%}。")

        # 性能指标
        if "metric_total_reward" in df.columns:
            avg_reward = df["metric_total_reward"].mean()
            findings.append(f"平均总奖励为 {avg_reward:.2f}。")

        # 风险分析
        if "init_combined_risk" in df.columns:
            avg_risk = df["init_combined_risk"].mean()
            findings.append(f"初始综合风险均值为 {avg_risk:.2f}。")

        # 显著性检验
        sig_tests = [h for h in hypotheses if h.is_significant]
        if sig_tests:
            findings.append(f"共 {len(sig_tests)} 个假设通过显著性检验。")

        return findings


# ================================================================================
# 便捷函数
# ================================================================================

def run_analysis(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> AnalysisReport:
    """
    便捷函数：执行分析

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        verbose: 是否打印详情

    Returns:
        分析报告
    """
    config = AnalysisConfig(
        data_dir=data_dir or DEFAULT_DATA_DIR,
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        verbose=verbose
    )

    engine = AnalysisEngine(config)
    return engine.run()


# ================================================================================
# 入口点
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M13 实验分析与开源工程化")
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"数据目录 (默认: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "-r", "--results",
        type=str,
        default=DEFAULT_EXPERIMENT_RESULTS_DIR,
        help=f"实验结果目录 (默认: {DEFAULT_EXPERIMENT_RESULTS_DIR})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.05,
        help=f"显著性水平 (默认: 0.05)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默模式"
    )

    args = parser.parse_args()

    config = AnalysisConfig(
        data_dir=args.input,
        results_dir=args.results,
        output_dir=args.output,
        significance_level=args.alpha,
        verbose=not args.quiet
    )

    engine = AnalysisEngine(config)

    print("=" * 60)
    print("M13 实验分析与开源工程化")
    print("=" * 60)

    try:
        report = engine.run()

        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)
        print(f"标题: {report.title}")
        print(f"总实验数: {report.data_summary['total_experiments']}")
        print(f"成功率: {report.data_summary['success_rate']:.2%}")
        print(f"假设检验数: {len(report.hypothesis_tests)}")
        print(f"关键发现: {len(report.key_findings)}")
        print(f"结论数: {len(report.conclusions)}")

        output_path = Path(config.output_dir)
        print(f"\n报告已保存至: {output_path}")

    except Exception as e:
        print(f"\n分析失败: {e}")
        import traceback
        traceback.print_exc()
