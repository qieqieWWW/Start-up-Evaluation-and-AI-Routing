#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
M14 & M15 模块：Streamlit 交互式演示界面
================================================================================

功能描述：
    - 使用 Streamlit 构建交互式演示界面
    - 加载实验结论并可视化展示仿真结果
    - 支持数据筛选、图表切换、报告导出

设计原则：
    - 每个逻辑模块对应一个独立文件
    - 响应式UI设计
    - 零配置启动

接口依赖：
    - 从 data_processor.py 导入 ProcessedExperiment

前置条件：
    - 需要安装 streamlit: pip install streamlit
    - 需要安装 plotly: pip install plotly

运行方式：
    - 命令行: streamlit run app_demo.py
    - 或: python app_demo.py

作者：阶段4开发团队
日期：2026-04-09
================================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Streamlit 相关
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("警告: Streamlit 未安装。请运行 'pip install streamlit plotly' 来安装依赖。")

# 数据处理相关
import pandas as pd
import numpy as np

# ================================================================================
# 配置常量
# ================================================================================

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 默认路径
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
DEFAULT_EXPERIMENT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiment_results")

# 页面配置
PAGE_TITLE = "Kickstarter 仿真实验分析平台"
PAGE_ICON = "🚀"
LAYOUT = "wide"


# ================================================================================
# 数据加载器
# ================================================================================

class ExperimentDataLoader:
    """
    实验数据加载器

    功能：
        - 从多个来源加载实验数据
        - 数据缓存与验证
    """

    def __init__(self, data_dir: Optional[str] = None, results_dir: Optional[str] = None):
        """
        初始化数据加载器

        Args:
            data_dir: 处理后数据目录
            results_dir: 实验结果目录
        """
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.results_dir = Path(results_dir or DEFAULT_EXPERIMENT_RESULTS_DIR)
        self._cache: Dict[str, Any] = {}

    def load_from_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载数据

        Args:
            file_path: JSONL文件路径

        Returns:
            实验数据列表
        """
        experiments = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        experiments.append(json.loads(line))
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
        return experiments

    def load_from_results(self) -> List[Dict[str, Any]]:
        """
        从实验结果目录加载数据

        Returns:
            实验数据列表
        """
        if not self.results_dir.exists():
            return []

        experiments = []
        result_files = list(self.results_dir.glob("experiment_results_*.json"))

        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    experiments.extend(data.get("experiments", []))
            except Exception as e:
                print(f"加载结果文件失败 {file_path}: {e}")

        return experiments

    def load_all(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """
        加载所有可用数据

        Returns:
            (原始实验列表, DataFrame格式)
        """
        all_experiments = []

        # 从JSONL文件加载
        jsonl_files = list(self.data_dir.glob("experiments_*.jsonl"))
        for file_path in jsonl_files:
            all_experiments.extend(self.load_from_jsonl(file_path))

        # 从结果目录加载
        all_experiments.extend(self.load_from_results())

        # 去重
        seen_ids = set()
        unique_experiments = []
        for exp in all_experiments:
            exp_id = exp.get("experiment_id") or exp.get("original_id")
            if exp_id not in seen_ids:
                seen_ids.add(exp_id)
                unique_experiments.append(exp)

        # 转换为DataFrame
        df = self._to_dataframe(unique_experiments)

        return unique_experiments, df

    def _to_dataframe(self, experiments: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        转换为DataFrame

        Args:
            experiments: 实验数据列表

        Returns:
            DataFrame
        """
        if not experiments:
            return pd.DataFrame()

        # 展平嵌套数据
        flattened = []
        for exp in experiments:
            row = {
                "experiment_id": exp.get("experiment_id", ""),
                "timestamp": exp.get("timestamp", ""),
                "duration_ms": exp.get("duration_ms", 0),
                "status": exp.get("status", "unknown"),
            }

            # 添加指标
            metrics = exp.get("metrics", {})
            for key, value in metrics.items():
                row[f"metric_{key}"] = value

            # 添加初始状态特征
            initial = exp.get("initial_state", {})
            for key, value in initial.items():
                row[f"init_{key}"] = value

            # 添加最终状态特征
            final = exp.get("final_state", {})
            for key, value in final.items():
                row[f"final_{key}"] = value

            flattened.append(row)

        return pd.DataFrame(flattened)


# ================================================================================
# 可视化组件
# ================================================================================

class VisualizationEngine:
    """
    可视化引擎

    功能：
        - 生成各类统计图表
        - 支持Plotly交互式图表
    """

    @staticmethod
    def plot_success_rate(df: pd.DataFrame) -> go.Figure:
        """
        绘制成功率统计图

        Args:
            df: 实验数据DataFrame

        Returns:
            Plotly Figure对象
        """
        if df.empty or "status" not in df.columns:
            return go.Figure()

        status_counts = df["status"].value_counts()

        fig = go.Figure(data=[
            go.Bar(
                x=status_counts.index,
                y=status_counts.values,
                marker_color=['#2ecc71' if s == 'success' else '#e74c3c' for s in status_counts.index]
            )
        ])

        fig.update_layout(
            title="实验状态分布",
            xaxis_title="状态",
            yaxis_title="数量",
            template="plotly_white",
            height=400
        )

        return fig

    @staticmethod
    def plot_duration_distribution(df: pd.DataFrame) -> go.Figure:
        """
        绘制执行时间分布图

        Args:
            df: 实验数据DataFrame

        Returns:
            Plotly Figure对象
        """
        if df.empty or "duration_ms" not in df.columns:
            return go.Figure()

        fig = px.histogram(
            df,
            x="duration_ms",
            nbins=30,
            title="实验执行时间分布",
            labels={"duration_ms": "执行时间 (ms)"},
            color_discrete_sequence=["#3498db"]
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_metrics_heatmap(df: pd.DataFrame) -> go.Figure:
        """
        绘制指标相关性热力图

        Args:
            df: 实验数据DataFrame

        Returns:
            Plotly Figure对象
        """
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [c for c in numeric_cols if c.startswith("metric_")]

        if len(metric_cols) < 2:
            return go.Figure()

        # 计算相关性矩阵
        corr_matrix = df[metric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu_r",
            hoverongaps=False
        ))

        fig.update_layout(
            title="指标相关性热力图",
            template="plotly_white",
            height=500
        )

        return fig

    @staticmethod
    def plot_trajectory_trend(experiments: List[Dict[str, Any]], 
                            sample_size: int = 10) -> go.Figure:
        """
        绘制轨迹趋势图

        Args:
            experiments: 实验数据列表
            sample_size: 采样数量

        Returns:
            Plotly Figure对象
        """
        # 采样
        sampled = experiments[:min(sample_size, len(experiments))]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("风险趋势", "奖励趋势", "目标比率", "时间惩罚")
        )

        for idx, exp in enumerate(sampled):
            trajectory = exp.get("trajectory", [])
            if not trajectory:
                continue

            steps = [t.get("step", i) for i, t in enumerate(trajectory)]

            # 风险趋势
            risks = []
            for step_data in trajectory:
                state = step_data.get("state", {})
                risks.append(state.get("combined_risk", 0))
            fig.add_trace(
                go.Scatter(x=steps, y=risks, mode='lines+markers', 
                          name=f'Exp {idx}', showlegend=False),
                row=1, col=1
            )

            # 奖励趋势
            rewards = [t.get("reward", 0) for t in trajectory]
            fig.add_trace(
                go.Scatter(x=steps, y=rewards, mode='lines+markers',
                          name=f'Exp {idx}', showlegend=False),
                row=1, col=2
            )

            # 目标比率
            goal_ratios = []
            for step_data in trajectory:
                state = step_data.get("state", {})
                goal_ratios.append(state.get("goal_ratio", 0))
            fig.add_trace(
                go.Scatter(x=steps, y=goal_ratios, mode='lines+markers',
                          name=f'Exp {idx}', showlegend=False),
                row=2, col=1
            )

            # 时间惩罚
            time_penalties = []
            for step_data in trajectory:
                state = step_data.get("state", {})
                time_penalties.append(state.get("time_penalty", 0))
            fig.add_trace(
                go.Scatter(x=steps, y=time_penalties, mode='lines+markers',
                          name=f'Exp {idx}', showlegend=False),
                row=2, col=2
            )

        fig.update_layout(
            title="仿真轨迹趋势分析",
            template="plotly_white",
            height=700,
            showlegend=False
        )

        return fig

    @staticmethod
    def plot_feature_importance(df: pd.DataFrame) -> go.Figure:
        """
        绘制特征重要性图

        Args:
            df: 实验数据DataFrame

        Returns:
            Plotly Figure对象
        """
        # 获取初始状态特征
        init_cols = [c for c in df.columns if c.startswith("init_")]
        if not init_cols:
            return go.Figure()

        # 计算各特征的变异性（作为重要性代理）
        importance = df[init_cols].std().sort_values(ascending=True)

        fig = go.Figure(data=[
            go.Bar(
                x=importance.values,
                y=[c.replace("init_", "") for c in importance.index],
                orientation='h',
                marker_color="#9b59b6"
            )
        ])

        fig.update_layout(
            title="特征变异性分析（重要性代理）",
            xaxis_title="标准差",
            yaxis_title="特征",
            template="plotly_white",
            height=400
        )

        return fig


# ================================================================================
# Streamlit 页面
# ================================================================================

def setup_page_config():
    """配置页面"""
    if STREAMLIT_AVAILABLE:
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout=LAYOUT
        )


def render_sidebar() -> Tuple[str, int, List[str]]:
    """
    渲染侧边栏

    Returns:
        (数据目录, 采样大小, 选中的图表类型)
    """
    if not STREAMLIT_AVAILABLE:
        return "", 10, []

    with st.sidebar:
        st.title("📊 配置面板")

        # 数据路径配置
        st.subheader("数据源")
        data_dir = st.text_input(
            "处理后数据目录",
            value=DEFAULT_DATA_DIR
        )
        results_dir = st.text_input(
            "实验结果目录",
            value=DEFAULT_EXPERIMENT_RESULTS_DIR
        )

        # 显示选项
        st.subheader("显示选项")
        sample_size = st.slider(
            "采样数量",
            min_value=5,
            max_value=100,
            value=20
        )

        # 图表选择
        st.subheader("图表类型")
        chart_types = st.multiselect(
            "选择要显示的图表",
            options=[
                "状态分布",
                "执行时间分布",
                "指标热力图",
                "轨迹趋势",
                "特征重要性"
            ],
            default=["状态分布", "执行时间分布"]
        )

        # 关于信息
        st.divider()
        st.caption("Kickstarter 仿真实验分析平台 v1.0")
        st.caption("Powered by Streamlit + Plotly")

    return data_dir, sample_size, chart_types


def render_overview_metrics(experiments: List[Dict[str, Any]], df: pd.DataFrame):
    """
    渲染概览指标卡片

    Args:
        experiments: 原始实验数据
        df: DataFrame数据
    """
    if not STREAMLIT_AVAILABLE:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total = len(experiments)
        st.metric("总实验数", total)

    with col2:
        success_count = sum(1 for e in experiments if e.get("status") == "success")
        rate = success_count / total * 100 if total > 0 else 0
        st.metric("成功率", f"{rate:.1f}%")

    with col3:
        avg_duration = df["duration_ms"].mean() if not df.empty else 0
        st.metric("平均耗时", f"{avg_duration:.1f} ms")

    with col4:
        success_count = sum(1 for e in experiments if e.get("status") == "success")
        st.metric("成功实验数", success_count)


def render_charts(experiments: List[Dict[str, Any]], 
                  df: pd.DataFrame, 
                  chart_types: List[str]):
    """
    渲染图表

    Args:
        experiments: 原始实验数据
        df: DataFrame数据
        chart_types: 要显示的图表类型列表
    """
    if not STREAMLIT_AVAILABLE:
        return

    viz = VisualizationEngine()

    for chart_type in chart_types:
        with st.container():
            if chart_type == "状态分布":
                fig = viz.plot_success_rate(df)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "执行时间分布":
                fig = viz.plot_duration_distribution(df)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "指标热力图":
                fig = viz.plot_metrics_heatmap(df)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "轨迹趋势":
                fig = viz.plot_trajectory_trend(experiments)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "特征重要性":
                fig = viz.plot_feature_importance(df)
                st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame, sample_size: int = 20):
    """
    渲染数据表格

    Args:
        df: DataFrame数据
        sample_size: 显示行数
    """
    if not STREAMLIT_AVAILABLE:
        return

    st.subheader("📋 实验数据表")

    # 显示选项
    show_columns = st.multiselect(
        "选择显示列",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:10]
    )

    if show_columns:
        display_df = df[show_columns].head(sample_size)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

        # 导出CSV
        csv = df[show_columns].to_csv(index=False)
        st.download_button(
            label="下载 CSV",
            data=csv,
            file_name=f"experiment_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def render_experiment_detail(experiments: List[Dict[str, Any]]):
    """
    渲染单个实验详情

    Args:
        experiments: 实验数据列表
    """
    if not STREAMLIT_AVAILABLE:
        return

    st.subheader("🔍 实验详情")

    if not experiments:
        st.info("暂无实验数据")
        return

    # 选择实验
    exp_options = [f"{i}: {e.get('experiment_id', 'N/A')}" for i, e in enumerate(experiments)]
    selected_idx = st.selectbox("选择实验", options=range(len(exp_options)), format_func=lambda x: exp_options[x])

    if 0 <= selected_idx < len(experiments):
        exp = experiments[selected_idx]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**基本信息**")
            st.json({
                "experiment_id": exp.get("experiment_id"),
                "status": exp.get("status"),
                "timestamp": exp.get("timestamp"),
                "duration_ms": exp.get("duration_ms")
            })

        with col2:
            st.markdown("**指标**")
            st.json(exp.get("metrics", {}))

        st.markdown("**初始状态**")
        st.json(exp.get("initial_state", {}))

        st.markdown("**最终状态**")
        st.json(exp.get("final_state", {}))


def main():
    """主函数"""
    # 设置页面配置
    setup_page_config()

    if not STREAMLIT_AVAILABLE:
        print("错误: Streamlit 不可用")
        print("请运行以下命令安装: pip install streamlit plotly")
        return

    # 页面标题
    st.title("🚀 Kickstarter 仿真实验分析平台")
    st.markdown("基于 Streamlit 的交互式数据可视化系统")

    # 渲染侧边栏
    data_dir, sample_size, chart_types = render_sidebar()

    # 加载数据
    try:
        loader = ExperimentDataLoader(
            data_dir=data_dir or DEFAULT_DATA_DIR,
            results_dir=DEFAULT_EXPERIMENT_RESULTS_DIR
        )
        experiments, df = loader.load_all()
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        experiments, df = [], pd.DataFrame()

    # 渲染概览指标
    render_overview_metrics(experiments, df)

    # 渲染图表
    if chart_types:
        st.divider()
        st.header("📈 数据可视化")
        render_charts(experiments, df, chart_types)

    # 渲染数据表格
    st.divider()
    if not df.empty:
        render_data_table(df, sample_size)

    # 渲染实验详情
    st.divider()
    render_experiment_detail(experiments)


# ================================================================================
# 命令行模式（非Streamlit）
# ================================================================================

def run_cli_mode():
    """命令行模式（无Streamlit时）"""
    print("=" * 60)
    print("Kickstarter 仿真实验分析平台 - CLI 模式")
    print("=" * 60)

    loader = ExperimentDataLoader()
    experiments, df = loader.load_all()

    print(f"\n加载了 {len(experiments)} 个实验数据")

    if not df.empty:
        print("\n数据概览:")
        print(df.describe())

        viz = VisualizationEngine()

        # 生成图表（保存为HTML）
        output_dir = Path(PROJECT_ROOT) / "visualizations"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            fig = viz.plot_success_rate(df)
            fig.write_html(str(output_dir / f"status_distribution_{timestamp}.html"))
            print(f"状态分布图已保存: {output_dir / f'status_distribution_{timestamp}.html'}")
        except Exception as e:
            print(f"生成图表失败: {e}")

    print("\n提示: 安装 Streamlit 后可使用 Web 界面")
    print("运行命令: streamlit run app_demo.py")


# ================================================================================
# 入口点
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M14/M15 Streamlit 演示界面")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="使用命令行模式（无需Streamlit）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit 服务端口"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="启动后自动打开浏览器"
    )

    args = parser.parse_args()

    if args.cli or not STREAMLIT_AVAILABLE:
        run_cli_mode()
    else:
        # 启动Streamlit服务
        import subprocess
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            __file__,
            "--server.port", str(args.port),
            "--server.headless", "true"
        ]

        if args.browser:
            cmd.append("--server.auto_open_browser")

        print(f"启动 Streamlit 服务: http://localhost:{args.port}")
        subprocess.run(cmd)
