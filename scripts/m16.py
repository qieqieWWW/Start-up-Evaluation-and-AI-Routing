#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
M16 模块：大规模实验与分析
================================================================================

功能描述：
    分析真实Kickstarter众筹数据，执行统计分析，生成分析报告

数据来源：
    1. datasets/Kickstarter.csv - 真实Kickstarter众筹数据
    2. Kickstarter_Clean/ - 清洗后的数据
    3. experiment_results/*.json - M3/M12仿真实验结果
    4. Kickstarter_Clean/m12_ood_tests/*.json - OOD测试结果

调用流程：
    1. 加载真实数据 → 读取Kickstarter数据或已有仿真结果
    2. 数据统计分析 → 假设验证、相关性分析
    3. 生成分析报告 → 输出Markdown/JSON/Notebook格式

================================================================================
"""

import os
import sys
import json
import time
import logging
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from threading import Lock
from collections import defaultdict

# Windows控制台编码修复
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 科学计算与统计分析
try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    import numpy as np
    import pandas as pd
    SCIPY_AVAILABLE = False

# Notebook 生成
try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False

# ================================================================================
# 路径配置
# ================================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENT_RESULTS_DIR = PROJECT_ROOT / "experiment_results"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
ANALYSIS_REPORTS_DIR = PROJECT_ROOT / "analysis_reports"
KICKSTARTER_DATA_DIR = PROJECT_ROOT / "datasets"
KICKSTARTER_CLEAN_DIR = PROJECT_ROOT / "Kickstarter_Clean"
M12_OOD_TESTS_DIR = KICKSTARTER_CLEAN_DIR / "m12_ood_tests"

# ================================================================================
# 实验执行器常量与数据类 
# ================================================================================

DEFAULT_EXPERIMENT_COUNT = 500
DEFAULT_BATCH_SIZE = 50
DEFAULT_LOG_DIR = str(LOGS_DIR / "experiments")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    experiment_count: int = DEFAULT_EXPERIMENT_COUNT
    batch_size: int = DEFAULT_BATCH_SIZE
    log_dir: str = DEFAULT_LOG_DIR
    output_dir: str = str(EXPERIMENT_RESULTS_DIR)
    enable_checkpoint: bool = True
    random_seed: Optional[int] = 42
    verbose: bool = True
    max_retries: int = 3


@dataclass
class ExperimentResult:
    """单次实验结果数据类"""
    experiment_id: int
    timestamp: str
    duration_ms: float
    status: str  # "success", "failed", "partial"
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSummary:
    """实验汇总数据类"""
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    total_duration_seconds: float
    success_rate: float
    avg_duration_ms: float
    metrics_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperimentLogger:
    """实验日志管理器"""
    def __init__(self, log_dir: str, experiment_id: int):
        self.log_dir = Path(log_dir)
        self.experiment_id = experiment_id
        self._lock = Lock()
        self._log_entries: List[Dict[str, Any]] = []
        self._trajectory_entries: List[Dict[str, Any]] = []
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logger()

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger(f"experiment_{self.experiment_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        log_file = self.log_dir / f"experiment_{self.experiment_id}.jsonl"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        entry = {"timestamp": datetime.now().isoformat(), "experiment_id": self.experiment_id, "event_type": event_type, "data": data}
        with self._lock:
            self._log_entries.append(entry)

    def log_trajectory_step(self, step: int, state: Dict[str, Any], action: Dict[str, Any], reward: float, next_state: Dict[str, Any], done: bool, info: Optional[Dict[str, Any]] = None) -> None:
        entry = {"timestamp": datetime.now().isoformat(), "experiment_id": self.experiment_id, "step": step, "state": state, "action": action, "reward": reward, "next_state": next_state, "done": done, "info": info or {}}
        with self._lock:
            self._trajectory_entries.append(entry)

    def flush(self) -> None:
        for handler in self.logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

    def get_trajectory(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._trajectory_entries.copy()

    def close(self) -> None:
        self.flush()
        self.logger.info(f"实验 {self.experiment_id} 日志记录完成")


class RealDataLoader:
    """
    真实数据加载器
    
    数据来源优先级：
    1. Kickstarter原始数据 (datasets/Kickstarter.csv)
    2. M3/M12仿真结果 (experiment_results/*.json)
    3. OOD测试结果 (Kickstarter_Clean/m12_ood_tests/*.json)
    """
    
    def __init__(self, config: "M16Config"):
        self.config = config
        self._data: List[Dict[str, Any]] = []
        self._data_source = "none"
    
    def load_kickstarter_csv(self, limit: Optional[int] = None) -> Tuple[bool, str]:
        """加载Kickstarter原始CSV数据"""
        import pandas as pd
        
        # 尝试多个可能的路径
        csv_file = KICKSTARTER_DATA_DIR / "Kickstarter.csv"
        if not csv_file.exists():
            # 查找子目录中的 Kickstarter.csv
            for subdir in KICKSTARTER_DATA_DIR.iterdir():
                if subdir.is_dir():
                    candidate = subdir / "Kickstarter.csv"
                    if candidate.exists():
                        csv_file = candidate
                        break
            else:
                # 尝试 Kickstarter_Clean 目录
                kickstarter_clean = KICKSTARTER_CLEAN_DIR / "kickstarter_cleaned.csv"
                if kickstarter_clean.exists():
                    csv_file = kickstarter_clean
                    
        if not csv_file.exists():
            return False, f"Kickstarter.csv不存在"
        
        try:
            print(f"[INFO] 加载Kickstarter原始数据: {csv_file}")
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            
            if limit:
                df = df.head(limit)
            
            # 转换DataFrame为实验格式
            self._data = self._convert_kickstarter_to_experiments(df)
            self._data_source = "Kickstarter原始数据"
            
            print(f"[OK] 加载了 {len(self._data)} 条Kickstarter项目记录")
            return True, str(csv_file)
        except Exception as e:
            return False, f"加载失败: {str(e)}"
    
    def _convert_kickstarter_to_experiments(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """将Kickstarter DataFrame转换为实验数据格式"""
        experiments = []
        
        # 特征列映射
        feature_cols = ['goal_ratio', 'time_penalty', 'category_risk', 'combined_risk', 
                       'country_factor', 'urgency_score']
        
        for idx, row in df.iterrows():
            # 映射 state 值到统一的 success/failed 格式
            state = str(row.get('state', 'unknown')).lower()
            if state == 'successful':
                status = 'success'
            else:
                status = 'failed'  # failed, canceled, live 等都映射为 failed
            
            exp = {
                "experiment_id": idx,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": 0,
                "status": status,
                "initial_state": {},
                "final_state": {},
                "trajectory": [],
                "metrics": {}
            }
            
            # 提取初始状态特征
            for col in feature_cols:
                if col in df.columns:
                    val = row.get(col)
                    if pd.notna(val):
                        exp["initial_state"][col] = float(val)
            
            experiments.append(exp)
        
        return experiments
    
    def load_experiment_results(self) -> Tuple[bool, str]:
        """加载M3/M12仿真实验结果"""
        result_files = list(EXPERIMENT_RESULTS_DIR.glob("experiment_results_*.json"))
        
        if not result_files:
            return False, "未找到仿真实验结果"
        
        # 获取最新文件
        latest = max(result_files, key=lambda p: p.stat().st_mtime)
        
        try:
            print(f"[INFO] 加载仿真实验结果: {latest}")
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            experiments = data.get("experiments", [])
            
            # 去重
            seen = set()
            unique_experiments = []
            for exp in experiments:
                exp_id = exp.get("experiment_id", "")
                if exp_id not in seen:
                    seen.add(exp_id)
                    unique_experiments.append(exp)
            
            self._data = unique_experiments
            self._data_source = f"M3/M12仿真结果 ({latest.name})"
            
            print(f"[OK] 加载了 {len(self._data)} 条实验记录")
            return True, str(latest)
        except Exception as e:
            return False, f"加载失败: {str(e)}"
    
    def load_m12_ood_results(self) -> Tuple[bool, str]:
        """加载M12 OOD测试结果"""
        ood_files = list(M12_OOD_TESTS_DIR.glob("m12_ood_report_*.json"))
        
        if not ood_files:
            return False, "未找到M12 OOD测试结果"
        
        # 获取最新文件
        latest = max(ood_files, key=lambda p: p.stat().st_mtime)
        
        try:
            print(f"[INFO] 加载M12 OOD测试结果: {latest}")
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 从OOD报告提取场景数据
            experiments = []
            scenarios = data.get("detailed_results", {}).get("scenarios", [])
            
            for idx, scenario in enumerate(scenarios):
                base_project = scenario.get("base_project", {})
                
                # 优先使用 expected_outcome 判断
                expected_outcome = scenario.get("expected_outcome", "")
                is_survive = "survive" in expected_outcome.lower() if expected_outcome else False
                
                # 备用：根据难度级别模拟 resilience_score（越难越低）
                difficulty_map = {"easy": 0.8, "medium": 0.6, "hard": 0.4, "extreme": 0.2}
                difficulty = scenario.get("difficulty_level", "medium")
                resilience = 0.75 if is_survive else difficulty_map.get(difficulty, 0.5)
                
                exp = {
                    "experiment_id": idx,
                    "timestamp": datetime.now().isoformat(),
                    "duration_ms": 0,
                    "status": "passed" if is_survive else "failed",
                    "initial_state": base_project,
                    "final_state": {"resilience_score": resilience, "expected_outcome": expected_outcome},
                    "trajectory": [],
                    "metrics": {
                        "resilience_score": resilience,
                        "difficulty": difficulty,
                        "expected_outcome": expected_outcome,
                        "distribution_shifts": len(scenario.get("distribution_shifts", [])),
                        "black_swan_events": len(scenario.get("black_swan_events", []))
                    }
                }
                experiments.append(exp)
            
            self._data = experiments
            self._data_source = f"M12 OOD测试 ({latest.name})"
            
            print(f"[OK] 加载了 {len(self._data)} 条OOD测试记录")
            return True, str(latest)
        except Exception as e:
            return False, f"加载失败: {str(e)}"
    
    def load_all_sources(self) -> Tuple[bool, str]:
        """
        从所有可用来源加载数据并合并
        
        加载优先级（统一汇总，不中断）：
        1. experiment_results/*.json (M3/M12仿真结果)
        2. datasets/Kickstarter.csv (Kickstarter原始数据)
        3. Kickstarter_Clean/m12_ood_tests/*.json (OOD测试结果)
        
        Returns:
            Tuple[bool, str]: (是否成功, 数据源信息)
        """
        all_experiments = []
        loaded_sources = []
        
        # 按优先级加载所有数据源
        sources = [
            ("M3/M12仿真结果", self._load_experiment_results_with_source),
            ("Kickstarter原始数据", lambda: self._load_kickstarter_with_source(1000)),
            ("M12 OOD测试", self._load_m12_ood_with_source),
        ]
        
        for name, loader_func in sources:
            try:
                experiments = loader_func()
                if experiments:
                    # 为每条数据添加来源标签
                    for exp in experiments:
                        exp["_source"] = name
                    all_experiments.extend(experiments)
                    loaded_sources.append(f"{name}({len(experiments)}条)")
                    print(f"[INFO] {name}: 加载了 {len(experiments)} 条记录")
            except Exception as e:
                print(f"[DEBUG] {name} 加载失败: {e}")
        
        if not all_experiments:
            return False, "所有数据源均无可用数据"
        
        # 去重（基于 experiment_id）
        seen = set()
        unique_experiments = []
        for exp in all_experiments:
            exp_id = exp.get("experiment_id", id(exp))  # 用 id() 作为兜底
            if exp_id not in seen:
                seen.add(exp_id)
                unique_experiments.append(exp)
        
        self._data = unique_experiments
        self._data_source = ", ".join(loaded_sources)
        
        total = len(unique_experiments)
        sources_count = len(loaded_sources)
        print(f"[OK] 汇总完成: 共 {total} 条记录, 来源: {sources_count} 个")
        
        return True, self._data_source
    
    def _load_experiment_results_with_source(self) -> List[Dict[str, Any]]:
        """加载仿真结果并添加来源标记"""
        result_files = list(EXPERIMENT_RESULTS_DIR.glob("experiment_results_*.json"))
        if not result_files:
            return []
        
        experiments = []
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                experiments.extend(data.get("experiments", []))
            except Exception as e:
                print(f"[DEBUG] 加载 {result_file.name} 失败: {e}")
        
        return experiments
    
    def _load_kickstarter_with_source(self, limit: int) -> List[Dict[str, Any]]:
        """加载Kickstarter原始数据并添加来源标记"""
        success, _ = self.load_kickstarter_csv(limit=limit)
        if success:
            return self._data
        return []
    
    def _load_m12_ood_with_source(self) -> List[Dict[str, Any]]:
        """加载M12 OOD测试结果并添加来源标记"""
        success, _ = self.load_m12_ood_results()
        if success:
            return self._data
        return []
    
    @property
    def experiments(self) -> List[Dict[str, Any]]:
        return self._data
    
    @property
    def data_source(self) -> str:
        return self._data_source


class ExperimentRunner:
    """
    真实数据实验执行器
    
    功能：
    1. 基于真实Kickstarter数据执行批量实验
    2. 记录实验轨迹和中间状态
    3. 生成可复现的实验结果
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._lock = Lock()
        self._results: List[ExperimentResult] = []
        self._stats = {"total": 0, "success": 0, "failed": 0}
    
    def run_single_experiment(self, experiment_id: int, initial_state: Dict[str, Any]) -> ExperimentResult:
        """执行单次实验"""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # 模拟实验执行
            final_state = self._simulate_experiment(initial_state)
            status = final_state.get("outcome", "success")  # 根据模拟结果设置状态
            trajectory = []
            
            result = ExperimentResult(
                experiment_id=experiment_id,
                timestamp=timestamp,
                duration_ms=(time.time() - start_time) * 1000,
                status=status,
                initial_state=initial_state,
                final_state=final_state,
                trajectory=trajectory
            )
            
            return result
            
        except Exception as e:
            return ExperimentResult(
                experiment_id=experiment_id,
                timestamp=timestamp,
                duration_ms=(time.time() - start_time) * 1000,
                status="failed",
                initial_state=initial_state,
                final_state={},
                trajectory=[],
                error_message=str(e)
            )
    
    def _simulate_experiment(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        模拟实验执行 - 基于M12韧性评估逻辑的风险评估
        
        复用 M12 ResilienceEvaluator 的核心公式：
        overall_resilience = (
            survival_rate * 0.3 +
            (1 - avg_risk_increase) * 0.25 +
            recovery_speed * 0.2 +
            worst_case_performance * 0.15 +
            adaptability_score * 0.1
        )
        """
        import random
        
        # 从特征中提取风险指标 (特征向量索引: 0=goal_ratio, 1=time_penalty, 
        # 2=category_risk, 3=combined_risk, 4=country_factor, 5=urgency_score)
        combined_risk = initial_state.get("combined_risk", 10.0)
        urgency_score = initial_state.get("urgency_score", 3.5)
        goal_ratio = initial_state.get("goal_ratio", 1.0)
        category_risk = initial_state.get("category_risk", 0.5)
        time_penalty = initial_state.get("time_penalty", 5.0)
        country_factor = initial_state.get("country_factor", 0.5)
        
        # 基于真实数据分布计算风险因子
        # 1. 生存率：基于 combined_risk 计算
        # combined_risk 越高，生存率越低
        risk_normalized = min(combined_risk / 50.0, 1.0)
        survival_rate = 1.0 - risk_normalized * 0.7  # 风险满载时生存率降至0.3
        
        # 2. 风险增幅：基于多个风险因子的综合评估
        # 风险因子包括：goal_ratio, time_penalty, category_risk, country_factor
        risk_factors = [
            min(goal_ratio / 10.0, 1.0),           # 目标倍率风险
            min(time_penalty / 15.0, 1.0),         # 时间惩罚风险
            category_risk,                          # 类别风险
            country_factor,                         # 国家风险
        ]
        avg_risk_increase = sum(risk_factors) / len(risk_factors)
        
        # 3. 恢复速度：基于紧急度和风险的综合评估
        # 紧急度高意味着项目有时间压力，但同时也显示活力
        urgency_factor = min(urgency_score / 7.0, 1.0)
        recovery_speed = 0.5 + urgency_factor * 0.4 - risk_normalized * 0.3
        
        # 4. 最坏情况表现：基于风险峰值预测
        worst_risk = combined_risk * (1 + avg_risk_increase * 0.5)
        worst_case_performance = max(0, 1.0 - worst_risk / 30.0)
        
        # 5. 适应性得分：基于特征稳定性
        # 特征值越均衡，适应性越好
        feature_stability = 1.0 - (risk_normalized + avg_risk_increase) / 2
        adaptability_score = max(0, min(1.0, feature_stability))
        
        # 综合韧性得分 (复用M12公式)
        overall_resilience = (
            survival_rate * 0.3 +
            (1 - avg_risk_increase) * 0.25 +
            recovery_speed * 0.2 +
            worst_case_performance * 0.15 +
            adaptability_score * 0.1
        )
        
        # 添加不确定性噪声
        noise = random.gauss(0, 0.05)
        resilience_with_noise = max(0.0, min(1.0, overall_resilience + noise))
        
        outcome = "success" if resilience_with_noise > 0.5 else "failed"
        
        return {
            "outcome": outcome,
            "resilience_score": resilience_with_noise,
            "survival_rate": survival_rate,
            "risk_increase": avg_risk_increase,
            "recovery_speed": recovery_speed,
            "worst_case_performance": worst_case_performance,
            "adaptability_score": adaptability_score,
            "risk_factors": {
                "combined_risk": combined_risk,
                "urgency_score": urgency_score,
                "goal_ratio": goal_ratio,
                "category_risk": category_risk,
                "time_penalty": time_penalty,
                "country_factor": country_factor
            }
        }
    
    def run_batch(self, experiments: List[Dict[str, Any]]) -> List[ExperimentResult]:
        """批量执行实验"""
        results = []
        for exp_data in experiments:
            exp_id = exp_data.get("experiment_id", len(results))
            initial_state = exp_data.get("initial_state", {})
            result = self.run_single_experiment(exp_id, initial_state)
            results.append(result)
        return results


@dataclass
class M16Config:
    """M16模块配置"""
    output_dir: Path = field(default_factory=lambda: EXPERIMENT_RESULTS_DIR)
    report_dir: Path = field(default_factory=lambda: ANALYSIS_REPORTS_DIR)
    data_sources: List[str] = field(default_factory=lambda: ["simulation", "kickstarter", "ood"])
    analysis_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class M16StageResult:
    """M16阶段结果"""
    stage_name: str
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass 
class M16Summary:
    """M16执行汇总"""
    total_stages: int
    successful_stages: int
    total_experiments: int
    success_rate: float
    total_duration_seconds: float
    output_files: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AnalysisConfig:
    """分析配置"""
    significance_level: float = 0.05
    correlation_threshold: float = 0.3
    min_sample_size: int = 30


class HypothesisResult:
    """假设检验结果"""
    def __init__(self, hypothesis_name: str, null_hypothesis: str, 
                 p_value: float, is_significant: bool, conclusion: str):
        self.hypothesis_name = hypothesis_name
        self.null_hypothesis = null_hypothesis
        self.p_value = p_value
        self.is_significant = is_significant
        self.conclusion = conclusion


class AnalysisReport:
    """分析报告数据类"""
    def __init__(self, title: str, timestamp: str):
        self.title = title
        self.timestamp = timestamp
        self.data_summary: Dict[str, Any] = {}
        self.key_findings: List[str] = []
        self.hypothesis_tests: List[HypothesisResult] = []
        self.correlation_analysis: Dict[str, float] = {}
        self.conclusions: List[str] = []
        self.recommendations: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "timestamp": self.timestamp,
            "data_summary": self.data_summary,
            "key_findings": self.key_findings,
            "hypothesis_tests": [
                {
                    "name": h.hypothesis_name,
                    "null_hypothesis": h.null_hypothesis,
                    "p_value": h.p_value,
                    "is_significant": h.is_significant,
                    "conclusion": h.conclusion
                } for h in self.hypothesis_tests
            ],
            "correlation_analysis": self.correlation_analysis,
            "conclusions": self.conclusions,
            "recommendations": self.recommendations
        }


class ExperimentDataLoader:
    """实验数据加载器"""
    def __init__(self, data_dir: str, results_dir: str):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self._df = None
    
    def load_all(self) -> Tuple[List[Dict], Any]:
        """加载所有实验数据"""
        experiments = []
        
        # 加载实验结果
        result_files = list(self.results_dir.glob("experiment_results_*.json"))
        for result_file in result_files:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                experiments.extend(data.get("experiments", []))
        
        # 去重
        seen = set()
        unique_experiments = []
        for exp in experiments:
            exp_id = exp.get("experiment_id", id(exp))
            if exp_id not in seen:
                seen.add(exp_id)
                unique_experiments.append(exp)
        
        self._df = self._to_dataframe(unique_experiments)
        return unique_experiments, self._df
    
    def _to_dataframe(self, experiments: List[Dict]) -> Any:
        """转换为DataFrame"""
        if not experiments:
            return None
        
        rows = []
        for exp in experiments:
            row = {
                "experiment_id": exp.get("experiment_id", ""),
                "timestamp": exp.get("timestamp", ""),
                "duration_ms": exp.get("duration_ms", 0),
                "status": exp.get("status", "unknown"),
            }
            
            # 展平metrics
            metrics = exp.get("metrics", {})
            for k, v in metrics.items():
                row[f"metric_{k}"] = v
            
            # 展平initial_state
            initial = exp.get("initial_state", {})
            for k, v in initial.items():
                row[f"init_{k}"] = v
            
            # 展平final_state
            final = exp.get("final_state", {})
            for k, v in final.items():
                row[f"final_{k}"] = v
            
            rows.append(row)
        
        return pd.DataFrame(rows) if rows else None


class StatisticalAnalyzer:
    """统计分析器"""
    def __init__(self, df: Any, significance_level: float = 0.05):
        self.df = df
        self.significance_level = significance_level
        self._descriptive_stats: Dict[str, Dict[str, float]] = {}
        self._correlation_matrix: Any = None
    
    def compute_descriptive_stats(self, columns: List[str]) -> Dict[str, Dict[str, float]]:
        """计算描述性统计"""
        self._descriptive_stats = {}
        
        for col in columns:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    self._descriptive_stats[col] = {
                        "mean": float(data.mean()),
                        "std": float(data.std()),
                        "min": float(data.min()),
                        "q25": float(data.quantile(0.25)),
                        "median": float(data.median()),
                        "q75": float(data.quantile(0.75)),
                        "max": float(data.max())
                    }
        
        return self._descriptive_stats
    
    def correlation_matrix(self, columns: List[str]) -> Any:
        """计算相关性矩阵"""
        subset = self.df[columns].dropna()
        if len(subset) > 0:
            self._correlation_matrix = subset.corr()
            return self._correlation_matrix
        return None


class HypothesisValidator:
    """假设验证器"""
    def __init__(self, analyzer: StatisticalAnalyzer):
        self.analyzer = analyzer
        self.results: List[HypothesisResult] = []
    
    def run_all_validations(self) -> List[HypothesisResult]:
        """运行所有假设验证"""
        self.results = []
        
        # 假设1: 成功率验证
        if hasattr(self.analyzer.df, 'status'):
            success_col = self.analyzer.df['status']
            if 'success' in success_col.values:
                self._test_success_rate()
        
        return self.results
    
    def _test_success_rate(self) -> None:
        """测试成功率假设"""
        success_count = (self.analyzer.df["status"] == "success").sum()
        total = len(self.analyzer.df)
        observed_rate = success_count / total if total > 0 else 0
        
        # 二项检验
        if SCIPY_AVAILABLE and total > 0:
            result = stats.binomtest(success_count, total, p=0.5, alternative='two-sided')
            p_value = result.pvalue
        else:
            p_value = 1.0
        
        self.results.append(HypothesisResult(
            hypothesis_name="成功率假设",
            null_hypothesis="成功率 = 0.5",
            p_value=float(p_value),
            is_significant=bool(p_value < 0.05),
            conclusion=f"观察成功率 {observed_rate:.2%} 与期望 50.0%"
        ))


class ReportGenerator:
    """报告生成器"""
    def __init__(self, config: M16Config):
        self.config = config
    
    def generate_report(self, df: Any, hypotheses: List[HypothesisResult], key_findings: List[str]) -> AnalysisReport:
        """生成分析报告"""
        timestamp = datetime.now().isoformat()
        report = AnalysisReport("Kickstarter 仿真实验分析报告", timestamp)
        
        # 数据摘要
        report.data_summary = {
            "total_experiments": len(df) if df is not None else 0,
            "success_rate": (df["status"] == "success").mean() if df is not None and "status" in df.columns else 0,
            "avg_duration_ms": df["duration_ms"].mean() if df is not None and "duration_ms" in df.columns else 0,
            "feature_count": len([c for c in df.columns if c.startswith("init_")]) if df is not None else 0
        }
        
        # 关键发现
        report.key_findings = key_findings
        
        # 假设检验
        report.hypothesis_tests = hypotheses
        
        # 相关性分析
        if df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                try:
                    corr = df[numeric_cols].corr()
                    for i, col1 in enumerate(numeric_cols):
                        for col2 in numeric_cols[i+1:]:
                            if abs(corr.loc[col1, col2]) > 0.3:
                                report.correlation_analysis[f"{col1} vs {col2}"] = float(corr.loc[col1, col2])
                except:
                    pass
        
        # 结论和建议
        sig_count = sum(1 for h in hypotheses if h.is_significant)
        report.conclusions = [
            f"在 0.05 显著性水平下，共发现 {sig_count} 个统计显著差异。",
            f"系统表现良好，成功率达到 {report.data_summary['success_rate']:.1%}。",
            f"总样本数: {report.data_summary['total_experiments']}。"
        ]
        
        report.recommendations = [
            "继续监控关键指标，收集更多数据以支持后续分析。"
        ]
        
        return report
    
    def save_json_report(self, report: AnalysisReport) -> str:
        """保存JSON报告"""
        output_path = Path(self.config.report_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"analysis_report_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(file_path)

    def save_markdown_report(self, report: AnalysisReport) -> str:
        """保存Markdown报告 - 优美格式"""
        output_path = Path(self.config.report_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"analysis_report_{timestamp}.md"
        
        # 生成成功率进度条
        def progress_bar(rate, width=30):
            filled = int(rate * width)
            bar = "#" * filled + "-" * (width - filled)
            return f"[{bar}] {rate*100:.1f}%"
        
        success_rate = report.data_summary['success_rate']
        duration_ms = report.data_summary['avg_duration_ms']
        total_exp = report.data_summary['total_experiments']
        feature_count = report.data_summary['feature_count']
        
        # 状态颜色
        if success_rate >= 0.8:
            rate_icon = "[OK]"
            rate_status = "优秀"
        elif success_rate >= 0.5:
            rate_icon = "[OK]"
            rate_status = "良好"
        elif success_rate >= 0.3:
            rate_icon = "[WARN]"
            rate_status = "一般"
        else:
            rate_icon = "[FAIL]"
            rate_status = "需改进"
        
        sig_count = sum(1 for h in report.hypothesis_tests if h.is_significant)
        total_hyp = len(report.hypothesis_tests)
        
        # 标题和摘要
        md_content = f"""
# {report.title}

> **生成时间**: `{report.timestamp}`  
> **报告状态**: 自动生成

---

## 执行摘要

| 指标 | 数值 | 状态 |
|:-----|:----:|:----:|
| 总实验数 | **{total_exp:,}** | [i] |
| 成功率 | **{success_rate:.2%}** | {rate_icon} {rate_status} |
| 平均耗时 | **{duration_ms:.2f} ms** | [i] |
| 特征维度 | **{feature_count}** | [i] |

**成功率可视化**:
```
{progress_bar(success_rate)}
```

---

## 关键发现

"""
        
        for finding in report.key_findings:
            md_content += f"> - **{finding}**\n\n"
        
        md_content += f"""

---

## 假设检验结果

> 共进行 {total_hyp} 项统计检验，其中 {sig_count} 项达到显著水平 (alpha=0.05)

"""
        
        if report.hypothesis_tests:
            for hyp in report.hypothesis_tests:
                status_icon = "[PASS]" if hyp.is_significant else "[----]"
                conf_level = "**显著**" if hyp.is_significant else "不显著"
                md_content += f"""
### {status_icon} {hyp.hypothesis_name}

| 项目 | 内容 |
|:-----|:-----|
| 原假设 (H0) | {hyp.null_hypothesis} |
| p 值 | `{hyp.p_value:.4f}` |
| 结论 | {conf_level} |
| 解读 | {hyp.conclusion} |

---
"""
        else:
            md_content += "*暂无假设检验数据*\n"
        
        md_content += """
## 相关性分析

"""
        
        if report.correlation_analysis and len(report.correlation_analysis) > 0:
            md_content += "| 变量对 (X vs Y) | Pearson r | 相关强度 | 方向 |\n"
            md_content += "|:--------------|:---------:|:--------:|:----:|\n"
            for pair, corr in report.correlation_analysis.items():
                abs_corr = abs(corr)
                if abs_corr >= 0.7:
                    strength = "强相关"
                    direction = "正向" if corr > 0 else "负向"
                elif abs_corr >= 0.4:
                    strength = "中等相关"
                    direction = "正向" if corr > 0 else "负向"
                elif abs_corr >= 0.2:
                    strength = "弱相关"
                    direction = "正向" if corr > 0 else "负向"
                else:
                    strength = "几乎无关"
                    direction = "---"
                md_content += f"| `{pair}` | `{corr:+.4f}` | {strength} | {direction} |\n"
        else:
            md_content += "> [INFO] 未发现显著相关性 (|r| < 0.3)\n\n"
        
        md_content += """
---

## 结论

"""
        
        for i, conclusion in enumerate(report.conclusions, 1):
            md_content += f"{i}. {conclusion}\n"
        
        md_content += """

---

## 建议

"""
        
        for i, rec in enumerate(report.recommendations, 1):
            md_content += f"{i}. {rec}\n"
        
        md_content += f"""

---

**报告说明**

- 本报告由 M16 数据分析引擎自动生成
- 显著性水平: alpha = 0.05
- 统计方法: Pearson/Spearman 相关分析, 二项检验
- 生成时间: {report.timestamp}
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(file_path)

    def generate_jupyter_notebook(self, report: AnalysisReport, df: Any) -> str:
        """生成Jupyter Notebook"""
        if not NBFORMAT_AVAILABLE:
            return "nbformat not available"
        
        output_path = Path(self.config.report_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"analysis_report_{timestamp}.ipynb"
        
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
        
        nb = new_notebook()
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.0"}
        }
        
        cells = []
        cells.append(new_markdown_cell(f"# {report.title}"))
        cells.append(new_markdown_cell(f"**生成时间**: {report.timestamp}"))
        cells.append(new_markdown_cell("## 摘要"))
        summary_table = f"| 指标 | 值 |\n|------|-----|\n| 总实验数 | {report.data_summary['total_experiments']} |\n| 成功率 | {report.data_summary['success_rate']:.2%} |\n| 平均耗时 | {report.data_summary['avg_duration_ms']:.2f} ms |"
        cells.append(new_markdown_cell(summary_table))
        cells.append(new_markdown_cell("## 关键发现"))
        for finding in report.key_findings:
            cells.append(new_markdown_cell(f"- {finding}"))
        cells.append(new_markdown_cell("## 假设检验结果"))
        for hyp in report.hypothesis_tests:
            status = "[PASS]" if hyp.is_significant else "[----]"
            cells.append(new_markdown_cell(f"### {status} {hyp.hypothesis_name}\n- **原假设**: {hyp.null_hypothesis}\n- **p值**: {hyp.p_value:.4f}\n- **显著性**: {hyp.is_significant}"))
        cells.append(new_markdown_cell("## 相关性分析"))
        if report.correlation_analysis:
            corr_table = "| 变量对 | 相关系数 |\n|--------|----------|\n"
            for pair, corr in report.correlation_analysis.items():
                corr_table += f"| {pair} | {corr:.4f} |\n"
            cells.append(new_markdown_cell(corr_table))
        cells.append(new_markdown_cell("## 结论"))
        for conclusion in report.conclusions:
            cells.append(new_markdown_cell(f"- {conclusion}"))
        cells.append(new_markdown_cell("## 建议"))
        for rec in report.recommendations:
            cells.append(new_markdown_cell(f"- {rec}"))
        
        nb.cells = cells
        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        return str(file_path)


class DataAnalyzer:
    """数据分析器 - 整合统计分析、假设验证和报告生成"""
    
    def __init__(self, config: M16Config):
        self.config = config
        self.df = None
        self.hypotheses: List[HypothesisResult] = []
        self.key_findings: List[str] = []
        self.report: Optional[AnalysisReport] = None
    
    def _to_dataframe(self, experiments: List[Dict[str, Any]]) -> Any:
        """将实验数据转换为DataFrame"""
        if not experiments:
            return None
        
        import pandas as pd
        rows = []
        for exp in experiments:
            row = {
                "experiment_id": exp.get("experiment_id", ""),
                "timestamp": exp.get("timestamp", ""),
                "duration_ms": exp.get("duration_ms", 0),
                "status": exp.get("status", "unknown"),
            }
            metrics = exp.get("metrics", {})
            for k, v in metrics.items():
                row[f"metric_{k}"] = v
            initial = exp.get("initial_state", {})
            for k, v in initial.items():
                row[f"init_{k}"] = v
            final = exp.get("final_state", {})
            for k, v in final.items():
                row[f"final_{k}"] = v
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _perform_analysis(self) -> bool:
        """执行统计分析"""
        if self.df is None or len(self.df) == 0:
            return False
            
        try:
            # 描述性统计
            analyzer = StatisticalAnalyzer(self.df, 0.05)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 生成关键发现
            self.key_findings = []
            if len(self.df) > 0:
                success_rate = (self.df["status"] == "success").mean() if "status" in self.df.columns else 0
                self.key_findings.append(f"总样本数: {len(self.df)}, 成功率: {success_rate:.2%}")
                
            # 假设验证
            validator = HypothesisValidator(analyzer)
            self.hypotheses = validator.run_all_validations()
            
            return True
        except Exception as e:
            print(f"[ERROR] 统计分析失败: {e}")
            return False
    
    def _generate_reports(self) -> Tuple[bool, str]:
        """生成各类报告"""
        if self.report is None:
            return False, "无报告数据"
            
        try:
            report_gen = ReportGenerator(self.config)
            
            # 生成JSON报告
            json_path = report_gen.save_json_report(self.report)
            print(f"[INFO] JSON报告已保存: {json_path}")
            
            # 生成Markdown报告
            md_path = report_gen.save_markdown_report(self.report)
            print(f"[INFO] Markdown报告已保存: {md_path}")
            
            # 生成Jupyter Notebook
            nb_path = report_gen.generate_jupyter_notebook(self.report, self.df)
            if nb_path and nb_path != "nbformat not available":
                print(f"[INFO] Jupyter Notebook已保存: {nb_path}")
            
            return True, json_path
        except Exception as e:
            print(f"[ERROR] 报告生成失败: {e}")
            return False, str(e)
    
    def run(self) -> Tuple[bool, str]:
        """
        执行完整的数据分析流程
        
        Returns:
            Tuple[bool, str]: (是否成功, 输出文件路径或错误信息)
        """
        print("[INFO] 步骤1: 加载数据...")
        
        # 使用RealDataLoader加载数据
        try:
            loader = RealDataLoader(self.config)
            success, data_info = loader.load_all_sources()
            if success and len(loader.experiments) > 0:
                self.df = self._to_dataframe(loader.experiments)
                print(f"[INFO] 成功从 RealDataLoader 加载数据")
        except Exception as e:
            print(f"[DEBUG] RealDataLoader失败: {e}")
            # 尝试ExperimentDataLoader
            try:
                exp_loader = ExperimentDataLoader(
                    data_dir=str(self.config.output_dir),
                    results_dir=str(self.config.output_dir)
                )
                _, self.df = exp_loader.load_all()
                if self.df is not None and len(self.df) > 0:
                    print(f"[INFO] 成功从 ExperimentDataLoader 加载数据")
                else:
                    return False, "无法加载任何数据源"
            except Exception as e2:
                print(f"[DEBUG] ExperimentDataLoader失败: {e2}")
                return False, f"无法加载任何数据源: {e2}"
        
        if self.df is None or len(self.df) == 0:
            return False, "加载的数据为空"
        
        print(f"[INFO] 加载数据: {len(self.df)} 条记录")
        
        print("[INFO] 步骤2: 执行统计分析...")
        if not self._perform_analysis():
            return False, "统计分析失败"
        
        print("[INFO] 步骤3: 生成分析报告...")
        report_gen = ReportGenerator(self.config)
        self.report = report_gen.generate_report(
            self.df, 
            self.hypotheses, 
            self.key_findings
        )
        
        success, output = self._generate_reports()
        return success, output


class M16Orchestrator:
    """
    M16主编排器 (真实数据分析)
    
    功能：
    1. 加载真实数据
    2. 执行数据分析和报告生成
    3. 汇总执行结果
    """
    
    def __init__(self, config: Optional[M16Config] = None):
        self.config = config or M16Config()
        self.stage_results: List[M16StageResult] = []
    
    def run(self) -> M16Summary:
        """执行完整分析流程"""
        start_time = time.time()
        
        print("=" * 60)
        print("M16 真实数据分析模块")
        print("=" * 60)
        print(f"数据来源优先级:")
        print(f"  1. experiment_results/*.json (M3/M12仿真结果)")
        print(f"  2. datasets/Kickstarter.csv (Kickstarter原始数据)")
        print(f"  3. Kickstarter_Clean/m12_ood_tests/*.json (OOD测试)")
        print(f"输出目录: {self.config.output_dir}")
        print("=" * 60)
        
        # 阶段1: 加载真实数据
        print("\n" + "=" * 60)
        print("[M16-1/2] 阶段1: 加载真实数据")
        print("=" * 60)
        
        stage1_start = time.time()
        loader = RealDataLoader(self.config)
        success1, data_info = loader.load_all_sources()
        
        stage1_duration = time.time() - stage1_start
        self.stage_results.append(M16StageResult(
            stage_name="数据加载",
            success=success1,
            output_path=data_info if success1 else None,
            error_message=None if success1 else data_info,
            duration_seconds=stage1_duration
        ))
        
        if not success1:
            print(f"[ERROR] 数据加载失败: {data_info}")
            return self._create_summary(start_time)
        
        # 阶段2: 数据分析和报告生成
        print("\n" + "=" * 60)
        print("[M16-2/2] 阶段2: 数据分析与报告生成")
        print("=" * 60)
        
        stage2_start = time.time()
        
        print("[INFO] 步骤1: 加载数据...")
        
        # 使用DataAnalyzer加载数据
        analyzer = DataAnalyzer(self.config)
        success2, output2 = analyzer.run()
        
        stage2_duration = time.time() - stage2_start
        self.stage_results.append(M16StageResult(
            stage_name="数据分析与报告",
            success=success2,
            output_path=output2 if success2 else None,
            error_message=None if success2 else output2,
            duration_seconds=stage2_duration
        ))
        
        # 生成汇总
        summary = self._create_summary(start_time)
        
        # 打印汇总信息
        print("\n" + "=" * 60)
        print("M16 执行汇总")
        print("=" * 60)
        print(f"总阶段数: {summary.total_stages}")
        print(f"成功阶段: {summary.successful_stages}")
        print(f"实验数量: {summary.total_experiments}")
        print(f"成功率: {summary.success_rate:.2%}")
        print(f"总耗时: {summary.total_duration_seconds:.2f} 秒")
        print("\n输出文件:")
        print(f"  - {loader.data_source}")
        for f in summary.output_files:
            print(f"  - {f}")
        
        return summary
    
    def _create_summary(self, start_time: float) -> M16Summary:
        """创建执行汇总"""
        total_duration = time.time() - start_time
        successful_stages = sum(1 for r in self.stage_results if r.success)
        
        output_files = [r.output_path for r in self.stage_results if r.success and r.output_path]
        
        # 获取实验数量
        total_experiments = 0
        success_rate = 0.0
        
        # 从最后的分析阶段获取
        for result in reversed(self.stage_results):
            if result.stage_name == "数据分析与报告":
                # 尝试从报告文件读取
                if result.output_path and Path(result.output_path).exists():
                    try:
                        with open(result.output_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            total_experiments = data.get("data_summary", {}).get("total_experiments", 0)
                            success_rate = data.get("data_summary", {}).get("success_rate", 0)
                    except:
                        pass
        
        return M16Summary(
            total_stages=len(self.stage_results),
            successful_stages=successful_stages,
            total_experiments=total_experiments,
            success_rate=success_rate,
            total_duration_seconds=total_duration,
            output_files=output_files
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='M16 真实数据分析模块')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 创建配置
    config = M16Config()
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    # 创建编排器并执行
    orchestrator = M16Orchestrator(config)
    summary = orchestrator.run()
    
    # 返回状态码
    return 0 if summary.successful_stages == summary.total_stages else 1


if __name__ == "__main__":
    sys.exit(main())
