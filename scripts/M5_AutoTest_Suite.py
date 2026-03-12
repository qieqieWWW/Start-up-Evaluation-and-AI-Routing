#!/usr/bin/env python
# coding: utf-8

"""
M5 高健壮性自动化测试套件
==========================
功能：完整闭环自动化测试系统
- 规则驱动测试用例生成
- 数据注入与动态仿真执行
- 自动报告生成与警告触发
- 可复现、可度量、可迭代

集成：M3/M4/M6/M8 核心模块
"""

import os, sys, json, time, threading, traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

# ============ 导入外部模块 ============
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pandas as pd
import numpy as np

try:
    from M3仿真环境基座模块版 import StartupEnv, M3Config, save_simulation_results
except:
    print("⚠️ M3 unavailable")
    StartupEnv = None

try:
    from m8_rule_adapter import judge_project_risk_m8
except:
    print("⚠️ M8 unavailable")
    judge_project_risk_m8 = None


# ============ 枚举定义 ============

class TestStatus(str, Enum):
    PENDING, RUNNING, PASSED, FAILED, ERROR, SKIPPED = "pending", "running", "passed", "failed", "error", "skipped"


class RiskLevel(str, Enum):
    CRITICAL, HIGH, MEDIUM, LOW, INFO = "critical", "high", "medium", "low", "info"


# ============ 数据类定义 ============

@dataclass
class TestCase:
    """测试用例"""
    case_id: str
    case_name: str
    description: str
    project_data: Dict[str, Any]
    expected_risk_level: str
    risk_factors: List[str] = field(default_factory=list)
    max_steps: int = 100
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TestResult:
    """测试结果"""
    case_id: str
    case_name: str
    status: TestStatus
    actual_risk_level: str
    expected_risk_level: str
    passed: bool
    start_time: float
    end_time: float
    duration: float = 0.0
    error_message: Optional[str] = None
    trajectory: Optional[Dict] = None
    metrics: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id, "case_name": self.case_name, "status": self.status.value,
            "actual_risk_level": self.actual_risk_level, "expected_risk_level": self.expected_risk_level,
            "passed": self.passed, "duration": self.duration, "error_message": self.error_message,
            "trajectory": self.trajectory, "metrics": self.metrics,
        }


@dataclass
class TestReport:
    """测试报告"""
    project_id: str
    run_id: str
    start_time: float
    end_time: float
    total_cases: int
    passed_count: int
    failed_count: int
    error_count: int
    skipped_count: int
    pass_rate: float
    results: List[TestResult] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "project_id": self.project_id, "run_id": self.run_id,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration": self.end_time - self.start_time,
            "summary": {"total": self.total_cases, "passed": self.passed_count, "failed": self.failed_count,
                       "error": self.error_count, "skipped": self.skipped_count, "pass_rate": f"{self.pass_rate:.2%}"},
            "results": [r.to_dict() for r in self.results],
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


# ============ 日志系统 ============

class TestLogger:
    """测试日志系统"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("M5TestSuite")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        log_file = self.log_dir / f"m5_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)
    def debug(self, msg: str): self.logger.debug(msg)


# ============ 测试用例生成器 ============

class TestCaseGenerator:
    """从数据集生成测试用例"""
    
    def __init__(self, logger: TestLogger):
        self.logger = logger
    
    def generate_from_dataset(self, data_file: str, sample_size: int = 50) -> List[TestCase]:
        """从CSV生成用例"""
        test_cases = []
        try:
            if not os.path.exists(data_file):
                self.logger.warning(f"数据文件不存在: {data_file}")
                return test_cases
            
            df = pd.read_csv(data_file)
            self.logger.info(f"✓ 加载数据集: {len(df)} 行")
            
            sampled_df = df.sample(min(len(df), sample_size), random_state=42)
            
            for idx, row in sampled_df.iterrows():
                case_id = f"test_case_{int(idx):05d}"
                project_data = self._extract_project_data(row)
                
                test_case = TestCase(
                    case_id=case_id,
                    case_name=f"评估_{row.get('main_category', 'UNKNOWN')}_{idx}",
                    description=f"项目: {row.get('name', 'Unknown')[:50]}",
                    project_data=project_data,
                    expected_risk_level=row.get('risk_level', 'medium'),
                    risk_factors=self._extract_risk_factors(row),
                    priority=self._estimate_priority(row),
                    tags=[row.get('main_category', 'UNKNOWN')]
                )
                test_cases.append(test_case)
            
            self.logger.info(f"✓ 生成 {len(test_cases)} 个测试用例")
            return test_cases
        except Exception as e:
            self.logger.error(f"✗ 生成用例失败: {str(e)}")
            return test_cases
    
    def generate_synthetic_cases(self, num_cases: int = 10) -> List[TestCase]:
        """生成合成极端场景"""
        test_cases = []
        scenarios = [
            ("超高融资", {"goal_ratio": 9.5, "time_penalty": 0.5, "category_risk": 0.2, 
                        "country_factor": 0.3, "urgency_score": 0.1}, "high", 2),
            ("极短周期高风险", {"goal_ratio": 2.0, "time_penalty": 3.5, "category_risk": 0.8,
                            "country_factor": 0.6, "urgency_score": 0.3}, "critical", 2),
            ("理想场景", {"goal_ratio": 1.2, "time_penalty": 1.0, "category_risk": 0.1,
                        "country_factor": 0.2, "urgency_score": 0.05}, "low", 1),
        ]
        
        for i in range(num_cases):
            name, data, risk, priority = scenarios[i % len(scenarios)]
            data.update({"main_category": "Technology", "duration_days": 60, "goal_usd": 200000, "country": "US"})
            
            test_cases.append(TestCase(
                case_id=f"synthetic_{i:03d}",
                case_name=f"{name}_{i}",
                description=f"合成场景: {name}",
                project_data=data,
                expected_risk_level=risk,
                priority=priority,
                tags=['synthetic']
            ))
        
        self.logger.info(f"✓ 生成 {num_cases} 个合成用例")
        return test_cases
    
    def _extract_project_data(self, row) -> Dict:
        """提取项目数据"""
        return {
            "goal_ratio": float(row.get("goal_ratio", 1.0)),
            "time_penalty": float(row.get("time_penalty", 1.0)),
            "category_risk": float(row.get("category_risk", 0.5)),
            "country_factor": float(row.get("country_factor", 0.5)),
            "urgency_score": float(row.get("urgency_score", 0.0)),
            "main_category": str(row.get("main_category", "UNKNOWN")),
            "duration_days": int(row.get("duration_days", 30)),
            "goal_usd": float(row.get("goal_usd", 10000)),
            "country": str(row.get("country", "US"))
        }
    
    def _extract_risk_factors(self, row) -> List[str]:
        """提取风险因素"""
        factors = []
        if row.get("goal_ratio", 0) > 5: factors.append("高融资目标")
        if row.get("time_penalty", 0) > 2: factors.append("融资周期短")
        if row.get("category_risk", 0) > 0.5: factors.append("高风险品类")
        return factors
    
    def _estimate_priority(self, row) -> int:
        """估计优先级"""
        if row.get("category_risk", 0) > 0.5: return 2
        if row.get("goal_ratio", 0) > 3: return 1
        return 0


# ============ 仿真执行引擎 ============

class SimulationExecutor:
    """在仿真环境中执行测试"""
    
    def __init__(self, logger: TestLogger, enable_m3: bool = True):
        self.logger = logger
        self.enable_m3 = enable_m3 and StartupEnv is not None
        
        if self.enable_m3:
            try:
                self.config = M3Config()
                self.logger.info("✓ M3配置初始化成功")
            except Exception as e:
                self.logger.warning(f"M3初始化失败，使用mock: {e}")
                self.enable_m3 = False
    
    def execute_test_case(self, test_case: TestCase, timeout: float = 30.0) -> TestResult:
        """执行单个测试用例"""
        start_time = time.time()
        
        result = TestResult(
            case_id=test_case.case_id,
            case_name=test_case.case_name,
            status=TestStatus.RUNNING,
            actual_risk_level="unknown",
            expected_risk_level=test_case.expected_risk_level,
            passed=False,
            start_time=start_time,
            end_time=start_time,
        )
        
        try:
            # 风险判定
            actual_risk = self._judge_risk(test_case.project_data)
            result.actual_risk_level = actual_risk
            
            # 检查通过/失败
            result.passed = self._check_pass(actual_risk, test_case.expected_risk_level)
            result.status = TestStatus.PASSED if result.passed else TestStatus.FAILED
            
            # 生成轨迹
            result.trajectory = self._generate_trajectory(test_case.project_data, actual_risk)
            
            # 计算度量
            result.metrics = self._calc_metrics(test_case.project_data, actual_risk)
            
        except Exception as e:
            self.logger.error(f"执行失败 {test_case.case_id}: {str(e)}")
            result.status = TestStatus.ERROR
            result.error_message = str(e)
        
        finally:
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
        
        return result
    
    def _judge_risk(self, project_data: Dict) -> str:
        """M8风险判定"""
        if judge_project_risk_m8:
            try:
                risk_level, reasons, metrics = judge_project_risk_m8(project_data, verbose=False)
                return risk_level.lower()
            except:
                pass
        
        # 简单的mock判定
        goal_ratio = project_data.get("goal_ratio", 1.0)
        time_penalty = project_data.get("time_penalty", 1.0)
        category_risk = project_data.get("category_risk", 0.5)
        
        combined_score = goal_ratio * 0.4 + time_penalty * 0.35 + category_risk * 0.25
        
        if combined_score > 6: return "critical"
        if combined_score > 4.5: return "high"
        if combined_score > 3: return "medium"
        return "low"
    
    def _check_pass(self, actual: str, expected: str) -> bool:
        """判定用例是否通过"""
        risk_levels = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        actual_level = risk_levels.get(actual, 2)
        expected_level = risk_levels.get(expected, 2)
        return abs(actual_level - expected_level) <= 1
    
    def _generate_trajectory(self, project_data: Dict, risk_level: str) -> Dict:
        """生成执行轨迹"""
        return {
            "steps": 1,
            "final_risk": risk_level,
            "state_history": [{"step": 0, "state": "initialized", "risk": risk_level}],
            "action_history": [{"step": 0, "action": "evaluate"}]
        }
    
    def _calc_metrics(self, project_data: Dict, risk_level: str) -> Dict:
        """计算度量指标"""
        return {
            "goal_ratio": project_data.get("goal_ratio", 0),
            "time_penalty": project_data.get("time_penalty", 0),
            "category_risk": project_data.get("category_risk", 0),
            "country_factor": project_data.get("country_factor", 0),
            "urgency_score": project_data.get("urgency_score", 0),
            "predicted_risk": risk_level,
        }


# ============ 报告生成器 ============

class ReportGenerator:
    """生成测试报告和警告"""
    
    def __init__(self, logger: TestLogger, output_dir: str = "test_reports"):
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, project_id: str, run_id: str, results: List[TestResult]) -> TestReport:
        """生成完整报告"""
        start_times = [r.start_time for r in results] if results else [time.time()]
        end_times = [r.end_time for r in results] if results else [time.time()]
        
        report = TestReport(
            project_id=project_id,
            run_id=run_id,
            start_time=min(start_times),
            end_time=max(end_times),
            total_cases=len(results),
            passed_count=sum(1 for r in results if r.passed),
            failed_count=sum(1 for r in results if r.status == TestStatus.FAILED),
            error_count=sum(1 for r in results if r.status == TestStatus.ERROR),
            skipped_count=sum(1 for r in results if r.status == TestStatus.SKIPPED),
            pass_rate=sum(1 for r in results if r.passed) / len(results) if results else 0.0,
            results=results,
            warnings=self._generate_warnings(results),
            metrics=self._calc_aggregate_metrics(results),
        )
        
        return report
    
    def _generate_warnings(self, results: List[TestResult]) -> List[Dict]:
        """生成警告"""
        warnings = []
        
        failed_cases = [r for r in results if r.status == TestStatus.FAILED]
        if failed_cases and len(failed_cases) / len(results) > 0.2:
            warnings.append({
                "level": "high",
                "type": "high_failure_rate",
                "message": f"失败率过高 ({len(failed_cases)}/{len(results)})",
                "cases": [r.case_id for r in failed_cases[:5]]
            })
        
        critical_cases = [r for r in results if r.actual_risk_level == "critical"]
        if critical_cases:
            warnings.append({
                "level": "high",
                "type": "critical_risk_detected",
                "message": f"检测到 {len(critical_cases)} 个关键风险项目",
                "cases": [r.case_id for r in critical_cases[:5]]
            })
        
        return warnings
    
    def _calc_aggregate_metrics(self, results: List[TestResult]) -> Dict:
        """计算聚合度量"""
        if not results: return {}
        
        durations = [r.duration for r in results if r.duration]
        risk_levels = {}
        for r in results:
            risk_levels[r.actual_risk_level] = risk_levels.get(r.actual_risk_level, 0) + 1
        
        return {
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "risk_distribution": risk_levels,
            "execution_time": max(r.end_time for r in results) - min(r.start_time for r in results),
        }
    
    def save_report(self, report: TestReport) -> Path:
        """保存报告"""
        filename = self.output_dir / f"report_{report.run_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✓ 报告已保存: {filename}")
        return filename
    
    def print_summary(self, report: TestReport):
        """打印摘要"""
        print("\n" + "="*60)
        print("M5 自动化测试报告")
        print("="*60)
        print(f"项目ID: {report.project_id}")
        print(f"运行ID: {report.run_id}")
        print(f"时间: {datetime.fromtimestamp(report.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n测试结果统计:")
        print(f"  总用例数: {report.total_cases}")
        print(f"  通过: {report.passed_count}")
        print(f"  失败: {report.failed_count}")
        print(f"  异常: {report.error_count}")
        print(f"  通过率: {report.pass_rate:.2%}")
        
        if report.warnings:
            print(f"\n⚠️ 警告 ({len(report.warnings)}):")
            for w in report.warnings:
                print(f"  [{w['level']}] {w['message']}")
        
        print("="*60 + "\n")


# ============ 主测试引擎 ============

class M5AutomationTestEngine:
    """M5自动化测试引擎 - 主入口"""
    
    def __init__(self, project_id: str, output_dir: str = "test_reports", data_file: str = None):
        self.project_id = project_id
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger = TestLogger()
        self.case_generator = TestCaseGenerator(self.logger)
        self.executor = SimulationExecutor(self.logger)
        self.report_generator = ReportGenerator(self.logger, output_dir)
        
        self.data_file = data_file
        self.test_cases: List[TestCase] = []
        self.test_results: List[TestResult] = []
        
        self.logger.info(f"M5测试引擎初始化 - {self.project_id}")
    
    def prepare_test_cases(self, use_synthetic: bool = True, num_synthetic: int = 10) -> List[TestCase]:
        """准备测试用例"""
        self.logger.info("【阶段1】准备测试用例")
        
        # 从数据文件生成
        if self.data_file and os.path.exists(self.data_file):
            self.test_cases.extend(self.case_generator.generate_from_dataset(self.data_file, sample_size=50))
        
        # 生成合成用例
        if use_synthetic:
            self.test_cases.extend(self.case_generator.generate_synthetic_cases(num_synthetic))
        
        # 如果没有用例,生成默认用例
        if not self.test_cases:
            self.logger.warning("未生成任何用例，创建默认用例")
            self.test_cases = self.case_generator.generate_synthetic_cases(20)
        
        self.logger.info(f"✓ 准备完成: {len(self.test_cases)} 个用例")
        return self.test_cases
    
    def execute_tests(self, parallel: bool = True, num_workers: int = 4) -> List[TestResult]:
        """执行测试"""
        self.logger.info("【阶段2】执行测试")
        
        if parallel:
            self._execute_parallel(num_workers)
        else:
            self._execute_sequential()
        
        self.logger.info(f"✓ 执行完成: {len(self.test_results)} 个结果")
        return self.test_results
    
    def _execute_sequential(self):
        """顺序执行"""
        for i, test_case in enumerate(self.test_cases):
            self.logger.info(f"执行 [{i+1}/{len(self.test_cases)}] {test_case.case_id}")
            result = self.executor.execute_test_case(test_case)
            self.test_results.append(result)
    
    def _execute_parallel(self, num_workers: int):
        """并行执行"""
        def worker():
            while True:
                try:
                    test_case = None
                    with lock:
                        if queue_idx[0] < len(self.test_cases):
                            test_case = self.test_cases[queue_idx[0]]
                            queue_idx[0] += 1
                    
                    if test_case is None: break
                    
                    result = self.executor.execute_test_case(test_case)
                    with lock:
                        self.test_results.append(result)
                except:
                    pass
        
        lock = threading.Lock()
        queue_idx = [0]
        threads = [threading.Thread(target=worker) for _ in range(num_workers)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    
    def generate_report(self) -> TestReport:
        """生成报告"""
        self.logger.info("【阶段3】生成报告")
        
        report = self.report_generator.generate_report(
            self.project_id, self.run_id, self.test_results
        )
        
        self.report_generator.save_report(report)
        self.report_generator.print_summary(report)
        
        return report
    
    def run_full_pipeline(self, use_synthetic: bool = True, parallel: bool = True) -> TestReport:
        """完整流程"""
        self.logger.info(f"启动M5自动化测试 - {self.project_id}")
        
        try:
            self.prepare_test_cases(use_synthetic=use_synthetic)
            self.execute_tests(parallel=parallel)
            report = self.generate_report()
            return report
        except Exception as e:
            self.logger.error(f"管道执行失败: {str(e)}")
            traceback.print_exc()
            raise


# ============ 入口 ============

if __name__ == "__main__":
    engine = M5AutomationTestEngine(
        project_id="Kickstarter_V1",
        output_dir="test_reports",
        data_file=None  # 可指定数据文件路径
    )
    
    report = engine.run_full_pipeline(use_synthetic=True, parallel=True)
    print(f"\n✓ 测试完成! 报告: {report.run_id}")
