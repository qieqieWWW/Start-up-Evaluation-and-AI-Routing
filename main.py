#!/usr/bin/env python
# coding: utf-8

"""
启动项目评估与AI路由 - 完整集成脚本 v2.0
============================================
整合所有模块：M2 → M4 → M3 → M6 → M8 → M7 → M9 → M10 → M12 → M16

新增功能：
- 整合所有未被调用的模块（M8、M9、M10、M12）
- 支持通过Web界面（main_ui.py）启动
- 输出完整的JSON分析结果

作者：集成脚本
日期：2026-03-25
"""

import os
import sys
import json
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# ==========================================
# 配置与路径管理
# ==========================================

PROJECT_ROOT = Path(__file__).parent.absolute()
print(f"项目根目录: {PROJECT_ROOT}")

DATASET_FOLDER_NAME = "Kickstarter_2025-12-18T03_20_24_296Z"
DATASET_FOLDER_ROOT = PROJECT_ROOT / DATASET_FOLDER_NAME
DATASET_FOLDER_IN_DATASETS = PROJECT_ROOT / "datasets" / DATASET_FOLDER_NAME

if DATASET_FOLDER_IN_DATASETS.exists():
    DATASET_FOLDER = DATASET_FOLDER_IN_DATASETS
elif DATASET_FOLDER_ROOT.exists():
    DATASET_FOLDER = DATASET_FOLDER_ROOT
else:
    DATASET_FOLDER = DATASET_FOLDER_ROOT

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UI_SCRIPT = PROJECT_ROOT / "main_ui.py"

OUTPUT_DIR = PROJECT_ROOT / "Kickstarter_Clean"
DATASETS_DIR = PROJECT_ROOT / "datasets"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENT_RESULTS_DIR = PROJECT_ROOT / "experiment_results"
ANALYSIS_REPORTS_DIR = PROJECT_ROOT / "analysis_reports"

# ==========================================
# 日志标准化函数
# ==========================================

def log_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def log_step(step_num, description):
    print(f"\n【步骤 {step_num}】 {description}")
    print("-" * 60)

def log_success(message):
    print(f"✓ {message}")

def log_error(message):
    print(f"✗ {message}")

def log_info(message):
    print(f"  {message}")

def log_warning(message):
    print(f"⚠️ {message}")


class TeeLogger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message):
        self.log_file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.log_file.flush()
        self.stdout.flush()

    def close(self):
        self.log_file.close()


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = LOGS_DIR / f"run_{timestamp}.log"
    logger = TeeLogger(log_file_path)
    sys.stdout = logger
    sys.stderr = logger
    return logger, log_file_path


def find_latest_file(directory, prefix, suffix):
    if not directory.exists():
        return None
    candidates = [p for p in directory.iterdir() if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ==========================================
# 模块导入辅助函数
# ==========================================

def import_module_from_path(module_name: str, file_path: Path):
    """从指定路径导入模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
    except Exception as e:
        log_error(f"导入模块 {module_name} 失败: {e}")
    return None


# ==========================================
# M8 风险规则判定（新增整合）
# ==========================================

def run_m8_risk_judgment(project_data: Dict[str, Any] = None, verbose: bool = True) -> Tuple[str, List[str], Dict]:
    """
    执行M8风险规则判定
    包含增强的文本风险分析功能
    """
    log_step("M8", "执行M8风险规则判定")
    
    try:
        m8_module = import_module_from_path("m8_rule_adapter", SCRIPTS_DIR / "m8_rule_adapter.py")
        
        if m8_module is None:
            log_error("M8模块加载失败")
            return "风险中等", ["默认判定"], {}
        
        judge_function = getattr(m8_module, "judge_project_risk_m8", None)
        if judge_function is None:
            log_error("M8模块中未找到judge_project_risk_m8函数")
            return "风险中等", ["默认判定"], {}
        
        if project_data is None:
            project_data = {
                "goal_ratio": 2.0,
                "time_penalty": 1.5,
                "category_risk": 0.5,
                "country_factor": 0.3,
                "urgency_score": 0.5,
                "main_category": "Technology",
                "country": "US"
            }
        
        risk_level, reasons, intermediate = judge_function(project_data, verbose=verbose)
        log_success(f"M8风险判定完成: {risk_level}")
        
        return risk_level, reasons, intermediate
        
    except Exception as e:
        log_error(f"M8执行异常: {str(e)}")
        return "风险中等", [f"异常: {str(e)}"], {}


# ==========================================
# M7 智能体路由（已存在）
# ==========================================

def run_m7_processing():
    """执行M7智能体池路由模块"""
    log_step(4, "执行M7智能体池路由 (m7/m7_demo.py)")

    try:
        script_path = SCRIPTS_DIR / "m7" / "m7_demo.py"
        if not script_path.exists():
            log_error(f"找不到脚本: {script_path}")
            return False

        if importlib.util.find_spec("matplotlib") is None:
            log_info("未检测到matplotlib，M7将跳过可视化")

        log_info(f"执行脚本: {script_path}")

        original_cwd = os.getcwd()
        try:
            os.chdir(SCRIPTS_DIR)
            result = subprocess.run(
                [sys.executable, "m7/m7_demo.py"],
                capture_output=False,
                timeout=600
            )
            if result.returncode != 0:
                log_error(f"M7处理失败，返回码: {result.returncode}")
                return False
            log_success("M7智能体池路由完成")
            return True
        finally:
            os.chdir(original_cwd)

    except subprocess.TimeoutExpired:
        log_error("M7处理超时")
        return False
    except Exception as e:
        log_error(f"M7处理异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# M9 终极引擎（新增整合）
# ==========================================

def run_m9_ultimate_engine(
    user_query: str = "",
    project_data: Dict[str, Any] = None,
    enable_ood: bool = False,
    api_key: str = ""
) -> Dict[str, Any]:
    """
    执行M9终极引擎
    整合M7+M8+M10+M12的完整决策流程
    """
    log_step("M9", "执行M9终极决策引擎")
    
    try:
        m9_module = import_module_from_path("m9", SCRIPTS_DIR / "m9.py")
        
        if m9_module is None:
            log_error("M9模块加载失败")
            return {"error": "M9模块加载失败"}
        
        M9EngineClass = getattr(m9_module, "M9UltimateRiskEngine", None)
        if M9EngineClass is None:
            log_error("M9模块中未找到M9UltimateRiskEngine类")
            return {"error": "M9类未找到"}
        
        # 设置API密钥
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        
        # 初始化引擎
        engine = M9EngineClass(api_key=api_key if api_key else None, verbose=False)
        
        # 默认项目数据
        if project_data is None:
            project_data = {
                "goal_usd": 15000,
                "duration_days": 60,
                "main_category": "Technology",
                "country": "US",
                "country_factor": 0.3,
                "goal_ratio": 2.0,
                "time_penalty": 1.5,
                "category_risk": 0.5,
                "combined_risk": 5.0,
                "urgency_score": 0.5,
            }
        
        # 默认用户查询
        if not user_query:
            user_query = "分析我的项目风险、抗压能力，并给出落地建议"
        
        # 执行完整决策
        result = engine.run_full_decision(
            user_query=user_query,
            project_data=project_data,
            user_id="main_script_user",
            enable_ood_test=enable_ood,
            enable_perf_monitor=True
        )
        
        log_success("M9终极决策完成")
        
        return result
        
    except Exception as e:
        log_error(f"M9执行异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ==========================================
# M10 性能监控（新增整合）
# ==========================================

def run_m10_performance_monitor(log_file_path: str = None) -> Dict[str, Any]:
    """
    执行M10性能监控
    监控Token消耗、定位瓶颈、生成Cost Report
    """
    log_step("M10", "执行M10性能监控")
    
    try:
        m10_module = import_module_from_path("M10_PerformanceMonitor", SCRIPTS_DIR / "M10_PerformanceMonitor.py")
        
        if m10_module is None:
            log_warning("M10模块加载失败，跳过性能监控")
            return {"status": "skipped", "reason": "module_load_failed"}
        
        # 默认日志路径
        if log_file_path is None:
            log_files = list(LOGS_DIR.glob("*.log"))
            if log_files:
                log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                log_file_path = str(log_files[0])
            else:
                log_warning("未找到日志文件，跳过性能监控")
                return {"status": "skipped", "reason": "no_log_file"}
        
        # 初始化监控器
        MonitorClass = getattr(m10_module, "M10_PerformanceMonitor", None)
        if MonitorClass is None:
            log_warning("M10模块中未找到M10_PerformanceMonitor类")
            return {"status": "skipped", "reason": "class_not_found"}
        
        monitor = MonitorClass(
            project_id="Kickstarter_Integration_Test",
            output_dir=str(OUTPUT_DIR / "performance_reports"),
            log_dir=str(LOGS_DIR)
        )
        
        # 执行监控
        if os.path.exists(log_file_path):
            report = monitor.monitor_log_stream(log_file_path)
            if report:
                log_success(f"M10性能监控完成，报告ID: {report.report_id}")
                return {
                    "status": "success",
                    "report_id": report.report_id,
                    "total_cost": report.total_cost,
                    "token_cost": report.total_token_cost,
                    "bottleneck_count": len(report.bottlenecks)
                }
        
        return {"status": "completed", "note": "no_data"}
        
    except Exception as e:
        log_warning(f"M10执行异常: {str(e)}")
        return {"status": "error", "error": str(e)}


# ==========================================
# M12 OOD高压测试（新增整合）
# ==========================================

def run_m12_ood_test(project_data: Dict[str, Any] = None, difficulty: str = "medium") -> Dict[str, Any]:
    """
    执行M12环境增强与OOD高压测试
    模拟黑天鹅事件，测试项目韧性
    """
    log_step("M12", "执行M12环境增强与OOD测试")
    
    try:
        m12_module = import_module_from_path("M12环境增强与OOD测试", SCRIPTS_DIR / "M12环境增强与OOD测试.py")
        
        if m12_module is None:
            log_warning("M12模块加载失败，跳过OOD测试")
            return {"status": "skipped", "reason": "module_load_failed"}
        
        # 导入所需类
        OODConfigClass = getattr(m12_module, "OODConfig", None)
        ScenarioGeneratorClass = getattr(m12_module, "OODScenarioGenerator", None)
        ResilienceEvaluatorClass = getattr(m12_module, "ResilienceEvaluator", None)
        
        if not all([OODConfigClass, ScenarioGeneratorClass, ResilienceEvaluatorClass]):
            log_warning("M12模块中未找到所需类")
            return {"status": "skipped", "reason": "class_not_found"}
        
        # 默认项目数据
        if project_data is None:
            project_data = {
                "goal_ratio": 2.0,
                "time_penalty": 1.5,
                "category_risk": 0.5,
                "combined_risk": 10.0,
                "country_factor": 0.5,
                "urgency_score": 1.0,
                "main_category": "Technology",
                "duration_days": 60,
                "goal_usd": 100000,
                "country": "US"
            }
        
        import logging
        logger = logging.getLogger("m12_test")
        
        # 初始化组件
        config = OODConfigClass()
        generator = ScenarioGeneratorClass(config=config, logger=logger)
        evaluator = ResilienceEvaluatorClass(config=config, logger=logger)
        
        # 生成测试场景
        scenario = generator.generate_scenario_from_project(project_data, difficulty=difficulty)
        
        # 生成报告摘要
        result = {
            "status": "success",
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "difficulty": scenario.difficulty_level,
            "distribution_shifts_count": len(scenario.distribution_shifts),
            "black_swan_events": [
                {
                    "name": e.name,
                    "type": e.event_type.value,
                    "severity": e.severity
                }
                for e in scenario.black_swan_events
            ],
            "expected_outcome": scenario.expected_outcome
        }
        
        log_success(f"M12 OOD测试完成，场景: {scenario.name}")
        
        return result
        
    except Exception as e:
        log_warning(f"M12执行异常: {str(e)}")
        return {"status": "error", "error": str(e)}


# ==========================================
# M16 大规模实验与分析
# ==========================================

def run_m16_processing():
    """执行M16大规模实验与分析模块"""
    log_step(5, "执行M16大规模实验与分析 (M16.py)")

    try:
        script_path = SCRIPTS_DIR / "M16.py"
        if not script_path.exists():
            log_error(f"找不到脚本: {script_path}")
            return False

        log_info(f"加载M16模块: {script_path}")

        original_cwd = os.getcwd()

        try:
            os.chdir(SCRIPTS_DIR)
            import importlib.util
            spec = importlib.util.spec_from_file_location("M16", script_path)
            M16 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(M16)

            log_info("初始化M16实验配置...")

            results_df, analysis, report = M16.run_m16_experiments(
                n_runs_per_config=20,
                n_workers=os.cpu_count() or 4,
                batch_size=50,
                resume=True,
                output_dir=str(EXPERIMENT_RESULTS_DIR)
            )

            log_success(f"M16实验完成，共执行 {len(results_df)} 次实验")
            
            if report:
                report_path = ANALYSIS_REPORTS_DIR / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                log_success(f"分析报告已保存: {report_path}")

            results_json_path = EXPERIMENT_RESULTS_DIR / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results_dict = {
                'timestamp': datetime.now().isoformat(),
                'total_experiments': len(results_df),
                'analysis_summary': analysis if isinstance(analysis, dict) else str(analysis)
            }
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            log_success(f"实验结果JSON已保存: {results_json_path}")

            return True

        finally:
            os.chdir(original_cwd)

    except subprocess.TimeoutExpired:
        log_error("M16处理超时")
        return False
    except Exception as e:
        log_error(f"M16处理异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# 数据清洗（M2）
# ==========================================

def run_data_processing():
    """执行数据清洗模块"""
    log_step(1, "执行数据清洗模块 (data_process.py)")
    
    try:
        script_path = SCRIPTS_DIR / "data_process.py"
        if not script_path.exists():
            log_error(f"找不到脚本: {script_path}")
            return False
        
        log_info(f"执行脚本: {script_path}")
        original_cwd = os.getcwd()
        
        try:
            os.chdir(SCRIPTS_DIR)
            result = subprocess.run(
                [sys.executable, "data_process.py"],
                capture_output=False,
                timeout=600
            )
            
            if result.returncode != 0:
                log_error(f"数据清洗失败，返回码: {result.returncode}")
                return False
            
            log_success("数据清洗完成")
            cleaned_data = OUTPUT_DIR / "kickstarter_cleaned.csv"
            if cleaned_data.exists():
                log_success(f"清洗数据已保存: {cleaned_data}")
                return True
            else:
                log_error(f"未找到清洗后的数据文件: {cleaned_data}")
                return False
                
        finally:
            os.chdir(original_cwd)
        
    except subprocess.TimeoutExpired:
        log_error("数据清洗超时")
        return False
    except Exception as e:
        log_error(f"数据清洗异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# M4特征工程与模型训练
# ==========================================

def run_m4_processing():
    """执行M4特征工程与模型训练模块"""
    log_step(2, "执行M4特征工程与模型训练 (M4代码.py)")
    
    try:
        script_path = SCRIPTS_DIR / "M4代码.py"
        if not script_path.exists():
            log_error(f"找不到脚本: {script_path}")
            return False
        
        log_info(f"执行脚本: {script_path}")
        original_cwd = os.getcwd()
        
        try:
            os.chdir(SCRIPTS_DIR)
            result = subprocess.run(
                [sys.executable, "M4代码.py"],
                capture_output=False,
                timeout=1800
            )
            
            if result.returncode != 0:
                log_error(f"M4处理失败，返回码: {result.returncode}")
                return False
            
            log_success("M4特征工程与模型训练完成")
            return True
            
        finally:
            os.chdir(original_cwd)
        
    except subprocess.TimeoutExpired:
        log_error("M4处理超时")
        return False
    except Exception as e:
        log_error(f"M4处理异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# M3仿真环境
# ==========================================

def run_m3_simulation():
    """执行M3仿真模块"""
    log_step(3, "执行M3仿真环境 (M3仿真环境基座模块版.py)")
    
    try:
        script_path = SCRIPTS_DIR / "M3仿真环境基座模块版.py"
        if not script_path.exists():
            log_error(f"找不到脚本: {script_path}")
            return False
        
        log_info(f"执行脚本: {script_path}")
        original_cwd = os.getcwd()
        
        try:
            os.chdir(SCRIPTS_DIR)
            result = subprocess.run(
                [sys.executable, "M3仿真环境基座模块版.py"],
                capture_output=False,
                timeout=1800
            )
            
            if result.returncode != 0:
                log_error(f"M3仿真失败，返回码: {result.returncode}")
                return False
            
            log_success("M3仿真完成")
            return True
            
        finally:
            os.chdir(original_cwd)
        
    except subprocess.TimeoutExpired:
        log_error("M3仿真超时")
        return False
    except Exception as e:
        log_error(f"M3仿真异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# 主工作流程
# ==========================================

def run_full_pipeline(
    skip_m2: bool = False,
    skip_m3: bool = False,
    skip_m4: bool = False,
    skip_m7: bool = False,
    skip_m16: bool = False,
    run_m8: bool = True,
    run_m9: bool = True,
    run_m10: bool = True,
    run_m12: bool = True,
    api_key: str = ""
) -> Dict[str, Any]:
    """
    执行完整流水线
    支持选择性运行各模块
    """
    logger, log_file_path = setup_logging()
    
    log_section("项目评估与AI路由 - 完整流水线")
    log_info(f"执行开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"项目根目录: {PROJECT_ROOT}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "modules": {},
        "final_output": {}
    }
    
    # 检查数据集
    log_step(0, "环境检查")
    if not DATASET_FOLDER.exists():
        log_warning(f"数据集文件夹不存在: {DATASET_FOLDER}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 核心项目数据（用于M8/M9/M12）
    project_data = {
        "goal_usd": 15000,
        "duration_days": 60,
        "main_category": "Technology",
        "country": "US",
        "country_factor": 0.3,
        "goal_ratio": 2.0,
        "time_penalty": 1.5,
        "category_risk": 0.5,
        "combined_risk": 5.0,
        "urgency_score": 0.5,
        "actual_funding_usd": 5000,
        "planned_duration_days": 45,
    }
    
    user_query = "分析我的项目风险、抗压能力，并给出落地建议"
    
    # 依次执行模块
    success_count = 0
    total_modules = 0
    
    # M2 数据清洗
    if not skip_m2:
        total_modules += 1
        success = run_data_processing()
        results["modules"]["M2_数据清洗"] = "成功" if success else "失败"
        if success: success_count += 1
    
    # M4 特征工程
    if not skip_m4:
        total_modules += 1
        success = run_m4_processing()
        results["modules"]["M4_特征工程"] = "成功" if success else "失败"
        if success: success_count += 1
    
    # M3 仿真
    if not skip_m3:
        total_modules += 1
        success = run_m3_simulation()
        results["modules"]["M3_仿真"] = "成功" if success else "失败"
        if success: success_count += 1
    
    # M8 风险判定（新增）
    if run_m8:
        total_modules += 1
        try:
            risk_level, reasons, intermediate = run_m8_risk_judgment(project_data)
            results["modules"]["M8_风险判定"] = "成功"
            results["final_output"]["risk_level"] = risk_level
            results["final_output"]["risk_reasons"] = reasons
            results["final_output"]["intermediate"] = intermediate
            success_count += 1
        except Exception as e:
            results["modules"]["M8_风险判定"] = f"失败: {str(e)}"
    
    # M9 终极引擎（新增）
    if run_m9:
        total_modules += 1
        try:
            m9_result = run_m9_ultimate_engine(user_query, project_data, enable_ood=False, api_key=api_key)
            results["modules"]["M9_终极引擎"] = "成功" if "error" not in m9_result else f"失败: {m9_result.get('error')}"
            results["final_output"]["m9_decision"] = {
                "intent": m9_result.get("intent", {}),
                "risk": m9_result.get("risk", {}),
                "routing": m9_result.get("routing", {}),
                "experts_count": len(m9_result.get("experts", []))
            }
            success_count += 1
        except Exception as e:
            results["modules"]["M9_终极引擎"] = f"失败: {str(e)}"
    
    # M7 智能体路由
    if not skip_m7:
        total_modules += 1
        success = run_m7_processing()
        results["modules"]["M7_专家路由"] = "成功" if success else "失败"
        if success: success_count += 1
    
    # M10 性能监控（新增）
    if run_m10:
        total_modules += 1
        try:
            m10_result = run_m10_performance_monitor()
            results["modules"]["M10_性能监控"] = "成功" if m10_result.get("status") == "success" else f"完成({m10_result.get('status')})"
            success_count += 1
        except Exception as e:
            results["modules"]["M10_性能监控"] = f"完成: {str(e)}"
    
    # M12 OOD测试（新增）
    if run_m12:
        total_modules += 1
        try:
            m12_result = run_m12_ood_test(project_data)
            results["modules"]["M12_OOD测试"] = "成功" if m12_result.get("status") == "success" else f"完成({m12_result.get('status')})"
            results["final_output"]["ood_scenario"] = m12_result
            success_count += 1
        except Exception as e:
            results["modules"]["M12_OOD测试"] = f"完成: {str(e)}"
    
    # M16 大规模实验
    if not skip_m16:
        total_modules += 1
        success = run_m16_processing()
        results["modules"]["M16_大规模实验"] = "成功" if success else "失败"
        if success: success_count += 1
    
    # 保存最终结果JSON
    output_json_path = OUTPUT_DIR / f"full_pipeline_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_info(f"完整结果JSON已保存: {output_json_path}")
    
    # 打印执行总结
    log_section("执行总结")
    log_info(f"总共 {total_modules} 个模块，成功 {success_count} 个")
    print()
    
    for module_name, status in results["modules"].items():
        print(f"  {module_name:<20} {status}")
    
    print()
    log_info(f"执行结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"完整结果: {output_json_path}")
    
    logger.close()
    
    return results


def start_web_ui():
    """启动Web界面（统一前端入口，串联全部模块）"""
    log_info("启动Web界面...")
    log_info(f"UI脚本路径: {UI_SCRIPT}")
    
    if not UI_SCRIPT.exists():
        log_error(f"UI脚本不存在: {UI_SCRIPT}")
        return False
    
    try:
        log_info("Web界面整合模块：M2→M4→M8→M7→M9→M10→M12→M16 + MAS黑板架构")
        log_info("支持文字输入和数据文件分析，输出完整JSON")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(UI_SCRIPT), "--server.port", "8501"],
            cwd=str(PROJECT_ROOT)
        )
        return True
    except KeyboardInterrupt:
        log_info("Web界面已关闭")
        return True
    except Exception as e:
        log_error(f"启动Web界面失败: {str(e)}")
        return False


# ==========================================
# 主入口
# ==========================================

def main():
    """主工作流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description="项目评估与AI路由系统")
    parser.add_argument("--mode", choices=["pipeline", "web", "m8", "m9", "m10", "m12", "all"], 
                       default="web", help="运行模式（默认启动Web界面）")
    parser.add_argument("--skip-m2", action="store_true", help="跳过M2数据清洗")
    parser.add_argument("--skip-m3", action="store_true", help="跳过M3仿真")
    parser.add_argument("--skip-m4", action="store_true", help="跳过M4特征工程")
    parser.add_argument("--skip-m7", action="store_true", help="跳过M7专家路由")
    parser.add_argument("--skip-m16", action="store_true", help="跳过M16大规模实验")
    parser.add_argument("--no-m8", action="store_true", help="禁用M8风险判定")
    parser.add_argument("--no-m9", action="store_true", help="禁用M9终极引擎")
    parser.add_argument("--no-m10", action="store_true", help="禁用M10性能监控")
    parser.add_argument("--no-m12", action="store_true", help="禁用M12 OOD测试")
    parser.add_argument("--api-key", type=str, default="", help="DeepSeek API密钥")
    parser.add_argument("--port", type=int, default=8501, help="Web服务端口")
    
    args = parser.parse_args()
    
    if args.mode == "web":
        # 启动Web界面（默认模式）
        start_web_ui()
    elif args.mode == "pipeline":
        # 仅运行命令行流水线
        run_full_pipeline(
            skip_m2=args.skip_m2,
            skip_m3=args.skip_m3,
            skip_m4=args.skip_m4,
            skip_m7=args.skip_m7,
            skip_m16=args.skip_m16,
            run_m8=not args.no_m8,
            run_m9=not args.no_m9,
            run_m10=not args.no_m10,
            run_m12=not args.no_m12,
            api_key=args.api_key
        )
    elif args.mode == "all":
        # 先运行流水线，再启动Web界面
        run_full_pipeline(
            skip_m2=args.skip_m2,
            skip_m3=args.skip_m3,
            skip_m4=args.skip_m4,
            skip_m7=args.skip_m7,
            skip_m16=args.skip_m16,
            run_m8=not args.no_m8,
            run_m9=not args.no_m9,
            run_m10=not args.no_m10,
            run_m12=not args.no_m12,
            api_key=args.api_key
        )
        print("\n流水线执行完成，启动Web界面...")
        start_web_ui()
    elif args.mode == "m8":
        # 仅运行M8
        risk_level, reasons, intermediate = run_m8_risk_judgment()
        print(f"\n结果: {risk_level}")
        print(f"原因: {reasons}")
    elif args.mode == "m9":
        # 仅运行M9
        result = run_m9_ultimate_engine(api_key=args.api_key)
        print(f"\nM9结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    elif args.mode == "m10":
        # 仅运行M10
        result = run_m10_performance_monitor()
        print(f"\nM10结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    elif args.mode == "m12":
        # 仅运行M12
        result = run_m12_ood_test()
        print(f"\nM12结果: {json.dumps(result, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n执行被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n致命错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
