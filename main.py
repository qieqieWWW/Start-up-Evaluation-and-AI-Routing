#!/usr/bin/env python
# coding: utf-8

"""
启动项目评估与AI路由 - 完整测试集成脚本
===========================================
整合所有模块：数据清洗(M2) → 特征工程/ML(M4) → 仿真(M3)

执行流程：
1. 数据清洗：原始数据集 → 标准化数据
2. M4特征/训练：特征工程 → 模型训练与预测
3. M3仿真：基于M4输出结果进行仿真

作者：集成脚本
日期：2026-02-06
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ==========================================
# 配置与路径管理
# ==========================================

# 获取项目根目录（main.py所在的目录）
PROJECT_ROOT = Path(__file__).parent.absolute()
print(f"项目根目录: {PROJECT_ROOT}")

# 原始数据集位置（可能在根目录或datasets目录中）
DATASET_FOLDER_NAME = "Kickstarter_2025-12-18T03_20_24_296Z"
DATASET_FOLDER_ROOT = PROJECT_ROOT / DATASET_FOLDER_NAME
DATASET_FOLDER_IN_DATASETS = PROJECT_ROOT / "datasets" / DATASET_FOLDER_NAME

# 优先检查datasets目录，其次检查根目录
if DATASET_FOLDER_IN_DATASETS.exists():
    DATASET_FOLDER = DATASET_FOLDER_IN_DATASETS
    print(f"数据集位置: {DATASET_FOLDER} (在datasets目录中)")
elif DATASET_FOLDER_ROOT.exists():
    DATASET_FOLDER = DATASET_FOLDER_ROOT
    print(f"数据集位置: {DATASET_FOLDER} (在项目根目录中)")
else:
    DATASET_FOLDER = DATASET_FOLDER_ROOT  # 默认位置，用于错误提示
    print(f"数据集位置: {DATASET_FOLDER}")

# 脚本位置
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "Kickstarter_Clean"
DATASETS_DIR = PROJECT_ROOT / "datasets"
LOGS_DIR = PROJECT_ROOT / "logs"

# ==========================================
# 日志标准化函数
# ==========================================

def log_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def log_step(step_num, description):
    """打印执行步骤"""
    print(f"\n【步骤 {step_num}】 {description}")
    print("-" * 60)

def log_success(message):
    """打印成功信息"""
    print(f"✓ {message}")

def log_error(message):
    """打印错误信息"""
    print(f"✗ {message}")

def log_info(message):
    """打印信息"""
    print(f"  {message}")


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
    """将输出同时写入控制台和日志文件"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = LOGS_DIR / f"run_{timestamp}.log"
    logger = TeeLogger(log_file_path)
    sys.stdout = logger
    sys.stderr = logger
    return logger, log_file_path


def find_latest_file(directory, prefix, suffix):
    """查找目录中符合前缀/后缀的最新文件"""
    if not directory.exists():
        return None
    candidates = [p for p in directory.iterdir() if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

# ==========================================
# 模块1：数据清洗（M2）
# ==========================================

def run_data_processing():
    """
    执行数据清洗模块
    原始数据 → 清洗后的CSV文件
    """
    log_step(1, "执行数据清洗模块 (data_process.py)")
    
    try:
        # 检查脚本存在性
        script_path = SCRIPTS_DIR / "data_process.py"
        if not script_path.exists():
            log_error(f"找不到脚本: {script_path}")
            return False
        
        log_info(f"执行脚本: {script_path}")
        
        # 保存原始路径
        original_cwd = os.getcwd()
        
        try:
            # 切换到脚本所在目录，以便相对路径正确解析
            os.chdir(SCRIPTS_DIR)
            
            # 使用subprocess执行脚本，避免import导致的命名空间污染
            result = subprocess.run(
                [sys.executable, "data_process.py"],
                capture_output=False,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode != 0:
                log_error(f"数据清洗失败，返回码: {result.returncode}")
                return False
            
            log_success("数据清洗完成")
            
            # 检查输出文件
            cleaned_data = OUTPUT_DIR / "kickstarter_cleaned.csv"
            if cleaned_data.exists():
                log_success(f"清洗数据已保存: {cleaned_data}")
                return True
            else:
                log_error(f"未找到清洗后的数据文件: {cleaned_data}")
                return False
                
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
        
    except subprocess.TimeoutExpired:
        log_error("数据清洗超时（>10分钟）")
        return False
    except Exception as e:
        log_error(f"数据清洗异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ==========================================
# 模块2：M4特征工程与模型训练
# ==========================================

def run_m4_processing():
    """
    执行M4特征工程与模型训练模块
    清洗数据 → 特征工程 → 模型训练与预测
    """
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
            
            # 执行M4代码
            result = subprocess.run(
                [sys.executable, "M4代码.py"],
                capture_output=False,
                timeout=1800  # 30分钟超时
            )
            
            if result.returncode != 0:
                log_error(f"M4处理失败，返回码: {result.returncode}")
                return False
            
            log_success("M4特征工程与模型训练完成")
            return True
            
        finally:
            os.chdir(original_cwd)
        
    except subprocess.TimeoutExpired:
        log_error("M4处理超时（>30分钟）")
        return False
    except Exception as e:
        log_error(f"M4处理异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ==========================================
# 模块3：M3仿真环境
# ==========================================

def run_m3_simulation():
    """
    执行M3仿真模块
    基于M4的输出结果进行仿真
    """
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
            
            # 执行M3仿真
            result = subprocess.run(
                [sys.executable, "M3仿真环境基座模块版.py"],
                capture_output=False,
                timeout=1800  # 30分钟超时
            )
            
            if result.returncode != 0:
                log_error(f"M3仿真失败，返回码: {result.returncode}")
                return False
            
            log_success("M3仿真完成")
            return True
            
        finally:
            os.chdir(original_cwd)
        
    except subprocess.TimeoutExpired:
        log_error("M3仿真超时（>30分钟）")
        return False
    except Exception as e:
        log_error(f"M3仿真异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ==========================================
# 主工作流程
# ==========================================

def main():
    """主工作流程 - 依次执行各个模块"""
    logger, log_file_path = setup_logging()
    
    log_section("启动项目评估与AI路由 - 完整测试集成")
    
    log_info(f"执行开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"项目根目录: {PROJECT_ROOT}")
    
    # 验证必要的目录与文件存在性
    log_step(0, "环境检查")
    
    if not DATASET_FOLDER.exists():
        log_error(f"原始数据集文件夹不存在: {DATASET_FOLDER}")
        log_info(f"请确保数据集位置正确（可在 {PROJECT_ROOT} 或 {PROJECT_ROOT / 'datasets'} 目录下）")
        return False
    
    log_success(f"数据集文件夹存在: {DATASET_FOLDER}")
    
    if not SCRIPTS_DIR.exists():
        log_error(f"脚本目录不存在: {SCRIPTS_DIR}")
        return False
    
    log_success(f"脚本目录存在: {SCRIPTS_DIR}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 依次执行各个模块
    modules = [
        ("数据清洗（M2）", run_data_processing),
        ("M4特征工程与训练", run_m4_processing),
        ("M3仿真", run_m3_simulation),
    ]
    
    results = {}
    success_count = 0
    
    for module_name, module_func in modules:
        try:
            success = module_func()
            results[module_name] = "✓ 成功" if success else "✗ 失败"
            if success:
                success_count += 1
        except Exception as e:
            log_error(f"执行 {module_name} 时发生意外错误: {str(e)}")
            results[module_name] = "✗ 异常"
    
    # 打印执行总结
    log_section("执行总结")
    
    log_info(f"总共 {len(modules)} 个模块，成功 {success_count} 个")
    print()
    
    for module_name, status in results.items():
        print(f"  {module_name:<20} {status}")
    
    print()
    log_info(f"执行结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 生成执行报告
    report_path = OUTPUT_DIR / "latest_run_report.txt"
    cleaned_data = OUTPUT_DIR / "kickstarter_cleaned.csv"
    latest_summary = find_latest_file(OUTPUT_DIR, "full_prediction_summary_", ".csv")
    latest_m3 = find_latest_file(OUTPUT_DIR, "m3_simulation_results_", ".csv")

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("运行报告\n")
        report.write("=" * 40 + "\n")
        report.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"项目根目录: {PROJECT_ROOT}\n")
        report.write(f"日志文件: {log_file_path}\n")
        report.write("\n模块状态:\n")
        for module_name, status in results.items():
            report.write(f"- {module_name}: {status}\n")
        report.write("\n输出文件:\n")
        report.write(f"- 清洗数据: {cleaned_data if cleaned_data.exists() else '未生成'}\n")
        report.write(f"- M4汇总: {latest_summary if latest_summary else '未生成'}\n")
        report.write(f"- M3结果: {latest_m3 if latest_m3 else '未生成'}\n")

    log_info(f"运行日志已保存: {log_file_path}")
    log_info(f"运行报告已保存: {report_path}")
    
    all_success = success_count == len(modules)
    
    if all_success:
        log_success("所有模块执行成功！")
    else:
        log_error("部分模块执行失败或异常，请查看上方输出")
    
    logger.close()
    return all_success

# ==========================================
# 入口点
# ==========================================

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        log_section("执行被用户中断")
        sys.exit(1)
    except Exception as e:
        log_section("致命错误")
        log_error(f"未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
