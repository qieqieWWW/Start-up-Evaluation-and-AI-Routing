"""
M5 自动化测试体系 - 核心处理代码
功能：规则驱动 → 数据注入 → 仿真执行 → 智能体交互 → 结果收集 → 度量反馈
"""

import time
import threading
from typing import List, Dict, Any

# 依赖模块（M1~M9 接口）
from M1 import RuleEngine       # 规则库、用例生成
from M2 import TestDataManager   # 测试数据准备
from M3 import EnvManager    # 仿真环境
from M4 import ProjectLoader   # 项目规范
from M6 import StaticValidator # 静态校验
from M8 import TestLogger            # 日志与快照
from M9 import MetricReporter       # 指标与报告

class M5_AutomationTestEngine:
    def __init__(self, project_id: str):
        """
        M5 测试引擎初始化
        :param project_id: 项目唯一标识
        """
        self.project_id = project_id
        self.rule_engine = RuleEngine()
        self.data_manager = TestDataManager()
        self.env_manager = EnvManager()
        self.logger = TestLogger()
        self.reporter = MetricReporter()

        # 全局上下文
        self.test_cases: List[Dict] = []
        self.test_datasets: List[Dict] = []
        self.sim_env = None
        self.agent = None
        self.execution_results: List[Dict] = []

    # ==============================
    # 阶段1：规则与用例生成（M1 → M5）
    # ==============================
    def stage1_gen_test_cases(self) -> List[Dict]:
        # 输入：M1 规则库、失效场景、业务接口
        rule_lib = self.rule_engine.get_rule_library(self.project_id)
        fail_scenarios = self.rule_engine.get_fail_scenarios(self.project_id)
        
        # 处理：解析 → 生成用例 → 优先级排序
        self.test_cases = self.rule_engine.parse_and_generate_cases(
            rule_lib=rule_lib,
            fail_scenarios=fail_scenarios
        )
        self.test_cases = self.rule_engine.sort_by_risk(self.test_cases)
        
        # 输出：结构化测试用例集
        self.logger.log_stage("STAGE1", "测试用例生成完成", len(self.test_cases))
        return self.test_cases

    # ==============================
    # 阶段2：测试数据准备（M2 → M5）
    # ==============================
    def stage2_prepare_data(self) -> List[Dict]:
        # 输入：M2 结构化数据、特征工程 pipeline
        self.test_datasets = self.data_manager.prepare_dataset(
            test_cases=self.test_cases,
            project_id=self.project_id,
            desensitize=True,
            batch_size=32
        )
        
        # 输出：可注入仿真环境的数据集
        self.logger.log_stage("STAGE2", "测试数据准备完成", len(self.test_datasets))
        return self.test_datasets

    # ==============================
    # 阶段3：仿真环境初始化（M3 → M5）
    # ==============================
    def stage3_init_simulation(self):
        # 输入：M3 环境API、状态空间、重置脚本
        self.sim_env = self.env_manager.create_env(
            project_id=self.project_id,
            reset_on_start=True
        )
        # 加载待测试智能体
        self.agent = self.sim_env.load_agent()
        # 挂载日志钩子
        self.sim_env.attach_logger(self.logger)
        
        # 输出：就绪仿真环境
        self.logger.log_stage("STAGE3", "仿真环境初始化完成")
        return self.sim_env, self.agent

    # ==============================
    # 阶段4：自动化执行与交互（核心）
    # ==============================
    def stage4_auto_execution(self, parallel: bool = True):
        if parallel:
            threads = []
            for case, data in zip(self.test_cases, self.test_datasets):
                t = threading.Thread(
                    target=self._run_single_case,
                    args=(case, data)
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
        else:
            for case, data in zip(self.test_cases, self.test_datasets):
                self._run_single_case(case, data)

        self.logger.log_stage("STAGE4", "全部用例执行完成")

    def _run_single_case(self, test_case: Dict, test_data: Dict):
        """单个用例执行原子流程"""
        case_id = test_case["case_id"]
        self.logger.start_case(case_id)

        try:
            # 1）静态校验 M6
            static_ok = StaticValidator.validate_agent_logic(
                agent=self.agent,
                rule=test_case["rule"]
            )
            if not static_ok:
                self.execution_results.append({
                    "case_id": case_id,
                    "result": "FAILED",
                    "reason": "静态规则校验不通过"
                })
                return

            # 2）动态仿真执行
            self.sim_env.reset()
            self.sim_env.inject_data(test_data)
            trajectory = self.sim_env.run_episode(
                agent=self.agent,
                max_steps=test_case["max_steps"]
            )

            # 3）结果判定
            passed = self.rule_engine.judge_result(
                trajectory=trajectory,
                expected=test_case["expected"]
            )

            self.execution_results.append({
                "case_id": case_id,
                "result": "PASS" if passed else "FAIL",
                "trajectory": trajectory,
                "risk_level": test_case["risk_level"]
            })

        except Exception as e:
            self.execution_results.append({
                "case_id": case_id,
                "result": "ERROR",
                "error": str(e)
            })
        finally:
            self.logger.finish_case(case_id)

    # ==============================
    # 阶段5：结果收集 & 度量反馈
    # ==============================
    def stage5_report_and_feedback(self) -> Dict:
        # 聚合结果
        report = self.reporter.generate_report(
            results=self.execution_results,
            test_cases=self.test_cases
        )
        # 计算指标
        metrics = self.reporter.calc_metrics(report)
        # 高风险告警
        self.reporter.alert_high_risk_failures(report)
        # 生成缺陷工单
        defects = self.reporter.create_defect_tickets(report)
        
        # 输出：报告 + 度量 + 缺陷 + 反馈数据
        output = {
            "test_report": report,
            "metrics": metrics,
            "defects": defects,
            "feedback_data": self.reporter.extract_feedback(report)
        }
        self.logger.log_stage("STAGE5", "测试报告与反馈生成完成")
        return output

    # ==============================
    # M5 总入口：完整闭环链路
    # ==============================
    def run_full_pipeline(self) -> Dict:
        """M5 自动化测试完整流程：输入 → 处理 → 输出"""
        self.logger.start_m5(self.project_id)
        
        # 1. 用例生成
        self.stage1_gen_test_cases()
        # 2. 数据准备
        self.stage2_prepare_data()
        # 3. 环境初始化
        self.stage3_init_simulation()
        # 4. 自动执行
        self.stage4_auto_execution(parallel=True)
        # 5. 报告与反馈
        result = self.stage5_report_and_feedback()
        
        self.logger.finish_m5(self.project_id)
        return result

# ==============================
# 调用示例
# ==============================
if __name__ == "__main__":
    m5_engine = M5_AutomationTestEngine(project_id="AGENT_2025_V1")
    final_report = m5_engine.run_full_pipeline()
    print("M5 测试闭环完成，报告：", final_report)
