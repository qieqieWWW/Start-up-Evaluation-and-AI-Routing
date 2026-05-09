#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#API并未填写
#开始实质性测试时请开启“enable_perf_monitor=False”以启动M10
import json
import uuid
import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional

# ======================== 导入全部模块（你提供的所有文件） ========================
# 注：若以下模块未实际存在，需先创建空占位文件避免导入错误

# 首先添加项目路径到sys.path
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
M7_DIR = SCRIPT_DIR / "m7"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(M7_DIR))  # m7模块目录
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # M7 核心 - 注意路径：m7模块在scripts/m7/目录下
    from m7.m7_expert_pool import get_expert_map
    from m7.m7_global_kb import retrieve_global_kb
    from m7.m7_intent_engine import recognize_intent
    from m8_rule_adapter import judge_project_risk_m8
    from m7.m7_router import route_experts
    from m7.m7_context_analyzer import build_layer1_context, build_layer2_context, build_layer3_context, build_layer4_context
    from m7.m7_prompt_builder import build_system_prompt, build_user_prompt
    from m7.m7_llm_client import DeepSeekClient
    from m7.m7_inference_runner import run_expert_llm_inference_with_blender
    from m7.m7_blender import blend_candidates

    # M6 状态管理与日志
    from M6状态管理与日志系统 import EnvStateManager, EnvState

    # M10 性能监控
    from M10_PerformanceMonitor import (
        LogStreamProcessor,
        TokenAnalyzer,
        BottleneckLocator,
        CostReport
    )

    # M12 OOD 高压测试与环境增强
    from M12环境增强与OOD测试 import (
        OODScenarioGenerator,
        EnhancedStartupEnv,
        ResilienceEvaluator,
        OODConfig
    )
    # 补充导入M3基础环境（M12依赖）
    try:
        from M3仿真环境基座模块版 import StartupEnv
    except ImportError:
        StartupEnv = None
        print("⚠️ M3仿真环境未找到，OOD测试功能受限")

except ImportError as e:
    raise ImportError(f"缺少依赖模块：{e}")

# ======================== M9 终极引擎主类 ========================
class M9UltimateRiskEngine:
    LOG_BASE_DIR = "./m9_batch_logs"

    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        os.makedirs(self.LOG_BASE_DIR, exist_ok=True)
        self.verbose = verbose
        self.session_id = str(uuid.uuid4())[:8]
        self.user_id = "default_user"
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key

        self.expert_map = get_expert_map()
        self.llm_client = DeepSeekClient(api_key=api_key) if api_key else None
        self.state_manager = EnvStateManager(project_id=self.session_id)

        self.log_processor = LogStreamProcessor(log_dir="logs")
        self.token_analyzer = TokenAnalyzer()
        self.bottleneck_locator = BottleneckLocator()

        self.ood_config = OODConfig()
        self.ood_logger = logging.getLogger(f"M9_OOD_{self.session_id}")
        self.ood_logger.setLevel(logging.INFO)
        if not self.ood_logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.ood_logger.addHandler(console_handler)

        self.ood_generator = OODScenarioGenerator(config=self.ood_config, logger=self.ood_logger)
        self.resilience_evaluator = ResilienceEvaluator(config=self.ood_config, logger=self.ood_logger)

        self._log("✅ M9 终极引擎初始化完成")

    def _generate_ood_project_data(self) -> Dict:
        if not hasattr(self, 'project_info'):
            self.project_info = {}

        # ====================== 修复 1：全部强转数字类型，解决批量数据类型报错 ======================
        return {
            "goal_ratio": float(self.project_info.get("goal_ratio", 2.0)),
            "time_penalty": float(self.project_info.get("time_penalty", 1.5)),
            "category_risk": float(self.project_info.get("category_risk", 0.5)),
            "combined_risk": float(self.project_info.get("combined_risk", 10.0)),
            "country_factor": float(self.project_info.get("country_factor", 0.5)),
            "urgency_score": float(self.project_info.get("urgency_score", 1.0)),
            "main_category": self.project_info.get("main_category", "Technology"),
            "duration_days": int(self.project_info.get("duration_days", 60)),
            "goal_usd": float(self.project_info.get("goal_usd", 100000)),
            "country": self.project_info.get("country", "US")
        }

    def _create_project_logger(self) -> logging.Logger:
        log_file = os.path.join(self.LOG_BASE_DIR, self.project_id, f"{self.project_id}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger = logging.getLogger(f"M9_Project_{self.project_id}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

    def _log(self, msg: str):
        if self.verbose:
            print(f"[M9-{self.session_id}] {msg}")

    # ======================== 全流程决策主入口 ========================
    def run_full_decision(
        self,
        user_query: str,
        project_data: Dict[str, Any],
        user_id: str = "default_user",
        enable_ood_test: bool = False,
        enable_perf_monitor: bool = True,
        # 新增：外部已计算的M8/M7结果，避免重复执行
        external_m8_result: Dict = None,
        external_m7_result: Dict = None,
    ) -> Dict[str, Any]:
        try:
            self.user_id = user_id
            self.project_id = project_data.get("project_id", self.session_id)
            self.project_info = project_data
            self.ood_config.project_id = self.project_id
            self.logger = self._create_project_logger()
            self._log("启动 M9 全流程决策引擎")

            init_state = EnvState(project_id=user_id, observation=project_data)
            self.state_manager.on_reset(init_state.to_dict())

            self._log("1/9 意图识别中...")
            intent = recognize_intent(user_query)

            # 检查是否有外部传入的M8结果
            if external_m8_result is not None:
                self._log("2/9 使用外部M8风险结果...")
                risk_level = external_m8_result.get("risk_level", "风险中等")
                risk_reasons = external_m8_result.get("risk_reasons", [])
                intermediate = external_m8_result.get("feature_values", {})
                # 从risk_level提取normalized级别
                if "很高" in risk_level or "high" in risk_level.lower():
                    normalized_risk = "high"
                elif "低" in risk_level or "low" in risk_level.lower():
                    normalized_risk = "low"
                elif "中" in risk_level or "medium" in risk_level.lower():
                    normalized_risk = "medium"
                else:
                    normalized_risk = "medium"
            else:
                self._log("2/9 M8风险规则计算...")
                # 传入user_query供M8分析文本风险
                project_data_with_text = {**project_data, 'user_input': user_query}
                risk_level, risk_reasons, intermediate = judge_project_risk_m8(project_data_with_text)
                # 根据risk_level确定normalized_risk
                if "很高" in risk_level or "high" in risk_level.lower():
                    normalized_risk = "high"
                elif "低" in risk_level or "low" in risk_level.lower():
                    normalized_risk = "low"
                else:
                    normalized_risk = "medium"

            # M12 OOD
            ood_result = None
            if enable_ood_test:
                self._log("3/9 开启 M12 极端场景高压测试...")
                try:
                    scenario = self.ood_generator.generate_scenario_from_project(
                        project_data=self._generate_ood_project_data(),
                        difficulty=3
                    )
                    if StartupEnv:
                        base_env = StartupEnv()
                        env = EnhancedStartupEnv(base_env=base_env, config=self.ood_config, logger=self.ood_logger)
                        obs, info = env.reset_with_scenario(scenario)
                        trajectory = []
                        done = False
                        while not done and env.current_step < self.ood_config.max_ood_steps:
                            action = self._generate_default_ood_action(obs)
                            next_obs, reward, terminated, truncated, step_info = env.step(action)
                            trajectory.append({
                                "step": env.current_step,
                                "observation": obs.tolist(),
                                "action": action.tolist(),
                                "reward": float(reward),
                                "terminated": terminated,
                                "truncated": truncated,
                                "info": step_info
                            })
                            obs = next_obs
                            done = terminated or truncated
                        ood_result = self.resilience_evaluator.evaluate_scenario(scenario, trajectory).to_dict()
                        self._log(f"✓ M12 OOD测试完成 | 场景ID: {scenario.scenario_id} | 综合韧性: {ood_result['overall_resilience']:.3f}")
                    else:
                        self._log("⚠️ M3仿真环境未加载，跳过OOD测试")
                except Exception as ood_e:
                    self._log(f"⚠️ M12 OOD测试执行异常：{str(ood_e)}")
                    ood_result = {"error": str(ood_e)}

            # ====================== 修复 2：补上必须的 intermediate 参数 ======================
            # 检查是否有外部传入的M7结果
            if external_m7_result is not None:
                self._log("4/9 使用外部M7路由结果...")
                # 构建与route_experts返回格式相同的结构
                route = {
                    "selected_experts": [
                        {"name": name} for name in external_m7_result.get("selected_experts", [])
                    ],
                    "route_reason": external_m7_result.get("route_reason", ""),
                    "confidence": external_m7_result.get("confidence", 0.8),
                    "routing_scores": external_m7_result.get("routing_scores", {}),
                    "normalized_risk_level": external_m7_result.get("normalized_risk_level", normalized_risk),
                    "intent_result": external_m7_result.get("intent_analysis", {})
                }
            else:
                self._log("4/9 智能专家路由匹配...")
                route = route_experts(
                    risk_level=risk_level,
                    project_data=project_data,
                    intermediate=intermediate
                )

            self._log("5/9 构建多维度上下文...")
            ctx1 = build_layer1_context(user_input=user_query)
            ctx2 = build_layer2_context(conversation_turns=[])
            ctx3 = build_layer3_context(user_id=user_id, current_query=user_query)
            ctx4 = build_layer4_context(current_query=user_query)
        
            self._log("6/9 多专家并行推理、融合与闸门校验...")
            blend = run_expert_llm_inference_with_blender(
                risk_level=risk_level,
                reasons=risk_reasons,
                intermediate=intermediate,
                project_data=project_data,
                route_result=route,
                user_id=user_id,
                user_input=user_query,
                uploaded_snippets=None,
                conversation_turns=[],
                summary_buffer="",
                session_max_turns=6,
                session_strategy="sliding_window",
                profile_db_path=None,
                profile_top_k=5,
                global_kb_path=None,
                global_kb_top_k=5,
                auto_profile_log=False,
                top_k=2,
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                use_llm_fuser=bool(self.llm_client),
            )
            candidates = blend.get("candidates", [])

            # M10
            perf_report = None
            if enable_perf_monitor:
                self._log("8/9 M10 性能与成本分析...")
                log_process_result = self.log_processor.process_log_stream("mock_log.txt")
                token_analysis = self.token_analyzer.analyze_token_consumption(self.log_processor.token_records)
                bottlenecks = self.bottleneck_locator.locate_bottlenecks(
                    self.log_processor.token_records,
                    self.log_processor.resource_records,
                    self.log_processor.event_buffer
                )
                perf_report = {
                    "token_analysis": token_analysis,
                    "bottlenecks": bottlenecks,
                    "log_process_status": log_process_result,
                    "cost_report": CostReport(
                        report_id="mock_report_001",
                        start_time=token_analysis["summary"]["start_time"],
                        end_time=token_analysis["summary"]["end_time"],
                        duration=token_analysis["summary"]["duration"],
                        total_token_cost=token_analysis["summary"]["total_cost"],
                        api_call_cost=0.0,
                        compute_cost=0.0,
                        storage_cost=0.0,
                        total_cost=token_analysis["summary"]["total_cost"],
                        token_breakdown=token_analysis["model_breakdown"],
                        cost_breakdown={},
                        bottlenecks=bottlenecks,
                        optimization_suggestions=token_analysis["suggestions"],
                        storage_optimization={}
                    ).to_dict()
                }

            final_payload = blend.get("final_result", blend.get("fused_result", {}))
            gate_info = final_payload.get("gate", {}) if isinstance(final_payload, dict) else {}
            final_reward = 1.0 if final_payload and not gate_info.get("blocked", False) else 0.0
            final_state = EnvState(project_id=user_id, action="decision_completed", reward=final_reward)
            self.state_manager.on_step(final_state.to_dict())

            self._log("✅ 9/9 决策完成！")
            
            # 构建完整的intent信息
            intent_result = {
                "primary_intent": intent.get("primary_intent", "") if isinstance(intent, dict) else str(intent),
                "sub_intent": intent.get("sub_intent", "") if isinstance(intent, dict) else "",
                "urgency": intent.get("urgency", "") if isinstance(intent, dict) else "",
                "raw_intent": intent
            }
            
            # 如果有外部M7结果，也提取意图分析
            if external_m7_result is not None:
                m7_intent = external_m7_result.get("intent_analysis", {})
                if m7_intent.get("primary_intent"):
                    intent_result["primary_intent"] = m7_intent.get("primary_intent", "")
                if m7_intent.get("sub_intent"):
                    intent_result["sub_intent"] = m7_intent.get("sub_intent", "")
            
            return {
                "code": 0,
                "msg": "成功",
                "session_id": self.session_id,
                "user_query": user_query,
                "intent": intent_result,
                "risk": {"level": risk_level, "reasons": risk_reasons, "score": intermediate},
                "ood_test": ood_result,
                "routing": route,
                "experts": candidates,
                "final": final_payload,
                "gate": blend.get("gate", {}),
                "performance": perf_report,
                "state_trace": self.state_manager.get_trace()
            }

        except Exception as e:
            self._log(f"❌ 引擎执行失败：{str(e)}")
            return {"code": 500, "msg": f"异常：{str(e)}", "session_id": self.session_id}

    def _generate_default_ood_action(self, obs: Any) -> Any:
        if not isinstance(obs, (list, tuple, np.ndarray)):
            obs = [2.0, 1.5, 0.5, 10.0, 0.5, 1.0]
        action = np.array(obs, dtype=np.float32) if not isinstance(obs, np.ndarray) else obs.copy()
        risk_level = min(action[3] / 20.0, 1.0)
        magnitude = 0.1 + (risk_level * 0.3)
        action[0] -= magnitude * 0.8
        action[1] -= magnitude * 0.7
        action[2] -= magnitude * 0.6
        action[3] -= magnitude * 1.0
        action[4] -= magnitude * 0.5
        action[5] += magnitude * 0.9
        feature_lows = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        feature_highs = np.array([50.0, 20.0, 1.0, 100.0, 1.0, 7.0], dtype=np.float32)
        action = np.clip(action, feature_lows, feature_highs)
        return action

    @staticmethod
    def run_single_project(project_info: Dict, api_key: Optional[str] = " ") -> Dict:
        project_id = project_info["project_id"]
        try:
            print(f"\n===== 开始分析项目：{project_id} =====")
            engine = M9UltimateRiskEngine(api_key=api_key, verbose=True)
            user_query = "分析我的项目风险、抗压能力、并给出落地建议"
            result = engine.run_full_decision(
                user_query=user_query,
                project_data=project_info,
                enable_ood_test=True,
                enable_perf_monitor=False
            )
            print(f"✅ 项目{project_id}分析完成")
            return result
        except Exception as e:
            error_msg = f"❌ 项目{project_id}分析失败：{str(e)}"
            print(error_msg)
            return {"code": 500, "msg": error_msg, "project_id": project_id}

    @staticmethod
    def run_batch_test(project_list_path: str = "project_list.json", api_key: Optional[str] = None):
        try:
            with open(project_list_path, "r", encoding="utf-8") as f:
                project_list = json.load(f)
            print(f"成功读取{len(project_list)}个测试项目")
        except Exception as e:
            print(f"❌ 读取项目文件失败：{e}")
            return

        batch_results = []
        for project_info in project_list:
            result = M9UltimateRiskEngine.run_single_project(project_info, api_key)
            batch_results.append(result)

        output_file = "m9_batch_test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

        success_num = len([r for r in batch_results if r.get("code") == 0])
        fail_num = len(batch_results) - success_num
        print(f"\n==================== 批量测试完成 ====================")
        print(f"统计：共{len(batch_results)}个项目 | 成功{success_num}个 | 失败{fail_num}个")
        print(f"结果已保存到：{output_file}")
        print(f"项目日志存储路径：{M9UltimateRiskEngine.LOG_BASE_DIR}")

if __name__ == "__main__":
    M9UltimateRiskEngine.run_batch_test(
        project_list_path="project_list.json",
        api_key=" "
    )

