#!/usr/bin/env python
# coding: utf-8

"""
M12 环境增强与 OOD 测试模块
==========================

核心功能：
1. 将基础仿真环境升级为高压仿真环境
2. 实现 OOD（Out-of-Distribution）分布偏移生成能力
3. 模拟黑天鹅事件：市场崩盘、资源断供、监管剧变等极端场景
4. 集成 M3 仿真环境、M5 测试框架和 M6 状态管理系统
5. 测试智能体在极端困境下的决策韧性和生存能力

输入：
- 基础 Env API（来自 M3）
- 极端案例逻辑（来自知识库，如 Autopsy.io, Failory, CB Insights）

输出：
- 增强版高压仿真环境
- OOD 测试报告与韧性评估结果

作者：M12 模块
日期：2026-03-19
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

# 导入相关模块
try:
    from M3仿真环境基座模块版 import StartupEnv, M3Config
except ImportError:
    print("⚠️ M3 模块未找到，部分功能将受限")
    StartupEnv = None
    M3Config = None

try:
    from M5_AutoTest_Suite import TestLogger, TestCase, TestResult, TestReport
except ImportError:
    print("⚠️ M5 模块未找到，部分功能将受限")
    TestLogger = None
    TestCase = None
    TestResult = None
    TestReport = None

try:
    from M6状态管理与日志系统 import (
        EnvStateManager, EnvState, EnvEventType,
        QuantifiedRule, RuleLibrary, RouteDispatcher
    )
except ImportError:
    print("⚠️ M6 模块未找到，部分功能将受限")
    EnvStateManager = None


# =============================
# 配置与常量
# =============================

class OODConfig:
    """OOD 测试配置"""
    
    def __init__(self):
        # 路径配置
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent
        self.output_dir = self.project_root / "Kickstarter_Clean"
        self.ood_output_dir = self.output_dir / "m12_ood_tests"
        self.ood_output_dir.mkdir(parents=True, exist_ok=True)
        
        # OOD 生成参数
        self.drift_magnitude_range = (0.5, 3.0)  # 分布偏移幅度范围
        self.black_swan_probability = 0.1  # 黑天鹅事件触发概率
        self.extreme_case_ratio = 0.2  # 极端案例在测试集中的比例
        
        # 测试参数
        self.max_ood_steps = 24  # OOD 环境下最大步数（加倍测试）
        self.resilience_threshold = 0.6  # 韧性评估阈值
        
        # 日志配置
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"M12 配置初始化完成")
        print(f"  - 输出目录: {self.ood_output_dir}")
        print(f"  - 日志目录: {self.log_dir}")


# =============================
# 枚举与数据类
# =============================

class BlackSwanEventType(str, Enum):
    """黑天鹅事件类型"""
    MARKET_CRASH = "market_crash"           # 市场崩盘
    FUNDING_CUTOFF = "funding_cutoff"        # 资金断供
    REGULATION_CHANGE = "regulation_change"  # 监管剧变
    SUPPLY_CHAIN = "supply_chain"            # 供应链断裂
    COMPETITIVE_ATTACK = "competitive_attack"  # 竞争者突袭
    TECH_FAILURE = "tech_failure"            # 技术故障
    PANIC_SELLING = "panic_selling"          # 恐慌抛售
    LIQUIDITY_CRISIS = "liquidity_crisis"    # 流动性危机


class DistributionShiftType(str, Enum):
    """分布偏移类型"""
    COVARIATE_SHIFT = "covariate_shift"     # 协变量偏移
    CONCEPT_SHIFT = "concept_shift"         # 概念偏移
    PRIOR_SHIFT = "prior_shift"             # 先验偏移
    HYBRID_SHIFT = "hybrid_shift"           # 混合偏移


@dataclass
class BlackSwanEvent:
    """黑天鹅事件定义"""
    event_id: str
    event_type: BlackSwanEventType
    name: str
    description: str
    severity: float  # 严重程度 0-1
    trigger_conditions: Dict[str, Any]
    impact_parameters: Dict[str, float]  # 对各参数的影响
    duration: int  # 持续步数
    recovery_difficulty: float  # 恢复难度 0-1
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DistributionShift:
    """分布偏移配置"""
    shift_id: str
    shift_type: DistributionShiftType
    target_feature: str  # 目标特征
    drift_magnitude: float  # 偏移幅度
    shift_direction: str  # "increase" 或 "decrease"
    apply_step: int  # 在第几步应用偏移
    is_gradual: bool  # 是否渐变
    gradual_steps: int = 1  # 渐变步数
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OODScenario:
    """OOD 测试场景"""
    scenario_id: str
    name: str
    description: str
    base_project: Dict[str, Any]  # 基础项目数据
    distribution_shifts: List[DistributionShift]
    black_swan_events: List[BlackSwanEvent]
    expected_outcome: str  # "survive" 或 "fail"
    difficulty_level: str  # "easy", "medium", "hard", "extreme"
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['distribution_shifts'] = [s.to_dict() for s in self.distribution_shifts]
        data['black_swan_events'] = [e.to_dict() for e in self.black_swan_events]
        return data


@dataclass
class ResilienceMetrics:
    """韧性度量指标"""
    scenario_id: str
    survival_rate: float  # 生存率
    avg_risk_increase: float  # 平均风险增幅
    recovery_speed: float  # 恢复速度
    worst_case_performance: float  # 最坏情况表现
    adaptability_score: float  # 适应性得分
    overall_resilience: float  # 综合韧性得分
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================
# 黑天鹅事件库（来自失败案例知识库）
# =============================

class BlackSwanLibrary:
    """黑天鹅事件库 - 基于 Autopsy.io, Failory, CB Insights 等失败案例"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.events = self._initialize_events()
        self.logger.info(f"✓ 黑天鹅事件库初始化完成，共 {len(self.events)} 个事件")
    
    def _initialize_events(self) -> List[BlackSwanEvent]:
        """初始化黑天鹅事件库"""
        events = []
        
        # 1. 市场崩盘（基于 2008 金融危机、2020 疫情等案例）
        events.append(BlackSwanEvent(
            event_id="bs_001",
            event_type=BlackSwanEventType.MARKET_CRASH,
            name="融资环境突然恶化",
            description="由于宏观经济危机，投资人突然撤资，融资轮次取消",
            severity=0.9,
            trigger_conditions={
                "step_range": [6, 12],
                "funding_status": "seeking",
                "market_stability": 0.3
            },
            impact_parameters={
                "goal_ratio": 5.0,  # 目标相对值飙升
                "country_factor": 0.8,  # 市场环境恶化
                "urgency_score": 3.0  # 紧迫度激增
            },
            duration=6,
            recovery_difficulty=0.8
        ))
        
        # 2. 资金断供（基于 Quibi, Jawbone 等烧钱过快案例）
        events.append(BlackSwanEvent(
            event_id="bs_002",
            event_type=BlackSwanEventType.FUNDING_CUTOFF,
            name="主力投资者突然撤资",
            description="关键投资者因内部问题撤回承诺的资金支持",
            severity=0.95,
            trigger_conditions={
                "step_range": [3, 10],
                "burn_rate": "high",
                "funding_round": "Series B or later"
            },
            impact_parameters={
                "time_penalty": 5.0,  # 时间惩罚急剧增加
                "goal_ratio": 8.0,  # 融资缺口扩大
                "urgency_score": 5.0
            },
            duration=8,
            recovery_difficulty=0.9
        ))
        
        # 3. 监管剧变（基于 Airbnb, Uber 初期案例）
        events.append(BlackSwanEvent(
            event_id="bs_003",
            event_type=BlackSwanEventType.REGULATION_CHANGE,
            name="核心业务被监管限制",
            description="新法规突然出台，直接限制核心商业模式",
            severity=0.85,
            trigger_conditions={
                "step_range": [5, 15],
                "business_model": "platform",
                "regulatory_risk": 0.7
            },
            impact_parameters={
                "category_risk": 0.9,  # 品类风险飙升
                "country_factor": 0.7,
                "time_penalty": 3.0
            },
            duration=10,
            recovery_difficulty=0.75
        ))
        
        # 4. 供应链断裂（基于 Theranos, Juicero 等案例）
        events.append(BlackSwanEvent(
            event_id="bs_004",
            event_type=BlackSwanEventType.SUPPLY_CHAIN,
            name="关键技术供应商破产",
            description="依赖的核心技术或供应商突然停止服务",
            severity=0.8,
            trigger_conditions={
                "step_range": [4, 12],
                "dependency": "critical",
                "supplier_concentration": 0.8
            },
            impact_parameters={
                "time_penalty": 4.0,
                "urgency_score": 4.0,
                "goal_ratio": 3.0
            },
            duration=7,
            recovery_difficulty=0.7
        ))
        
        # 5. 竞争者突袭（基于 Google+, Amazon 等）
        events.append(BlackSwanEvent(
            event_id="bs_005",
            event_type=BlackSwanEventType.COMPETITIVE_ATTACK,
            name="巨头入场竞争",
            description="行业巨头突然推出同类产品，免费抢占市场",
            severity=0.75,
            trigger_conditions={
                "step_range": [6, 14],
                "market_position": "leader",
                "competitor_threat": 0.9
            },
            impact_parameters={
                "category_risk": 0.7,
                "time_penalty": 2.0,
                "country_factor": 0.6
            },
            duration=5,
            recovery_difficulty=0.65
        ))
        
        # 6. 技术故障（基于 Theranos 案例）
        events.append(BlackSwanEvent(
            event_id="bs_006",
            event_type=BlackSwanEventType.TECH_FAILURE,
            name="核心技术被证明不可行",
            description="核心产品技术方案被发现存在根本性缺陷",
            severity=0.95,
            trigger_conditions={
                "step_range": [2, 8],
                "tech_maturity": "early",
                "technical_risk": 0.8
            },
            impact_parameters={
                "goal_ratio": 10.0,  # 需要完全重构
                "time_penalty": 8.0,
                "urgency_score": 6.0
            },
            duration=12,
            recovery_difficulty=0.95
        ))
        
        # 7. 恐慌抛售（基于市场情绪案例）
        events.append(BlackSwanEvent(
            event_id="bs_007",
            event_type=BlackSwanEventType.PANIC_SELLING,
            name="用户群体恐慌性流失",
            description="负面舆论爆发，用户大规模取消订阅/退款",
            severity=0.8,
            trigger_conditions={
                "step_range": [5, 11],
                "user_base": "growing",
                "public_sentiment": 0.2
            },
            impact_parameters={
                "category_risk": 0.8,
                "goal_ratio": 4.0,
                "urgency_score": 3.0
            },
            duration=6,
            recovery_difficulty=0.7
        ))
        
        # 8. 流动性危机（基于 2000 互联网泡沫等案例）
        events.append(BlackSwanEvent(
            event_id="bs_008",
            event_type=BlackSwanEventType.LIQUIDITY_CRISIS,
            name="现金流突然枯竭",
            description="收款延迟、支出提前，现金流在短期内归零",
            severity=0.9,
            trigger_conditions={
                "step_range": [4, 10],
                "cash_reserve": "low",
                "payment_terms": "adverse"
            },
            impact_parameters={
                "time_penalty": 6.0,
                "goal_ratio": 7.0,
                "urgency_score": 5.0
            },
            duration=9,
            recovery_difficulty=0.85
        ))
        
        return events
    
    def get_event_by_id(self, event_id: str) -> Optional[BlackSwanEvent]:
        """根据ID获取事件"""
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None
    
    def get_events_by_type(self, event_type: BlackSwanEventType) -> List[BlackSwanEvent]:
        """根据类型获取事件列表"""
        return [e for e in self.events if e.event_type == event_type]
    
    def get_random_event(self, severity_threshold: float = 0.5) -> BlackSwanEvent:
        """获取随机事件（可过滤严重程度）"""
        import random
        eligible = [e for e in self.events if e.severity >= severity_threshold]
        return random.choice(eligible) if eligible else random.choice(self.events)
    
    def get_all_events(self) -> List[BlackSwanEvent]:
        """获取所有事件"""
        return self.events.copy()


# =============================
# 分布偏移生成器
# =============================

class DriftGenerator:
    """分布偏移生成器 - 生成 OOD 场景的分布偏移"""
    
    def __init__(self, config: OODConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.feature_names = [
            "goal_ratio", "time_penalty", "category_risk",
            "combined_risk", "country_factor", "urgency_score"
        ]
        self.feature_ranges = {
            "goal_ratio": (0.0, 50.0),
            "time_penalty": (0.0, 20.0),
            "category_risk": (0.0, 1.0),
            "combined_risk": (0.0, 100.0),
            "country_factor": (0.0, 1.0),
            "urgency_score": (0.0, 7.0)
        }
        self.logger.info("✓ 分布偏移生成器初始化完成")
    
    def generate_covariate_shift(self, feature_name: str, magnitude: float, 
                                 direction: str = "increase") -> DistributionShift:
        """生成协变量偏移（输入特征分布变化）"""
        if feature_name not in self.feature_names:
            raise ValueError(f"未知特征: {feature_name}")
        
        return DistributionShift(
            shift_id=f"covariate_{feature_name}_{int(magnitude*100)}",
            shift_type=DistributionShiftType.COVARIATE_SHIFT,
            target_feature=feature_name,
            drift_magnitude=magnitude,
            shift_direction=direction,
            apply_step=0,
            is_gradual=True,
            gradual_steps=3
        )
    
    def generate_concept_shift(self, risk_mapping: Dict[str, float]) -> List[DistributionShift]:
        """生成概念偏移（风险判定逻辑变化）"""
        shifts = []
        for feature, multiplier in risk_mapping.items():
            if feature in self.feature_names:
                shifts.append(DistributionShift(
                    shift_id=f"concept_{feature}_{int(multiplier*100)}",
                    shift_type=DistributionShiftType.CONCEPT_SHIFT,
                    target_feature=feature,
                    drift_magnitude=multiplier,
                    shift_direction="increase" if multiplier > 1 else "decrease",
                    apply_step=6,
                    is_gradual=False
                ))
        return shifts
    
    def generate_prior_shift(self, category_shift: Dict[str, float]) -> List[DistributionShift]:
        """生成先验偏移（品类风险分布变化）"""
        shifts = []
        for feature, shift_amount in category_shift.items():
            if feature == "category_risk":
                shifts.append(DistributionShift(
                    shift_id=f"prior_category_{int(shift_amount*100)}",
                    shift_type=DistributionShiftType.PRIOR_SHIFT,
                    target_feature=feature,
                    drift_magnitude=shift_amount,
                    shift_direction="increase" if shift_amount > 0 else "decrease",
                    apply_step=0,
                    is_gradual=False
                ))
        return shifts
    
    def generate_random_drift(self) -> DistributionShift:
        """生成随机分布偏移"""
        import random
        feature = random.choice(self.feature_names)
        magnitude = random.uniform(*self.config.drift_magnitude_range)
        direction = random.choice(["increase", "decrease"])
        
        shift_type = random.choice(list(DistributionShiftType))
        
        return DistributionShift(
            shift_id=f"random_{shift_type.value}_{feature}_{int(magnitude*100)}",
            shift_type=shift_type,
            target_feature=feature,
            drift_magnitude=magnitude,
            shift_direction=direction,
            apply_step=random.randint(0, 6),
            is_gradual=random.choice([True, False]),
            gradual_steps=random.randint(1, 5)
        )
    
    def apply_drift(self, feature_vector: np.ndarray, 
                    drift: DistributionShift, 
                    current_step: int) -> np.ndarray:
        """应用分布偏移到特征向量"""
        feature_idx = self.feature_names.index(drift.target_feature)
        result = feature_vector.copy()
        
        # 检查是否到达应用步骤
        if current_step < drift.apply_step:
            return result
        
        # 计算偏移量
        shift_amount = drift.drift_magnitude
        
        if drift.is_gradual:
            # 渐变应用
            steps_since_apply = current_step - drift.apply_step
            progress = min(steps_since_apply / drift.gradual_steps, 1.0)
            effective_shift = shift_amount * progress
        else:
            # 立即应用
            effective_shift = shift_amount
        
        # 应用偏移
        if drift.shift_direction == "increase":
            result[feature_idx] += effective_shift
        else:
            result[feature_idx] -= effective_shift
        
        # 约束在合理范围内
        min_val, max_val = self.feature_ranges[drift.target_feature]
        result[feature_idx] = np.clip(result[feature_idx], min_val, max_val)
        
        return result


# =============================
# OOD 场景生成器
# =============================

class OODScenarioGenerator:
    """OOD 场景生成器 - 生成极端测试场景"""
    
    def __init__(self, config: OODConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.black_swan_library = BlackSwanLibrary(logger)
        self.drift_generator = DriftGenerator(config, logger)
        self.scenario_counter = 0
        self.logger.info("✓ OOD 场景生成器初始化完成")
    
    def generate_scenario_from_project(self, project_data: Dict[str, Any],
                                       difficulty: str = "medium") -> OODScenario:
        """从基础项目生成 OOD 场景"""
        self.scenario_counter += 1
        
        # 确定难度级别
        difficulty_map = {
            "easy": 1, "medium": 2, "hard": 3, "extreme": 4
        }
        difficulty_level = difficulty_map.get(difficulty, 2)
        
        # 生成分布偏移
        num_shifts = difficulty_level * 2
        shifts = []
        for i in range(num_shifts):
            shifts.append(self.drift_generator.generate_random_drift())
        
        # 生成黑天鹅事件
        import random
        num_events = difficulty_level
        events = []
        for i in range(num_events):
            severity_threshold = 0.5 + (difficulty_level * 0.1)
            event = self.black_swan_library.get_random_event(severity_threshold)
            events.append(event)
        
        # 预期结果
        expected_outcome = "survive" if difficulty_level <= 2 else "fail"
        
        scenario = OODScenario(
            scenario_id=f"ood_scenario_{self.scenario_counter:04d}",
            name=f"OOD_{difficulty}_{self.scenario_counter}",
            description=f"难度{difficulty}的极端测试场景，包含{len(shifts)}个分布偏移和{len(events)}个黑天鹅事件",
            base_project=project_data,
            distribution_shifts=shifts,
            black_swan_events=events,
            expected_outcome=expected_outcome,
            difficulty_level=difficulty
        )
        
        self.logger.info(f"✓ 生成 OOD 场景: {scenario.scenario_id}")
        return scenario
    
    def generate_scenario_batch(self, projects: List[Dict[str, Any]],
                               num_per_difficulty: int = 5) -> List[OODScenario]:
        """批量生成 OOD 场景"""
        scenarios = []
        difficulties = ["easy", "medium", "hard", "extreme"]
        
        for difficulty in difficulties:
            for i in range(num_per_difficulty):
                if projects:
                    project = projects[i % len(projects)]
                else:
                    project = self._generate_mock_project()
                
                scenario = self.generate_scenario_from_project(project, difficulty)
                scenarios.append(scenario)
        
        self.logger.info(f"✓ 批量生成 {len(scenarios)} 个 OOD 场景")
        return scenarios
    
    def _generate_mock_project(self) -> Dict[str, Any]:
        """生成模拟项目数据"""
        return {
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


# =============================
# 增强版仿真环境
# =============================

class EnhancedStartupEnv:
    """增强版高压仿真环境 - 基于 M3 环境的 OOD 扩展"""
    
    def __init__(self, base_env: StartupEnv, config: OODConfig, 
                 logger: logging.Logger):
        self.base_env = base_env
        self.config = config
        self.logger = logger
        self.state_manager = None
        self.current_scenario = None
        self.applied_events = []
        self.max_steps = self.config.max_ood_steps
        self.current_step = 0
        self.feature_dim = 6
        self.feature_names = [
            "goal_ratio", "time_penalty", "category_risk",
            "combined_risk", "country_factor", "urgency_score"
        ]
        
        self.logger.info("✓ 增强版高压仿真环境初始化完成")
    
    def attach_state_manager(self, state_manager):
        """挂载状态管理器"""
        self.state_manager = state_manager
        self.logger.info("✓ 状态管理器已挂载")
    
    def reset_with_scenario(self, scenario: OODScenario) -> Tuple[np.ndarray, Dict]:
        """使用 OOD 场景重置环境"""
        self.current_scenario = scenario
        self.current_step = 0
        self.applied_events = []
        
        # 提取基础特征向量
        base_data = scenario.base_project
        feature_vector = np.array([
            base_data.get("goal_ratio", 2.0),
            base_data.get("time_penalty", 1.5),
            base_data.get("category_risk", 0.5),
            base_data.get("combined_risk", 10.0),
            base_data.get("country_factor", 0.5),
            base_data.get("urgency_score", 1.0)
        ], dtype=np.float32)
        
        # 应用初始分布偏移
        for drift in scenario.distribution_shifts:
            if drift.apply_step == 0:
                feature_vector = self.drift_generator.apply_drift(
                    feature_vector, drift, 0
                )
        
        # 重置基础环境
        obs, info = self.base_env.reset(options={"feature_vector": feature_vector})
        
        # 记录状态
        if self.state_manager:
            self.state_manager.on_reset({
                "project_id": scenario.scenario_id,
                "case_id": scenario.scenario_id,
                "scenario_id": scenario.scenario_id,
                "step": 0,
                "done": False,
                "observation": obs.tolist(),
                "info": info,
                "timestamp": datetime.now().timestamp()
            })
        
        self.logger.info(f"✓ OOD 场景重置: {scenario.scenario_id}")
        return obs, info
    
    @property
    def drift_generator(self):
        """获取分布偏移生成器"""
        if not hasattr(self, '_drift_gen'):
            self._drift_gen = DriftGenerator(self.config, self.logger)
        return self._drift_gen
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步（考虑 OOD 场景）"""
        self.current_step += 1
        
        # 应用分布偏移
        modified_action = action.copy()
        if self.current_scenario:
            for drift in self.current_scenario.distribution_shifts:
                modified_action = self.drift_generator.apply_drift(
                    modified_action, drift, self.current_step
                )
        
        # 触发黑天鹅事件
        black_swan_impact = self._check_and_trigger_black_swan()
        if black_swan_impact is not None:
            modified_action = self._apply_black_swan_impact(
                modified_action, black_swan_impact
            )
        
        # 执行基础环境 step
        obs, reward, terminated, truncated, info = self.base_env.step(modified_action)
        
        # 更新终止条件（OOD 环境可能更严格）
        if self.current_scenario and self.current_scenario.difficulty_level >= 3:
            # 高难度场景：风险过高提前终止
            if obs[3] > 25.0 and self.current_step >= 3:  # combined_risk > 25
                terminated = True
                reward -= 10.0  # 额外惩罚
        elif self.current_step >= self.max_steps:
            truncated = True
        
        # 记录状态
        if self.state_manager:
            self.state_manager.on_step({
                "project_id": self.current_scenario.scenario_id if self.current_scenario else "unknown",
                "case_id": self.current_scenario.scenario_id if self.current_scenario else "unknown",
                "scenario_id": self.current_scenario.scenario_id if self.current_scenario else None,
                "step": self.current_step,
                "done": terminated or truncated,
                "action": action.tolist(),
                "observation": obs.tolist(),
                "reward": float(reward),
                "info": {
                    **info,
                    "black_swan_triggered": black_swan_impact is not None,
                    "active_events": [e.event_id for e in self.applied_events]
                },
                "timestamp": datetime.now().timestamp()
            })
        
        return obs, float(reward), terminated, truncated, info
    
    def _check_and_trigger_black_swan(self) -> Optional[BlackSwanEvent]:
        """检查并触发黑天鹅事件"""
        import random
        
        if not self.current_scenario:
            return None
        
        # 随机触发（根据配置概率）
        if random.random() > self.config.black_swan_probability:
            return None
        
        # 检查每个事件的触发条件
        for event in self.current_scenario.black_swan_events:
            if event.event_id in [e.event_id for e in self.applied_events]:
                continue  # 已经触发过
            
            trigger_cond = event.trigger_conditions
            step_range = trigger_cond.get("step_range", [0, self.max_steps])
            
            if step_range[0] <= self.current_step <= step_range[1]:
                self.applied_events.append(event)
                self.logger.warning(f"⚠️ 黑天鹅事件触发: {event.name} ({event.event_id})")
                return event
        
        return None
    
    def _apply_black_swan_impact(self, feature_vector: np.ndarray, 
                                 event: BlackSwanEvent) -> np.ndarray:
        """应用黑天鹅事件影响"""
        result = feature_vector.copy()
        impact = event.impact_parameters
        
        # 应用各项参数影响
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in impact:
                result[i] += impact[feature_name]
        
        # 约束在合理范围内
        feature_ranges = {
            "goal_ratio": (0.0, 50.0),
            "time_penalty": (0.0, 20.0),
            "category_risk": (0.0, 1.0),
            "combined_risk": (0.0, 100.0),
            "country_factor": (0.0, 1.0),
            "urgency_score": (0.0, 7.0)
        }
        
        for i, feature_name in enumerate(self.feature_names):
            min_val, max_val = feature_ranges.get(feature_name, (0, 100))
            result[i] = np.clip(result[i], min_val, max_val)
        
        return result
    
    def close(self):
        """关闭环境"""
        if self.base_env:
            self.base_env.close()


# =============================
# 韧性评估器
# =============================

class ResilienceEvaluator:
    """韧性评估器 - 评估智能体在 OOD 场景下的表现"""
    
    def __init__(self, config: OODConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.logger.info("✓ 韧性评估器初始化完成")
    
    def evaluate_scenario(self, scenario: OODScenario,
                         trajectory: List[Dict]) -> ResilienceMetrics:
        """评估单个场景的韧性"""
        if not trajectory:
            return self._empty_metrics(scenario.scenario_id)
        
        # 1. 生存率：是否存活到预期步数
        final_step = trajectory[-1].get("step", 0)
        survival_rate = min(final_step / 12, 1.0)  # 基准是12步
        
        # 2. 平均风险增幅：相比初始状态的风险增幅
        initial_risk = trajectory[0].get("observation", [0]*6)[3]
        max_risk = max(step.get("observation", [0]*6)[3] for step in trajectory)
        avg_risk_increase = max(0, (max_risk - initial_risk) / max(initial_risk, 1))
        
        # 3. 恢复速度：从黑天鹅事件中恢复的速度
        recovery_speed = self._calculate_recovery_speed(trajectory)
        
        # 4. 最坏情况表现：风险峰值时的表现
        worst_risk_step = max(trajectory, 
                             key=lambda x: x.get("observation", [0]*6)[3])
        worst_case_performance = 1.0 - min(
            worst_risk_step.get("observation", [0]*6)[3] / 30.0, 1.0
        )
        
        # 5. 适应性得分：动作的合理性
        adaptability_score = self._calculate_adaptability(trajectory)
        
        # 6. 综合韧性得分
        overall_resilience = (
            survival_rate * 0.3 +
            (1 - avg_risk_increase) * 0.25 +
            recovery_speed * 0.2 +
            worst_case_performance * 0.15 +
            adaptability_score * 0.1
        )
        
        metrics = ResilienceMetrics(
            scenario_id=scenario.scenario_id,
            survival_rate=survival_rate,
            avg_risk_increase=avg_risk_increase,
            recovery_speed=recovery_speed,
            worst_case_performance=worst_case_performance,
            adaptability_score=adaptability_score,
            overall_resilience=overall_resilience
        )
        
        self.logger.info(f"✓ 韧性评估: {scenario.scenario_id} -> {overall_resilience:.3f}")
        return metrics
    
    def _calculate_recovery_speed(self, trajectory: List[Dict]) -> float:
        """计算恢复速度"""
        black_swan_steps = [
            i for i, step in enumerate(trajectory)
            if step.get("info", {}).get("black_swan_triggered", False)
        ]
        
        if not black_swan_steps:
            return 1.0  # 未触发黑天鹅，视为完美恢复
        
        # 找到最晚的黑天鹅事件
        last_event_step = black_swan_steps[-1]
        if last_event_step >= len(trajectory) - 2:
            return 0.0  # 刚触发就结束，无法评估恢复
        
        # 计算事件后的风险变化
        event_risk = trajectory[last_event_step].get("observation", [0]*6)[3]
        final_risk = trajectory[-1].get("observation", [0]*6)[3]
        
        if event_risk <= 0:
            return 0.5
        
        recovery_ratio = max(0, 1 - (final_risk - event_risk) / event_risk)
        return min(recovery_ratio, 1.0)
    
    def _calculate_adaptability(self, trajectory: List[Dict]) -> float:
        """计算适应性得分"""
        if len(trajectory) < 2:
            return 0.5
        
        # 计算动作的平滑性（剧烈变化过多得分低）
        actions = [step.get("action", [0]*6) for step in trajectory]
        action_changes = []
        
        for i in range(1, len(actions)):
            prev_action = actions[i-1]
            curr_action = actions[i]
            
            if isinstance(prev_action, list) and isinstance(curr_action, list):
                change = sum(abs(p - c) for p, c in zip(prev_action, curr_action))
                action_changes.append(change)
        
        if not action_changes:
            return 0.5
        
        avg_change = sum(action_changes) / len(action_changes)
        # 适度变化得分高，过度波动或完全不变化得分低
        adaptability = max(0, 1 - avg_change / 10.0)
        return min(adaptability, 1.0)
    
    def _empty_metrics(self, scenario_id: str) -> ResilienceMetrics:
        """空轨迹的默认指标"""
        return ResilienceMetrics(
            scenario_id=scenario_id,
            survival_rate=0.0,
            avg_risk_increase=1.0,
            recovery_speed=0.0,
            worst_case_performance=0.0,
            adaptability_score=0.0,
            overall_resilience=0.0
        )
    
    def evaluate_batch(self, scenarios: List[OODScenario],
                     trajectories: List[List[Dict]]) -> List[ResilienceMetrics]:
        """批量评估"""
        metrics_list = []
        
        for scenario, trajectory in zip(scenarios, trajectories):
            metrics = self.evaluate_scenario(scenario, trajectory)
            metrics_list.append(metrics)
        
        # 计算聚合指标
        avg_resilience = sum(m.overall_resilience for m in metrics_list) / len(metrics_list)
        survival_rate = sum(1 for m in metrics_list 
                          if m.overall_resilience > self.config.resilience_threshold) / len(metrics_list)
        
        self.logger.info(f"✓ 批量评估完成: {len(metrics_list)} 个场景")
        self.logger.info(f"  - 平均韧性: {avg_resilience:.3f}")
        self.logger.info(f"  - 生存率: {survival_rate:.2%}")
        
        return metrics_list


# =============================
# OOD 测试执行器
# =============================

class OODTestExecutor:
    """OOD 测试执行器 - 执行 OOD 测试并生成报告"""
    
    def __init__(self, config: OODConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scenario_generator = OODScenarioGenerator(config, logger)
        self.resilience_evaluator = ResilienceEvaluator(config, logger)
        self.logger.info("✓ OOD 测试执行器初始化完成")
    
    def run_ood_test(self, scenario: OODScenario, 
                     agent_action_fn: Optional[Callable] = None) -> Tuple[Dict, List]:
        """执行单个 OOD 测试"""
        if StartupEnv is None:
            self.logger.error("M3 环境未初始化，无法执行测试")
            return {}, []
        
        # 创建基础环境
        base_env = StartupEnv()
        
        # 创建增强环境
        enhanced_env = EnhancedStartupEnv(base_env, self.config, self.logger)
        
        # 挂载状态管理器（如果可用）
        if EnvStateManager:
            state_manager = EnvStateManager(
                project_id=scenario.scenario_id,
                logger=self.logger
            )
            enhanced_env.attach_state_manager(state_manager)
        
        # 重置环境
        obs, info = enhanced_env.reset_with_scenario(scenario)
        
        # 执行仿真
        trajectory = []
        done = False
        
        while not done:
            # 生成动作
            if agent_action_fn:
                action = agent_action_fn(obs, info)
            else:
                # 默认动作：简单的风险降低策略
                action = self._default_action(obs)
            
            # 执行 step
            next_obs, reward, terminated, truncated, step_info = enhanced_env.step(action)
            
            # 记录轨迹
            trajectory.append({
                "step": enhanced_env.current_step,
                "observation": obs.tolist(),
                "action": action.tolist(),
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "info": step_info
            })
            
            obs = next_obs
            done = terminated or truncated
        
        # 获取状态轨迹（如果使用状态管理器）
        state_trace = []
        if EnvStateManager and state_manager:
            state_trace = state_manager.get_trace()
        
        # 关闭环境
        enhanced_env.close()
        
        self.logger.info(f"✓ OOD 测试完成: {scenario.scenario_id}, 步数: {len(trajectory)}")
        
        return {
            "scenario_id": scenario.scenario_id,
            "trajectory": trajectory,
            "state_trace": state_trace
        }, trajectory
    
    def _default_action(self, obs: np.ndarray) -> np.ndarray:
        """默认动作策略：风险降低"""
        action = obs.copy()
        
        # 风险越高，调整幅度越大
        risk_level = min(obs[3] / 20.0, 1.0)
        magnitude = 0.1 + (risk_level * 0.3)
        
        # 降低各项风险指标
        action[0] -= magnitude * 0.8  # goal_ratio
        action[1] -= magnitude * 0.7  # time_penalty
        action[2] -= magnitude * 0.6  # category_risk
        action[3] -= magnitude * 1.0  # combined_risk (核心)
        action[4] -= magnitude * 0.5  # country_factor
        action[5] += magnitude * 0.9  # urgency_score (提升紧迫度)
        
        # 约束在合理范围
        feature_lows = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        feature_highs = np.array([50.0, 20.0, 1.0, 100.0, 1.0, 7.0], dtype=np.float32)
        action = np.clip(action, feature_lows, feature_highs)
        
        return action
    
    def run_ood_test_batch(self, scenarios: List[OODScenario],
                          agent_action_fn: Optional[Callable] = None) -> Dict:
        """批量执行 OOD 测试"""
        results = {}
        all_trajectories = []
        
        for scenario in scenarios:
            result, trajectory = self.run_ood_test(scenario, agent_action_fn)
            results[scenario.scenario_id] = result
            all_trajectories.append(trajectory)
        
        # 评估韧性
        resilience_metrics = self.resilience_evaluator.evaluate_batch(
            scenarios, all_trajectories
        )
        
        return {
            "results": results,
            "trajectories": all_trajectories,
            "resilience_metrics": [m.to_dict() for m in resilience_metrics]
        }


# =============================
# 报告生成器
# =============================

class OODReportGenerator:
    """OOD 测试报告生成器"""
    
    def __init__(self, config: OODConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.logger.info("✓ OOD 报告生成器初始化完成")
    
    def generate_report(self, test_results: Dict, 
                       scenarios: List[OODScenario]) -> Dict:
        """生成完整的 OOD 测试报告"""
        resilience_metrics = test_results.get("resilience_metrics", [])
        
        # 计算汇总统计
        avg_resilience = sum(m["overall_resilience"] for m in resilience_metrics) / len(resilience_metrics) if resilience_metrics else 0
        survival_rate = sum(1 for m in resilience_metrics 
                          if m["overall_resilience"] > self.config.resilience_threshold) / len(resilience_metrics) if resilience_metrics else 0
        
        # 按难度分组统计
        difficulty_stats = {}
        for scenario, metrics in zip(scenarios, resilience_metrics):
            difficulty = scenario.difficulty_level
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {
                    "count": 0,
                    "total_resilience": 0,
                    "survived": 0
                }
            
            difficulty_stats[difficulty]["count"] += 1
            difficulty_stats[difficulty]["total_resilience"] += metrics["overall_resilience"]
            if metrics["overall_resilience"] > self.config.resilience_threshold:
                difficulty_stats[difficulty]["survived"] += 1
        
        # 计算平均值
        for diff in difficulty_stats:
            stats = difficulty_stats[diff]
            stats["avg_resilience"] = stats["total_resilience"] / stats["count"]
            stats["survival_rate"] = stats["survived"] / stats["count"]
        
        report = {
            "report_id": f"m12_ood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_ood_steps": self.config.max_ood_steps,
                "resilience_threshold": self.config.resilience_threshold,
                "drift_magnitude_range": self.config.drift_magnitude_range,
                "black_swan_probability": self.config.black_swan_probability
            },
            "summary": {
                "total_scenarios": len(scenarios),
                "avg_resilience": round(avg_resilience, 4),
                "survival_rate": round(survival_rate, 4),
                "pass_threshold": self.config.resilience_threshold,
                "passed": sum(1 for m in resilience_metrics 
                            if m["overall_resilience"] > self.config.resilience_threshold)
            },
            "difficulty_breakdown": difficulty_stats,
            "detailed_results": {
                "scenarios": [s.to_dict() for s in scenarios],
                "resilience_metrics": resilience_metrics
            },
            "recommendations": self._generate_recommendations(resilience_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, resilience_metrics: List[Dict]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if not resilience_metrics:
            return recommendations
        
        avg_resilience = sum(m["overall_resilience"] for m in resilience_metrics) / len(resilience_metrics)
        
        if avg_resilience < 0.4:
            recommendations.append("整体韧性严重不足，建议重新设计核心决策逻辑")
        elif avg_resilience < 0.6:
            recommendations.append("韧性有待提升，建议加强对黑天鹅事件的应对机制")
        
        # 检查具体维度
        avg_recovery = sum(m["recovery_speed"] for m in resilience_metrics) / len(resilience_metrics)
        if avg_recovery < 0.5:
            recommendations.append("恢复速度较慢，建议增加快速响应机制")
        
        avg_adaptability = sum(m["adaptability_score"] for m in resilience_metrics) / len(resilience_metrics)
        if avg_adaptability < 0.5:
            recommendations.append("适应性不足，建议优化动作平滑性")
        
        avg_risk_increase = sum(m["avg_risk_increase"] for m in resilience_metrics) / len(resilience_metrics)
        if avg_risk_increase > 0.5:
            recommendations.append("风险控制能力较弱，建议加强风险预测和预防")
        
        if not recommendations:
            recommendations.append("整体表现良好，建议继续优化极端场景下的决策能力")
        
        return recommendations
    
    def save_report(self, report: Dict, filename: Optional[str] = None) -> Path:
        """保存报告到文件"""
        if filename is None:
            filename = f"m12_ood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = self.config.ood_output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✓ OOD 测试报告已保存: {report_path}")
        return report_path
    
    def print_summary(self, report: Dict):
        """打印报告摘要"""
        print("\n" + "="*70)
        print("M12 环境增强与 OOD 测试报告")
        print("="*70)
        print(f"报告ID: {report['report_id']}")
        print(f"生成时间: {report['timestamp']}")
        
        summary = report['summary']
        print(f"\n📊 测试摘要:")
        print(f"  总场景数: {summary['total_scenarios']}")
        print(f"  平均韧性: {summary['avg_resilience']:.4f}")
        print(f"  生存率: {summary['survival_rate']:.2%}")
        print(f"  通过场景: {summary['passed']}/{summary['total_scenarios']}")
        
        print(f"\n📈 难度分组:")
        for difficulty, stats in report['difficulty_breakdown'].items():
            print(f"  {difficulty:8s}: {stats['count']} 个场景, "
                  f"平均韧性={stats['avg_resilience']:.4f}, "
                  f"生存率={stats['survival_rate']:.2%}")
        
        print(f"\n💡 改进建议:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
        
        print("="*70 + "\n")


# =============================
# M12 主入口
# =============================

class M12_OOD_Test_Module:
    """M12 环境增强与 OOD 测试模块 - 主入口"""
    
    def __init__(self):
        self.config = OODConfig()
        self.logger = self._setup_logger()
        self.scenario_generator = OODScenarioGenerator(self.config, self.logger)
        self.test_executor = OODTestExecutor(self.config, self.logger)
        self.report_generator = OODReportGenerator(self.config, self.logger)
        
        self.logger.info("✓ M12 模块初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger("M12_OOD_Test")
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 文件处理器
        log_file = self.config.log_dir / f"m12_ood_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_full_ood_test(self, num_projects: int = 20,
                         num_per_difficulty: int = 5,
                         use_mock_data: bool = True) -> Dict:
        """运行完整的 OOD 测试流程"""
        self.logger.info("="*70)
        self.logger.info("启动 M12 环境增强与 OOD 测试")
        self.logger.info("="*70)
        
        # 1. 生成测试项目数据
        if use_mock_data:
            projects = [
                self.scenario_generator._generate_mock_project()
                for _ in range(num_projects)
            ]
            self.logger.info(f"✓ 生成 {len(projects)} 个模拟项目")
        else:
            # 从真实数据加载（如果有）
            projects = []
            self.logger.warning("真实数据加载未实现，使用模拟数据")
        
        # 2. 生成 OOD 场景
        scenarios = self.scenario_generator.generate_scenario_batch(
            projects, num_per_difficulty
        )
        self.logger.info(f"✓ 生成 {len(scenarios)} 个 OOD 测试场景")
        
        # 3. 执行 OOD 测试
        self.logger.info("开始执行 OOD 测试...")
        test_results = self.test_executor.run_ood_test_batch(scenarios)
        self.logger.info("✓ OOD 测试执行完成")
        
        # 4. 生成报告
        self.logger.info("生成测试报告...")
        report = self.report_generator.generate_report(test_results, scenarios)
        
        # 5. 保存报告
        report_path = self.report_generator.save_report(report)
        self.logger.info(f"✓ 报告已保存: {report_path}")
        
        # 6. 打印摘要
        self.report_generator.print_summary(report)
        
        self.logger.info("="*70)
        self.logger.info("M12 环境增强与 OOD 测试完成")
        self.logger.info("="*70)
        
        return report
    
    def test_single_scenario(self, scenario: OODScenario) -> Dict:
        """测试单个场景（用于调试）"""
        result, trajectory = self.test_executor.run_ood_test(scenario)
        metrics = self.test_executor.resilience_evaluator.evaluate_scenario(
            scenario, trajectory
        )
        
        return {
            "result": result,
            "trajectory": trajectory,
            "metrics": metrics.to_dict()
        }


# =============================
# 入口点
# =============================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("M12 环境增强与 OOD 测试模块")
    print("="*70)
    print("\n功能说明:")
    print("1. 分布偏移生成（Drift Generation）")
    print("2. 黑天鹅场景模拟（Black Swan Simulation）")
    print("3. 高压环境测试（High-Pressure Testing）")
    print("4. 韧性评估（Resilience Evaluation）")
    print("\n集成模块:")
    print("- M3 仿真环境")
    print("- M5 自动化测试")
    print("- M6 状态管理")
    print("="*70 + "\n")
    
    # 创建 M12 模块实例
    m12_module = M12_OOD_Test_Module()
    
    # 运行完整测试
    report = m12_module.run_full_ood_test(
        num_projects=20,
        num_per_difficulty=5,
        use_mock_data=True
    )
    
    print(f"\n✓ M12 模块执行完成！")
    print(f"  - 测试场景数: {report['summary']['total_scenarios']}")
    print(f"  - 平均韧性: {report['summary']['avg_resilience']:.4f}")
    print(f"  - 报告ID: {report['report_id']}")
