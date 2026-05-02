#!/usr/bin/env python
# coding: utf-8
"""
M16 大规模实验与分析模块
从"单次仿真"走向"统计学意义上的稳健性评估"

核心功能：
1. 规模化仿真：不同参数配置下执行大规模实验
2. 假设验证：通过统计分析验证路由策略优劣
3. 归因分析：自动生成分析报告
4. 断点续传：支持中断后继续执行

依赖模块：
- M3: 仿真环境基座
- M7: AI路由策略
- M8: 风险规则判定
"""

import os
import sys
import json
import time
import uuid
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ======================== 路径配置 ========================
import sys
import io

# 修复Windows控制台输出编码
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# 导入M3仿真环境
gymnasium_available = False
try:
    import gymnasium as gym
    from gymnasium import spaces
    gymnasium_available = True
except ImportError:
    print("[WARN] gymnasium not installed, M3 simulation will use mock mode")

try:
    from M3仿真环境基座模块版 import StartupEnv
except ImportError as e:
    raise ImportError(
        "M16 requires M3 simulation environment. "
        "Please ensure 'M3仿真环境基座模块版.py' is in the scripts directory."
    ) from e

# 导入M8风险判定
try:
    from m8_rule_adapter import judge_project_risk_m8
except ImportError:
    judge_project_risk_m8 = None
    print("[INFO] M8 rule adapter not loaded, risk evaluation will use built-in rules")


# ======================== 日志配置 ========================
def setup_m16_logger(log_dir: str = None) -> logging.Logger:
    """配置M16专用日志器"""
    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, "logs", "m16_experiments")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("M16")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 文件处理器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"m16_run_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


# ======================== 数据类定义 ========================
@dataclass
class ExperimentConfig:
    """实验参数配置"""
    market_volatility: float  # 市场波动率: 0.1, 0.5, 1.0
    agent_routing_strategy: str  # 路由策略: random, rule-based, m7-agent-dynamic
    budget_range: str  # 预算范围: small, medium, large
    budget_value: float  # 预算金额（USD）
    seed: int  # 随机种子
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @staticmethod
    def from_dict(d: Dict) -> 'ExperimentConfig':
        return ExperimentConfig(**d)


@dataclass
class SingleExperimentResult:
    """单次实验结果"""
    experiment_id: str
    config: ExperimentConfig
    success: bool
    final_risk: float
    total_reward: float
    steps_completed: int
    roi: float  # 投资回报率
    final_state: List[float]
    execution_time_ms: float
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        result = {
            "experiment_id": self.experiment_id,
            "success": self.success,
            "final_risk": self.final_risk,
            "total_reward": self.total_reward,
            "steps_completed": self.steps_completed,
            "roi": self.roi,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
        }
        result.update(self.config.to_dict())
        return result


# ======================== 路由策略实现 ========================
class RoutingStrategy:
    """路由策略基类"""
    
    @staticmethod
    def select_action(env: StartupEnv, obs: np.ndarray, config: ExperimentConfig, 
                      step: int, rng: np.random.Generator) -> np.ndarray:
        """选择动作向量"""
        raise NotImplementedError


class RandomRouting(RoutingStrategy):
    """随机路由策略 - Baseline"""
    
    @staticmethod
    def select_action(env: StartupEnv, obs: np.ndarray, config: ExperimentConfig,
                      step: int, rng: np.random.Generator) -> np.ndarray:
        """随机生成动作"""
        # 在业务范围内生成随机动作
        noise_scale = config.market_volatility * 0.5
        noise = rng.normal(0, noise_scale, size=obs.shape).astype(np.float32)
        
        action_direction = np.array([
            -0.5,  # goal_ratio
            -0.5,  # time_penalty
            -0.4,  # category_risk
            -0.7,  # combined_risk
            -0.3,  # country_factor
            +0.6   # urgency_score
        ], dtype=np.float32)
        
        action = obs + action_direction * (0.1 + rng.random() * 0.3) + noise
        action = np.clip(action, env.feature_lows, env.feature_highs)
        return action


class RuleBasedRouting(RoutingStrategy):
    """基于规则的路由策略 - M8"""
    
    @staticmethod
    def select_action(env: StartupEnv, obs: np.ndarray, config: ExperimentConfig,
                      step: int, rng: np.random.Generator) -> np.ndarray:
        """基于M8规则生成动作"""
        risk_level = min(obs[3] / 20.0, 1.0)
        base_magnitude = 0.15 + (risk_level * 0.35)
        step_decay = 1.0 - (step / env.max_steps)
        magnitude = base_magnitude * step_decay
        
        # 加入市场波动影响
        volatility_factor = 1.0 + config.market_volatility * rng.uniform(-0.2, 0.2)
        magnitude *= volatility_factor
        
        action_direction = np.array([
            -0.8, -0.7, -0.6, -1.0, -0.5, +0.9
        ], dtype=np.float32)
        
        action = obs + action_direction * magnitude
        action = np.clip(action, env.feature_lows, env.feature_highs)
        return action


class M7DynamicRouting(RoutingStrategy):
    """M7动态智能路由策略 - 最优方案"""
    
    @staticmethod
    def select_action(env: StartupEnv, obs: np.ndarray, config: ExperimentConfig,
                      step: int, rng: np.random.Generator) -> np.ndarray:
        """M7动态智能路由"""
        # 基于当前状态自适应调整
        risk_level = min(obs[3] / 20.0, 1.0)
        urgency_level = obs[5] / 7.0
        
        # 动态计算动作幅度
        base_magnitude = 0.1 + (risk_level * 0.4)
        step_decay = 1.0 - (step / env.max_steps)
        magnitude = base_magnitude * step_decay
        
        # 市场波动自适应
        vol_adjusted = config.market_volatility
        if vol_adjusted > 0.5:
            magnitude *= 1.2  # 高波动环境下加大调整
        else:
            magnitude *= 0.9  # 低波动环境保守调整
        
        # 预算约束影响
        if config.budget_range == "small":
            magnitude *= 0.8  # 小预算保守
        elif config.budget_range == "large":
            magnitude *= 1.1  # 大预算激进
        
        # 动态方向权重
        action_direction = np.array([
            -0.8 - (urgency_level * 0.1),  # goal_ratio
            -0.7 - (urgency_level * 0.1),   # time_penalty
            -0.6,                            # category_risk
            -1.0 - (risk_level * 0.3),      # combined_risk (核心)
            -0.5,                            # country_factor
            +0.9 + (urgency_level * 0.2)    # urgency_score
        ], dtype=np.float32)
        
        action = obs + action_direction * magnitude
        
        # 添加随机扰动（探索）
        if rng.random() < 0.1:
            noise = rng.normal(0, 0.1 * vol_adjusted, size=obs.shape).astype(np.float32)
            action += noise
        
        action = np.clip(action, env.feature_lows, env.feature_highs)
        return action


# ======================== 策略注册表 ========================
ROUTING_STRATEGIES = {
    "random": RandomRouting,
    "rule-based": RuleBasedRouting,
    "m7-agent-dynamic": M7DynamicRouting
}


# ======================== 文本分析模块 ========================
# 导入文本转数据的核心模块
_text_analyzer_module = None
_intent_engine_module = None

try:
    from mas_blackboard.classifier import ComplexityClassifier
    _text_analyzer_module = True
except ImportError:
    ComplexityClassifier = None
    print("[INFO] ComplexityClassifier not loaded")

try:
    from m7.m7_intent_engine import recognize_intent, semantic_intent_matches
    _intent_engine_module = True
except ImportError:
    recognize_intent = None
    semantic_intent_matches = None
    print("[INFO] Intent engine not loaded")


@dataclass
class TextAnalysisResult:
    """文本分析结果"""
    raw_text: str
    text_length: int
    complexity_score: float
    tier: str
    intent: Dict[str, Any]
    semantic_matches: List[Dict[str, Any]]
    risk_keywords: Dict[str, List[str]]
    extracted_contacts: Dict[str, List[str]]
    recommended_experiment_config: Dict[str, Any]
    confidence: float
    analysis_timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            "raw_text": self.raw_text,
            "text_length": self.text_length,
            "complexity_score": self.complexity_score,
            "tier": self.tier,
            "intent": self.intent,
            "semantic_matches": self.semantic_matches,
            "risk_keywords": self.risk_keywords,
            "extracted_contacts": self.extracted_contacts,
            "recommended_experiment_config": self.recommended_experiment_config,
            "confidence": self.confidence,
            "analysis_timestamp": self.analysis_timestamp,
        }


class TextAnalyzer:
    """
    文本分析器 - 将用户输入的文本转化为结构化数据
    
    集成模块:
    - mas_blackboard.classifier: 复杂度分类
    - m7_intent_engine: 意图识别
    - 内置风险关键词提取
    """
    
    # 风险类别关键词映射
    RISK_KEYWORDS = {
        "现金流风险": {
            "keywords": ["现金流", "资金", "财务", "融资", "薪资", "支出", "回款", "营收", "支付", "冻结", "缺口"],
            "severity": 3
        },
        "知识产权风险": {
            "keywords": ["专利", "版权", "著作权", "商标", "侵权", "抄袭", "开源", "GPL", "GDPR", "知识产权"],
            "severity": 2
        },
        "合规/政策风险": {
            "keywords": ["合规", "政策", "出口管制", "VAT", "税务", "数据", "GDPR", "认证", "监管"],
            "severity": 2
        },
        "增长/市场风险": {
            "keywords": ["获客", "转化", "流失", "续费", "竞品", "市场", "营收", "增长", "客户"],
            "severity": 2
        },
        "团队/运营风险": {
            "keywords": ["离职", "团队", "宕机", "服务器", "外包", "bug", "客服", "招聘"],
            "severity": 2
        }
    }
    
    def __init__(self):
        self.classifier = None
        self._init_classifier()
    
    def _init_classifier(self):
        """初始化复杂度分类器"""
        if ComplexityClassifier is not None:
            try:
                self.classifier = ComplexityClassifier()
            except Exception as e:
                print(f"[WARN] Failed to init ComplexityClassifier: {e}")
                self.classifier = None
    
    def extract_contacts(self, text: str) -> Dict[str, List[str]]:
        """提取联系方式"""
        import re
        
        # 提取手机号
        phone_pattern = r'(?:手机号|电话)[：:]?\s*(\+?[\d\s\-\(\)]{7,20})'
        phones = re.findall(phone_pattern, text)
        
        # 提取邮箱
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text)
        
        return {
            "phones": [p.strip() for p in phones if p.strip()],
            "emails": list(set(emails))
        }
    
    def extract_risk_keywords(self, text: str) -> Dict[str, List[str]]:
        """提取风险关键词"""
        text_lower = text.lower()
        result = {}
        
        for category, info in self.RISK_KEYWORDS.items():
            matched = [kw for kw in info["keywords"] if kw in text_lower]
            if matched:
                result[category] = matched
        
        return result
    
    def calculate_risk_level(self, risk_keywords: Dict[str, List[str]]) -> Tuple[str, float, float]:
        """
        计算风险等级
        返回: (risk_level, severity_score, volatility)
        """
        total_keywords = sum(len(v) for v in risk_keywords.values())
        
        # 计算加权严重度
        weighted_severity = 0
        for category, keywords in risk_keywords.items():
            severity = self.RISK_KEYWORDS.get(category, {}).get("severity", 1)
            weighted_severity += severity * len(keywords)
        
        avg_severity = weighted_severity / max(total_keywords, 1)
        
        if total_keywords >= 15 or avg_severity >= 2.5:
            return "极高风险", 3.0, 1.0
        elif total_keywords >= 10 or avg_severity >= 2.0:
            return "高风险", 2.0, 0.8
        elif total_keywords >= 5 or avg_severity >= 1.5:
            return "中等风险", 1.5, 0.5
        else:
            return "低风险", 1.0, 0.2
    
    def get_recommended_config(self, text: str, tier: str, risk_level: str, 
                               volatility: float) -> Dict[str, Any]:
        """根据分析结果推荐实验配置"""
        text_lower = text.lower()
        
        # 确定路由策略
        if tier == "L3" or risk_level in ["极高风险", "高风险"]:
            strategy = "m7-agent-dynamic"
            budget = "large"
            budget_value = 80000
        elif tier == "L2" or risk_level == "中等风险":
            strategy = "rule-based"
            budget = "medium"
            budget_value = 40000
        else:
            strategy = "random"
            budget = "small"
            budget_value = 20000
        
        return {
            "market_volatility": volatility,
            "agent_routing_strategy": strategy,
            "budget_range": budget,
            "budget_value": budget_value,
            "suggested_tier": tier
        }
    
    def analyze(self, text: str) -> TextAnalysisResult:
        """
        执行完整的文本分析
        
        Args:
            text: 用户输入的文本
        
        Returns:
            TextAnalysisResult: 结构化的分析结果
        """
        timestamp = datetime.now().isoformat()
        
        # 1. 复杂度分类
        complexity_score = 5.0
        tier = "L2"
        confidence = 0.5
        
        if self.classifier is not None:
            try:
                decision = self.classifier.classify(text)
                complexity_score = decision.complexity_score
                tier = decision.tier
                # 尝试获取置信度，可能字段名不同
                confidence = getattr(decision, 'confidence', 
                           getattr(decision, 'confidence_score', 
                           getattr(decision, 'score', 0.5)))
            except Exception as e:
                print(f"[WARN] Classifier failed: {e}")
        
        # 2. 意图识别
        intent = {
            "primary_intent": "risk_assessment",
            "sub_intent": "general",
            "urgency": "medium",
            "confidence_score": confidence
        }
        semantic_matches = []
        
        if recognize_intent is not None:
            try:
                intent_result = recognize_intent(text)
                intent = {
                    "primary_intent": intent_result.get("primary_intent", "risk_assessment"),
                    "sub_intent": intent_result.get("sub_intent", "general"),
                    "urgency": intent_result.get("urgency", "medium"),
                    "confidence_score": intent_result.get("confidence_score", confidence)
                }
                semantic_matches = intent_result.get("semantic_matches", [])
            except Exception as e:
                print(f"[WARN] Intent recognition failed: {e}")
        
        # 3. 风险关键词提取
        risk_keywords = self.extract_risk_keywords(text)
        
        # 4. 风险等级评估
        risk_level, severity_score, volatility = self.calculate_risk_level(risk_keywords)
        
        # 5. 联系方式提取
        contacts = self.extract_contacts(text)
        
        # 6. 推荐实验配置
        recommended_config = self.get_recommended_config(
            text, tier, risk_level, volatility
        )
        
        return TextAnalysisResult(
            raw_text=text[:500],  # 截断保存
            text_length=len(text),
            complexity_score=complexity_score,
            tier=tier,
            intent=intent,
            semantic_matches=semantic_matches[:3],  # 只保留top 3
            risk_keywords=risk_keywords,
            extracted_contacts=contacts,
            recommended_experiment_config=recommended_config,
            confidence=confidence,
            analysis_timestamp=timestamp
        )
    
    def analyze_file(self, file_path: str) -> List[TextAnalysisResult]:
        """分析文件中的所有文本"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按行分段分析
        lines = content.split('\n')
        results = []
        current_text = ""
        
        for line in lines:
            if line.strip().startswith('# 【') or line.strip().startswith('# 【'):
                # 遇到新段落，先分析之前的文本
                if current_text.strip():
                    results.append(self.analyze(current_text))
                current_text = ""
            current_text += line + "\n"
        
        # 分析最后一段
        if current_text.strip():
            results.append(self.analyze(current_text))
        
        return results


# ======================== 仿真引擎 ========================
class SimulationEngine:
    """仿真引擎 - 执行单次实验"""
    
    def __init__(self, config: ExperimentConfig, timeout_seconds: int = 60):
        self.config = config
        self.timeout_seconds = timeout_seconds
    
    def run(self, experiment_id: str = None) -> SingleExperimentResult:
        """运行单次仿真实验"""
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        
        try:
            # 初始化环境
            if StartupEnv is None:
                raise RuntimeError("M3仿真环境未加载")
            
            env = StartupEnv()
            rng = np.random.default_rng(self.config.seed)
            
            # 生成初始特征向量（基于预算范围调整）
            initial_features = self._generate_initial_features(rng)
            
            # 重置环境
            obs, info = env.reset(options={"feature_vector": initial_features})
            
            # 仿真主循环
            done = False
            step = 0
            total_reward = 0.0
            prev_risk = None
            
            # 获取路由策略
            strategy_class = ROUTING_STRATEGIES.get(
                self.config.agent_routing_strategy, 
                RandomRouting
            )
            
            while not done and step < env.max_steps:
                step += 1
                
                # 选择动作
                action = strategy_class.select_action(
                    env, obs, self.config, step, rng
                )
                
                # 执行步进
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 计算奖励
                current_risk = float(obs[3])
                step_reward = self._calculate_reward(current_risk, prev_risk, terminated, truncated, step, env.max_steps)
                total_reward += step_reward
                prev_risk = current_risk
                
                # 判断终止
                done = terminated or truncated
            
            # 计算ROI（简化版）
            roi = self._calculate_roi(obs, self.config)
            
            execution_time = (time.time() - start_time) * 1000
            
            return SingleExperimentResult(
                experiment_id=experiment_id,
                config=self.config,
                success=True,
                final_risk=float(obs[3]),
                total_reward=total_reward,
                steps_completed=step,
                roi=roi,
                final_state=obs.tolist(),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return SingleExperimentResult(
                experiment_id=experiment_id,
                config=self.config,
                success=False,
                final_risk=-1,
                total_reward=0,
                steps_completed=0,
                roi=0,
                final_state=[],
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def _generate_initial_features(self, rng: np.random.Generator) -> np.ndarray:
        """生成初始特征向量"""
        # 基于预算范围和波动率生成
        budget_multiplier = {
            "small": 1.5,
            "medium": 2.5,
            "large": 4.0
        }.get(self.config.budget_range, 2.0)
        
        vol_factor = 1.0 + self.config.market_volatility * rng.uniform(-0.5, 0.5)
        
        base_features = np.array([
            budget_multiplier * vol_factor * rng.uniform(0.8, 2.0),  # goal_ratio
            rng.uniform(1.0, 2.5),  # time_penalty
            rng.uniform(0.2, 0.6),  # category_risk
            budget_multiplier * rng.uniform(5.0, 15.0),  # combined_risk
            rng.uniform(0.2, 0.5),  # country_factor
            rng.uniform(1.0, 3.0),   # urgency_score
        ], dtype=np.float32)
        
        return np.clip(base_features, [0, 0, 0, 0, 0, 0], [50, 20, 1, 100, 1, 7])
    
    def _calculate_reward(self, current_risk: float, prev_risk: Optional[float],
                          terminated: bool, truncated: bool, step: int, max_steps: int) -> float:
        """计算奖励"""
        reward = 0.5  # 存活奖励
        
        if prev_risk is not None:
            risk_change = prev_risk - current_risk
            reward += min(risk_change * 2.0, 2.0)
        
        if terminated:
            if current_risk > 20.0:
                reward -= 5.0
            elif current_risk < 0.1:
                reward += 2.0
        elif truncated:
            reward += 2.0
        
        return reward
    
    def _calculate_roi(self, final_state: np.ndarray, config: ExperimentConfig) -> float:
        """计算投资回报率"""
        initial_risk = config.budget_value * 0.1 if config.budget_value else 10.0
        final_risk = float(final_state[3])
        
        if initial_risk > 0:
            roi = (initial_risk - final_risk) / initial_risk * 100
        else:
            roi = 0.0
        
        return roi


# ======================== 实验参数生成器 ========================
class ExperimentDesign:
    """实验设计 - 参数空间构建"""
    
    # 参数空间定义
    MARKET_VOLATILITIES = [0.1, 0.5, 1.0]
    ROUTING_STRATEGIES = ["random", "rule-based", "m7-agent-dynamic"]
    BUDGET_RANGES = {
        "small": 5000,
        "medium": 50000,
        "large": 500000
    }
    
    @classmethod
    def generate_experiment_configs(cls, n_runs_per_config: int = 20,
                                    random_seeds: List[int] = None) -> List[ExperimentConfig]:
        """
        生成实验配置组合
        
        Args:
            n_runs_per_config: 每个参数组合的重复次数
            random_seeds: 随机种子列表
        
        Returns:
            实验配置列表
        """
        configs = []
        
        # 生成参数网格
        param_grid = list(itertools.product(
            cls.MARKET_VOLATILITIES,
            cls.ROUTING_STRATEGIES,
            list(cls.BUDGET_RANGES.keys())
        ))
        
        # 生成随机种子
        if random_seeds is None:
            random_seeds = list(range(n_runs_per_config))
        
        # 生成所有配置
        for volatility, strategy, budget in param_grid:
            for seed in random_seeds:
                config = ExperimentConfig(
                    market_volatility=volatility,
                    agent_routing_strategy=strategy,
                    budget_range=budget,
                    budget_value=cls.BUDGET_RANGES[budget],
                    seed=seed
                )
                configs.append(config)
        
        return configs
    
    @classmethod
    def get_config_summary(cls) -> Dict:
        """获取实验配置摘要"""
        n_strategies = len(cls.ROUTING_STRATEGIES)
        n_volatilities = len(cls.MARKET_VOLATILITIES)
        n_budgets = len(cls.BUDGET_RANGES)
        
        return {
            "volatilities": cls.MARKET_VOLATILITIES,
            "strategies": cls.ROUTING_STRATEGIES,
            "budgets": cls.BUDGET_RANGES,
            "param_combinations": n_volatilities * n_strategies * n_budgets
        }


# ======================== 检查点管理器 ========================
class CheckpointManager:
    """断点续传管理器"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pkl")
        self.results_file = os.path.join(checkpoint_dir, "partial_results.csv")
    
    def load_checkpoint(self) -> Tuple[Optional[int], List[Dict]]:
        """
        加载检查点
        
        Returns:
            (已完成数量, 结果列表)
        """
        if not os.path.exists(self.checkpoint_file):
            return None, []
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            completed_count = checkpoint.get("completed_count", 0)
            results = checkpoint.get("results", [])
            
            # 加载已保存的部分结果
            if os.path.exists(self.results_file):
                partial_df = pd.read_csv(self.results_file)
                results = partial_df.to_dict('records')
            
            print(f"📂 检查点加载成功: 已完成 {completed_count} 个实验")
            return completed_count, results
            
        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}")
            return None, []
    
    def save_checkpoint(self, completed_count: int, results: List[Dict]):
        """保存检查点"""
        checkpoint = {
            "completed_count": completed_count,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results)
        }
        
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # 保存部分结果
            if results:
                df = pd.DataFrame(results)
                df.to_csv(self.results_file, index=False)
                
        except Exception as e:
            print(f"⚠️ 检查点保存失败: {e}")
    
    def get_checkpoint_info(self) -> Optional[Dict]:
        """获取检查点信息"""
        if not os.path.exists(self.checkpoint_file):
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None


# ======================== M16实验运行器 ========================
class M16ExperimentRunner:
    """M16大规模实验运行器"""
    
    def __init__(self, 
                 output_dir: str = None,
                 checkpoint_dir: str = None,
                 n_workers: int = None,
                 batch_size: int = 50,
                 resume: bool = True):
        
        self.output_dir = output_dir or os.path.join(PROJECT_ROOT, "m16_results")
        self.checkpoint_dir = checkpoint_dir or os.path.join(self.output_dir, "checkpoints")
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.batch_size = batch_size
        self.resume = resume
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.logger = setup_m16_logger()
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        
        self.experiment_configs = []
        self.results = []
        self.start_time = None
        self.experiment_id = str(uuid.uuid4())[:8]
    
    def generate_tasks(self, n_runs_per_config: int = 20) -> int:
        """生成实验任务"""
        self.experiment_configs = ExperimentDesign.generate_experiment_configs(
            n_runs_per_config=n_runs_per_config
        )
        
        # 检查断点续传
        completed_count = 0
        if self.resume:
            completed_count, self.results = self.checkpoint_manager.load_checkpoint()
            if completed_count and completed_count > 0:
                # 过滤掉已完成的配置
                self.experiment_configs = self.experiment_configs[completed_count:]
        
        self.logger.info(f"任务生成完成: 共 {len(self.experiment_configs)} + {completed_count or 0} 个实验")
        return len(self.experiment_configs)
    
    def run_single_experiment(self, config: ExperimentConfig, exp_id: str) -> Dict:
        """运行单个实验（独立进程调用）"""
        engine = SimulationEngine(config)
        result = engine.run(experiment_id=exp_id)
        return result.to_dict()
    
    def run_parallel(self, progress_callback: Callable = None) -> List[Dict]:
        """并行运行所有实验"""
        if not self.experiment_configs:
            self.logger.warning("没有待执行的实验任务")
            return self.results
        
        self.start_time = time.time()
        total_tasks = len(self.experiment_configs)
        
        self.logger.info(f"🚀 开始并行执行 {total_tasks} 个实验 ({self.n_workers} 工作进程)")
        
        all_results = list(self.results)  # 保留之前的结果
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # 提交所有任务
            future_to_config = {}
            for i, config in enumerate(self.experiment_configs):
                exp_id = f"exp_{self.experiment_id}_{i:05d}"
                future = executor.submit(self.run_single_experiment, config, exp_id)
                future_to_config[future] = (i, config)
            
            # 处理完成的任务
            completed = 0
            batch_results = []
            
            for future in as_completed(future_to_config):
                try:
                    result = future.result(timeout=120)
                    all_results.append(result)
                    batch_results.append(result)
                    completed += 1
                    
                    # 进度回调
                    if progress_callback:
                        progress_callback(completed, total_tasks, result)
                    
                    # 批次保存检查点
                    if completed % self.batch_size == 0:
                        self.checkpoint_manager.save_checkpoint(completed, all_results)
                        elapsed = time.time() - self.start_time
                        eta = (elapsed / completed) * (total_tasks - completed)
                        self.logger.info(
                            f"进度: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%) | "
                            f"ETA: {eta/60:.1f}分钟"
                        )
                        
                except TimeoutError:
                    self.logger.error(f"实验超时: {future_to_config[future][1]}")
                except Exception as e:
                    self.logger.error(f"实验执行失败: {e}")
            
            # 最终保存
            self.checkpoint_manager.save_checkpoint(len(all_results), all_results)
        
        self.results = all_results
        elapsed = time.time() - self.start_time
        self.logger.info(f"✅ 全部实验完成! 耗时: {elapsed/60:.1f} 分钟")
        
        return self.results
    
    def aggregate_results(self, save: bool = True) -> pd.DataFrame:
        """合并所有实验结果"""
        if not self.results:
            raise ValueError("没有实验结果可合并")
        
        df = pd.DataFrame(self.results)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"m16_unified_results_{timestamp}.csv")
            df.to_csv(save_path, index=False)
            self.logger.info(f"结果已保存: {save_path}")
        
        return df
    
    def perform_statistical_analysis(self, df: pd.DataFrame = None) -> Dict:
        """
        执行统计分析
        
        Returns:
            统计分析结果字典
        """
        if df is None:
            df = pd.DataFrame(self.results)
        
        analysis = {
            "summary": {},
            "by_strategy": {},
            "by_volatility": {},
            "by_budget": {},
            "hypothesis_tests": {},
            "top_bottom": {}
        }
        
        # 1. 总体摘要
        analysis["summary"] = {
            "total_experiments": len(df),
            "success_rate": df["success"].mean() * 100,
            "mean_roi": df["roi"].mean(),
            "mean_reward": df["total_reward"].mean(),
            "mean_steps": df["steps_completed"].mean(),
            "std_roi": df["roi"].std(),
            "std_reward": df["total_reward"].std()
        }
        
        # 2. 按策略分组统计
        for strategy in ExperimentDesign.ROUTING_STRATEGIES:
            strategy_df = df[df["agent_routing_strategy"] == strategy]
            if len(strategy_df) > 0:
                analysis["by_strategy"][strategy] = {
                    "count": len(strategy_df),
                    "success_rate": strategy_df["success"].mean() * 100,
                    "mean_roi": strategy_df["roi"].mean(),
                    "std_roi": strategy_df["roi"].std(),
                    "mean_reward": strategy_df["total_reward"].mean(),
                    "std_reward": strategy_df["total_reward"].std(),
                    "mean_final_risk": strategy_df["final_risk"].mean()
                }
        
        # 3. 按波动率分组统计
        for vol in ExperimentDesign.MARKET_VOLATILITIES:
            vol_df = df[df["market_volatility"] == vol]
            if len(vol_df) > 0:
                analysis["by_volatility"][str(vol)] = {
                    "count": len(vol_df),
                    "success_rate": vol_df["success"].mean() * 100,
                    "mean_roi": vol_df["roi"].mean(),
                    "mean_reward": vol_df["total_reward"].mean()
                }
        
        # 4. 按预算分组统计
        for budget in ExperimentDesign.BUDGET_RANGES.keys():
            budget_df = df[df["budget_range"] == budget]
            if len(budget_df) > 0:
                analysis["by_budget"][budget] = {
                    "count": len(budget_df),
                    "success_rate": budget_df["success"].mean() * 100,
                    "mean_roi": budget_df["roi"].mean(),
                    "mean_reward": budget_df["total_reward"].mean()
                }
        
        # 5. 假设检验: M7 vs Random
        m7_results = df[df["agent_routing_strategy"] == "m7-agent-dynamic"]["roi"].values
        random_results = df[df["agent_routing_strategy"] == "random"]["roi"].values
        rule_results = df[df["agent_routing_strategy"] == "rule-based"]["roi"].values
        
        if len(m7_results) > 0 and len(random_results) > 0:
            # T检验
            t_stat, p_value = stats.ttest_ind(m7_results, random_results)
            analysis["hypothesis_tests"]["m7_vs_random"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "m7_mean": float(m7_results.mean()),
                "random_mean": float(random_results.mean()),
                "improvement": float((m7_results.mean() - random_results.mean()) / abs(random_results.mean()) * 100)
                    if random_results.mean() != 0 else 0
            }
        
        if len(m7_results) > 0 and len(rule_results) > 0:
            t_stat, p_value = stats.ttest_ind(m7_results, rule_results)
            analysis["hypothesis_tests"]["m7_vs_rule"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "m7_mean": float(m7_results.mean()),
                "rule_mean": float(rule_results.mean())
            }
        
        # 6. 置信区间 (95%)
        for strategy in ExperimentDesign.ROUTING_STRATEGIES:
            strategy_df = df[df["agent_routing_strategy"] == strategy]["roi"]
            if len(strategy_df) > 1:
                ci = stats.t.interval(0.95, len(strategy_df)-1, 
                                      loc=strategy_df.mean(), 
                                      scale=stats.sem(strategy_df))
                analysis["by_strategy"][strategy]["ci_95"] = {
                    "lower": float(ci[0]),
                    "upper": float(ci[1])
                }
        
        # 7. Top/Bottom 5% 分析
        n_top_bottom = max(1, int(len(df) * 0.05))
        
        top_5 = df.nlargest(n_top_bottom, "roi")
        bottom_5 = df.nsmallest(n_top_bottom, "roi")
        
        analysis["top_bottom"] = {
            "top_5_percent": {
                "count": len(top_5),
                "mean_config": {
                    "strategy": top_5["agent_routing_strategy"].mode().iloc[0] 
                        if len(top_5) > 0 else "N/A",
                    "volatility": top_5["market_volatility"].mean(),
                    "budget_range": top_5["budget_range"].mode().iloc[0]
                        if len(top_5) > 0 else "N/A"
                },
                "mean_roi": top_5["roi"].mean()
            },
            "bottom_5_percent": {
                "count": len(bottom_5),
                "mean_config": {
                    "strategy": bottom_5["agent_routing_strategy"].mode().iloc[0]
                        if len(bottom_5) > 0 else "N/A",
                    "volatility": bottom_5["market_volatility"].mean(),
                    "budget_range": bottom_5["budget_range"].mode().iloc[0]
                        if len(bottom_5) > 0 else "N/A"
                },
                "mean_roi": bottom_5["roi"].mean()
            }
        }
        
        return analysis
    
    def generate_report(self, df: pd.DataFrame = None, analysis: Dict = None,
                        output_format: str = "markdown") -> str:
        """生成实验报告"""
        if df is None:
            df = pd.DataFrame(self.results)
        if analysis is None:
            analysis = self.perform_statistical_analysis(df)
        
        if output_format == "markdown":
            return self._generate_markdown_report(df, analysis)
        elif output_format == "json":
            return json.dumps(analysis, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的报告格式: {output_format}")
    
    def _generate_markdown_report(self, df: pd.DataFrame, analysis: Dict) -> str:
        """生成Markdown报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# M16 大规模实验与分析报告

生成时间: {timestamp}
实验批次ID: {self.experiment_id}

---

## 1. 实验概述

| 指标 | 数值 |
|------|------|
| 总实验数 | {analysis['summary']['total_experiments']} |
| 总体成功率 | {analysis['summary']['success_rate']:.2f}% |
| 平均ROI | {analysis['summary']['mean_roi']:.2f}% |
| ROI标准差 | {analysis['summary']['std_roi']:.2f}% |
| 平均奖励 | {analysis['summary']['mean_reward']:.2f} |

---

## 2. 按路由策略分组统计

| 策略 | 样本数 | 成功率 | 平均ROI | ROI标准差 | 95%置信区间 |
|------|--------|--------|---------|-----------|------------|
"""
        
        for strategy, stats_dict in analysis["by_strategy"].items():
            ci = stats_dict.get("ci_95", {"lower": "N/A", "upper": "N/A"})
            ci_str = f"[{ci['lower']:.2f}, {ci['upper']:.2f}]" if isinstance(ci.get('lower'), float) else "N/A"
            
            report += f"| {strategy} | {stats_dict['count']} | {stats_dict['success_rate']:.1f}% | "
            report += f"{stats_dict['mean_roi']:.2f}% | {stats_dict['std_roi']:.2f}% | {ci_str} |\n"
        
        report += f"""

---

## 3. 按市场波动率分组统计

| 波动率 | 样本数 | 成功率 | 平均ROI | 平均奖励 |
|--------|--------|--------|---------|----------|
"""
        
        for vol, stats_dict in analysis["by_volatility"].items():
            report += f"| {vol} | {stats_dict['count']} | {stats_dict['success_rate']:.1f}% | "
            report += f"{stats_dict['mean_roi']:.2f}% | {stats_dict['mean_reward']:.2f} |\n"
        
        # 安全格式化函数
        def safe_fmt(val, default='N/A', fmt='.2f'):
            """安全格式化数字"""
            if val is None or val == default:
                return str(default)
            try:
                return f"{float(val):{fmt}}"
            except (ValueError, TypeError):
                return str(val)
        
        m7_random = analysis['hypothesis_tests'].get('m7_vs_random', {})
        m7_rule = analysis['hypothesis_tests'].get('m7_vs_rule', {})
        
        report += f"""

---

## 4. 按预算范围分组统计

| 预算范围 | 预算金额 | 样本数 | 成功率 | 平均ROI |
|----------|----------|--------|--------|---------|
| small | $5,000 | {analysis['by_budget'].get('small', {}).get('count', 0)} | {analysis['by_budget'].get('small', {}).get('success_rate', 0):.1f}% | {analysis['by_budget'].get('small', {}).get('mean_roi', 0):.2f}% |
| medium | $50,000 | {analysis['by_budget'].get('medium', {}).get('count', 0)} | {analysis['by_budget'].get('medium', {}).get('success_rate', 0):.1f}% | {analysis['by_budget'].get('medium', {}).get('mean_roi', 0):.2f}% |
| large | $500,000 | {analysis['by_budget'].get('large', {}).get('count', 0)} | {analysis['by_budget'].get('large', {}).get('success_rate', 0):.1f}% | {analysis['by_budget'].get('large', {}).get('mean_roi', 0):.2f}% |

---

## 5. 假设检验结果

### M7动态路由 vs 随机路由

| 指标 | 数值 |
|------|------|
| M7平均ROI | {safe_fmt(m7_random.get('m7_mean'))}% |
| 随机路由平均ROI | {safe_fmt(m7_random.get('random_mean'))}% |
| T统计量 | {safe_fmt(m7_random.get('t_statistic'), 'N/A', '.4f')} |
| P值 | {safe_fmt(m7_random.get('p_value'), 'N/A', '.6f')} |
| 显著性 (α=0.05) | {'是' if m7_random.get('significant', False) else '否'} |
| 性能提升 | {safe_fmt(m7_random.get('improvement'), '0', '.1f')}% |

### M7动态路由 vs 规则路由

| 指标 | 数值 |
|------|------|
| M7平均ROI | {safe_fmt(m7_rule.get('m7_mean'))}% |
| 规则路由平均ROI | {safe_fmt(m7_rule.get('rule_mean'))}% |
| T统计量 | {safe_fmt(m7_rule.get('t_statistic'), 'N/A', '.4f')} |
| P值 | {safe_fmt(m7_rule.get('p_value'), 'N/A', '.6f')} |
| 显著性 (α=0.05) | {'是' if m7_rule.get('significant', False) else '否'} |

---

## 6. 异常样本分析

### Top 5% 最佳表现样本

- 样本数: {analysis['top_bottom']['top_5_percent']['count']}
- 平均ROI: {analysis['top_bottom']['top_5_percent']['mean_roi']:.2f}%
- 主要策略: {analysis['top_bottom']['top_5_percent']['mean_config']['strategy']}
- 平均波动率: {analysis['top_bottom']['top_5_percent']['mean_config']['volatility']:.2f}

### Bottom 5% 最差表现样本

- 样本数: {analysis['top_bottom']['bottom_5_percent']['count']}
- 平均ROI: {analysis['top_bottom']['bottom_5_percent']['mean_roi']:.2f}%
- 主要策略: {analysis['top_bottom']['bottom_5_percent']['mean_config']['strategy']}
- 平均波动率: {analysis['top_bottom']['bottom_5_percent']['mean_config']['volatility']:.2f}

---

## 7. 核心结论

"""
        
        # 自动生成结论
        m7_vs_random = analysis['hypothesis_tests'].get('m7_vs_random', {})
        if m7_vs_random.get('significant', False):
            improvement = m7_vs_random.get('improvement', 0)
            report += f"""✅ **M7动态路由策略显著优于随机路由**

在95%置信度下，M7动态路由策略相比随机路由策略，ROI提升了 **{improvement:.1f}%**。
该差异具有统计学意义（P值 < 0.05）。

"""
        else:
            report += """⚠️ **M7动态路由策略与随机路由策略无显著差异**

统计检验显示两种策略的表现差异不显著。建议进一步分析实验配置和参数空间。

"""
        
        # 保存报告
        report_path = os.path.join(self.output_dir, f"m16_report_{self.experiment_id}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"报告已保存: {report_path}")
        
        return report
    
    def generate_visualizations(self, df: pd.DataFrame = None, output_dir: str = None) -> Dict[str, str]:
        """生成可视化图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
        except ImportError:
            self.logger.warning("matplotlib未安装，跳过可视化生成")
            return {}
        
        if df is None:
            df = pd.DataFrame(self.results)
        
        output_dir = output_dir or self.output_dir
        plots = {}
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. ROI箱线图 - 按策略
        fig, ax = plt.subplots(figsize=(10, 6))
        strategies = ExperimentDesign.ROUTING_STRATEGIES
        data = [df[df["agent_routing_strategy"] == s]["roi"].values for s in strategies]
        bp = ax.boxplot(data, labels=strategies, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xlabel('Routing Strategy')
        ax.set_ylabel('ROI (%)')
        ax.set_title('ROI Distribution by Routing Strategy')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        boxplot_path = os.path.join(output_dir, 'roi_boxplot_by_strategy.png')
        plt.savefig(boxplot_path, dpi=150)
        plt.close()
        plots['roi_boxplot'] = boxplot_path
        
        # 2. 成功率柱状图 - 按策略和波动率
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(ExperimentDesign.MARKET_VOLATILITIES))
        width = 0.25
        for i, strategy in enumerate(ExperimentDesign.ROUTING_STRATEGIES):
            success_rates = []
            for vol in ExperimentDesign.MARKET_VOLATILITIES:
                mask = (df["agent_routing_strategy"] == strategy) & (df["market_volatility"] == vol)
                rate = df[mask]["success"].mean() * 100 if mask.sum() > 0 else 0
                success_rates.append(rate)
            ax.bar(x + i * width, success_rates, width, label=strategy)
        ax.set_xlabel('Market Volatility')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Strategy and Volatility')
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(v) for v in ExperimentDesign.MARKET_VOLATILITIES])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        barplot_path = os.path.join(output_dir, 'success_rate_by_strategy_vol.png')
        plt.savefig(barplot_path, dpi=150)
        plt.close()
        plots['success_bar'] = barplot_path
        
        # 3. ROI分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy, color in zip(ExperimentDesign.ROUTING_STRATEGIES, colors):
            mask = df["agent_routing_strategy"] == strategy
            if mask.sum() > 0:
                ax.hist(df[mask]["roi"], bins=30, alpha=0.6, label=strategy, color=color)
        ax.set_xlabel('ROI (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('ROI Distribution Histogram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = os.path.join(output_dir, 'roi_histogram.png')
        plt.savefig(hist_path, dpi=150)
        plt.close()
        plots['roi_hist'] = hist_path
        
        # 4. 热力图 - 策略 vs 波动率
        try:
            pivot_data = df.pivot_table(
                values='roi', 
                index='agent_routing_strategy', 
                columns='market_volatility', 
                aggfunc='mean'
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_xticklabels(pivot_data.columns)
            ax.set_yticklabels(pivot_data.index)
            ax.set_xlabel('Market Volatility')
            ax.set_ylabel('Routing Strategy')
            ax.set_title('Mean ROI Heatmap')
            plt.colorbar(im, ax=ax, label='ROI (%)')
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    text = ax.text(j, i, f'{pivot_data.values[i, j]:.1f}',
                                   ha="center", va="center", color="black")
            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, 'roi_heatmap.png')
            plt.savefig(heatmap_path, dpi=150)
            plt.close()
            plots['roi_heatmap'] = heatmap_path
        except Exception as e:
            self.logger.warning(f"热力图生成失败: {e}")
        
        self.logger.info(f"可视化图表已保存至: {output_dir}")
        
        return plots


# ======================== 主程序入口 ========================
def run_m16_experiments(
    n_runs_per_config: int = 20,
    n_workers: int = None,
    batch_size: int = 50,
    resume: bool = True,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict, str]:
    """
    运行M16大规模实验的主入口函数
    
    Args:
        n_runs_per_config: 每个参数组合的重复次数
        n_workers: 并行工作进程数
        batch_size: 批次大小（用于检查点保存）
        resume: 是否从检查点恢复
        output_dir: 输出目录
    
    Returns:
        (results_df, analysis, report)
    """
    print("=" * 60)
    print("M16 大规模实验与分析系统")
    print("=" * 60)
    
    # 初始化运行器
    runner = M16ExperimentRunner(
        output_dir=output_dir,
        n_workers=n_workers,
        batch_size=batch_size,
        resume=resume
    )
    
    # 显示实验设计摘要
    config_summary = ExperimentDesign.get_config_summary()
    print(f"\n📊 实验设计摘要:")
    print(f"   - 市场波动率: {config_summary['volatilities']}")
    print(f"   - 路由策略: {config_summary['strategies']}")
    print(f"   - 预算范围: {list(config_summary['budgets'].keys())}")
    print(f"   - 参数组合数: {config_summary['param_combinations']}")
    print(f"   - 每组合重复: {n_runs_per_config} 次")
    
    # 生成任务
    n_tasks = runner.generate_tasks(n_runs_per_config=n_runs_per_config)
    print(f"\n📋 任务生成完成: {n_tasks} 个待执行实验")
    
    # 定义进度回调
    def progress_callback(completed, total, result):
        if completed % 100 == 0 or completed == total:
            status = "✅" if result.get("success") else "❌"
            print(f"   [{completed}/{total}] {status} {result.get('experiment_id')}")
    
    # 运行实验
    print(f"\n🚀 开始执行实验 (使用 {runner.n_workers} 个工作进程)...")
    runner.run_parallel(progress_callback=progress_callback)
    
    # 合并结果
    print("\n📊 合并实验结果...")
    df = runner.aggregate_results()
    
    # 统计分析
    print("📈 执行统计分析...")
    analysis = runner.perform_statistical_analysis(df)
    
    # 生成报告
    print("📝 生成实验报告...")
    report = runner.generate_report(df, analysis)
    
    # 生成可视化
    print("📊 生成可视化图表...")
    plots = runner.generate_visualizations(df)
    
    print("\n" + "=" * 60)
    print("✅ M16 实验完成!")
    print("=" * 60)
    
    return df, analysis, report


# ======================== 快速测试入口 ========================
def quick_test(n_experiments: int = 10):
    """快速测试模式 - 运行少量实验验证流程"""
    print(f"🧪 快速测试模式: 运行 {n_experiments} 个实验")
    
    runner = M16ExperimentRunner(output_dir=os.path.join(PROJECT_ROOT, "m16_test_results"))
    
    # 生成少量配置
    test_configs = []
    for i in range(min(n_experiments, 9)):
        from itertools import product
        vols = ExperimentDesign.MARKET_VOLATILITIES[:2]
        strats = ExperimentDesign.ROUTING_STRATEGIES[:2]
        budgets = list(ExperimentDesign.BUDGET_RANGES.keys())[:1]
        
        for vol, strat, budget in list(product(vols, strats, budgets))[:n_experiments]:
            config = ExperimentConfig(
                market_volatility=vol,
                agent_routing_strategy=strat,
                budget_range=budget,
                budget_value=ExperimentDesign.BUDGET_RANGES[budget],
                seed=i
            )
            test_configs.append(config)
            if len(test_configs) >= n_experiments:
                break
        if len(test_configs) >= n_experiments:
            break
    
    runner.experiment_configs = test_configs
    
    def progress_callback(completed, total, result):
        print(f"   [{completed}/{total}] 完成")
    
    runner.run_parallel(progress_callback=progress_callback)
    
    if runner.results:
        df = runner.aggregate_results()
        analysis = runner.perform_statistical_analysis(df)
        report = runner.generate_report(df, analysis)
        print("\n" + report)
        return df, analysis, report
    
    return None, None, None


# ======================== 文本分析入口 ========================
def analyze_text_input(
    text: str = None,
    input_file: str = None,
    output_file: str = None
) -> List[TextAnalysisResult]:
    """
    文本分析入口函数 - 将用户文本转化为结构化数据
    
    Args:
        text: 要分析的文本字符串
        input_file: 要分析的文本文件路径
        output_file: JSON输出文件路径
    
    Returns:
        List[TextAnalysisResult]: 分析结果列表
    """
    print("=" * 60)
    print("M16 文本分析系统")
    print("=" * 60)
    
    analyzer = TextAnalyzer()
    results = []
    
    if input_file:
        # 从文件读取
        print(f"\n📂 读取文件: {input_file}")
        if os.path.exists(input_file):
            results = analyzer.analyze_file(input_file)
            print(f"✓ 完成 {len(results)} 个文本段落的分析")
        else:
            print(f"❌ 文件不存在: {input_file}")
            return []
    
    elif text:
        # 分析单个文本
        print(f"\n📝 分析文本 ({len(text)} 字符)")
        result = analyzer.analyze(text)
        results = [result]
        print(f"✓ 分析完成")
    else:
        print("\n⚠️ 未提供文本或文件，请使用 --text 或 --file 参数")
        return []
    
    # 输出结果摘要
    print("\n" + "-" * 60)
    print("📊 分析结果摘要")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n【段落 {i}】")
        print(f"  复杂度评分: {result.complexity_score:.2f} (Tier: {result.tier})")
        print(f"  风险等级: {result.risk_keywords}")
        print(f"  推荐策略: {result.recommended_experiment_config['agent_routing_strategy']}")
        print(f"  置信度: {result.confidence:.2f}")
    
    # 保存JSON结果
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        output_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_segments": len(results),
            "results": [r.to_dict() for r in results]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 结果已保存至: {output_file}")
    
    return results


# ======================== 脚本入口 ========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="M16 大规模实验与分析系统")
    parser.add_argument("--runs", type=int, default=20, help="每个参数组合的重复次数")
    parser.add_argument("--workers", type=int, default=None, help="并行工作进程数")
    parser.add_argument("--batch", type=int, default=50, help="批次大小")
    parser.add_argument("--no-resume", action="store_true", help="禁用断点续传")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--test", action="store_true", help="快速测试模式")
    parser.add_argument("--test-n", type=int, default=10, help="测试模式实验数")
    
    # 文本分析参数
    parser.add_argument("--analyze", action="store_true", help="启用文本分析模式")
    parser.add_argument("--text", type=str, default=None, help="要分析的文本内容")
    parser.add_argument("--file", type=str, default=None, help="要分析的文本文件路径")
    parser.add_argument("--output-json", type=str, default=None, help="JSON输出文件路径")
    
    args = parser.parse_args()
    
    # 文本分析模式
    if args.analyze or args.text or args.file:
        analyze_text_input(
            text=args.text,
            input_file=args.file,
            output_file=args.output_json
        )
    elif args.test:
        quick_test(n_experiments=args.test_n)
    else:
        run_m16_experiments(
            n_runs_per_config=args.runs,
            n_workers=args.workers,
            batch_size=args.batch,
            resume=not args.no_resume,
            output_dir=args.output
        )
