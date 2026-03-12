"""
M5 测试套件配置文件
====================
管理所有测试参数、阈值和规则
支持不同环境的配置切换
"""

from enum import Enum
from typing import Dict, Any


class Environment(str, Enum):
    """执行环境"""
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class TestConfig:
    """测试配置基类"""
    
    def __init__(self, env: Environment = Environment.DEV):
        self.env = env
        self._init_config()
    
    def _init_config(self):
        """初始化配置"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class M5TestConfig(TestConfig):
    """M5测试配置"""
    
    def _init_config(self):
        # ============ 通用配置 ============
        self.project_id = "Kickstarter_AutoTest"
        self.output_dir = "test_reports"
        self.log_dir = "logs"
        
        # ============ 测试用例生成 ============
        self.num_synthetic_cases = 20 if self.env != Environment.PROD else 50
        self.data_sample_size = 50 if self.env != Environment.PROD else 200
        self.stratify_by_category = True
        self.seed = 42  # 保证可复现性
        
        # ============ 执行参数 ============
        self.enable_parallel = True
        self.num_workers = 2 if self.env == Environment.DEV else 4
        self.timeout_per_case = 30.0  # 秒
        self.max_retries = 2
        
        # ============ 风险判定阈值 ============
        self.risk_thresholds = {
            "critical": 6.0,
            "high": 4.5,
            "medium": 3.0,
            "low": 1.0
        }
        
        # ============ 警告阈值 ============
        self.warning_config = {
            "high_failure_rate_threshold": 0.2,  # 20%
            "critical_risk_alert_enabled": True,
            "execution_timeout_alert": 60.0,  # 秒
        }
        
        # ============ 环境特定配置 ============
        if self.env == Environment.DEV:
            self.enable_parallel = False
            self.num_workers = 1
            self.num_synthetic_cases = 5
        elif self.env == Environment.TEST:
            self.num_workers = 4
            self.num_synthetic_cases = 20
        elif self.env == Environment.PROD:
            self.enable_parallel = True
            self.num_workers = 8
            self.num_synthetic_cases = 100


class RiskJudgmentConfig(TestConfig):
    """M8风险判定配置"""
    
    def _init_config(self):
        # ============ 高失败品类 ============
        self.high_fail_categories = ['Journalism', 'Food', 'Crafts']
        
        # ============ 高风险阈值 ============
        self.high_risk_threshold = 5.5
        self.precision_compensate_threshold = 6.3
        
        # ============ 特征权重 ============
        self.feature_weights = {
            'goal_ratio': 0.36,
            'time_penalty': 0.30,
            'category_risk': 0.19,
            'country_factor': 0.09,
            'urgency_score': 0.05,
            'combined_risk': 0.95
        }
        
        # ============ 场景约束阈值 ============
        self.scenario_thresholds = {
            'goal_ratio': 1.8,
            'time_penalty': 2.4,
            'category_risk': 0.32,
            'country_factor': 0.26,
            'urgency_score': 0.10
        }
        
        # ============ 特征系数 ============
        self.feature_coefficients = {
            'goal_ratio': 7,
            'time_penalty': 7,
            'category_risk': 6,
            'country_factor': 6,
            'urgency_score': 1
        }
        
        # ============ 融合比例 ============
        self.normalized_ratio = 0.8
        self.raw_sum_ratio = 0.2


class SimulationConfig(TestConfig):
    """M3仿真配置"""
    
    def _init_config(self):
        # ============ 环境参数 ============
        self.max_steps = 100
        self.episode_timeout = 300.0  # 秒
        
        # ============ 特征维度 ============
        self.num_features = 30
        self.observation_space = "continuous"
        
        # ============ 奖励函数 ============
        self.reward_success = 1.0
        self.reward_failure = -1.0
        self.reward_timeout = -0.5
        
        # ============ 随机性 ============
        self.random_seed = 42
        self.noise_level = 0.01


class DataProcessingConfig(TestConfig):
    """M2数据处理配置"""
    
    def _init_config(self):
        # ============ 数据路径 ============
        self.raw_data_dir = "Kickstarter_2025-12-18T03_20_24_296Z"
        self.processed_data_dir = "Kickstarter_Clean"
        
        # ============ 数据清洗参数 ============
        self.handle_missing = "drop"  # drop, mean, forward_fill
        self.outlier_method = "iqr"   # iqr, zscore
        self.outlier_threshold = 3.0
        
        # ============ 特征工程 ============
        self.normalize_features = True
        self.scale_method = "standard"  # standard, minmax
        self.feature_selection = True
        self.num_selected_features = 25
        
        # ============ 分层参数 ============
        self.stratify_column = "main_category"
        self.test_size = 0.2
        self.random_state = 42


class ReportConfig(TestConfig):
    """报告生成配置"""
    
    def _init_config(self):
        # ============ 报告格式 ============
        self.export_formats = ["json", "html", "csv"]
        self.include_charts = True
        self.include_metrics = True
        self.include_warnings = True
        
        # ============ 报告详细程度 ============
        self.detail_level = "standard"  # minimal, standard, detailed
        self.max_cases_in_summary = 10
        self.include_full_trajectories = False
        
        # ============ 保存配置 ============
        self.auto_save = True
        self.archive_old_reports = True
        self.max_archived_reports = 100


class AlertConfig(TestConfig):
    """告警系统配置"""
    
    def _init_config(self):
        # ============ 告警级别 ============
        self.alert_levels = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "info": 0
        }
        
        # ============ 告警规则 ============
        self.alert_rules = {
            "high_failure_rate": {
                "enabled": True,
                "threshold": 0.2,
                "level": "high"
            },
            "critical_risk_detected": {
                "enabled": True,
                "threshold": 1,  # 至少1个critical risk
                "level": "high"
            },
            "execution_timeout": {
                "enabled": True,
                "threshold": 60.0,
                "level": "medium"
            },
            "anomaly_detected": {
                "enabled": True,
                "threshold": 2,
                "level": "medium"
            }
        }
        
        # ============ 通知配置 ============
        self.enable_notifications = False
        self.notification_channels = ["console", "log"]  # console, log, email, slack
        self.email_recipients = []
        self.slack_webhook = ""


# ============ 全局配置管理器 ============

class ConfigManager:
    """配置管理器 - 单例模式"""
    
    _instance = None
    _configs = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._configs:
            self._init_default_configs()
    
    def _init_default_configs(self):
        """初始化默认配置"""
        env = Environment.DEV
        
        self._configs = {
            "m5_test": M5TestConfig(env),
            "risk_judgment": RiskJudgmentConfig(env),
            "simulation": SimulationConfig(env),
            "data_processing": DataProcessingConfig(env),
            "report": ReportConfig(env),
            "alert": AlertConfig(env),
        }
    
    def get_config(self, config_name: str) -> TestConfig:
        """获取配置对象"""
        return self._configs.get(config_name)
    
    def set_environment(self, env: Environment):
        """切换环境"""
        for config_name in self._configs:
            self._configs[config_name] = eval(
                f"{self._configs[config_name].__class__.__name__}(env)"
            )
    
    def get_all_configs(self) -> Dict[str, TestConfig]:
        """获取所有配置"""
        return self._configs
    
    def print_config(self, config_name: str = None):
        """打印配置"""
        if config_name:
            config = self.get_config(config_name)
            if config:
                print(f"\n{config.__class__.__name__}:")
                for k, v in config.to_dict().items():
                    print(f"  {k}: {v}")
        else:
            for name, config in self._configs.items():
                self.print_config(name)


# ============ 快速访问函数 ============

def get_m5_config() -> M5TestConfig:
    """获取M5测试配置"""
    return ConfigManager().get_config("m5_test")


def get_risk_config() -> RiskJudgmentConfig:
    """获取风险判定配置"""
    return ConfigManager().get_config("risk_judgment")


def get_simulation_config() -> SimulationConfig:
    """获取仿真配置"""
    return ConfigManager().get_config("simulation")


def set_test_environment(env: Environment):
    """设置测试环境"""
    ConfigManager().set_environment(env)


# ============ 示例使用 ============

if __name__ == "__main__":
    # 切换到生产环境
    set_test_environment(Environment.PROD)
    
    # 获取配置
    m5_cfg = get_m5_config()
    print(f"项目ID: {m5_cfg.project_id}")
    print(f"Worker数: {m5_cfg.num_workers}")
    print(f"合成用例数: {m5_cfg.num_synthetic_cases}")
    
    # 打印所有配置
    print("\n全局配置:")
    ConfigManager().print_config()
