#!/usr/bin/env python
# coding: utf-8

"""
M10 性能监控与优化模块
==========================
功能：监控与优化性能，Token消耗分析与瓶颈定位，输出Cost Report
主要职责：
1. 输入：初步仿真日志流
2. 处理：Token消耗分析与瓶颈定位  
3. 输出：Cost Report，优化存储结构

集成模块：M3/M5/M6/M8日志系统
作者：性能监控引擎
日期：2025-03-10
"""

import os
import sys
import json
import time
import threading
import psutil
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import pandas as pd
import numpy as np


# ============ 导入项目其他模块 ============

try:
    # 尝试导入M3仿真环境
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from M3仿真环境基座模块版 import StartupEnv, save_simulation_results
except:
    print("⚠️ M3 unavailable, using mock performance monitoring")

try:
    # 尝试导入M6状态管理
    from M6状态管理与日志系统 import EnvState, EnvEventRecord
except:
    print("⚠️ M6 unavailable, using simplified event structure")


# ============ 枚举定义 ============

class PerformanceLevel(str, Enum):
    """性能级别枚举"""
    EXCELLENT = "excellent"      # < 70% utilization
    GOOD = "good"                # 70-85% utilization
    WARNING = "warning"          # 85-95% utilization
    CRITICAL = "critical"        # > 95% utilization


class ResourceType(str, Enum):
    """资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    TOKEN = "token"              # AI token消耗
    API_CALL = "api_call"        # API调用次数


class BottleneckType(str, Enum):
    """瓶颈类型枚举"""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    MODEL_LATENCY = "model_latency"
    DATA_LOADING = "data_loading"


# ============ 数据类定义 ============

@dataclass
class TokenConsumption:
    """Token消耗记录"""
    timestamp: float
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    api_latency: float  # API调用延迟（秒）
    context: Dict[str, Any] = field(default_factory=dict)  # 调用上下文
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResourceUsage:
    """资源使用记录"""
    timestamp: float
    cpu_percent: float            # CPU使用率
    memory_mb: float              # 内存使用（MB）
    memory_percent: float         # 内存使用率
    disk_io_read: float           # 磁盘读（MB/s）
    disk_io_write: float          # 磁盘写（MB/s）
    network_sent: float           # 网络发送（MB/s）
    network_received: float       # 网络接收（MB/s）
    open_files: int               # 打开文件数
    thread_count: int             # 线程数
    process_memory_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PerformanceMetric:
    """性能度量指标"""
    metric_name: str
    value: float
    unit: str
    timestamp: float
    level: PerformanceLevel
    threshold: float              # 阈值
    description: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BottleneckAnalysis:
    """瓶颈分析结果"""
    bottleneck_type: BottleneckType
    location: str                  # 瓶颈位置（文件:行号或函数名）
    severity: float               # 严重程度 0-1
    duration: float               # 持续时间（秒）
    samples: int                  # 采样次数
    improvement_suggestions: List[str] = field(default_factory=list)
    affected_operations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CostReport:
    """成本报告"""
    report_id: str
    start_time: float
    end_time: float
    duration: float
    total_token_cost: float
    api_call_cost: float
    compute_cost: float
    storage_cost: float
    total_cost: float
    token_breakdown: Dict[str, int]          # 按模型统计的token数
    cost_breakdown: Dict[str, float]         # 成本细分
    performance_metrics: List[PerformanceMetric] = field(default_factory=list)
    bottlenecks: List[BottleneckAnalysis] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    storage_optimization: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        report_dict = asdict(self)
        # 处理性能度量和瓶颈列表
        report_dict["performance_metrics"] = [m.to_dict() for m in self.performance_metrics]
        report_dict["bottlenecks"] = [b.to_dict() for b in self.bottlenecks]
        return report_dict


# ============ 日志流处理器 ============

class LogStreamProcessor:
    """仿真日志流处理器"""
    
    def __init__(self, log_dir: str = "logs", buffer_size: int = 1000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.event_buffer = deque(maxlen=buffer_size)
        self.token_records: List[TokenConsumption] = []
        self.resource_records: List[ResourceUsage] = []
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            "total_events": 0,
            "total_tokens": 0,
            "total_api_calls": 0,
            "peak_memory_mb": 0,
            "peak_cpu_percent": 0,
        }
        
        # 模式匹配模式
        self.token_patterns = [
            "prompt_tokens", "completion_tokens", "total_tokens",
            "token", "llm", "gpt", "model", "inference"
        ]
        
        self.api_patterns = [
            "api_call", "request", "response", "latency",
            "endpoint", "query", "invoke"
        ]
    
    def process_log_stream(self, log_file_path: str) -> Dict[str, Any]:
        """处理日志文件流"""
        print(f"📊 处理日志流: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            print(f"⚠️ 日志文件不存在: {log_file_path}")
            return {"status": "error", "message": "Log file not found"}
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines_processed = 0
                for line_num, line in enumerate(f, 1):
                    self._process_log_line(line.strip(), line_num)
                    lines_processed += 1
                    
                    # 每100行输出进度
                    if line_num % 100 == 0:
                        print(f"  已处理 {line_num} 行...")
            
            print(f"✓ 日志处理完成: {lines_processed} 行")
            
            return {
                "status": "success",
                "lines_processed": lines_processed,
                "events_captured": len(self.event_buffer),
                "token_records": len(self.token_records),
                "resource_records": len(self.resource_records),
                "stats": self.stats,
            }
            
        except Exception as e:
            print(f"✗ 日志处理失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _process_log_line(self, line: str, line_num: int):
        """处理单行日志"""
        try:
            # 尝试解析为JSON
            if line.startswith("{") and line.endswith("}"):
                event = json.loads(line)
                self._process_json_event(event, line_num)
            else:
                # 文本日志分析
                self._process_text_line(line, line_num)
                
        except json.JSONDecodeError:
            # 不是JSON，作为文本处理
            self._process_text_line(line, line_num)
        except Exception as e:
            print(f"日志行 {line_num} 处理异常: {e}")
    
    def _process_json_event(self, event: Dict, line_num: int):
        """处理JSON事件"""
        event_type = event.get("event_type", "unknown")
        timestamp = event.get("timestamp", time.time())
        
        # 检查是否为token相关事件
        if any(pattern in str(event).lower() for pattern in self.token_patterns):
            token_record = self._extract_token_consumption(event, timestamp)
            if token_record:
                with self.lock:
                    self.token_records.append(token_record)
                    self.stats["total_tokens"] += token_record.total_tokens
                    self.stats["total_api_calls"] += 1
        
        # 检查是否为资源使用事件
        if "cpu" in event or "memory" in event or "resource" in str(event).lower():
            resource_record = self._extract_resource_usage(event, timestamp)
            if resource_record:
                with self.lock:
                    self.resource_records.append(resource_record)
                    self.stats["peak_memory_mb"] = max(
                        self.stats["peak_memory_mb"], resource_record.memory_mb
                    )
                    self.stats["peak_cpu_percent"] = max(
                        self.stats["peak_cpu_percent"], resource_record.cpu_percent
                    )
        
        # 添加到事件缓冲区
        with self.lock:
            self.event_buffer.append({
                "line_num": line_num,
                "timestamp": timestamp,
                "event_type": event_type,
                "data": event
            })
            self.stats["total_events"] += 1
    
    def _process_text_line(self, line: str, line_num: int):
        """处理文本日志行"""
        timestamp = time.time()
        
        # 简单的文本匹配逻辑
        line_lower = line.lower()
        
        # 检测token相关
        token_match = False
        for pattern in self.token_patterns:
            if pattern in line_lower:
                token_match = True
                break
        
        if token_match:
            # 尝试从文本中提取token数量
            import re
            token_nums = re.findall(r'\b(\d+)\s*(?:tokens?|token count)', line_lower)
            if token_nums:
                token_count = int(token_nums[0])
                token_record = TokenConsumption(
                    timestamp=timestamp,
                    model_name="unknown",
                    prompt_tokens=token_count // 2,
                    completion_tokens=token_count // 2,
                    total_tokens=token_count,
                    estimated_cost=token_count * 0.000002,  # 假设价格
                    api_latency=0.5,
                    context={"source": "text_log", "line": line_num}
                )
                with self.lock:
                    self.token_records.append(token_record)
                    self.stats["total_tokens"] += token_count
    
    def _extract_token_consumption(self, event: Dict, timestamp: float) -> Optional[TokenConsumption]:
        """从事件中提取token消耗"""
        try:
            # 尝试不同格式的token数据
            prompt_tokens = event.get("prompt_tokens", event.get("input_tokens", 0))
            completion_tokens = event.get("completion_tokens", event.get("output_tokens", 0))
            total_tokens = event.get("total_tokens", prompt_tokens + completion_tokens)
            
            model_name = event.get("model", event.get("model_name", "unknown"))
            
            # 估计成本（根据模型定价）
            cost_per_token = self._get_model_cost(model_name)
            estimated_cost = total_tokens * cost_per_token
            
            api_latency = event.get("latency", event.get("response_time", 0.5))
            
            if total_tokens > 0:
                return TokenConsumption(
                    timestamp=timestamp,
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost=estimated_cost,
                    api_latency=api_latency,
                    context=event
                )
        except Exception as e:
            print(f"Token提取失败: {e}")
        
        return None
    
    def _extract_resource_usage(self, event: Dict, timestamp: float) -> Optional[ResourceUsage]:
        """从事件中提取资源使用"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return ResourceUsage(
                timestamp=timestamp,
                cpu_percent=event.get("cpu_percent", psutil.cpu_percent()),
                memory_mb=event.get("memory_mb", memory_info.rss / 1024 / 1024),
                memory_percent=event.get("memory_percent", psutil.virtual_memory().percent),
                disk_io_read=event.get("disk_read", 0),
                disk_io_write=event.get("disk_write", 0),
                network_sent=event.get("network_sent", 0),
                network_received=event.get("network_received", 0),
                open_files=event.get("open_files", len(process.open_files()) if hasattr(process, 'open_files') else 0),
                thread_count=event.get("thread_count", process.num_threads()),
                process_memory_info={
                    "rss": memory_info.rss,
                    "vms": memory_info.vms,
                    "shared": memory_info.shared if hasattr(memory_info, 'shared') else 0,
                    "text": memory_info.text if hasattr(memory_info, 'text') else 0,
                    "data": memory_info.data if hasattr(memory_info, 'data') else 0,
                }
            )
        except Exception as e:
            print(f"资源提取失败: {e}")
        
        return None
    
    def _get_model_cost(self, model_name: str) -> float:
        """获取模型token成本"""
        model_pricing = {
            "gpt-4": 0.00003,      # $0.03 per 1K tokens
            "gpt-3.5-turbo": 0.0000015,  # $0.0015 per 1K tokens
            "claude-3": 0.000025,
            "llama-3": 0.000005,
            "unknown": 0.000002,   # 默认价格
        }
        
        for key, price in model_pricing.items():
            if key in model_name.lower():
                return price
        
        return model_pricing["unknown"]
    
    def get_processed_data(self) -> Dict[str, Any]:
        """获取处理后的数据"""
        with self.lock:
            return {
                "event_buffer": list(self.event_buffer),
                "token_records": [tr.to_dict() for tr in self.token_records],
                "resource_records": [rr.to_dict() for rr in self.resource_records],
                "stats": self.stats,
            }


# ============ Token消耗分析器 ============

class TokenAnalyzer:
    """Token消耗深度分析器"""
    
    def __init__(self):
        self.token_thresholds = {
            "warning": 1000,      # 单次调用警告阈值
            "critical": 5000,     # 单次调用关键阈值
            "daily_warning": 100000,  # 每日警告阈值
            "daily_critical": 500000,  # 每日关键阈值
        }
        
        self.cost_thresholds = {
            "warning": 0.10,      # $0.10警告
            "critical": 1.00,     # $1.00关键
            "daily_warning": 10.00,  # $10每日警告
            "daily_critical": 50.00,  # $50每日关键
        }
    
    def analyze_token_consumption(self, token_records: List[TokenConsumption]) -> Dict[str, Any]:
        """分析Token消耗模式"""
        print("🧮 分析Token消耗...")
        
        if not token_records:
            return {"status": "no_data", "message": "No token records found"}
        
        # 转换为DataFrame便于分析
        df_data = []
        for record in token_records:
            df_data.append({
                "timestamp": record.timestamp,
                "model": record.model_name,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "total_tokens": record.total_tokens,
                "estimated_cost": record.estimated_cost,
                "api_latency": record.api_latency,
            })
        
        df = pd.DataFrame(df_data)
        
        # 基础统计
        total_tokens = df["total_tokens"].sum()
        total_cost = df["estimated_cost"].sum()
        avg_tokens = df["total_tokens"].mean()
        max_tokens = df["total_tokens"].max()
        avg_latency = df["api_latency"].mean()
        
        # 按模型统计
        model_stats = {}
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            model_stats[model] = {
                "total_tokens": model_df["total_tokens"].sum(),
                "total_cost": model_df["estimated_cost"].sum(),
                "call_count": len(model_df),
                "avg_tokens": model_df["total_tokens"].mean(),
                "avg_latency": model_df["api_latency"].mean(),
            }
        
        # 时间序列分析
        df["hour"] = pd.to_datetime(df["timestamp"], unit='s').dt.hour
        hourly_stats = df.groupby("hour").agg({
            "total_tokens": "sum",
            "estimated_cost": "sum",
            "api_latency": "mean"
        }).to_dict(orient="index")
        
        # 检测异常模式
        anomalies = self._detect_token_anomalies(df)
        
        # 成本优化建议
        suggestions = self._generate_cost_suggestions(df, model_stats)
        
        print(f"✓ Token分析完成: {total_tokens:,} tokens, ${total_cost:.4f} 成本")
        
        return {
            "summary": {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "avg_tokens_per_call": avg_tokens,
                "max_tokens_per_call": max_tokens,
                "avg_api_latency": avg_latency,
                "total_calls": len(token_records),
            },
            "model_breakdown": model_stats,
            "hourly_pattern": hourly_stats,
            "anomalies": anomalies,
            "suggestions": suggestions,
            "threshold_violations": self._check_thresholds(total_tokens, total_cost),
        }
    
    def _detect_token_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测Token消耗异常"""
        anomalies = []
        
        # 检测异常大的token消耗
        token_threshold = self.token_thresholds["critical"]
        large_calls = df[df["total_tokens"] > token_threshold]
        
        for idx, row in large_calls.iterrows():
            anomalies.append({
                "type": "high_token_usage",
                "timestamp": row["timestamp"],
                "model": row["model"],
                "tokens": row["total_tokens"],
                "cost": row["estimated_cost"],
                "severity": "critical" if row["total_tokens"] > 10000 else "warning",
                "suggestion": "考虑使用更小模型或优化prompt"
            })
        
        # 检测异常延迟
        latency_threshold = 5.0  # 5秒
        slow_calls = df[df["api_latency"] > latency_threshold]
        
        for idx, row in slow_calls.iterrows():
            anomalies.append({
                "type": "high_latency",
                "timestamp": row["timestamp"],
                "model": row["model"],
                "latency": row["api_latency"],
                "tokens": row["total_tokens"],
                "severity": "critical" if row["api_latency"] > 10.0 else "warning",
                "suggestion": "检查网络连接或使用更低延迟模型"
            })
        
        # 检测频繁调用模式（短时间内多次调用）
        if len(df) > 10:
            time_diff = df["timestamp"].diff()
            rapid_calls = df[time_diff < 0.1]  # 0.1秒内多次调用
            
            if len(rapid_calls) > 0:
                anomalies.append({
                    "type": "rapid_fire_calls",
                    "count": len(rapid_calls),
                    "timeframe": "sub-second",
                    "severity": "warning",
                    "suggestion": "考虑批处理请求或实现请求队列"
                })
        
        return anomalies
    
    def _generate_cost_suggestions(self, df: pd.DataFrame, model_stats: Dict) -> List[str]:
        """生成成本优化建议"""
        suggestions = []
        
        # 分析模型使用成本
        if len(model_stats) > 1:
            # 找到最贵的模型
            model_costs = {model: stats["total_cost"] for model, stats in model_stats.items()}
            most_expensive = max(model_costs.items(), key=lambda x: x[1])
            cheapest = min(model_costs.items(), key=lambda x: x[1])
            
            if most_expensive[1] > cheapest[1] * 5:  # 贵5倍以上
                suggestions.append(f"考虑将 {most_expensive[0]} 的部分任务迁移到 {cheapest[0]}")
        
        # 检查prompt与completion比例
        avg_prompt_ratio = (df["prompt_tokens"] / df["total_tokens"]).mean()
        if avg_prompt_ratio > 0.8:
            suggestions.append("Prompt token占比过高，考虑优化prompt长度")
        elif avg_prompt_ratio < 0.2:
            suggestions.append("Completion token占比过高，检查输出是否过长")
        
        # 批处理建议
        if len(df) > 50 and df["total_tokens"].mean() < 100:
            suggestions.append("大量小请求，考虑批处理以减少API调用次数")
        
        # 缓存建议
        duplicate_contexts = sum(1 for record in df.to_dict(orient="records") 
                               if record.get("context", {}).get("similarity_score", 0) > 0.9)
        if duplicate_contexts > len(df) * 0.3:
            suggestions.append("检测到重复的请求上下文，考虑实现结果缓存")
        
        return suggestions
    
    def _check_thresholds(self, total_tokens: int, total_cost: float) -> Dict[str, Any]:
        """检查阈值违规"""
        violations = {
            "token_violations": [],
            "cost_violations": [],
        }
        
        # Token阈值检查
        if total_tokens > self.token_thresholds["daily_critical"]:
            violations["token_violations"].append({
                "level": "critical",
                "threshold": self.token_thresholds["daily_critical"],
                "actual": total_tokens,
                "message": f"每日Token消耗严重超标"
            })
        elif total_tokens > self.token_thresholds["daily_warning"]:
            violations["token_violations"].append({
                "level": "warning",
                "threshold": self.token_thresholds["daily_warning"],
                "actual": total_tokens,
                "message": f"每日Token消耗警告"
            })
        
        # 成本阈值检查
        if total_cost > self.cost_thresholds["daily_critical"]:
            violations["cost_violations"].append({
                "level": "critical",
                "threshold": self.cost_thresholds["daily_critical"],
                "actual": total_cost,
                "message": f"每日成本严重超标 ${total_cost:.2f}"
            })
        elif total_cost > self.cost_thresholds["daily_warning"]:
            violations["cost_violations"].append({
                "level": "warning",
                "threshold": self.cost_thresholds["daily_warning"],
                "actual": total_cost,
                "message": f"每日成本警告 ${total_cost:.2f}"
            })
        
        return violations


# ============ 瓶颈定位器 ============

class BottleneckLocator:
    """性能瓶颈定位器"""
    
    def __init__(self):
        self.bottleneck_patterns = {
            "cpu_bound": {
                "indicators": ["cpu_percent > 80", "slow_computation", "model_inference"],
                "suggestions": [
                    "使用更高效的算法",
                    "实现计算缓存",
                    "使用批处理减少调用次数",
                    "考虑使用GPU加速"
                ]
            },
            "memory_bound": {
                "indicators": ["memory_percent > 80", "frequent_gc", "large_dataset"],
                "suggestions": [
                    "优化数据结构",
                    "使用内存映射文件",
                    "实现分页加载",
                    "减少数据冗余"
                ]
            },
            "io_bound": {
                "indicators": ["high_disk_io", "file_operations", "database_queries"],
                "suggestions": [
                    "使用SSD存储",
                    "实现文件缓存",
                    "批量读写操作",
                    "使用内存数据库"
                ]
            },
            "model_latency": {
                "indicators": ["api_latency > 2s", "llm_response_time", "model_loading"],
                "suggestions": [
                    "使用更小模型",
                    "实现模型预热",
                    "使用本地模型部署",
                    "优化请求格式"
                ]
            },
        }
    
    def locate_bottlenecks(self, 
                          token_records: List[TokenConsumption],
                          resource_records: List[ResourceUsage],
                          event_buffer: List[Dict]) -> List[BottleneckAnalysis]:
        """定位性能瓶颈"""
        print("🔍 定位性能瓶颈...")
        
        bottlenecks = []
        
        # 分析资源使用模式
        if resource_records:
            resource_bottlenecks = self._analyze_resource_bottlenecks(resource_records)
            bottlenecks.extend(resource_bottlenecks)
        
        # 分析Token消耗模式
        if token_records:
            token_bottlenecks = self._analyze_token_bottlenecks(token_records)
            bottlenecks.extend(token_bottlenecks)
        
        # 分析事件流模式
        if event_buffer:
            event_bottlenecks = self._analyze_event_bottlenecks(event_buffer)
            bottlenecks.extend(event_bottlenecks)
        
        # 按严重程度排序
        bottlenecks.sort(key=lambda x: x.severity, reverse=True)
        
        print(f"✓ 定位到 {len(bottlenecks)} 个潜在瓶颈")
        
        return bottlenecks
    
    def _analyze_resource_bottlenecks(self, resource_records: List[ResourceUsage]) -> List[BottleneckAnalysis]:
        """分析资源瓶颈"""
        bottlenecks = []
        
        if not resource_records:
            return bottlenecks
        
        # 转换为DataFrame
        df_data = [rr.to_dict() for rr in resource_records]
        df = pd.DataFrame(df_data)
        
        # CPU瓶颈分析
        high_cpu_periods = df[df["cpu_percent"] > 80]
        if len(high_cpu_periods) > 0:
            cpu_severity = min(1.0, high_cpu_periods["cpu_percent"].mean() / 100)
            
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type=BottleneckType.CPU_BOUND,
                location="system_cpu",
                severity=cpu_severity,
                duration=len(high_cpu_periods) * 1.0,  # 假设每秒采样
                samples=len(high_cpu_periods),
                improvement_suggestions=self.bottleneck_patterns["cpu_bound"]["suggestions"],
                affected_operations=["model_inference", "data_processing", "simulation"]
            ))
        
        # 内存瓶颈分析
        high_memory_periods = df[df["memory_percent"] > 80]
        if len(high_memory_periods) > 0:
            memory_severity = min(1.0, high_memory_periods["memory_percent"].mean() / 100)
            
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type=BottleneckType.MEMORY_BOUND,
                location="system_memory",
                severity=memory_severity,
                duration=len(high_memory_periods) * 1.0,
                samples=len(high_memory_periods),
                improvement_suggestions=self.bottleneck_patterns["memory_bound"]["suggestions"],
                affected_operations=["data_loading", "model_loading", "caching"]
            ))
        
        return bottlenecks
    
    def _analyze_token_bottlenecks(self, token_records: List[TokenConsumption]) -> List[BottleneckAnalysis]:
        """分析Token相关瓶颈"""
        bottlenecks = []
        
        if not token_records:
            return bottlenecks
        
        df_data = [tr.to_dict() for tr in token_records]
        df = pd.DataFrame(df_data)
        
        # 高延迟API调用
        high_latency_calls = df[df["api_latency"] > 2.0]
        if len(high_latency_calls) > 0:
            latency_severity = min(1.0, high_latency_calls["api_latency"].mean() / 10.0)
            
            bottlenecks.append(BottleneckAnalysis(
                bottleneck_type=BottleneckType.MODEL_LATENCY,
                location="api_endpoint",
                severity=latency_severity,
                duration=high_latency_calls["api_latency"].sum(),
                samples=len(high_latency_calls),
                improvement_suggestions=self.bottleneck_patterns["model_latency"]["suggestions"],
                affected_operations=["llm_inference", "api_calls"]
            ))
        
        return bottlenecks
    
    def _analyze_event_bottlenecks(self, event_buffer: List[Dict]) -> List[BottleneckAnalysis]:
        """分析事件流瓶颈"""
        bottlenecks = []
        
        if not event_buffer:
            return bottlenecks
        
        # 分析事件类型分布
        event_types = defaultdict(int)
        for event in event_buffer:
            event_type = event.get("event_type", "unknown")
            event_types[event_type] += 1
        
        # 检测频繁事件（可能指示重复操作）
        total_events = len(event_buffer)
        for event_type, count in event_types.items():
            if count > total_events * 0.3:  # 占比超过30%
                bottlenecks.append(BottleneckAnalysis(
                    bottleneck_type=BottleneckType.DATA_LOADING,
                    location=f"event_{event_type}",
                    severity=0.5,
                    duration=count * 0.1,  # 估计时间
                    samples=count,
                    improvement_suggestions=["优化事件处理逻辑", "减少重复事件", "实现事件去重"],
                    affected_operations=[event_type]
                ))
        
        return bottlenecks


# ============ 存储优化器 ============

class StorageOptimizer:
    """存储结构优化器"""
    
    def __init__(self, output_dir: str = "optimized_storage"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def optimize_storage_structure(self, 
                                  token_records: List[TokenConsumption],
                                  resource_records: List[ResourceUsage],
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """优化存储结构"""
        print("💾 优化存储结构...")
        
        optimization = {
            "original_size": self._estimate_size(token_records, resource_records),
            "optimized_files": {},
            "compression_ratios": {},
            "partition_strategy": {},
        }
        
        # 1. 分区存储
        self._partition_storage(token_records, resource_records)
        optimization["partition_strategy"] = self._get_partition_strategy()
        
        # 2. 压缩数据
        compressed_files = self._compress_data(token_records, resource_records, analysis_results)
        optimization["optimized_files"] = compressed_files
        
        # 3. 计算压缩比
        original_mb = optimization["original_size"] / (1024 * 1024)
        optimized_mb = sum(compressed_files.values()) / (1024 * 1024)
        if original_mb > 0:
            compression_ratio = optimized_mb / original_mb
            optimization["compression_ratios"] = {
                "original_mb": original_mb,
                "optimized_mb": optimized_mb,
                "compression_ratio": compression_ratio,
                "space_saved_mb": original_mb - optimized_mb,
                "space_saved_percent": (1 - compression_ratio) * 100,
            }
        
        # 4. 生成索引文件
        index_file = self._create_index_file(token_records, resource_records, analysis_results)
        optimization["index_file"] = index_file
        
        # 5. 清理建议
        optimization["cleanup_suggestions"] = self._generate_cleanup_suggestions(
            token_records, resource_records
        )
        
        print(f"✓ 存储优化完成，压缩比: {optimization['compression_ratios'].get('compression_ratio', 0):.2f}")
        
        return optimization
    
    def _estimate_size(self, 
                      token_records: List[TokenConsumption],
                      resource_records: List[ResourceUsage]) -> int:
        """估计数据大小（字节）"""
        total_size = 0
        
        # 估计Token记录大小
        for record in token_records:
            # 粗略估计：每个记录约500字节
            total_size += 500
        
        # 估计资源记录大小
        for record in resource_records:
            # 每个资源记录约1KB
            total_size += 1024
        
        return total_size
    
    def _partition_storage(self, 
                          token_records: List[TokenConsumption],
                          resource_records: List[ResourceUsage]):
        """分区存储数据"""
        
        # 按时间分区（按小时）
        if token_records:
            token_by_hour = defaultdict(list)
            for record in token_records:
                hour = datetime.fromtimestamp(record.timestamp).hour
                token_by_hour[hour].append(record)
            
            for hour, records in token_by_hour.items():
                hour_dir = self.output_dir / "tokens" / f"hour_{hour:02d}"
                hour_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = hour_dir / "token_records.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([tr.to_dict() for tr in records], f, indent=2, ensure_ascii=False)
        
        # 按资源类型分区
        if resource_records:
            resource_dir = self.output_dir / "resources"
            resource_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = resource_dir / "resource_records.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([rr.to_dict() for rr in resource_records], f, indent=2, ensure_ascii=False)
    
    def _get_partition_strategy(self) -> Dict[str, Any]:
        """获取分区策略"""
        return {
            "temporal_partitioning": {
                "granularity": "hourly",
                "directory_structure": "tokens/hour_{HH}/token_records.json",
                "benefits": ["快速时间范围查询", "减少单个文件大小", "便于归档"]
            },
            "type_partitioning": {
                "resources": "resources/resource_records.json",
                "metadata": "metadata/analysis_results.json",
                "benefits": ["按类型组织", "差异化压缩策略", "独立备份"]
            },
            "compression_strategy": {
                "tokens": "gzip压缩",
                "resources": "原始JSON",
                "metadata": "msgpack压缩",
                "benefits": ["平衡性能与存储", "快速读取常用数据", "节省空间"]
            }
        }
    
    def _compress_data(self, 
                      token_records: List[TokenConsumption],
                      resource_records: List[ResourceUsage],
                      analysis_results: Dict[str, Any]) -> Dict[str, int]:
        """压缩数据并返回文件大小"""
        compressed_files = {}
        
        # 压缩Token记录（使用gzip）
        if token_records:
            token_file = self.output_dir / "compressed" / "tokens.json.gz"
            token_file.parent.mkdir(parents=True, exist_ok=True)
            
            import gzip
            token_data = [tr.to_dict() for tr in token_records]
            with gzip.open(token_file, 'wt', encoding='utf-8') as f:
                json.dump(token_data, f)
            
            compressed_files["tokens_gz"] = token_file.stat().st_size
        
        # 保存分析结果
        if analysis_results:
            analysis_file = self.output_dir / "analysis" / "full_analysis.json"
            analysis_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
            compressed_files["analysis_json"] = analysis_file.stat().st_size
        
        # 保存摘要数据（小文件，快速加载）
        summary_data = {
            "token_summary": {
                "total_records": len(token_records),
                "total_tokens": sum(tr.total_tokens for tr in token_records),
                "total_cost": sum(tr.estimated_cost for tr in token_records),
            },
            "resource_summary": {
                "total_records": len(resource_records),
                "peak_memory_mb": max((rr.memory_mb for rr in resource_records), default=0),
                "peak_cpu_percent": max((rr.cpu_percent for rr in resource_records), default=0),
            },
            "timestamp": time.time(),
        }
        
        summary_file = self.output_dir / "summary" / "performance_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        compressed_files["summary_json"] = summary_file.stat().st_size
        
        return compressed_files
    
    def _create_index_file(self, 
                          token_records: List[TokenConsumption],
                          resource_records: List[ResourceUsage],
                          analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建索引文件"""
        index = {
            "metadata": {
                "created_at": time.time(),
                "total_files": len(list(self.output_dir.rglob("*"))),
                "total_size_bytes": sum(f.stat().st_size for f in self.output_dir.rglob("*") if f.is_file()),
            },
            "data_files": [],
            "query_examples": [
                "find_tokens_by_hour(10, 12) - 查找10-12点的token记录",
                "get_resource_peak('memory') - 获取内存使用峰值",
                "analyze_cost_by_model() - 按模型分析成本",
                "locate_bottlenecks('cpu') - 定位CPU瓶颈",
            ]
        }
        
        # 列出所有数据文件
        for file_path in self.output_dir.rglob("*.json"):
            if file_path.is_file():
                index["data_files"].append({
                    "path": str(file_path.relative_to(self.output_dir)),
                    "size_bytes": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                })
        
        # 保存索引文件
        index_file = self.output_dir / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        return {
            "index_file": str(index_file),
            "total_entries": len(index["data_files"]),
            "total_size_mb": index["metadata"]["total_size_bytes"] / (1024 * 1024),
        }
    
    def _generate_cleanup_suggestions(self, 
                                     token_records: List[TokenConsumption],
                                     resource_records: List[ResourceUsage]) -> List[str]:
        """生成存储清理建议"""
        suggestions = []
        
        # 检查旧数据
        current_time = time.time()
        one_week_ago = current_time - 7 * 24 * 3600
        
        old_token_records = [tr for tr in token_records if tr.timestamp < one_week_ago]
        if len(old_token_records) > len(token_records) * 0.5:
            suggestions.append("超过50%的token记录超过1周，考虑归档或删除旧数据")
        
        # 检查数据冗余
        if len(token_records) > 1000:
            suggestions.append("Token记录超过1000条，考虑聚合统计信息，保留原始样本")
        
        # 检查存储效率
        if resource_records:
            avg_interval = self._calculate_average_interval(resource_records)
            if avg_interval < 1.0:  # 小于1秒采样间隔
                suggestions.append("资源采样过于频繁，考虑降低采样频率或使用滑动窗口聚合")
        
        return suggestions
    
    def _calculate_average_interval(self, records: List[ResourceUsage]) -> float:
        """计算平均采样间隔"""
        if len(records) < 2:
            return 0
        
        timestamps = sorted([r.timestamp for r in records])
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        return sum(intervals) / len(intervals) if intervals else 0


# ============ 主性能监控器 ============

class M10_PerformanceMonitor:
    """M10主性能监控器"""
    
    def __init__(self, 
                 project_id: str = "default_project",
                 output_dir: str = "performance_reports",
                 log_dir: str = "logs"):
        
        self.project_id = project_id
        self.report_id = f"perf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化组件
        self.log_processor = LogStreamProcessor(log_dir=log_dir)
        self.token_analyzer = TokenAnalyzer()
        self.bottleneck_locator = BottleneckLocator()
        self.storage_optimizer = StorageOptimizer(output_dir=output_dir)
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = 0
        
        print(f"🚀 M10性能监控器初始化 - 项目: {project_id}")
    
    def monitor_log_stream(self, log_file_path: str) -> CostReport:
        """监控日志流并生成成本报告"""
        print(f"\n📈 开始监控日志流: {log_file_path}")
        
        # 1. 处理日志流
        process_result = self.log_processor.process_log_stream(log_file_path)
        if process_result["status"] != "success":
            print(f"✗ 日志处理失败: {process_result.get('message', 'Unknown error')}")
            return None
        
        # 2. 获取处理后的数据
        processed_data = self.log_processor.get_processed_data()
        token_records = [TokenConsumption(**tr) for tr in processed_data["token_records"]]
        resource_records = [ResourceUsage(**rr) for rr in processed_data["resource_records"]]
        event_buffer = processed_data["event_buffer"]
        
        # 3. Token消耗分析
        token_analysis = self.token_analyzer.analyze_token_consumption(token_records)
        
        # 4. 瓶颈定位
        bottlenecks = self.bottleneck_locator.locate_bottlenecks(
            token_records, resource_records, event_buffer
        )
        
        # 5. 存储优化
        analysis_results = {
            "token_analysis": token_analysis,
            "bottlenecks": [b.to_dict() for b in bottlenecks],
            "processed_stats": process_result,
        }
        
        storage_optimization = self.storage_optimizer.optimize_storage_structure(
            token_records, resource_records, analysis_results
        )
        
        # 6. 生成成本报告
        total_token_cost = token_analysis["summary"]["total_cost"]
        total_api_calls = token_analysis["summary"]["total_calls"]
        
        # 估算其他成本
        api_call_cost = total_api_calls * 0.001  # 假设每次API调用$0.001
        compute_cost = len(resource_records) * 0.0001  # 假设每记录$0.0001
        storage_cost = storage_optimization["original_size"] * 0.000000001  # 假设$0.01 per GB
        
        total_cost = total_token_cost + api_call_cost + compute_cost + storage_cost
        
        # 生成性能度量
        performance_metrics = self._generate_performance_metrics(
            token_analysis, resource_records, bottlenecks
        )
        
        # 生成优化建议
        optimization_suggestions = self._generate_optimization_suggestions(
            token_analysis, bottlenecks, storage_optimization
        )
        
        # 创建成本报告
        cost_report = CostReport(
            report_id=self.report_id,
            start_time=self.start_time,
            end_time=time.time(),
            duration=time.time() - self.start_time,
            total_token_cost=total_token_cost,
            api_call_cost=api_call_cost,
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            total_cost=total_cost,
            token_breakdown=token_analysis.get("model_breakdown", {}),
            cost_breakdown={
                "tokens": total_token_cost,
                "api_calls": api_call_cost,
                "compute": compute_cost,
                "storage": storage_cost,
            },
            performance_metrics=performance_metrics,
            bottlenecks=bottlenecks,
            optimization_suggestions=optimization_suggestions,
            storage_optimization=storage_optimization,
        )
        
        # 7. 保存报告
        self._save_cost_report(cost_report)
        
        print(f"\n✅ 性能监控完成!")
        print(f"   总成本: ${total_cost:.4f}")
        print(f"   Token成本: ${total_token_cost:.4f}")
        print(f"   发现瓶颈: {len(bottlenecks)} 个")
        print(f"   存储优化: {storage_optimization.get('compression_ratios', {}).get('space_saved_percent', 0):.1f}% 空间节省")
        
        return cost_report
    
    def start_realtime_monitoring(self, log_dir: str = "logs", interval: float = 5.0):
        """启动实时监控"""
        print(f"🔄 启动实时性能监控 (间隔: {interval}s)")
        
        self.monitoring_active = True
        self.start_time = time.time()
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # 查找最新的日志文件
                    log_files = list(Path(log_dir).glob("*.log"))
                    if log_files:
                        log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        latest_log = log_files[0]
                        
                        # 处理最新的日志
                        self.monitor_log_stream(str(latest_log))
                    
                    # 等待下一个间隔
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"监控循环异常: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("✓ 实时监控已启动")
    
    def stop_realtime_monitoring(self):
        """停止实时监控"""
        print("🛑 停止实时性能监控")
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        print("✓ 实时监控已停止")
    
    def _generate_performance_metrics(self, 
                                     token_analysis: Dict,
                                     resource_records: List[ResourceUsage],
                                     bottlenecks: List[BottleneckAnalysis]) -> List[PerformanceMetric]:
        """生成性能度量"""
        metrics = []
        
        # Token相关度量
        token_summary = token_analysis.get("summary", {})
        if token_summary:
            metrics.append(PerformanceMetric(
                metric_name="token_per_call",
                value=token_summary.get("avg_tokens_per_call", 0),
                unit="tokens",
                timestamp=time.time(),
                level=PerformanceLevel.GOOD if token_summary.get("avg_tokens_per_call", 0) < 1000 else PerformanceLevel.WARNING,
                threshold=1000,
                description="平均每次调用的Token数量"
            ))
            
            metrics.append(PerformanceMetric(
                metric_name="api_latency",
                value=token_summary.get("avg_api_latency", 0),
                unit="seconds",
                timestamp=time.time(),
                level=PerformanceLevel.EXCELLENT if token_summary.get("avg_api_latency", 0) < 1.0 else PerformanceLevel.WARNING,
                threshold=2.0,
                description="平均API调用延迟"
            ))
        
        # 资源使用度量
        if resource_records:
            cpu_values = [rr.cpu_percent for rr in resource_records]
            memory_values = [rr.memory_mb for rr in resource_records]
            
            metrics.append(PerformanceMetric(
                metric_name="cpu_utilization",
                value=max(cpu_values) if cpu_values else 0,
                unit="percent",
                timestamp=time.time(),
                level=PerformanceLevel.EXCELLENT if max(cpu_values) < 70 else PerformanceLevel.WARNING,
                threshold=85,
                description="CPU使用率峰值"
            ))
            
            metrics.append(PerformanceMetric(
                metric_name="memory_usage",
                value=max(memory_values) if memory_values else 0,
                unit="MB",
                timestamp=time.time(),
                level=PerformanceLevel.GOOD,
                threshold=1024,  # 1GB
                description="内存使用峰值"
            ))
        
        # 瓶颈严重程度度量
        if bottlenecks:
            max_severity = max([b.severity for b in bottlenecks]) if bottlenecks else 0
            metrics.append(PerformanceMetric(
                metric_name="bottleneck_severity",
                value=max_severity,
                unit="ratio",
                timestamp=time.time(),
                level=PerformanceLevel.CRITICAL if max_severity > 0.8 else PerformanceLevel.WARNING,
                threshold=0.7,
                description="最大瓶颈严重程度"
            ))
        
        return metrics
    
    def _generate_optimization_suggestions(self,
                                          token_analysis: Dict,
                                          bottlenecks: List[BottleneckAnalysis],
                                          storage_optimization: Dict) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 从Token分析获取建议
        token_suggestions = token_analysis.get("suggestions", [])
        suggestions.extend(token_suggestions)
        
        # 从瓶颈分析获取建议
        for bottleneck in bottlenecks:
            suggestions.extend(bottleneck.improvement_suggestions)
        
        # 从存储优化获取建议
        storage_suggestions = storage_optimization.get("cleanup_suggestions", [])
        suggestions.extend(storage_suggestions)
        
        # 去重并排序
        unique_suggestions = list(dict.fromkeys(suggestions))  # 保持顺序去重
        
        # 根据潜在影响排序（简单启发式）
        high_impact_keywords = ["成本", "critical", "严重", "瓶颈", "内存", "优化"]
        medium_impact_keywords = ["警告", "warning", "建议", "考虑", "提高"]
        
        def suggestion_priority(suggestion):
            if any(keyword in suggestion for keyword in high_impact_keywords):
                return 0
            elif any(keyword in suggestion for keyword in medium_impact_keywords):
                return 1
            else:
                return 2
        
        unique_suggestions.sort(key=suggestion_priority)
        
        return unique_suggestions[:10]  # 返回前10个最重要的建议
    
    def _save_cost_report(self, cost_report: CostReport):
        """保存成本报告"""
        report_dir = Path("performance_reports") / self.project_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"{cost_report.report_id}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(cost_report.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"📄 成本报告已保存: {report_file}")
        
        # 同时保存简洁版本
        summary_file = report_dir / f"{cost_report.report_id}_summary.json"
        summary = {
            "report_id": cost_report.report_id,
            "project_id": cost_report.project_id,
            "total_cost": cost_report.total_cost,
            "token_cost": cost_report.total_token_cost,
            "duration": cost_report.duration,
            "bottleneck_count": len(cost_report.bottlenecks),
            "suggestion_count": len(cost_report.optimization_suggestions),
            "storage_savings_percent": cost_report.storage_optimization.get(
                "compression_ratios", {}
            ).get("space_saved_percent", 0),
            "timestamp": cost_report.end_time,
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# ============ 使用示例 ============

if __name__ == "__main__":
    print("🔧 M10性能监控模块测试")
    print("="*60)
    
    # 创建性能监控器
    monitor = M10_PerformanceMonitor(
        project_id="Kickstarter_Performance_V1",
        output_dir="performance_reports",
        log_dir="../logs"  # 指向项目日志目录
    )
    
    # 查找日志文件示例
    log_files = list(Path("../logs").glob("*.log"))
    if log_files:
        # 使用最新的日志文件
        log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_log = log_files[0]
        
        print(f"找到日志文件: {latest_log}")
        
        # 监控日志流并生成报告
        report = monitor.monitor_log_stream(str(latest_log))
        
        if report:
            print("\n📊 报告摘要:")
            print(f"   报告ID: {report.report_id}")
            print(f"   总成本: ${report.total_cost:.4f}")
            print(f"   Token成本: ${report.total_token_cost:.4f}")
            print(f"   瓶颈数量: {len(report.bottlenecks)}")
            print(f"   优化建议: {len(report.optimization_suggestions)} 条")
            
            if report.optimization_suggestions:
                print("\n💡 优化建议:")
                for i, suggestion in enumerate(report.optimization_suggestions[:3], 1):
                    print(f"   {i}. {suggestion}")
            
            print(f"\n📁 存储优化: {report.storage_optimization.get('compression_ratios', {}).get('space_saved_percent', 0):.1f}% 空间节省")
    else:
        print("⚠️ 未找到日志文件，创建示例数据...")
        
        # 创建示例日志文件
        example_log = Path("../logs") / "example_m10_test.log"
        example_log.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成一些示例日志
        example_data = [
            '{"timestamp": 1741615200, "event_type": "token_usage", "model": "gpt-4", "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}',
            '{"timestamp": 1741615201, "event_type": "resource_usage", "cpu_percent": 45.2, "memory_mb": 256.7}',
            '{"timestamp": 1741615202, "event_type": "api_call", "endpoint": "/v1/chat/completions", "latency": 1.2}',
            '{"timestamp": 1741615203, "event_type": "token_usage", "model": "gpt-3.5-turbo", "prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120}',
        ]
        
        with open(example_log, 'w', encoding='utf-8') as f:
            f.write("\n".join(example_data))
        
        print(f"✓ 创建示例日志: {example_log}")
        
        # 使用示例日志
        report = monitor.monitor_log_stream(str(example_log))
        
        if report:
            print(f"\n✅ 示例监控完成，总成本: ${report.total_cost:.4f}")
    
    print("\n" + "="*60)
    print("M10性能监控模块测试完成")
    print("="*60)
