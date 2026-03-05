#!/usr/bin/env python
# coding: utf-8

"""
M6 状态管理与日志系统
====================

职责：
- 提供统一的 Env 状态定义与标准化接口
- 管理仿真环境在执行过程中的状态快照与轨迹
- 输出可被 M5 / M8 / M9 等模块消费的标准化日志数据

注意：
- 仅通过类 / 函数接口对外暴露能力，不在此文件中编写 main 入口或直接执行代码
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==============================
# 一、Env 状态与事件定义
# ==============================


class EnvEventType(str, Enum):
    """环境状态变更事件类型枚举"""

    RESET = "reset"
    STEP = "step"
    ERROR = "error"
    INFO = "info"


@dataclass
class EnvState:
    """
    Env 状态统一结构定义

    该结构用于“状态管理器 ← Env/M5”之间的数据传递，确保字段含义明确、可扩展。
    """

    project_id: str
    case_id: Optional[str] = None
    scenario_id: Optional[str] = None

    step: int = 0
    done: bool = False
    reward: Optional[float] = None

    action: Any = None          # 智能体动作（可为 dict / list / 原始类型）
    observation: Any = None     # 环境观测
    info: Dict[str, Any] = None # 额外信息（如风险标签、错误码等）

    timestamp: float = 0.0      # Unix 时间戳（秒）

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的字典"""
        data = asdict(self)
        if data.get("info") is None:
            data["info"] = {}
        return data


@dataclass
class EnvEventRecord:
    """
    单条状态事件记录：事件类型 + EnvState + 扩展字段
    """

    event_type: EnvEventType
    state: EnvState
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "event_type": self.event_type.value,
            "timestamp": self.state.timestamp,
        }
        base.update(self.state.to_dict())
        base.update(self.extra or {})
        return base


# ==============================
# 二、量化规则与路由分类（核心）
# ==============================


class RuleOperator(str, Enum):
    """规则比较运算符"""

    GT = "GT"
    GE = "GE"
    LT = "LT"
    LE = "LE"
    EQ = "EQ"
    NE = "NE"
    IN = "IN"
    NOT_IN = "NOT_IN"
    RANGE = "RANGE"  # 左闭右闭区间 [min_value, max_value]


class RuleTarget(str, Enum):
    """规则作用目标（从上下文中取值的命名空间）"""

    ENV = "env"        # Env 状态字段，如 step / reward
    ACTION = "action"  # 智能体动作
    METRIC = "metric"  # 统计指标 / 业务度量
    CONTEXT = "context"  # 直接从整体上下文取字段


@dataclass
class QuantifiedRule:
    """
    可量化业务规则结构

    示例：
    - 当 step > 100 时终止流程，路由到 "terminate"
    - 当 reward < 0 时路由到 "risk_control"
    """

    rule_id: str
    name: str
    target: RuleTarget
    field: str
    operator: RuleOperator

    # 单值或区间阈值
    value: Any = None
    min_value: Any = None
    max_value: Any = None

    route: str = "default"
    description: str = ""
    priority: int = 0
    enabled: bool = True
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def _safe_get(container: Dict[str, Any], key: str) -> Any:
        """容错获取字段，嵌套字段支持 a.b.c 形式"""
        if container is None:
            return None
        if "." not in key:
            return container.get(key)
        cur = container
        for part in key.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    def _extract_value(self, context: Dict[str, Any]) -> Any:
        """根据 target 从上下文中取出被比较的值"""
        if self.target == RuleTarget.ENV:
            source = context.get("env", {})
        elif self.target == RuleTarget.ACTION:
            source = context.get("action", {})
        elif self.target == RuleTarget.METRIC:
            source = context.get("metric", {})
        else:
            source = context
        if not isinstance(source, dict):
            return None
        return self._safe_get(source, self.field)

    def _compare(self, left: Any) -> bool:
        """执行具体的比较逻辑"""
        op = self.operator
        try:
            if op == RuleOperator.GT:
                return left > self.value
            if op == RuleOperator.GE:
                return left >= self.value
            if op == RuleOperator.LT:
                return left < self.value
            if op == RuleOperator.LE:
                return left <= self.value
            if op == RuleOperator.EQ:
                return left == self.value
            if op == RuleOperator.NE:
                return left != self.value
            if op == RuleOperator.IN:
                return left in (self.value or [])
            if op == RuleOperator.NOT_IN:
                return left not in (self.value or [])
            if op == RuleOperator.RANGE:
                if self.min_value is None or self.max_value is None:
                    return False
                return self.min_value <= left <= self.max_value
        except Exception:
            return False
        return False

    def matches(self, context: Dict[str, Any]) -> bool:
        """给定上下文，判断是否命中规则"""
        if not self.enabled:
            return False
        left = self._extract_value(context)
        if left is None:
            return False
        return self._compare(left)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantifiedRule":
        """从配置字典构造规则（容错）"""
        return cls(
            rule_id=str(data.get("rule_id") or data.get("id") or ""),
            name=str(data.get("name") or ""),
            target=RuleTarget(str(data.get("target", RuleTarget.CONTEXT.value))),
            field=str(data.get("field", "")),
            operator=RuleOperator(str(data.get("operator", RuleOperator.EQ.value))),
            value=data.get("value"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            route=str(data.get("route", "default")),
            description=str(data.get("description", "")),
            priority=int(data.get("priority", 0)),
            enabled=bool(data.get("enabled", True)),
            tags=list(data.get("tags", [])) or None,
            metadata=data.get("metadata") or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["target"] = self.target.value
        data["operator"] = self.operator.value
        return data


class RuleLibrary:
    """
    规则库：负责规则加载、管理与匹配

    - 支持从 JSON 配置文件加载量化规则
    - 支持在运行期间调用 refresh_from_file 实现动态更新（无需重启）
    """

    def __init__(
        self,
        project_id: str,
        config_path: Optional[Path | str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.project_id = project_id
        self._logger = logger or logging.getLogger(f"M6.RuleLibrary.{project_id}")
        self._logger.setLevel(logging.INFO)
        self._rules: List[QuantifiedRule] = []
        self._config_path: Optional[Path] = Path(config_path) if config_path else None
        self._last_mtime: Optional[float] = None

        if self._config_path and self._config_path.exists():
            self._load_from_file()

    # ------ 规则加载与动态刷新 ------

    def _load_from_file(self) -> None:
        try:
            if not self._config_path or not self._config_path.exists():
                return
            mtime = self._config_path.stat().st_mtime
            with self._config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rules: List[QuantifiedRule] = []
            for item in data or []:
                try:
                    rules.append(QuantifiedRule.from_dict(item))
                except Exception as exc:
                    self._logger.warning(
                        "Skip invalid rule config: %s (error=%s)", item, exc
                    )
            self._rules = sorted(rules, key=lambda r: r.priority, reverse=True)
            self._last_mtime = mtime
            self._logger.info(
                "[M6] RuleLibrary loaded %d rules from %s",
                len(self._rules),
                self._config_path,
            )
        except Exception as exc:
            self._logger.error("Failed to load rule config from %s: %s", self._config_path, exc)

    def refresh_from_file(self) -> None:
        """如配置文件发生变化，则自动重新加载规则"""
        if not self._config_path or not self._config_path.exists():
            return
        mtime = self._config_path.stat().st_mtime
        if self._last_mtime is None or mtime > self._last_mtime:
            self._load_from_file()

    # ------ 内存规则管理（可选：支持从代码侧直接维护规则） ------

    def add_rule(self, rule: QuantifiedRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> None:
        self._rules = [r for r in self._rules if r.rule_id != rule_id]

    def get_active_rules(self) -> List[QuantifiedRule]:
        return [r for r in self._rules if r.enabled]

    # ------ 规则匹配 ------

    def match_rules(
        self, context: Dict[str, Any], first_only: bool = False
    ) -> List[QuantifiedRule]:
        """
        根据上下文匹配规则，按优先级降序返回命中规则列表

        context 约定：
        - 可包含子字典：env / action / metric / ...
        """
        self.refresh_from_file()
        matched: List[QuantifiedRule] = []
        for rule in self.get_active_rules():
            if rule.matches(context):
                matched.append(rule)
                if first_only:
                    break
        return matched


class RouteDispatcher:
    """
    路由分发器：基于规则库完成基础路由分类

    - 支持多条规则命中，按优先级选择主路由
    - 将规则匹配的过程打日志，便于排查
    """

    def __init__(
        self,
        rule_library: RuleLibrary,
        default_route: str = "default",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.rule_library = rule_library
        self.default_route = default_route
        self._logger = logger or logging.getLogger(
            f"M6.RouteDispatcher.{rule_library.project_id}"
        )
        self._logger.setLevel(logging.INFO)

    def dispatch(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据上下文执行路由分类

        返回结构：
        {
            "route": "risk_control",
            "matched_rules": [...],
        }
        """
        matched = self.rule_library.match_rules(context)
        if not matched:
            route = self.default_route
        else:
            route = matched[0].route or self.default_route

        self._logger.info(
            "[M6] Route dispatch | route=%s | matched=%d",
            route,
            len(matched),
        )

        return {
            "route": route,
            "matched_rules": [r.to_dict() for r in matched],
        }


# ==============================
# 三、状态管理器（核心）
# ==============================


class EnvStateManager:
    """
    环境状态管理器

    - 负责接收来自 Env / M5 的状态更新（reset / step / error）
    - 做字段补全、统一打标签、写入标准化日志
    - 对外提供“当前状态快照 + 完整轨迹导出”能力
    """

    def __init__(
        self,
        project_id: str,
        logger: Optional[logging.Logger] = None,
        enable_in_memory_trace: bool = True,
        rule_library: Optional[RuleLibrary] = None,
        route_dispatcher: Optional[RouteDispatcher] = None,
    ) -> None:
        self.project_id = project_id
        self._logger = logger or logging.getLogger(f"M6.StateManager.{project_id}")
        self._logger.setLevel(logging.INFO)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self._enable_trace = enable_in_memory_trace
        self._trace: List[EnvEventRecord] = []
        self._latest_state: Optional[EnvState] = None
        self._rule_library = rule_library
        self._route_dispatcher = route_dispatcher

    # -------- 内部通用入口 --------

    def _record_event(
        self,
        event_type: EnvEventType,
        raw_state: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        标准化并记录一条状态事件

        :param event_type: 事件类型
        :param raw_state:  原始状态字典（来自 Env / M5）
        :param extra:      额外补充字段（如错误信息、阶段名称等）
        """
        state = EnvState(
            project_id=raw_state.get("project_id", self.project_id),
            case_id=raw_state.get("case_id"),
            scenario_id=raw_state.get("scenario_id"),
            step=int(raw_state.get("step", 0)),
            done=bool(raw_state.get("done", False)),
            reward=raw_state.get("reward"),
            action=raw_state.get("action"),
            observation=raw_state.get("observation"),
            info=raw_state.get("info") or {},
            timestamp=raw_state.get("timestamp") or time.time(),
        )

        record = EnvEventRecord(
            event_type=event_type,
            state=state,
            extra=extra or {},
        )

        self._latest_state = state

        if self._enable_trace:
            self._trace.append(record)

        # 写一条标准化日志（方便后续排查）
        payload = record.to_dict()
        self._logger.info(
            "[M6] EnvEvent | type=%s | project=%s | case=%s | step=%s | done=%s",
            payload.get("event_type"),
            payload.get("project_id"),
            payload.get("case_id"),
            payload.get("step"),
            payload.get("done"),
        )

    # -------- 对外事件接口 --------

    def on_reset(self, raw_state: Dict[str, Any]) -> None:
        """环境 reset 时调用"""
        self._record_event(EnvEventType.RESET, raw_state, extra={"stage": "RESET"})

    def on_step(self, raw_state: Dict[str, Any]) -> None:
        """环境每一个 step 调用"""
        self._record_event(EnvEventType.STEP, raw_state, extra={"stage": "STEP"})

    def on_error(self, raw_state: Dict[str, Any], error: Exception) -> None:
        """环境或智能体执行出错时调用"""
        extra = {
            "stage": "ERROR",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        self._record_event(EnvEventType.ERROR, raw_state, extra=extra)

    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """记录一条与具体状态弱相关的说明性事件"""
        raw_state: Dict[str, Any] = {
            "project_id": self.project_id,
            "step": self._latest_state.step if self._latest_state else 0,
            "done": self._latest_state.done if self._latest_state else False,
            "timestamp": time.time(),
        }
        extra = {"stage": "INFO", "message": message}
        if context:
            extra.update(context)
        self._record_event(EnvEventType.INFO, raw_state, extra=extra)

    # -------- 查询与导出 --------

    @property
    def latest_state(self) -> Optional[Dict[str, Any]]:
        """获取最新一条状态快照（字典形式）"""
        return self._latest_state.to_dict() if self._latest_state else None

    def get_trace(self) -> List[Dict[str, Any]]:
        """获取当前会话内所有状态轨迹（列表形式）"""
        return [r.to_dict() for r in self._trace]

    def export_trace(self, output_path: Path | str) -> Path:
        """
        将当前轨迹以 JSON Lines 格式导出到文件

        :param output_path: 目标路径（文件或目录）
        :return: 最终写入的文件路径
        """
        output_path = Path(output_path)
        if output_path.is_dir():
            output_file = output_path / f"m6_state_trace_{int(time.time())}.jsonl"
        else:
            output_file = output_path

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            for record in self._trace:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        self._logger.info("[M6] State trace exported to %s", output_file)
        return output_file

    # -------- 规则系统挂载与路由辅助 --------

    def attach_rule_system(
        self,
        rule_library: RuleLibrary,
        route_dispatcher: Optional[RouteDispatcher] = None,
    ) -> None:
        """在运行时挂载规则库与路由器"""
        self._rule_library = rule_library
        self._route_dispatcher = route_dispatcher

    def decide_route(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用挂载的 RouteDispatcher 进行一次路由决策。

        当尚未挂载路由器时，返回默认 route=default。
        """
        if self._route_dispatcher is None:
            return {"route": "default", "matched_rules": []}
        return self._route_dispatcher.dispatch(context)


# ==============================
# 三、静态规则校验（供 M5 调用）
# ==============================


class StaticValidator:
    """
    静态规则校验器（M6 → M5）

    目标：
    - 在执行仿真前，对“智能体实现是否满足基础接口约束、规则约束”做一次快速校验
    - 失败时返回 False，由 M5 记录原因并跳过该用例
    - 这里提供一个通用、稳健的默认实现，后续可按项目需要加规则
    """

    @staticmethod
    def validate_agent_logic(agent: Any, rule: Dict[str, Any]) -> bool:
        """
        校验智能体是否满足最基本的调用约束。

        当前默认规则（可按需扩展）：
        1. agent 不为 None
        2. agent 至少实现以下任意一种交互方法：
           - step(obs) / __call__(obs) / act(obs)
        3. 如果 rule 中声明了 max_steps / required_methods 等字段，则做一致性检查
        4. 可选：对接 M6 量化规则 / 动作约束的静态检查
        """
        if agent is None:
            return False

        # 1. 基础接口检查
        has_step = hasattr(agent, "step")
        has_call = callable(agent)
        has_act = hasattr(agent, "act")

        if not (has_step or has_call or has_act):
            return False

        # 2. 规则声明的约束检查（可选）
        if not isinstance(rule, dict):
            return True

        required_methods = rule.get("required_methods", [])
        for m in required_methods:
            if not hasattr(agent, m):
                return False

        max_steps = rule.get("max_steps")
        if max_steps is not None:
            try:
                if int(max_steps) <= 0:
                    return False
            except Exception:
                return False
        # 3. 业务动作约束 / 量化规则的静态校验（可选）
        action_constraints = rule.get("action_constraints")
        if action_constraints and not StaticValidator._validate_action_constraints_schema(
            action_constraints
        ):
            return False

        quantified_rules = rule.get("quantified_rules")
        if quantified_rules:
            # 确认量化规则配置结构合法（能被 QuantifiedRule 解析）
            try:
                for item in quantified_rules:
                    QuantifiedRule.from_dict(item)
            except Exception:
                return False

        return True

    @staticmethod
    def _validate_action_constraints_schema(constraints: Dict[str, Any]) -> bool:
        """
        校验 action_constraints 字段结构是否合理。

        支持的示例结构：
        {
            "type": "range", "min": -1, "max": 1
        }
        或
        {
            "type": "choices", "values": ["buy", "sell", "hold"]
        }
        """
        if not isinstance(constraints, dict):
            return False

        ctype = constraints.get("type")
        if ctype == "range":
            return "min" in constraints and "max" in constraints
        if ctype == "choices":
            values = constraints.get("values")
            return isinstance(values, (list, tuple)) and len(values) > 0
        # 其他类型暂时一律视为合法，避免过度限制
        return True


__all__ = [
    "EnvEventType",
    "EnvState",
    "EnvEventRecord",
    "EnvStateManager",
    "RuleOperator",
    "RuleTarget",
    "QuantifiedRule",
    "RuleLibrary",
    "RouteDispatcher",
    "StaticValidator",
]

