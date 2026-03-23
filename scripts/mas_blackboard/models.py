from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

try:
    _pydantic = importlib.import_module("pydantic")
    BaseModel = _pydantic.BaseModel
    Field = _pydantic.Field
except ImportError:
    # Offline fallback: keep Pydantic-like API surface for prototype execution.
    @dataclass
    class _FieldSpec:
        default: Any = ...
        default_factory: Optional[Any] = None

    def Field(default: Any = ..., default_factory: Optional[Any] = None, **_: Any) -> _FieldSpec:
        return _FieldSpec(default=default, default_factory=default_factory)

    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                if name in kwargs:
                    value = kwargs[name]
                else:
                    default_value = getattr(self.__class__, name, ...)
                    if isinstance(default_value, _FieldSpec):
                        if default_value.default_factory is not None:
                            value = default_value.default_factory()
                        elif default_value.default is not ...:
                            value = default_value.default
                        else:
                            raise TypeError(f"Missing required field: {name}")
                    elif default_value is not ...:
                        value = default_value
                    else:
                        raise TypeError(f"Missing required field: {name}")
                setattr(self, name, value)

        def model_dump(self) -> Dict[str, Any]:
            data: Dict[str, Any] = {}
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                value = getattr(self, name)
                if isinstance(value, BaseModel):
                    data[name] = value.model_dump()
                elif isinstance(value, list):
                    data[name] = [item.model_dump() if isinstance(item, BaseModel) else item for item in value]
                elif isinstance(value, dict):
                    copied: Dict[str, Any] = {}
                    for k, v in value.items():
                        copied[k] = v.model_dump() if isinstance(v, BaseModel) else v
                    data[name] = copied
                else:
                    data[name] = value
            return data


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
TaskTier = Literal["L1", "L2", "L3"]


class RoutingDecision(BaseModel):
    complexity_score: float = Field(..., ge=0.0, le=10.0)
    tier: TaskTier
    recommended_agents: List[str] = Field(default_factory=list)
    reason: str = ""


class ZoneEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: Dict[str, Any]
    evidence_refs: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ZoneState(BaseModel):
    zone_name: str
    entries: List[ZoneEntry] = Field(default_factory=list)


class GlobalState(BaseModel):
    status: Literal["RUNNING", "COMPLETED", "STOPPED"] = "RUNNING"
    current_phase: str = "INIT"
    risk_level: RiskLevel = "LOW"
    completed_agents: List[str] = Field(default_factory=list)


class AuditRecord(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str
    action: Literal["READ", "WRITE", "SUBSCRIBE", "EVENT"]
    zone: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    detail: str = ""


class BlackboardState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    global_state: GlobalState = Field(default_factory=GlobalState)
    zones: Dict[str, ZoneState] = Field(
        default_factory=lambda: {
            "legal_zone": ZoneState(zone_name="legal_zone"),
            "tech_zone": ZoneState(zone_name="tech_zone"),
            "finance_zone": ZoneState(zone_name="finance_zone"),
            "debate_zone": ZoneState(zone_name="debate_zone"),
            "general_zone": ZoneState(zone_name="general_zone"),
        }
    )
    audit_log: List[AuditRecord] = Field(default_factory=list)
