from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from .models import AuditRecord, BlackboardState, ZoneEntry, ZoneState

EventHandler = Callable[[str, Dict[str, Any]], None]


class SharedBlackboard:
    """Thread-safe shared blackboard.

    Control flow is owned by workflow/orchestrator; agents only interact through
    read/write operations on this data plane.
    """

    def __init__(self, state: Optional[BlackboardState] = None) -> None:
        self._state = state or BlackboardState()
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)

    @property
    def state(self) -> BlackboardState:
        with self._lock:
            return self._state

    def read(self, zone: str, tags: Optional[List[str]] = None, agent_id: str = "system") -> List[ZoneEntry]:
        with self._lock:
            zone_state = self._state.zones.get(zone)
            if zone_state is None:
                return []

            selected = zone_state.entries
            if tags:
                selected = [
                    entry
                    for entry in selected
                    if set(tags).issubset(set(entry.tags))
                ]

            self._append_audit(
                agent_id=agent_id,
                action="READ",
                zone=zone,
                tags=tags or [],
                detail=f"read_entries={len(selected)}",
            )
            return list(selected)

    def write(
        self,
        zone: str,
        content: Dict[str, Any],
        tags: Optional[List[str]] = None,
        evidence_refs: Optional[List[str]] = None,
        agent_id: str = "system",
    ) -> ZoneEntry:
        with self._lock:
            if zone not in self._state.zones:
                self._state.zones[zone] = ZoneState(zone_name=zone, entries=[])

            entry = ZoneEntry(
                agent_id=agent_id,
                content=content,
                tags=tags or [],
                evidence_refs=evidence_refs or [],
            )
            self._state.zones[zone].entries.append(entry)
            self._append_audit(
                agent_id=agent_id,
                action="WRITE",
                zone=zone,
                tags=tags or [],
                detail=f"entry_id={entry.entry_id}",
            )

        self._publish("zone_write", {"zone": zone, "entry_id": entry.entry_id, "agent_id": agent_id})
        return entry

    def subscribe(self, event: str, handler: EventHandler, agent_id: str = "system") -> None:
        with self._lock:
            self._subscribers[event].append(handler)
            self._append_audit(
                agent_id=agent_id,
                action="SUBSCRIBE",
                zone=None,
                tags=[],
                detail=f"event={event}",
            )

    def update_global_state(self, patch: Dict[str, Any], agent_id: str = "system") -> None:
        with self._lock:
            current = self._state.global_state.model_dump()
            current.update(patch)
            self._state.global_state = self._state.global_state.__class__(**current)
            self._append_audit(
                agent_id=agent_id,
                action="EVENT",
                zone=None,
                tags=[],
                detail=f"global_state_patch={patch}",
            )

        if patch.get("risk_level") == "CRITICAL":
            self._publish("critical_risk", {"agent_id": agent_id, "patch": patch})

    def _publish(self, event: str, payload: Dict[str, Any]) -> None:
        # Event callbacks run out of the lock to avoid deadlocks.
        handlers = list(self._subscribers.get(event, []))
        for handler in handlers:
            try:
                handler(event, payload)
            except Exception:
                pass

    def _append_audit(self, agent_id: str, action: str, zone: Optional[str], tags: List[str], detail: str) -> None:
        self._state.audit_log.append(
            AuditRecord(
                agent_id=agent_id,
                action=action,
                zone=zone,
                tags=tags,
                detail=detail,
            )
        )
