from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .agents import FinanceAgent, GeneralAgent, LegalAgent, TechAgent
from .blackboard import SharedBlackboard
from .classifier import ComplexityClassifier
from .models import BlackboardState, RoutingDecision


@dataclass
class WorkflowResult:
    decision: RoutingDecision
    blackboard_state: BlackboardState
    final_report: Dict[str, Any]


class AdaptiveTieredWorkflow:
    """Blackboard-based adaptive tiered routing workflow.

    Control-flow is centralized in this orchestrator while data-flow stays in the
    shared blackboard, enabling decentralized agent collaboration.
    """

    def __init__(self) -> None:
        self.classifier = ComplexityClassifier()
        self.agents = {
            "general_agent": GeneralAgent(),
            "legal_agent": LegalAgent(),
            "tech_agent": TechAgent(),
            "finance_agent": FinanceAgent(),
        }

    def run(self, user_input: str, session_id: str | None = None) -> WorkflowResult:
        decision = self.classifier.classify(user_input)
        bb = SharedBlackboard(
            BlackboardState(session_id=session_id) if session_id else BlackboardState()
        )

        bb.subscribe("critical_risk", self._on_critical_risk, agent_id="workflow")
        bb.update_global_state(
            {
                "status": "RUNNING",
                "current_phase": f"ROUTING_{decision.tier}",
                "risk_level": "LOW",
            },
            agent_id="workflow",
        )

        bb.write(
            zone="general_zone",
            content={"summary": user_input, "kind": "task_input"},
            tags=["input"],
            evidence_refs=[],
            agent_id="workflow",
        )

        self._execute_tier(bb=bb, decision=decision, user_input=user_input)

        if bb.state.global_state.status == "RUNNING":
            bb.update_global_state(
                {"status": "COMPLETED", "current_phase": "SYNTHESIZE"},
                agent_id="workflow",
            )

        report = self._synthesize_report(bb, decision)
        return WorkflowResult(decision=decision, blackboard_state=bb.state, final_report=report)

    def _execute_tier(self, bb: SharedBlackboard, decision: RoutingDecision, user_input: str) -> None:
        if decision.tier == "L1":
            self._run_agent(bb, "general_agent", user_input)
            return

        if decision.tier == "L2":
            for agent_id in ["legal_agent", "finance_agent", "tech_agent"]:
                if bb.state.global_state.status == "STOPPED":
                    return
                self._run_agent(bb, agent_id, user_input)
            return

        # L3: activate specialist pool + debate zone rounds.
        max_rounds = 3
        for round_idx in range(1, max_rounds + 1):
            if bb.state.global_state.status == "STOPPED":
                return

            bb.update_global_state(
                {"current_phase": f"DEBATE_ROUND_{round_idx}"},
                agent_id="workflow",
            )

            round_results: List[Tuple[str, Dict[str, Any]]] = []
            for agent_id in ["legal_agent", "finance_agent", "tech_agent"]:
                if bb.state.global_state.status == "STOPPED":
                    return
                result = self._run_agent(bb, agent_id, user_input)
                round_results.append((agent_id, result))

            # Agents exchange conclusions in debate zone (decentralized collaboration).
            for agent_id, result in round_results:
                bb.write(
                    zone="debate_zone",
                    content={
                        "summary": f"{agent_id} stance={result.get('stance')} risk={result.get('key_risk')}",
                        "stance": result.get("stance"),
                    },
                    tags=["debate", f"round_{round_idx}"],
                    evidence_refs=result.get("evidence", []),
                    agent_id=agent_id,
                )

            if self._debate_consensus(round_results):
                bb.update_global_state(
                    {"current_phase": f"DEBATE_CONSENSUS_{round_idx}"},
                    agent_id="workflow",
                )
                break

        if bb.state.global_state.status != "STOPPED":
            self._run_agent(bb, "general_agent", user_input)

    def _run_agent(self, bb: SharedBlackboard, agent_id: str, user_input: str) -> Dict[str, Any]:
        agent = self.agents[agent_id]

        # TODO(cost-optimization): plug semantic cache here to skip repeated calls.
        # TODO(latency-optimization): plug speculative sampling for parallel draft generation.
        return agent.run(blackboard=bb, project_text=user_input)

    @staticmethod
    def _debate_consensus(round_results: List[Tuple[str, Dict[str, Any]]]) -> bool:
        stances = [str(result.get("stance", "")) for _, result in round_results]
        if not stances:
            return False
        return len(set(stances)) == 1

    @staticmethod
    def _on_critical_risk(event: str, payload: Dict[str, Any]) -> None:
        _ = event
        _ = payload
        # Hook point for external notification, pager alerts, etc.
        return

    def _synthesize_report(self, bb: SharedBlackboard, decision: RoutingDecision) -> Dict[str, Any]:
        # Synthesis reads zone summaries only, not raw internals from agents.
        zone_summary: Dict[str, Any] = {}
        for zone_name, zone_state in bb.state.zones.items():
            zone_summary[zone_name] = [entry.content for entry in zone_state.entries]

        return {
            "session_id": bb.state.session_id,
            "routing_decision": decision.model_dump(),
            "global_state": bb.state.global_state.model_dump(),
            "zone_summary": zone_summary,
            "audit_count": len(bb.state.audit_log),
        }
