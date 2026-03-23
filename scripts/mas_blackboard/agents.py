from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .blackboard import SharedBlackboard


@dataclass
class MockLargeLLM:
    """Simulates a strong model (e.g., GPT-4/Claude).

    For prototype purpose we keep deterministic rules so behavior is testable and
    easy to swap with real API calls later.
    """

    model_name: str = "mock-gpt4"

    def analyze(self, role: str, project_text: str, context_snippets: List[str]) -> Dict[str, Any]:
        text = project_text or ""
        ctx = " ".join(context_snippets)

        fatal_tokens = ["违法", "侵权", "资金断裂", "破产", "致命漏洞"]
        is_critical = any(token in (text + ctx) for token in fatal_tokens)

        if role == "legal":
            key_risk = "知识产权与合规风险"
            evidence = ["用户提到知识产权/合规相关描述"] if ("知识产权" in text or "合规" in text) else []
            stance = "NO_GO" if is_critical else "CAUTION"
        elif role == "finance":
            key_risk = "现金流与融资节奏风险"
            evidence = ["用户提到现金流、预算或融资压力"] if any(t in text for t in ["现金流", "预算", "融资"]) else []
            stance = "NO_GO" if is_critical else "CAUTION"
        elif role == "tech":
            key_risk = "技术可交付与稳定性风险"
            evidence = ["用户提到技术或架构复杂度"] if any(t in text for t in ["技术", "架构", "系统"]) else []
            stance = "CAUTION"
        else:
            key_risk = "综合经营与执行风险"
            evidence = ["综合上下文进行通用评估"]
            stance = "GO" if not is_critical else "NO_GO"

        return {
            "role": role,
            "key_risk": key_risk,
            "stance": stance,
            "is_critical": is_critical,
            "summary": f"{role} agent 认为主要风险是：{key_risk}",
            "evidence": evidence,
            "tags": [role, "analysis", stance.lower()],
        }


class BaseSpecialistAgent:
    """Base class for blackboard-driven specialist agents.

    Agents must read only relevant zones and write to owned zones, which enforces
    data-flow isolation and decentralized collaboration.
    """

    agent_id: str = "base_agent"
    role: str = "general"
    read_zones: List[str] = ["general_zone"]
    write_zone: str = "general_zone"

    def __init__(self, llm: MockLargeLLM | None = None) -> None:
        self.llm = llm or MockLargeLLM()

    def run(self, blackboard: SharedBlackboard, project_text: str) -> Dict[str, Any]:
        snippets: List[str] = []
        for zone in self.read_zones:
            entries = blackboard.read(zone=zone, tags=None, agent_id=self.agent_id)
            for entry in entries[-3:]:
                text = str(entry.content.get("summary", ""))
                if text:
                    snippets.append(text)

        result = self.llm.analyze(role=self.role, project_text=project_text, context_snippets=snippets)

        blackboard.write(
            zone=self.write_zone,
            content={
                "summary": result["summary"],
                "key_risk": result["key_risk"],
                "stance": result["stance"],
            },
            tags=result.get("tags", []),
            evidence_refs=result.get("evidence", []),
            agent_id=self.agent_id,
        )

        gs = blackboard.state.global_state
        completed = list(gs.completed_agents)
        if self.agent_id not in completed:
            completed.append(self.agent_id)

        patch = {
            "completed_agents": completed,
            "current_phase": f"{self.agent_id}_done",
        }

        if result.get("is_critical"):
            patch["risk_level"] = "CRITICAL"
            patch["status"] = "STOPPED"

        blackboard.update_global_state(patch, agent_id=self.agent_id)
        return result


class GeneralAgent(BaseSpecialistAgent):
    agent_id = "general_agent"
    role = "general"
    read_zones = ["general_zone", "debate_zone"]
    write_zone = "general_zone"


class LegalAgent(BaseSpecialistAgent):
    agent_id = "legal_agent"
    role = "legal"
    read_zones = ["legal_zone", "debate_zone", "general_zone"]
    write_zone = "legal_zone"


class TechAgent(BaseSpecialistAgent):
    agent_id = "tech_agent"
    role = "tech"
    read_zones = ["tech_zone", "debate_zone", "general_zone"]
    write_zone = "tech_zone"


class FinanceAgent(BaseSpecialistAgent):
    agent_id = "finance_agent"
    role = "finance"
    read_zones = ["finance_zone", "debate_zone", "general_zone"]
    write_zone = "finance_zone"
