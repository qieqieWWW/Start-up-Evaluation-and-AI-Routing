from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import RoutingDecision


@dataclass
class MockSmallLLM:
    """Simulates a small model (e.g., Qwen-7B) for cheap routing decisions."""

    def score_complexity(self, user_input: str) -> float:
        text = user_input or ""
        score = 2.0
        length_factor = min(4.0, len(text) / 180.0)
        score += length_factor

        complexity_signals = ["知识产权", "现金流", "合规", "架构", "跨境", "融资", "多域", "争议"]
        score += 0.8 * sum(1 for token in complexity_signals if token in text)

        if any(token in text for token in ["同时", "但是", "不过", "一方面"]):
            score += 1.0

        return max(0.0, min(10.0, round(score, 2)))


class ComplexityClassifier:
    def __init__(self) -> None:
        self.model = MockSmallLLM()

    def classify(self, user_input: str) -> RoutingDecision:
        score = self.model.score_complexity(user_input)

        if score <= 3.5:
            tier = "L1"
            agents = ["general_agent"]
        elif score <= 6.5:
            tier = "L2"
            agents = ["legal_agent", "finance_agent", "tech_agent"]
        else:
            tier = "L3"
            agents = ["legal_agent", "finance_agent", "tech_agent", "general_agent"]

        return RoutingDecision(
            complexity_score=score,
            tier=tier,
            recommended_agents=agents,
            reason=f"Complexity score={score} leads to tier={tier}.",
        )
