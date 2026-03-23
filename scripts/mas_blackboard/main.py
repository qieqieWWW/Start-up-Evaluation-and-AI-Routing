from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from .workflow import AdaptiveTieredWorkflow
except ImportError:
    # Allow direct script execution: python scripts/mas_blackboard/main.py
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from mas_blackboard.workflow import AdaptiveTieredWorkflow  # type: ignore


def run_example() -> None:
    user_input = (
        "这个创业项目做跨境SaaS，增长看起来不错，但我担心知识产权、"
        "合规风险和现金流断裂，另外技术架构是否能支撑未来扩张也不确定，"
        "同时团队背景调查、市场验证和监管争议也需要同步评估。"
    )

    workflow = AdaptiveTieredWorkflow()
    result = workflow.run(user_input=user_input)

    print("=" * 72)
    print("Blackboard-based Adaptive Tiered Routing Demo")
    print("=" * 72)
    print("Routing Decision:")
    print(json.dumps(result.decision.model_dump(), ensure_ascii=False, indent=2))

    print("\nFinal Report:")
    print(json.dumps(result.final_report, ensure_ascii=False, indent=2))

    print("\nAudit Log (last 8 records):")
    for row in result.blackboard_state.audit_log[-8:]:
        print(
            f"- {row.timestamp.isoformat()} | agent={row.agent_id} | "
            f"action={row.action} | zone={row.zone} | detail={row.detail}"
        )


if __name__ == "__main__":
    run_example()
