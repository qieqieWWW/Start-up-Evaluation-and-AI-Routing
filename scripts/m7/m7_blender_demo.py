import json

from m7_blender import blend_candidates


def run_blender_demo() -> None:
    candidates = [
        {
            "expert": {"name": "finance_advisor", "role": "财务专家"},
            "parsed": {
                "risk_summary": "现金流存在 2 个月内断裂风险，需立即调整支出结构。",
                "actions": [
                    {"priority": 1, "title": "冻结低ROI投放", "owner": "增长运营", "due_days": 2, "expected_impact": "净流出下降"},
                    {"priority": 2, "title": "重排付款节奏", "owner": "财务", "due_days": 3, "expected_impact": "延长runway"},
                    {"priority": 3, "title": "制定周度现金流看板", "owner": "财务分析", "due_days": 5, "expected_impact": "提升可见性"},
                ],
                "alerts": ["若两周内净流出不下降，需启动应急融资预案"],
            },
        },
        {
            "expert": {"name": "growth_strategist", "role": "增长专家"},
            "parsed": {
                "risk_summary": "增长可以继续，但必须先修复留存漏斗避免无效获客。",
                "actions": [
                    {"priority": 1, "title": "优化Onboarding流程", "owner": "产品增长", "due_days": 7, "expected_impact": "提升激活率"},
                    {"priority": 2, "title": "聚焦高转化渠道", "owner": "增长团队", "due_days": 4, "expected_impact": "降低CAC"},
                    {"priority": 3, "title": "建立留存实验节奏", "owner": "增长分析", "due_days": 6, "expected_impact": "提升续费"},
                ],
                "alerts": ["若渠道CAC继续上升，应暂停扩量"],
            },
        },
    ]

    result = blend_candidates(candidates=candidates, use_llm_fuser=False)
    print("=== Ranked Candidates ===")
    print(json.dumps(result.get("ranked_candidates", []), ensure_ascii=False, indent=2))
    print("\n=== Fused Result ===")
    print(json.dumps(result.get("fused_result", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_blender_demo()
