import json

from m7_profile_rag import append_profile_record, load_profile_records


def run_profile_autolog_demo() -> None:
    before = load_profile_records()

    result = append_profile_record(
        {
            "user_id": "u_autolog_demo",
            "industry_tags": ["SaaS"],
            "preferences": ["auto_logged", "demo"],
            "risk_appetite": "medium",
            "assessment_summary": "自动写入演示：用户咨询增长与现金流平衡。",
            "preference_note": "demo record",
            "common_needs": "增长与风控平衡",
            "industry_comment": "",
            "meta": {"source": "m7_profile_autolog_demo"},
        }
    )

    after = load_profile_records()
    print("Before:", len(before))
    print("After:", len(after))
    print("Write Result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Last Record:")
    print(json.dumps(after[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_profile_autolog_demo()
