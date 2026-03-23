import json

from m7_context_analyzer import build_layer1_context, render_layer1_for_prompt


def run_layer1_demo() -> None:
    user_input = "我们本季度目标是把月活从 2k 提升到 5k，同时避免现金流断裂。"
    uploaded_snippets = [
        {
            "file_name": "founder_note.txt",
            "snippet": "现有团队 5 人，技术 3 人，运营 1 人，市场 1 人。",
        },
        {
            "file_name": "cashflow.csv",
            "snippet": "month,cash_in,cash_out\\n2026-04,5000,9000\\n2026-05,4800,9200",
        },
    ]

    layer1 = build_layer1_context(
        user_input=user_input,
        uploaded_snippets=uploaded_snippets,
    )

    print("L1 Context JSON:")
    print(json.dumps(layer1, ensure_ascii=False, indent=2))
    print("\nRendered Prompt Block:")
    print(render_layer1_for_prompt(layer1))


if __name__ == "__main__":
    run_layer1_demo()
