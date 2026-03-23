import json

from m7_context_analyzer import build_layer3_context, render_layer3_for_prompt


def run_layer3_demo() -> None:
    layer3 = build_layer3_context(
        user_id="u_001",
        current_query="我们最近两个月净流出较高，但想继续加大增长投放，如何平衡？",
        top_k=3,
    )

    print("=== Layer-3 User Profile Context ===")
    print(json.dumps(layer3, ensure_ascii=False, indent=2))
    print("\nRendered Prompt Block:")
    print(render_layer3_for_prompt(layer3))


if __name__ == "__main__":
    run_layer3_demo()
