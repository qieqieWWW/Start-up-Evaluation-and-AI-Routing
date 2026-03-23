import json

from m7_context_analyzer import build_layer4_context, render_layer4_for_prompt


def run_layer4_demo() -> None:
    layer4 = build_layer4_context(
        current_query="我们是跨境SaaS，现金流有压力，但希望提升留存并避免单平台风险。",
        top_k=3,
    )

    print("=== Layer-4 Global Knowledge Context ===")
    print(json.dumps(layer4, ensure_ascii=False, indent=2))
    print("\nRendered Prompt Block:")
    print(render_layer4_for_prompt(layer4))


if __name__ == "__main__":
    run_layer4_demo()
