import json

from m7_context_analyzer import build_layer2_context, render_layer2_for_prompt


def run_layer2_demo() -> None:
    conversation_turns = [
        {"role": "user", "content": "我们是做 B2B SaaS 的，主要客户是跨境电商卖家。"},
        {"role": "assistant", "content": "了解，你们当前更关注增长还是现金流安全？"},
        {"role": "user", "content": "两者都要，但优先现金流，另外下个月要扩招 2 个销售。"},
        {"role": "assistant", "content": "明白，是否有最近 2 个月收支数据？"},
        {"role": "user", "content": "有，最近两个月净流出都在 4k 美元左右。"},
        {"role": "assistant", "content": "你期望 90 天内最重要的目标是什么？"},
        {"role": "user", "content": "把客户留存提升到 85%，并防止现金流断裂。"},
    ]

    sliding = build_layer2_context(
        conversation_turns=conversation_turns,
        max_turns=4,
        strategy="sliding_window",
    )
    buffered = build_layer2_context(
        conversation_turns=conversation_turns,
        max_turns=4,
        summary_buffer="历史摘要: 团队关注现金流安全，目标是稳定续费并减少净流出。",
        strategy="summary_buffer",
    )

    print("=== Sliding Window ===")
    print(json.dumps(sliding, ensure_ascii=False, indent=2))
    print(render_layer2_for_prompt(sliding))

    print("\n=== Summary Buffer ===")
    print(json.dumps(buffered, ensure_ascii=False, indent=2))
    print(render_layer2_for_prompt(buffered))


if __name__ == "__main__":
    run_layer2_demo()
