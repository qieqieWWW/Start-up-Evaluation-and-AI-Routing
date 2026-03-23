import json

from m7_intent_engine import recognize_intent
from m7_trajectory_manager import build_trajectory_context


def run_demo() -> None:
    user_input = "这个项目看起来不错，但我担心他们的知识产权有隐患，而且现金流好像有点紧。"
    turns = [
        {"role": "user", "content": "我们做跨境SaaS，最近增长不错。"},
        {"role": "assistant", "content": "你更担心增长还是风险控制？"},
        {"role": "user", "content": "都有点担心，尤其是现金流和合规。"},
    ]

    intent = recognize_intent(user_input=user_input)
    trajectory = build_trajectory_context(
        current_input=user_input,
        conversation_turns=turns,
        previous_state="Waiting_For_Financials",
        user_id="u_001",
    )

    print("=== Intent Recognition ===")
    print(json.dumps(intent, ensure_ascii=False, indent=2))
    print("\n=== Trajectory Context ===")
    print(json.dumps(trajectory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_demo()
