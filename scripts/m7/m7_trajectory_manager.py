from typing import Any, Dict, List, Optional

from m7_profile_rag import build_profile_summary, retrieve_profile_records


PRONOUNS = ["它", "这个", "那个", "该项目", "这个公司", "那个项目"]


def _extract_entities(conversation_turns: List[Dict[str, str]], current_input: str) -> List[str]:
    entities: List[str] = []
    for turn in conversation_turns:
        content = str(turn.get("content", ""))
        if "SaaS" in content and "SaaS" not in entities:
            entities.append("SaaS")
        if "跨境" in content and "跨境电商" not in entities:
            entities.append("跨境电商")
        if "现金流" in content and "现金流" not in entities:
            entities.append("现金流")
        if "知识产权" in content and "知识产权" not in entities:
            entities.append("知识产权")
    if "团队" in current_input and "团队" not in entities:
        entities.append("团队")
    return entities


def _resolve_pronouns(current_input: str, entities: List[str]) -> Dict[str, str]:
    if not entities:
        return {}
    resolved: Dict[str, str] = {}
    last_entity = entities[-1]
    for p in PRONOUNS:
        if p in current_input:
            resolved[p] = last_entity
    return resolved


def _next_state(previous_state: str, current_input: str) -> str:
    text = current_input or ""
    if "这是财报" in text or "财报" in text or "现金流" in text:
        return "Analyzing_Financials"
    if "知识产权" in text or "专利" in text or "侵权" in text:
        return "Analyzing_IP"
    if "缺少" in text and "财务" in text:
        return "Waiting_For_Financials"
    if previous_state:
        return previous_state
    return "Routing_Ready"


def build_trajectory_context(
    current_input: str,
    conversation_turns: Optional[List[Dict[str, str]]] = None,
    previous_state: str = "",
    user_id: str = "",
    profile_db_path: Optional[str] = None,
) -> Dict[str, Any]:
    turns = conversation_turns or []
    entities = _extract_entities(turns, current_input)
    resolved_refs = _resolve_pronouns(current_input, entities)
    current_state = _next_state(previous_state=previous_state, current_input=current_input)

    long_term = retrieve_profile_records(
        user_id=user_id,
        query=current_input,
        top_k=5,
        db_path=profile_db_path,
    )
    profile_summary = build_profile_summary(long_term)

    forced_nodes: List[str] = []
    top_prefs = " ".join(str(x) for x in profile_summary.get("top_preferences", []))
    is_early_stage = any(k in current_input for k in ["早期", "种子轮", "初创", "0到1"])
    if is_early_stage and ("忽略团队" in top_prefs or "重市场轻团队" in top_prefs):
        forced_nodes.append("team_risk_strong_alert")

    return {
        "short_term_memory": {
            "window_turns": turns[-6:],
            "entities": entities,
            "resolved_references": resolved_refs,
            "state_machine": {
                "previous_state": previous_state or "",
                "current_state": current_state,
            },
        },
        "long_term_memory": {
            "retrieved_history": long_term,
            "profile_summary": profile_summary,
            "forced_nodes": forced_nodes,
        },
    }
