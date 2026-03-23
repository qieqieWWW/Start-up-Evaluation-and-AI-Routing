import json
from typing import Any, Dict, List, Optional

from m7_context_analyzer import build_layer1_context, build_layer2_context, build_layer3_context, build_layer4_context
from m7_blender import blend_candidates
from m7_llm_client import DeepSeekClient
from m7_profile_rag import append_profile_record, infer_risk_appetite_from_text
from m7_prompt_builder import build_system_prompt, build_user_prompt


def _parse_json_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"raw_text": "", "parse_error": "empty_response"}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        return {"raw_text": text, "parse_error": "invalid_json"}


def run_expert_llm_inference(
    risk_level: str,
    reasons: List[str],
    intermediate: Dict[str, float],
    project_data: Dict[str, object],
    route_result: Dict[str, Any],
    user_input: str = "",
    uploaded_snippets: Optional[List[Dict[str, str]]] = None,
    conversation_turns: Optional[List[Dict[str, str]]] = None,
    summary_buffer: str = "",
    session_max_turns: int = 6,
    session_strategy: str = "sliding_window",
    user_id: str = "",
    profile_db_path: Optional[str] = None,
    profile_top_k: int = 5,
    global_kb_path: Optional[str] = None,
    global_kb_top_k: int = 5,
    auto_profile_log: bool = True,
    top_k: int = 1,
    model: str = "deepseek-chat",
) -> List[Dict[str, Any]]:
    selected_experts = route_result.get("selected_experts", [])
    if not selected_experts:
        return []

    client = DeepSeekClient(model=model)
    outputs: List[Dict[str, Any]] = []
    layer1_context = build_layer1_context(
        user_input=user_input,
        uploaded_snippets=uploaded_snippets,
    )
    layer2_context = build_layer2_context(
        conversation_turns=conversation_turns,
        max_turns=session_max_turns,
        summary_buffer=summary_buffer,
        strategy=session_strategy,
    )
    layer3_context = build_layer3_context(
        user_id=user_id,
        current_query=user_input,
        top_k=profile_top_k,
        profile_db_path=profile_db_path,
    )
    layer4_context = build_layer4_context(
        current_query=user_input,
        top_k=global_kb_top_k,
        kb_path=global_kb_path,
    )

    for expert in selected_experts[: max(1, top_k)]:
        system_prompt = build_system_prompt(expert)
        user_prompt = build_user_prompt(
            risk_level=risk_level,
            reasons=reasons,
            intermediate=intermediate,
            project_data=project_data,
            route_reason=str(route_result.get("route_reason", "")),
            layer1_context=layer1_context,
            layer2_context=layer2_context,
            layer3_context=layer3_context,
            layer4_context=layer4_context,
        )

        resp = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=700,
            response_format={"type": "json_object"},
        )

        parsed = _parse_json_response(resp.content)
        outputs.append(
            {
                "expert": expert,
                "model": resp.model,
                "usage": resp.usage,
                "parsed": parsed,
                "raw_content": resp.content,
            }
        )

    if auto_profile_log and user_id:
        profile_summary = layer3_context.get("profile_summary", {})
        dominant_risk_appetite = ""
        if isinstance(profile_summary, dict):
            dominant_risk_appetite = str(profile_summary.get("dominant_risk_appetite", ""))

        record = {
            "user_id": user_id,
            "industry_tags": [str(project_data.get("main_category", ""))] if project_data.get("main_category") else [],
            "preferences": [
                "auto_logged",
                f"route:{str(route_result.get('normalized_risk_level', ''))}",
            ],
            "risk_appetite": infer_risk_appetite_from_text(user_input, fallback=dominant_risk_appetite),
            "assessment_summary": f"用户本轮查询: {(user_input or '').strip()[:180]}",
            "preference_note": f"route_reason={str(route_result.get('route_reason', ''))}",
            "common_needs": "",
            "industry_comment": "",
            "meta": {
                "source": "run_expert_llm_inference",
                "selected_experts": [str(item.get("expert", {}).get("name", "")) for item in outputs],
                "risk_level": risk_level,
            },
        }
        append_profile_record(record=record, db_path=profile_db_path)

    return outputs


def run_expert_llm_inference_with_blender(
    risk_level: str,
    reasons: List[str],
    intermediate: Dict[str, float],
    project_data: Dict[str, object],
    route_result: Dict[str, Any],
    user_input: str = "",
    uploaded_snippets: Optional[List[Dict[str, str]]] = None,
    conversation_turns: Optional[List[Dict[str, str]]] = None,
    summary_buffer: str = "",
    session_max_turns: int = 6,
    session_strategy: str = "sliding_window",
    user_id: str = "",
    profile_db_path: Optional[str] = None,
    profile_top_k: int = 5,
    global_kb_path: Optional[str] = None,
    global_kb_top_k: int = 5,
    auto_profile_log: bool = True,
    top_k: int = 2,
    model: str = "deepseek-chat",
    use_llm_fuser: bool = True,
) -> Dict[str, Any]:
    """Run expert inference then blend outputs via PairRanker + GenFuser."""
    candidates = run_expert_llm_inference(
        risk_level=risk_level,
        reasons=reasons,
        intermediate=intermediate,
        project_data=project_data,
        route_result=route_result,
        user_input=user_input,
        uploaded_snippets=uploaded_snippets,
        conversation_turns=conversation_turns,
        summary_buffer=summary_buffer,
        session_max_turns=session_max_turns,
        session_strategy=session_strategy,
        user_id=user_id,
        profile_db_path=profile_db_path,
        profile_top_k=profile_top_k,
        global_kb_path=global_kb_path,
        global_kb_top_k=global_kb_top_k,
        auto_profile_log=auto_profile_log,
        top_k=top_k,
        model=model,
    )

    blended = blend_candidates(
        candidates=candidates,
        use_llm_fuser=use_llm_fuser,
        model=model,
    )

    return {
        "candidates": candidates,
        "ranked_candidates": blended.get("ranked_candidates", []),
        "fused_result": blended.get("fused_result", {}),
    }
