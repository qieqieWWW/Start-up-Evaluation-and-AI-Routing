import json
from typing import Any, Dict, List

from m7_llm_client import DeepSeekClient
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
    top_k: int = 1,
    model: str = "deepseek-chat",
) -> List[Dict[str, Any]]:
    selected_experts = route_result.get("selected_experts", [])
    if not selected_experts:
        return []

    client = DeepSeekClient(model=model)
    outputs: List[Dict[str, Any]] = []

    for expert in selected_experts[: max(1, top_k)]:
        system_prompt = build_system_prompt(expert)
        user_prompt = build_user_prompt(
            risk_level=risk_level,
            reasons=reasons,
            intermediate=intermediate,
            project_data=project_data,
            route_reason=str(route_result.get("route_reason", "")),
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

    return outputs
