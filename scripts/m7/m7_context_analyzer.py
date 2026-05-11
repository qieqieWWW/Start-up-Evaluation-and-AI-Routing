import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompts.loader import load_prompt_dict

# 动态添加 m7 目录到 sys.path
_m7_dir = Path(__file__).parent
if str(_m7_dir) not in sys.path:
    sys.path.insert(0, str(_m7_dir))

from m7_global_kb import retrieve_global_kb, summarize_global_kb
from m7_knowledge_graph import retrieve_knowledge_graph_hits, summarize_knowledge_graph_hits
from m7_profile_rag import build_profile_summary, retrieve_profile_records


def build_layer1_context(
    user_input: str,
    uploaded_snippets: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build layer-1 context from raw user inputs with no semantic processing.

    The first layer keeps user's direct text and uploaded file snippets as-is,
    so downstream prompts can solve the immediate user request first.
    """
    snippets = uploaded_snippets or []
    return {
        "layer": "L1_user_direct_input",
        "user_input": user_input or "",
        "uploaded_snippets": snippets,
        "notes": "raw passthrough context, no transformation",
    }


def render_layer1_for_prompt(layer1_context: Dict[str, Any]) -> str:
    """Render layer-1 context for direct prompt embedding."""
    layers = load_prompt_dict("m7/context_layers.json")["layers"]
    return layers["L1"]["template"].format(
        layer_context=json.dumps(layer1_context, ensure_ascii=False),
    )


def _to_summary_lines(turns: List[Dict[str, str]]) -> List[str]:
    """Convert turns into concise role-tagged lines without semantic rewriting."""
    lines: List[str] = []
    for idx, turn in enumerate(turns, 1):
        role = (turn.get("role") or "unknown").strip()
        content = (turn.get("content") or "").strip()
        lines.append(f"{idx}. [{role}] {content}")
    return lines


def build_layer2_context(
    conversation_turns: Optional[List[Dict[str, str]]] = None,
    max_turns: int = 6,
    summary_buffer: str = "",
    strategy: str = "sliding_window",
) -> Dict[str, Any]:
    """Build layer-2 session context using sliding window or summary buffer.

    - sliding_window: keep latest N turns as lightweight summary lines.
    - summary_buffer: merge previous summary buffer + latest N turns summary lines.
    """
    turns = conversation_turns or []
    window = turns[-max(1, max_turns) :]
    window_lines = _to_summary_lines(window)

    merged_summary = ""
    if strategy == "summary_buffer":
        pieces = []
        if summary_buffer.strip():
            pieces.append(summary_buffer.strip())
        if window_lines:
            pieces.append("\n".join(window_lines))
        merged_summary = "\n".join(pieces).strip()

    return {
        "layer": "L2_session_context",
        "strategy": strategy,
        "max_turns": max(1, max_turns),
        "window_turns": window,
        "window_summary": "\n".join(window_lines),
        "summary_buffer": summary_buffer,
        "merged_summary": merged_summary,
        "notes": "session continuity context for reference resolution",
    }


def render_layer2_for_prompt(layer2_context: Dict[str, Any]) -> str:
    """Render layer-2 context for prompt embedding."""
    layers = load_prompt_dict("m7/context_layers.json")["layers"]
    return layers["L2"]["template"].format(
        layer_context=json.dumps(layer2_context, ensure_ascii=False),
    )


def build_layer3_context(
    user_id: str,
    current_query: str,
    top_k: int = 5,
    profile_db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build layer-3 context from user historical profile via local vector retrieval."""
    retrieved = retrieve_profile_records(
        user_id=user_id,
        query=current_query,
        top_k=top_k,
        db_path=profile_db_path,
    )
    profile_summary = build_profile_summary(retrieved)

    return {
        "layer": "L3_user_profile",
        "user_id": user_id,
        "retrieval_method": "local_vector_rag",
        "top_k": max(1, top_k),
        "retrieved_records": retrieved,
        "profile_summary": profile_summary,
        "notes": "personalized routing hints from historical records",
    }


def render_layer3_for_prompt(layer3_context: Dict[str, Any]) -> str:
    """Render layer-3 user profile context for prompt embedding."""
    layers = load_prompt_dict("m7/context_layers.json")["layers"]
    return layers["L3"]["template"].format(
        layer_context=json.dumps(layer3_context, ensure_ascii=False),
    )


def build_layer4_context(
    current_query: str,
    top_k: int = 5,
    kb_path: Optional[str] = None,
    knowledge_graph_path: Optional[str] = None,
    knowledge_graph_top_k: int = 3,
) -> Dict[str, Any]:
    """Build layer-4 context from static global knowledge base retrieval."""
    retrieved = retrieve_global_kb(
        query=current_query,
        top_k=top_k,
        kb_path=kb_path,
    )
    kb_summary = summarize_global_kb(retrieved)
    kg_hits = retrieve_knowledge_graph_hits(
        query=current_query,
        top_k=knowledge_graph_top_k,
        graph_path=knowledge_graph_path,
    )
    kg_summary = summarize_knowledge_graph_hits(kg_hits)

    return {
        "layer": "L4_global_knowledge",
        "retrieval_method": "static_kb_retrieval",
        "top_k": max(1, top_k),
        "retrieved_records": retrieved,
        "kb_summary": kb_summary,
        "knowledge_graph_hits": kg_hits,
        "knowledge_graph_summary": kg_summary,
        "knowledge_graph_provider": "placeholder_json_graph_adapter",
        "notes": "grounding references for professional baseline",
    }


def render_layer4_for_prompt(layer4_context: Dict[str, Any]) -> str:
    """Render layer-4 global knowledge grounding context for prompt embedding."""
    layers = load_prompt_dict("m7/context_layers.json")["layers"]
    return layers["L4"]["template"].format(
        layer_context=json.dumps(layer4_context, ensure_ascii=False),
    )
