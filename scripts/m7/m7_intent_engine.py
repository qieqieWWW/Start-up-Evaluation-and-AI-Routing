import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from m7_llm_client import DeepSeekClient


def _default_intent_library_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "config" / "m7_intent_library.json"


def load_intent_library(library_path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = Path(library_path) if library_path else _default_intent_library_path()
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception:
        return []

    return []


def _char_ngrams(text: str, n: int = 2) -> List[str]:
    cleaned = "".join((text or "").split())
    if not cleaned:
        return []
    if len(cleaned) < n:
        return [cleaned]
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def _vectorize(text: str) -> Counter:
    return Counter(_char_ngrams(text, n=2))


def _cosine_sim(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _intent_to_text(intent_item: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in ["intent_id", "primary_intent", "sub_intent", "description"]:
        value = intent_item.get(key)
        if isinstance(value, str):
            chunks.append(value)
    keywords = intent_item.get("keywords", [])
    if isinstance(keywords, list):
        chunks.append(" ".join(str(x) for x in keywords))
    return " ".join(chunks)


def semantic_intent_matches(
    user_input: str,
    library_path: Optional[str] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    intents = load_intent_library(library_path=library_path)
    if not intents:
        return []

    q_vec = _vectorize(user_input or "")
    q_tokens = set((user_input or "").lower().replace("，", " ").replace(",", " ").split())

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in intents:
        sim = _cosine_sim(q_vec, _vectorize(_intent_to_text(item)))
        keywords = item.get("keywords", [])
        if isinstance(keywords, list) and q_tokens:
            overlap = len(q_tokens & set(str(x).lower() for x in keywords))
            sim += 0.06 * overlap
        scored.append((sim, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[: max(1, top_k)]
    return [
        {
            "score": round(score, 4),
            "intent": item,
        }
        for score, item in chosen
    ]


def _safe_json_parse(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}

    return {}


def _build_intent_prompt(user_input: str, semantic_matches: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system_prompt = (
        "你是创业项目路由系统的意图识别引擎。"
        "你的任务是将用户非结构化输入解析成严格JSON格式的路由指令。"
        "必须只输出JSON对象，不允许任何额外文本。"
        "字段必须包含：primary_intent, sub_intent, urgency, required_experts, missing_info, confidence_score, reasoning_summary。"
        "reasoning_summary必须简短，聚焦证据，不要长篇思维链。"
    )

    few_shot = {
        "input": "这个项目看起来不错，但我担心他们的知识产权有隐患，而且现金流有点紧。",
        "output": {
            "primary_intent": "risk_assessment",
            "sub_intent": "cross_domain_legal_finance",
            "urgency": "high",
            "required_experts": ["risk_guardian", "finance_advisor"],
            "missing_info": ["ip_documents", "cashflow_statement"],
            "confidence_score": 0.9,
            "reasoning_summary": "用户同时提及知识产权和现金流，属于跨域高优先风险。",
        },
    }

    user_prompt = {
        "task_input": user_input,
        "semantic_matches": semantic_matches,
        "few_shot_example": few_shot,
        "output_schema": {
            "primary_intent": "string",
            "sub_intent": "string",
            "urgency": "low|medium|high",
            "required_experts": ["string"],
            "missing_info": ["string"],
            "confidence_score": 0.0,
            "reasoning_summary": "string",
        },
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]


def _fallback_intent(user_input: str, semantic_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    top_item = semantic_matches[0] if semantic_matches else {}
    intent = top_item.get("intent", {}) if isinstance(top_item, dict) else {}
    required = intent.get("default_required_experts", ["risk_guardian", "finance_advisor"])
    confidence = float(top_item.get("score", 0.0)) if isinstance(top_item, dict) else 0.0

    urgency = "medium"
    text = user_input or ""
    if any(token in text for token in ["紧急", "马上", "立刻", "致命", "严重"]):
        urgency = "high"
    elif any(token in text for token in ["先看看", "不着急", "后续"]):
        urgency = "low"

    missing_info: List[str] = []
    if "现金流" in text or "财务" in text:
        missing_info.append("financial_statements")
    if "知识产权" in text or "专利" in text:
        missing_info.append("ip_documents")

    return {
        "primary_intent": intent.get("primary_intent", "risk_assessment"),
        "sub_intent": intent.get("sub_intent", "general_risk"),
        "urgency": urgency,
        "required_experts": required,
        "missing_info": missing_info,
        "confidence_score": round(min(0.95, 0.5 + confidence), 2),
        "reasoning_summary": "基于语义相似度和关键词规则得到的回退意图判定。",
    }


def recognize_intent(
    user_input: str,
    library_path: Optional[str] = None,
    model: str = "deepseek-chat",
) -> Dict[str, Any]:
    semantic_matches = semantic_intent_matches(
        user_input=user_input,
        library_path=library_path,
        top_k=3,
    )

    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            client = DeepSeekClient(model=model)
            resp = client.chat(
                messages=_build_intent_prompt(user_input=user_input, semantic_matches=semantic_matches),
                temperature=0.1,
                max_tokens=450,
                response_format={"type": "json_object"},
            )
            parsed = _safe_json_parse(resp.content)
            if parsed:
                parsed.setdefault("required_experts", ["risk_guardian", "finance_advisor"])
                parsed.setdefault("missing_info", [])
                parsed.setdefault("confidence_score", 0.7)
                parsed["semantic_matches"] = semantic_matches
                return parsed
        except Exception:
            pass

    fallback = _fallback_intent(user_input=user_input, semantic_matches=semantic_matches)
    fallback["semantic_matches"] = semantic_matches
    return fallback
