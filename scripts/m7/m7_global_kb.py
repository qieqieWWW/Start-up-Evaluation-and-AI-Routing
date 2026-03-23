import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _default_kb_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "config" / "m7_global_knowledge_base.json"


def load_global_kb_records(kb_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load global knowledge base records from JSON file."""
    path = Path(kb_path) if kb_path else _default_kb_path()
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


def _record_to_text(record: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for key in [
        "title",
        "category",
        "industry",
        "risk_model_definition",
        "success_case",
        "guideline",
    ]:
        value = record.get(key)
        if isinstance(value, str):
            chunks.append(value)

    for key in ["keywords", "tags"]:
        value = record.get(key)
        if isinstance(value, list):
            chunks.append(" ".join(str(x) for x in value))

    return " ".join(chunks)


def retrieve_global_kb(
    query: str,
    top_k: int = 5,
    kb_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve global grounding knowledge with hybrid scoring.

    Score = cosine similarity + keyword overlap bonus.
    """
    records = load_global_kb_records(kb_path=kb_path)
    if not records:
        return []

    q_text = query or ""
    q_vec = _vectorize(q_text)
    query_tokens = set(q_text.lower().replace("，", " ").replace(",", " ").split())

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for record in records:
        r_text = _record_to_text(record)
        score = _cosine_sim(q_vec, _vectorize(r_text))

        keywords = record.get("keywords", [])
        if isinstance(keywords, list) and query_tokens:
            kw_tokens = set(str(x).lower() for x in keywords)
            overlap = len(query_tokens & kw_tokens)
            score += 0.05 * overlap

        if score > 0.0:
            scored.append((score, record))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[: max(1, top_k)]

    return [
        {
            "score": round(score, 4),
            "record": record,
        }
        for score, record in chosen
    ]


def summarize_global_kb(retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create concise grounding summary from retrieved KB items."""
    categories: Counter = Counter()
    industries: Counter = Counter()
    key_guidelines: List[str] = []

    for item in retrieved:
        record = item.get("record", {})
        if not isinstance(record, dict):
            continue

        category = record.get("category")
        if isinstance(category, str) and category.strip():
            categories.update([category.strip()])

        industry = record.get("industry")
        if isinstance(industry, str) and industry.strip():
            industries.update([industry.strip()])

        guideline = record.get("guideline")
        if isinstance(guideline, str) and guideline.strip():
            key_guidelines.append(guideline.strip())

    return {
        "top_categories": [k for k, _ in categories.most_common(5)],
        "top_industries": [k for k, _ in industries.most_common(5)],
        "key_guidelines": key_guidelines[:5],
    }
