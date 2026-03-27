import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _default_profile_db_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "config" / "m7_user_profile_records.json"


def load_profile_records(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load profile records from JSON file.

    File format: a JSON array of profile record objects.
    """
    path = Path(db_path) if db_path else _default_profile_db_path()
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
        "assessment_summary",
        "preference_note",
        "risk_appetite",
        "common_needs",
        "industry_comment",
    ]:
        value = record.get(key)
        if isinstance(value, str):
            chunks.append(value)

    for key in ["industry_tags", "preferences"]:
        value = record.get(key)
        if isinstance(value, list):
            chunks.append(" ".join(str(x) for x in value))

    return " ".join(chunks)


def retrieve_profile_records(
    user_id: str,
    query: str,
    top_k: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k user history/profile records using local vector similarity.

    This is a lightweight in-repo vector retrieval that can be replaced by a real
    vector database later with the same output contract.
    """
    records = load_profile_records(db_path=db_path)
    if not records:
        return []

    q_vec = _vectorize(query or "")
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for record in records:
        text = _record_to_text(record)
        score = _cosine_sim(q_vec, _vectorize(text))

        # Personalization boost for the same user
        if user_id and str(record.get("user_id", "")) == user_id:
            score += 0.15

        # Keep relevant records or same-user records
        if score > 0.0 or (user_id and str(record.get("user_id", "")) == user_id):
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


def build_profile_summary(retrieved_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate retrieved records to a compact personalization summary."""
    tag_counter: Counter = Counter()
    pref_counter: Counter = Counter()
    risk_counter: Counter = Counter()
    past_summaries: List[str] = []

    for item in retrieved_records:
        record = item.get("record", {})
        if not isinstance(record, dict):
            continue

        tags = record.get("industry_tags", [])
        if isinstance(tags, list):
            tag_counter.update(str(tag) for tag in tags)

        prefs = record.get("preferences", [])
        if isinstance(prefs, list):
            pref_counter.update(str(pref) for pref in prefs)

        risk_appetite = record.get("risk_appetite")
        if isinstance(risk_appetite, str) and risk_appetite.strip():
            risk_counter.update([risk_appetite.strip()])

        summary = record.get("assessment_summary")
        if isinstance(summary, str) and summary.strip():
            past_summaries.append(summary.strip())

    return {
        "common_industry_tags": [k for k, _ in tag_counter.most_common(5)],
        "top_preferences": [k for k, _ in pref_counter.most_common(5)],
        "dominant_risk_appetite": (risk_counter.most_common(1)[0][0] if risk_counter else ""),
        "past_assessment_summaries": past_summaries[:5],
    }


def infer_risk_appetite_from_text(text: str, fallback: str = "") -> str:
    """Infer user risk appetite from latest query text using simple keyword rules."""
    value = (text or "").strip()
    if not value:
        return fallback

    if any(token in value for token in ["高风险", "高回报", "激进", "快速扩张", "加杠杆"]):
        return "high"
    if any(token in value for token in ["稳健", "保守", "现金流安全", "降低风险", "兜底"]):
        return "low"
    return fallback or "medium"


def save_profile_records(records: List[Dict[str, Any]], db_path: Optional[str] = None) -> Path:
    """Persist profile records to JSON file, creating parent directories if missing."""
    path = Path(db_path) if db_path else _default_profile_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    return path


def append_profile_record(record: Dict[str, Any], db_path: Optional[str] = None) -> Dict[str, Any]:
    """Append a profile record to DB and return persistence metadata."""
    records = load_profile_records(db_path=db_path)
    enriched = dict(record)
    enriched.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    records.append(enriched)
    path = save_profile_records(records, db_path=db_path)
    return {"db_path": str(path), "total_records": len(records)}
