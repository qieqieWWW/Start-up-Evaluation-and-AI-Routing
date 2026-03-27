import json
import os
from typing import Any, Dict, List, Optional, Tuple

from m7_llm_client import DeepSeekClient


def _candidate_quality_score(candidate: Dict[str, Any]) -> float:
    parsed = candidate.get("parsed", {})
    if not isinstance(parsed, dict):
        return 0.1

    score = 0.2

    if parsed.get("parse_error"):
        score -= 0.15

    risk_summary = parsed.get("risk_summary")
    if isinstance(risk_summary, str) and risk_summary.strip():
        score += 0.18

    actions = parsed.get("actions", [])
    if isinstance(actions, list):
        score += min(0.3, 0.08 * len(actions))

    alerts = parsed.get("alerts", [])
    if isinstance(alerts, list):
        score += min(0.12, 0.04 * len(alerts))

    # Reward clearly structured action entries
    if isinstance(actions, list):
        valid_actions = 0
        for action in actions:
            if isinstance(action, dict) and action.get("title") and action.get("owner"):
                valid_actions += 1
        score += min(0.2, 0.05 * valid_actions)

    return round(max(0.0, min(1.0, score)), 4)


def pairrank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """PairRanker: do pairwise comparisons and produce ranking metadata."""
    if not candidates:
        return []

    base_scores = [_candidate_quality_score(item) for item in candidates]
    wins = [0 for _ in candidates]

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            si = base_scores[i]
            sj = base_scores[j]
            if si >= sj:
                wins[i] += 1
            else:
                wins[j] += 1

    ranked: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, candidate in enumerate(candidates):
        pair_score = 0.0
        if len(candidates) > 1:
            pair_score = wins[idx] / float(len(candidates) - 1)
        total_score = 0.55 * base_scores[idx] + 0.45 * pair_score
        ranked.append((total_score, wins[idx], candidate))

    ranked.sort(key=lambda x: x[0], reverse=True)

    result: List[Dict[str, Any]] = []
    for rank_idx, (score, win_count, candidate) in enumerate(ranked, 1):
        item = dict(candidate)
        item["pairrank"] = {
            "rank": rank_idx,
            "score": round(score, 4),
            "wins": int(win_count),
            "base_quality": _candidate_quality_score(candidate),
        }
        result.append(item)

    return result


def _dedup_actions(actions: List[Dict[str, Any]], max_actions: int = 6) -> List[Dict[str, Any]]:
    seen = set()
    merged: List[Dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        title = str(action.get("title", "")).strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(action)
        if len(merged) >= max(1, max_actions):
            break
    return merged


def _rule_based_fuse(ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ranked_candidates:
        return {
            "fused_risk_summary": "",
            "fused_actions": [],
            "fused_alerts": [],
            "source_experts": [],
            "fusion_method": "rule_based",
            "fusion_confidence": 0.0,
        }

    top = ranked_candidates[0]
    parsed_top = top.get("parsed", {}) if isinstance(top.get("parsed", {}), dict) else {}

    source_experts: List[str] = []
    all_actions: List[Dict[str, Any]] = []
    all_alerts: List[str] = []
    summary_parts: List[str] = []

    for candidate in ranked_candidates:
        expert = candidate.get("expert", {})
        if isinstance(expert, dict):
            name = str(expert.get("name", "")).strip()
            if name:
                source_experts.append(name)

        parsed = candidate.get("parsed", {})
        if not isinstance(parsed, dict):
            continue

        summary = parsed.get("risk_summary")
        if isinstance(summary, str) and summary.strip():
            summary_parts.append(summary.strip())

        actions = parsed.get("actions", [])
        if isinstance(actions, list):
            all_actions.extend(action for action in actions if isinstance(action, dict))

        alerts = parsed.get("alerts", [])
        if isinstance(alerts, list):
            all_alerts.extend(str(a).strip() for a in alerts if str(a).strip())

    fused_summary = parsed_top.get("risk_summary", "")
    if not fused_summary and summary_parts:
        fused_summary = summary_parts[0]
    if summary_parts:
        fused_summary = "；".join(dict.fromkeys(summary_parts))[:500]

    fused_actions = _dedup_actions(all_actions, max_actions=6)
    fused_alerts = list(dict.fromkeys(all_alerts))[:6]

    top_pair = top.get("pairrank", {})
    confidence = 0.65
    if isinstance(top_pair, dict):
        confidence = 0.55 + 0.4 * float(top_pair.get("score", 0.5))

    return {
        "fused_risk_summary": fused_summary,
        "fused_actions": fused_actions,
        "fused_alerts": fused_alerts,
        "source_experts": source_experts,
        "fusion_method": "rule_based",
        "fusion_confidence": round(max(0.0, min(0.99, confidence)), 4),
    }


def _llm_fuse_if_available(
    ranked_candidates: List[Dict[str, Any]],
    model: str = "deepseek-chat",
) -> Optional[Dict[str, Any]]:
    if not os.getenv("DEEPSEEK_API_KEY"):
        return None

    payload = []
    for item in ranked_candidates:
        payload.append(
            {
                "expert": item.get("expert", {}),
                "pairrank": item.get("pairrank", {}),
                "parsed": item.get("parsed", {}),
            }
        )

    system_prompt = (
        "你是GenFuser。请将多个专家候选建议融合为一个统一JSON。"
        "必须只输出JSON对象，字段为："
        "fused_risk_summary, fused_actions, fused_alerts, source_experts, fusion_method, fusion_confidence。"
    )
    user_prompt = json.dumps(
        {
            "ranked_candidates": payload,
            "constraints": {
                "fused_actions_min": 3,
                "fused_alerts_min": 1,
                "language": "zh-cn",
            },
        },
        ensure_ascii=False,
    )

    try:
        client = DeepSeekClient(model=model)
        resp = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.content)
        if isinstance(parsed, dict):
            parsed.setdefault("fusion_method", "llm_genfuser")
            parsed.setdefault("fusion_confidence", 0.78)
            return parsed
    except Exception:
        return None

    return None


def blend_candidates(
    candidates: List[Dict[str, Any]],
    use_llm_fuser: bool = True,
    model: str = "deepseek-chat",
) -> Dict[str, Any]:
    """LLM-Blender style pipeline: PairRanker -> GenFuser."""
    ranked = pairrank_candidates(candidates)
    fused = None
    if use_llm_fuser:
        fused = _llm_fuse_if_available(ranked_candidates=ranked, model=model)
    if fused is None:
        fused = _rule_based_fuse(ranked)

    return {
        "ranked_candidates": ranked,
        "fused_result": fused,
    }
