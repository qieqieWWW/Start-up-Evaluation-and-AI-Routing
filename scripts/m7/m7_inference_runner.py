import json
import hashlib
import importlib.util
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# 动态添加 m7 目录到 sys.path
_m7_dir = Path(__file__).parent
if str(_m7_dir) not in sys.path:
    sys.path.insert(0, str(_m7_dir))

from m7_context_analyzer import build_layer1_context, build_layer2_context, build_layer3_context, build_layer4_context
from m7_blender import blend_candidates
from m7_freshness_detector import FreshnessDetector
from m7_knowledge_graph import get_kg_engine, KnowledgeGraphEngine
from m7_llm_client import make_llm_client
from m7_profile_rag import append_profile_record, infer_risk_appetite_from_text
from m7_prompt_builder import build_system_prompt, build_user_prompt
from m7_search_arbiter import SearchArbiter, SearchDecision
from m7_web_retriever import WebRetriever

logger = logging.getLogger("m7_inference_runner")


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


def _load_accuracy_gate_class():
    gate_path = Path(__file__).resolve().parents[2] / "OPCcomp" / "accuracy_gate.py"
    if not gate_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("opc_accuracy_gate", str(gate_path))
    if not spec or not spec.loader:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception:
        return None

    return getattr(module, "AccuracyGate", None)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_text(value: Any, max_len: int = 240) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _make_evidence_id(evidence_type: str, source: str, snippet: str, date_str: str) -> str:
    raw = f"{evidence_type}|{source}|{snippet}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(raw).hexdigest()[:8]
    return f"EVD-{evidence_type}-{date_str}-{digest}"


def _append_evidence(
    registry: List[Dict[str, Any]],
    seen: set,
    evidence_type: str,
    source: str,
    source_label: str,
    snippet: str,
    collected_at: str,
    freshness_ttl_hours: int,
) -> str:
    date_str = collected_at[:10].replace("-", "")
    evidence_id = _make_evidence_id(evidence_type, source, snippet, date_str)
    if evidence_id in seen:
        return evidence_id

    seen.add(evidence_id)
    registry.append(
        {
            "evidence_id": evidence_id,
            "evidence_type": evidence_type,
            "source": source,
            "source_label": source_label,
            "collected_at": collected_at,
            "freshness_ttl_hours": freshness_ttl_hours,
            "snippet": _safe_text(snippet, 300),
        }
    )
    return evidence_id


def _is_evidence_stale(collected_at: str, freshness_ttl_hours: int) -> bool:
    if not collected_at or not freshness_ttl_hours or freshness_ttl_hours <= 0:
        return False
    try:
        collected_dt = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    age_hours = (datetime.now(timezone.utc) - collected_dt).total_seconds() / 3600.0
    return age_hours > float(freshness_ttl_hours)


def _build_evidence_bound_output(
    risk_level: str,
    reasons: List[str],
    intermediate: Dict[str, float],
    route_result: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    fused_result: Dict[str, Any],
) -> Dict[str, Any]:
    now_iso = _now_iso()
    registry: List[Dict[str, Any]] = []
    seen_ids: set = set()
    claims: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []

    routing_scores = route_result.get("routing_scores", {})
    route_evidence_id = _append_evidence(
        registry=registry,
        seen=seen_ids,
        evidence_type="rule",
        source="scripts/m7/m7_router.py",
        source_label="M7 Router Scores",
        snippet=json.dumps(routing_scores, ensure_ascii=False),
        collected_at=now_iso,
        freshness_ttl_hours=168,
    )

    risk_reasons_text = "；".join(_safe_text(item, 120) for item in reasons if _safe_text(item, 120))
    risk_evidence_id = _append_evidence(
        registry=registry,
        seen=seen_ids,
        evidence_type="rule",
        source="scripts/m8_rule_adapter.py",
        source_label="M8 Risk Reasons",
        snippet=risk_reasons_text,
        collected_at=now_iso,
        freshness_ttl_hours=168,
    )

    feature_evidence_id = _append_evidence(
        registry=registry,
        seen=seen_ids,
        evidence_type="dataset",
        source="Kickstarter_Clean/full_prediction_summary_*.csv",
        source_label="Engineered Risk Features",
        snippet=json.dumps(intermediate, ensure_ascii=False),
        collected_at=now_iso,
        freshness_ttl_hours=168,
    )

    knowledge_graph_hits = route_result.get("knowledge_graph_hits", [])
    kg_evidence_ids: List[str] = []
    if isinstance(knowledge_graph_hits, list):
        for hit in knowledge_graph_hits[:3]:
            if not isinstance(hit, dict):
                continue
            edge = hit.get("edge", {})
            if not isinstance(edge, dict):
                continue
            source_node = hit.get("source_node", {})
            target_node = hit.get("target_node", {})
            relation = _safe_text(edge.get("relation", ""), 80)
            src_label = _safe_text(source_node.get("label", ""), 120) if isinstance(source_node, dict) else ""
            dst_label = _safe_text(target_node.get("label", ""), 120) if isinstance(target_node, dict) else ""
            snippet = _safe_text(edge.get("evidence_snippet", ""), 260)
            graph_id = _safe_text(hit.get("graph_id", "kg-placeholder"), 80) or "kg-placeholder"

            if not relation or not snippet:
                continue

            kg_evidence_id = _append_evidence(
                registry=registry,
                seen=seen_ids,
                evidence_type="rule",
                source=f"knowledge_graph:{graph_id}",
                source_label=f"Knowledge Graph Hit ({src_label}->{relation}->{dst_label})",
                snippet=snippet,
                collected_at=now_iso,
                freshness_ttl_hours=72,
            )
            kg_evidence_ids.append(kg_evidence_id)

    summary = _safe_text(fused_result.get("fused_risk_summary", ""), 500)
    if summary:
        claims.append(
            {
                "claim_id": f"CLM-{len(claims) + 1:03d}",
                "text": summary,
                "evidence_ids": [route_evidence_id, risk_evidence_id, feature_evidence_id, *kg_evidence_ids[:1]],
                "confidence": float(fused_result.get("fusion_confidence", 0.55)),
                "scope": "short_term",
                "decision_type": "estimate",
            }
        )

    fused_actions = fused_result.get("fused_actions", [])
    if isinstance(fused_actions, list):
        for idx, item in enumerate(fused_actions[:6], 1):
            if not isinstance(item, dict):
                continue
            title = _safe_text(item.get("title", ""), 180)
            owner = _safe_text(item.get("owner", "unknown"), 60) or "unknown"
            if not title:
                continue
            claims.append(
                {
                    "claim_id": f"CLM-{len(claims) + 1:03d}",
                    "text": title,
                    "evidence_ids": [route_evidence_id, risk_evidence_id, *kg_evidence_ids[:1]],
                    "confidence": max(0.5, float(fused_result.get("fusion_confidence", 0.55)) - 0.1),
                    "scope": "short_term",
                    "decision_type": "recommendation",
                }
            )
            actions.append(
                {
                    "action_id": f"ACT-{idx:03d}",
                    "text": title,
                    "owner": owner,
                    "due_hint": _safe_text(item.get("eta", "T+7d"), 32) or "T+7d",
                    "depends_on_evidence_ids": [route_evidence_id, risk_evidence_id],
                }
            )

    # 记录候选专家摘要作为可追溯证据，便于后续回放。
    for candidate in candidates[:3]:
        expert = candidate.get("expert", {})
        parsed = candidate.get("parsed", {})
        expert_name = _safe_text(expert.get("name", "unknown"), 80) if isinstance(expert, dict) else "unknown"
        parsed_summary = _safe_text(parsed.get("risk_summary", ""), 240) if isinstance(parsed, dict) else ""
        if parsed_summary:
            _append_evidence(
                registry=registry,
                seen=seen_ids,
                evidence_type="profile",
                source=f"expert:{expert_name}",
                source_label="Expert Candidate Summary",
                snippet=parsed_summary,
                collected_at=now_iso,
                freshness_ttl_hours=72,
            )

    # 若 claim 缺证据，直接标记降级（协议红线）。
    degraded = False
    degrade_reason = ""
    for claim in claims:
        evidence_ids = claim.get("evidence_ids", [])
        if not isinstance(evidence_ids, list) or len(evidence_ids) == 0:
            degraded = True
            degrade_reason = "NO_EVIDENCE"
            break

    if not claims:
        degraded = True
        degrade_reason = "NO_EVIDENCE"

    registry_ids = {item.get("evidence_id") for item in registry if isinstance(item, dict)}
    dangling_links = any(
        any(eid not in registry_ids for eid in (claim.get("evidence_ids") or []))
        for claim in claims
        if isinstance(claim, dict)
    )
    if dangling_links:
        degraded = True
        degrade_reason = "NO_EVIDENCE"

    stale_exists = any(
        _is_evidence_stale(str(item.get("collected_at", "")), int(item.get("freshness_ttl_hours", 0) or 0))
        for item in registry
        if isinstance(item, dict)
    )

    if conflicts:
        degraded = True
        if not degrade_reason:
            degrade_reason = "EVIDENCE_CONFLICT"

    if stale_exists and not degrade_reason:
        degraded = True
        degrade_reason = "STALE_EVIDENCE"

    if risk_level and str(risk_level).strip() and not degraded:
        # 简单冲突占位：当融合摘要为空但动作很多时，提示一致性待验证。
        if not summary and len(actions) >= 3:
            conflicts.append(
                {
                    "conflict_id": "CFL-001",
                    "claim_ids": [item.get("claim_id") for item in claims if item.get("claim_id")][:2],
                    "evidence_ids": [route_evidence_id, risk_evidence_id],
                    "reason": "动作建议较多但总览摘要缺失，需人工复核一致性。",
                    "resolution": "补充统一风险总览后再进入用户侧主回复。",
                }
            )

    valid_claims = [
        item for item in claims
        if isinstance(item.get("evidence_ids"), list) and len(item.get("evidence_ids")) > 0
    ]
    coverage = 0.0 if not claims else round(len(valid_claims) / float(len(claims)), 4)

    return {
        "claims": claims,
        "evidence_registry": registry,
        "conflicts": conflicts,
        "actions": actions,
        "output_meta": {
            "coverage": coverage,
            "conflict_present": bool(conflicts),
            "degraded": degraded,
            "degrade_reason": degrade_reason,
        },
    }


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

    # 使用工厂函数创建客户端（当前固定为 DeepSeek 后端）
    client = make_llm_client(backend="auto", model=model)

    # ── 搜索决策（互联网检索增强） ──
    search_decision: Optional[SearchDecision] = None
    local_evidence_context: str = ""
    
    # ── 知识图谱上下文（结构化因果证据增强） ──
    kg_engine: Optional[KnowledgeGraphEngine] = None
    kg_context_for_prompt: str = ""
    kg_similar_project_context: str = ""
    try:
        arbiter = SearchArbiter()
        search_decision = arbiter.decide(
            user_input=user_input,
            intent_type="task",  # 默认任务型（inference_runner 主要处理评估任务）
            domain_context=str(project_data.get("main_category", "")),
            project_data=project_data,
        )

        if search_decision and search_decision.should_search:
            logger.info(
                f"[M7-Inference] 搜索决策: mode={search_decision.search_mode.value} "
                f"local_evidence={search_decision.need_local_evidence} "
                f"reason={search_decision.reason}"
            )

            # L2 本地补充证据（如果需要）
            if search_decision.need_local_evidence:
                retriever = WebRetriever()
                if retriever.is_available and search_decision.local_search_queries:
                    all_l2_evidence: List[Any] = []
                    for q in search_decision.local_search_queries[:2]:
                        results = retriever.search(q, top_k=3)
                        all_l2_evidence.extend(results)
                    if all_l2_evidence:
                        local_evidence_context = WebRetriever.format_evidence_for_prompt(all_l2_evidence, max_items=4)
                        logger.debug(f"[M7-Inference] L2 证据已获取: {len(all_l2_evidence)} 条")
    except Exception as exc:
        logger.warning(f"[M7-Inference] 搜索决策异常（已降级跳过）: {exc}")
        search_decision = None

    # ── 知识图谱增强 ──
    try:
        kg_engine = get_kg_engine()
        if kg_engine.is_loaded:
            # 1) 根据用户查询检索相关图谱命中 → 注入 Prompt
            kg_hits = kg_engine.search(query=user_input, top_k=3)
            if kg_hits:
                kg_context_for_prompt = kg_engine.format_hits_for_prompt(kg_hits, max_hits=3)
                logger.debug(f"[M7-Inference] KG 命中 {len(kg_hits)} 条, 已生成 Prompt 上下文")

            # 2) 根据项目属性找相似历史项目 → 注入 Prompt 作为基准锚点
            category = str(project_data.get("main_category", "") or project_data.get("category", ""))
            country = str(project_data.get("country", "") or "")
            if category or country:
                kg_similar_project_context = kg_engine.get_similar_project_context(
                    category=category or None,
                    country=country or None,
                    top_k=2,
                )
                if kg_similar_project_context:
                    logger.debug("[M7-Inference] KG 相似项目上下文已生成")
    except Exception as exc:
        logger.warning(f"[M7-Inference] 知识图谱增强异常（已降级跳过）: {exc}")
        kg_engine = None

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

        # ── Prompt 增强: 注入联网搜索指引 (L1) + L2 证据上下文 + KG 图谱上下文 ──
        system_prompt, user_prompt = _enhance_prompts_for_search(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            search_decision=search_decision,
            local_evidence_context=local_evidence_context,
            kg_context=kg_context_for_prompt,
            kg_similar_project_context=kg_similar_project_context,
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
    search_decision: Optional[SearchDecision] = None
    kg_engine: Optional[Any] = None
    client: Optional[Any] = None
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

    evidence_bound_output = _build_evidence_bound_output(
        risk_level=risk_level,
        reasons=reasons,
        intermediate=intermediate,
        route_result=route_result,
        candidates=candidates,
        fused_result=blended.get("fused_result", {}) if isinstance(blended, dict) else {},
    )

    gate_packet: Dict[str, Any] = {}
    final_result = dict(blended.get("fused_result", {})) if isinstance(blended, dict) else {}
    gate_class = _load_accuracy_gate_class()
    
    # ── 自我迭代闭环上下文（用于 SelfIterationLoop） ──
    iteration_context: Dict[str, Any] = {
        "user_input": user_input,
        "project_data": project_data,
        "risk_level": risk_level,
        "route_result": route_result,
        "llm_client": None,  # 在 gate 评估后设置
        "messages": [],     # 同上
    }
    
    if gate_class:
        try:
            gate = gate_class()
            evaluation = gate.evaluate_router_payload(
                evidence_bound_output,
                output_id=f"m7_blend_{route_result.get('normalized_risk_level', 'unknown')}",
            )
            gate_packet = gate.to_runtime_gate_packet(evaluation)
            final_result.update(
                {
                    "gate": gate_packet,
                    "gate_evaluation": evaluation.to_dict(),
                    "evidence_bound_output": evidence_bound_output,
                    "degraded": bool(gate_packet.get("blocked", False)),
                    "degrade_reason": gate_packet.get("decision_reason", ""),
                }
            )
            
            # ═════════════════════════════════
            # SelfIterationLoop: 当 Gate 决策为 REQUIRES_REVISION 时
            # 触发带修正信息的重新推理闭环
            # ═════════════════════════════════
            _GD = _get_gate_decision_enum()
            if _GD and evaluation.gate_decision == getattr(_GD, 'REQUIRES_REVISION', None):
                logger.info(
                    f"[M7-Blender] AccuracyGate 决策=REQUIRES_REVISION, "
                    f"触发 SelfIterationLoop | "
                    f"conflicts={evaluation.conflicting_statements} "
                    f"hallucinations={evaluation.hallucinated_statements}"
                )
                
                try:
                    from accuracy_gate import SelfIterationLoop as _SelfIterationLoop
                    
                    iteration_context["llm_client"] = client
                    loop = _SelfIterationLoop(llm_client=client)
                    
                    # 对 candidates 中的每个专家输出执行自我迭代
                    revised_candidates = []
                    for cand in candidates:
                        # 构建该 candidate 的简化 gate eval 用于迭代
                        cand_eval = _quick_evaluate_candidate(gate, cand)
                        
                        revised = loop.iterate_if_needed(
                            original_output=cand,
                            gate_evaluation=cand_eval or evaluation,
                            context=iteration_context,
                        )
                        revised_candidates.append(revised)
                    
                    # 如果有候选被修正，重新融合
                    any_revised = any(c.get("_revised") for c in revised_candidates)
                    if any_revised:
                        logger.info(f"[M7-Blender] 自我迭代完成, {sum(1 for c in revised_candidates if c.get('_revised'))}/{len(revised_candidates)} 个候选已修正")
                        
                        blended = blend_candidates(
                            candidates=revised_candidates,
                            use_llm_fuser=use_llm_fuser,
                            model=model,
                        )
                        evidence_bound_output = _build_evidence_bound_output(
                            risk_level=risk_level,
                            reasons=reasons,
                            intermediate=intermediate,
                            route_result=route_result,
                            candidates=revised_candidates,
                            fused_result=blended.get("fused_result", {}) if isinstance(blended, dict) else {},
                        )
                        
                        final_result["_self_iterated"] = True
                        final_result["_revision_counts"] = [
                            c.get("_iteration_count", 0) for c in revised_candidates
                        ]
                        
                except ImportError:
                    logger.warning("[M7-Blender] SelfIterationLoop 导入失败，跳过闭环")
                except Exception as loop_exc:
                    logger.warning(f"[M7-Blender] SelfIterationLoop 执行异常: {loop_exc}")
            
            if gate_packet.get("blocked", False):
                final_result.setdefault("fused_risk_summary", evidence_bound_output.get("claims", [{}])[0].get("text", ""))
                final_result.setdefault("fused_actions", evidence_bound_output.get("actions", []))
                final_result.setdefault("fused_alerts", [])
        except Exception as exc:
            gate_packet = {"available": False, "error": str(exc)}
            final_result.update(
                {
                    "gate": gate_packet,
                    "evidence_bound_output": evidence_bound_output,
                }
            )
    else:
        gate_packet = {"available": False, "error": "AccuracyGate unavailable"}
        final_result.update(
            {
                "gate": gate_packet,
                "evidence_bound_output": evidence_bound_output,
            }
        )

    return {
        "candidates": candidates,
        "ranked_candidates": blended.get("ranked_candidates", []),
        "fused_result": blended.get("fused_result", {}),
        "evidence_bound_output": evidence_bound_output,
        "gate": gate_packet,
        "final_result": final_result,
        "search_decision": search_decision.to_dict() if search_decision else None,
        "kg_stats": kg_engine.stats if kg_engine else None,
    }


# ════════════════════════════════════════════
#  Prompt 增强: 互联网检索指引注入
# ════════════════════════════════════════════

def _enhance_prompts_for_search(
    system_prompt: str,
    user_prompt: str,
    search_decision: Optional[SearchDecision] = None,
    local_evidence_context: str = "",
    kg_context: str = "",
    kg_similar_project_context: str = "",
) -> tuple:
    """
    将搜索决策 + 知识图谱上下文转化为 Prompt 增强。

        L1 增强（System Prompt）:
            - 注入 agent_search_hint，引导当前 LLM 主动执行联网检索

    L2 增强（User Prompt）:
      - 注入本地 WebRetriever 获取的结构化网络证据作为参考上下文

    L3 增强（User Prompt, Knowledge Graph）:
      - 注入知识图谱因果规律命中（来自 graph_schema/graph_index_builder）
      - 注入相似历史项目风险分布作为基准锚点

    Returns:
        (enhanced_system_prompt, enhanced_user_prompt)
    """
    if not search_decision or not search_decision.should_search:
        # 即使没有搜索决策，如果有 KG 上下文也要注入
        if kg_context or kg_similar_project_context:
            return _inject_kg_to_prompts(system_prompt, user_prompt, kg_context, kg_similar_project_context)
        return system_prompt, user_prompt

    # ── L1: System Prompt 注入联网搜索指引 ──
    if search_decision.agent_search_hint:
        search_block = (
            "\n\n"
            "╔" + "═" * 48 + "╗\n"
            "║ 【互联网检索指引 — 请务必执行】                                    ║\n"
            "╚" + "═" * 48 + "╝\n\n"
            f"{search_decision.agent_search_hint}\n\n"
            "╔" + "═" * 48 + "╗\n"
            "║ 【指引结束】                                                        ║\n"
            "╚" + "═" * 48 + "╝"
        )
        system_prompt = system_prompt + search_block

    # ── L2: User Prompt 注入结构化网络证据 ──
    if local_evidence_context:
        evidence_block = (
            "\n\n"
            "╔" + "═" * 50 + "╗\n"
            "║ 【互联网实时参考资料 — 回答时必须与以下信息保持一致】              ║\n"
            "╚" + "═" * 50 + "╝\n\n"
            f"{local_evidence_context}\n"
            "╔" + "═" * 50 + "╗\n"
            "║ 【参考资料结束】                                                    ║\n"
            "╚" + "═" * 50 + "╝"
        )
        user_prompt = user_prompt + evidence_block

    # ── L3: 知识图谱上下文注入 ──
    if kg_context or kg_similar_project_context:
        user_prompt = _inject_kg_to_user_prompt(user_prompt, kg_context, kg_similar_project_context)

    return system_prompt, user_prompt


def _inject_kg_to_prompts(
    system_prompt: str,
    user_prompt: str,
    kg_context: str,
    kg_similar_project_context: str,
) -> tuple:
    """仅注入知识图谱上下文（无搜索决策时调用）。"""
    if kg_similar_project_context:
        user_prompt = _inject_kg_to_user_prompt(user_prompt, kg_context, kg_similar_project_context)
    elif kg_context:
        user_prompt = _inject_kg_to_user_prompt(user_prompt, kg_context, "")
    return system_prompt, user_prompt


def _inject_kg_to_user_prompt(
    user_prompt: str,
    kg_context: str,
    kg_similar_project_context: str,
) -> str:
    """将知识图谱上下文块注入 User Prompt。"""
    parts = []

    if kg_context:
        kg_block = (
            "\n\n"
            "╔" + "═" * 56 + "╗\n"
            "║ 【知识图谱因果规律 — 历史数据驱动的风险关联】                        ║\n"
            "╚" + "═" * 56 + "╝\n\n"
            f"{kg_context}\n"
            "╔" + "═" * 56 + "╗\n"
            "║ 【因果规律参考结束】                                                  ║\n"
            "╚" + "═" * 56 + "╝"
        )
        parts.append(kg_block)

    if kg_similar_project_context:
        sim_block = (
            "\n\n"
            "╔" + "═" * 54 + "╗\n"
            "║ 【相似历史项目风险参照 — 来自知识图谱同类案例】                      ║\n"
            "╚" + "═" * 54 + "╝\n\n"
            f"{kg_similar_project_context}\n"
            "╔" + "═" * 54 + "╗\n"
            "║ 【相似案例参考结束】                                                  ║\n"
            "╚" + "═" * 54 + "╝"
        )
        parts.append(sim_block)

    if parts:
        user_prompt = user_prompt + "".join(parts)

    return user_prompt


# ── 辅助：快速评估单个 candidate 用于迭代闭环 ──

GateDecisionType = None  # 延迟导入，避免循环依赖

def _get_gate_decision_enum():
    """延迟获取 GateDecision 枚举"""
    global GateDecisionType
    if GateDecisionType is None:
        try:
            gate_module = _load_accuracy_gate_class()
            if gate_module:
                import sys as _sys
                for _mod_name, _mod in list(_sys.modules.items()):
                    if hasattr(_mod, 'GateDecision'):
                        GateDecisionType = _mod.GateDecision
                        return GateDecisionType
                # fallback: 直接从文件导入
                from OPCcomp.accuracy_gate import GateDecision as _GD
                GateDecisionType = _GD
                return GateDecisionType
        except Exception:
            pass
        # 最终 fallback: 用字符串代替
        class _FallbackGateDecision:
            REQUIRES_REVISION = "requires_revision"
            PASS = "pass"
            PASS_WITH_WARNING = "pass_with_warning"
        GateDecisionType = _FallbackGateDecision
    return GateDecisionType


def _quick_evaluate_candidate(gate_instance, candidate: Dict[str, Any]):
    """对单个候选输出做简化版 gate 评估，用于 SelfIterationLoop"""
    try:
        parsed = candidate.get("parsed", {})
        if not isinstance(parsed, dict):
            return None
        
        risk_summary = parsed.get("risk_summary", "")
        claims = []
        if isinstance(risk_summary, str) and risk_summary.strip():
            claims.append({
                "claim": risk_summary[:300],
                "evidence_ids": [],
                "statement_type": "conclusion",
                "verifiable": True,
            })
        
        actions = parsed.get("actions", [])
        if isinstance(actions, list):
            for a in actions[:3]:
                title = a.get("title", "") if isinstance(a, dict) else str(a)
                if title:
                    claims.append({
                        "claim": title,
                        "evidence_ids": [],
                        "statement_type": "recommendation",
                        "verifiable": False,
                    })
        
        if claims:
            eval_result = gate_instance.check_output(
                output_text="\n".join(c["claim"] for c in claims),
                output_id=f"candidate_quick_{id(candidate)}",
                structured_claims=claims,
            )
            return eval_result
        
        return None
    except Exception:
        return None
