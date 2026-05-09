"""
知识图谱检索引擎 v2 — 深度对接 graph_schema / graph_index_builder

核心能力:
  1. 从 graph_index_builder 导出的 CSV/JSON 构建内存知识图谱
  2. 基于多维度（属性匹配/关系路径/语义关键词）的图谱检索
  3. 输出结构化的图谱命中，可用于:
     - Prompt 增强（推理前注入领域上下文）
     - EvidenceStore 种子填充（Gate 事实核查）
     - 路由偏置校正（专家选择时的因果约束）
  4. 图谱因果链矛盾检测（输出声明 vs 已知关系的交叉验证）

数据来源优先级:
  1. graph_exports/ 目录下的 CSV (nodes.csv, edges.csv) + evidence_index.json
  2. config/m7_knowledge_graph_placeholder.json (向后兼容降级)
  3. 空图谱（所有操作优雅降级）

依赖:
  - OPCcomp.graph_schema: Node, Edge, NodeType, EdgeType 及解析函数
  - OPCcomp.graph_index_builder: build_graph_index 等索引构建函数
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("m7_knowledge_graph")

# ── 类型定义 ──────────────────────────────────────────────


@dataclass
class GraphNode:
    """内存中的图谱节点."""
    node_id: str
    label: str
    node_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return self.properties.get("name") or self.properties.get("text") or self.label

    @property
    def text_for_search(self) -> str:
        """拼接所有文本字段用于搜索匹配."""
        parts = [self.label, self.node_type]
        for v in self.properties.values():
            if isinstance(v, str) and v.strip():
                parts.append(v.strip)
        return " ".join(parts).lower()


@dataclass
class GraphEdge:
    """内存中的图谱边."""
    source: str
    target: str
    rel_type: str
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_snippet(self) -> str:
        return self.properties.get("evidence_snippet") or self.properties.get("relation", "")

    @property
    def keywords(self) -> List[str]:
        kw = self.properties.get("keywords", [])
        if isinstance(kw, list):
            return [str(k).strip().lower() for k in kw if str(k).strip()]
        if isinstance(kw, str):
            return [kw.strip().lower()]
        return []


@dataclass
class GraphHit:
    """一次图谱检索命中结果."""
    score: float
    edge: GraphEdge
    source_node: Optional[GraphNode] = None
    target_node: Optional[GraphNode] = None
    graph_id: str = "kg-v2"
    graph_version: str = "v2"
    match_reason: str = ""
    # 关联的 evidence_index 条目（如果有）
    evidence_ref: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "edge": {
                "source": self.edge.source,
                "target": self.edge.target,
                "relation": self.edge.rel_type,
                "keywords": self.edge.keywords,
                "evidence_snippet": self.edge.evidence_snippet,
            },
            "source_node": {
                "id": self.source_node.node_id,
                "label": self.source_node.label,
                "type": self.source_node.node_type,
            } if self.source_node else {},
            "target_node": {
                "id": self.target_node.node_id,
                "label": self.target_node.label,
                "type": self.target_node.node_type,
            } if self.target_node else {},
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "match_reason": self.match_reason,
            "evidence_ref": self.evidence_ref,
        }


@dataclass
class CausalChain:
    """因果链 — 用于检测输出声明是否与已知因果矛盾."""
    path: List[GraphNode]  # 节点路径
    edges: List[GraphEdge]  # 经过的边
    chain_text: str  # 可读的链描述
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": [{"id": n.node_id, "label": n.display_label, "type": n.node_type} for n in self.path],
            "edges": [{"source": e.source, "target": e.target, "rel": e.rel_type} for e in self.edges],
            "chain_text": self.chain_text,
            "confidence": round(self.confidence, 4),
        }


# ── 图谱引擎 ──────────────────────────────────────────────


class KnowledgeGraphEngine:
    """
    内存知识图谱引擎.

    使用方式:
        engine = KnowledgeGraphEngine()
        engine.load_from_exports(graph_exports_dir="OPCcomp/graph_exports")
        hits = engine.search(query="现金流不足对市场验证的影响", top_k=3)
        chains = engine.find_causal_chains("现金流跑道", "市场验证", max_depth=3)
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, List[GraphEdge]] = {}  # node_id -> outgoing edges
        self.reverse_adjacency: Dict[str, List[GraphEdge]] = {}  # node_id -> incoming edges
        self.evidence_index: Dict[str, Dict[str, Any]] = {}  # evidence_id -> evidence data
        self.graph_id: str = "kg-empty"
        self.graph_version: str = "v2"
        self._loaded_source: str = ""

    # ── 加载方法 ────────────────────────────────────────

    def load(self, graph_path: Optional[str] = None) -> bool:
        """
        自动发现并加载最佳数据源.
        
        优先级:
          1. 显式传入的 JSON 文件
          2. graph_exports/ 目录的 CSV 集合 + evidence_index
          3. config/m7_knowledge_graph_placeholder.json (兼容)
        """
        # 尝试显式路径
        if graph_path:
            p = Path(graph_path)
            if p.exists():
                if p.is_dir():
                    return self._load_from_exports_dir(p)
                elif p.suffix == ".json":
                    return self._load_from_json(p)

        # 尝试 graph_exports/
        exports_dir = self._find_exports_dir()
        if exports_dir:
            loaded = self._load_from_exports_dir(exports_dir)
            if loaded:
                return True

        # 降级到 placeholder
        placeholder = self._find_placeholder()
        if placeholder:
            return self._load_from_json(placeholder)

        logger.warning("[KG-Engine] 未找到任何可加载的知识图谱数据源")
        self.graph_id = "kg-empty"
        return False

    def _find_exports_dir(self) -> Optional[Path]:
        """查找 graph_exports 目录."""
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "OPCcomp" / "graph_exports",
            Path(__file__).resolve().parent.parent / "OPCcomp" / "graph_exports",
            Path(__file__).resolve().parent.parent.parent / "graph_exports",
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                # 至少有一个 nodes.csv 或 *_nodes.csv
                has_data = any(
                    f.suffix == ".csv" and "node" in f.name.lower()
                    for f in c.iterdir()
                )
                if has_data:
                    return c
        return None

    def _find_placeholder(self) -> Optional[Path]:
        """查找占位符 JSON."""
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "config" / "m7_knowledge_graph_placeholder.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_from_exports_dir(self, dir_path: Path) -> bool:
        """
        从 graph_exports/ 目录加载 CSV 数据.
        
        期望文件格式:
          * _nodes.csv: id, label, properties_json
          * _edges.csv: source, target, type, properties_json
          * _evidence_index.json: evidence_id -> {scenario_id, source, raw}
        """
        node_count = 0
        edge_count = 0
        ev_count = 0

        # 1. 加载节点
        for csv_file in sorted(dir_path.glob("*_nodes.csv")):
            try:
                imported = self._import_nodes_csv(csv_file)
                node_count += imported
            except Exception as exc:
                logger.warning(f"[KG-Engine] 加载节点失败 {csv_file.name}: {exc}")

        # 2. 加载边
        for csv_file in sorted(dir_path.glob("*_edges.csv")):
            try:
                imported = self._import_edges_csv(csv_file)
                edge_count += imported
            except Exception as exc:
                logger.warning(f"[KG-Engine] 加载边失败 {csv_file.name}: {exc}")

        # 3. 构建 adjacency
        self._rebuild_adjacency()

        # 4. 加载 evidence_index
        for json_file in sorted(dir_path.glob("*_evidence_index.json")):
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                if isinstance(idx_data, dict):
                    for ev_id, ev_data in idx_data.items():
                        self.evidence_index[ev_id] = ev_data
                        ev_count += 1
            except Exception as exc:
                logger.warning(f"[KG-Engine] 加载evidence_index失败 {json_file.name}: {exc}")

        total = node_count + edge_count + ev_count
        if total > 0:
            self.graph_id = f"kg-exports-{dir_path.name}"
            self._loaded_source = f"exports:{dir_path}"
            logger.info(
                f"[KG-Engine] 从 {dir_path.name} 加载完成: "
                f"{node_count} nodes, {edge_count} edges, {ev_count} evidence items"
            )
            return True
        return False

    def _load_from_json(self, json_path: Path) -> bool:
        """从 JSON 文件加载（placeholder 兼容格式）."""
        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(f"[KG-Engine] JSON加载失败 {json_path}: {exc}")
            return False

        if not isinstance(payload, dict):
            return False

        self.graph_id = payload.get("graph_id", json_path.stem)
        self.graph_version = payload.get("version", "v0")

        # 解析节点
        raw_nodes = payload.get("nodes", [])
        for n in raw_nodes:
            if not isinstance(n, dict):
                continue
            nid = str(n.get("id", "")).strip()
            if not nid:
                continue
            props = n.get("properties", {})
            if isinstance(props, str):
                try:
                    props = json.loads(props)
                except Exception:
                    props = {}
            self.nodes[nid] = GraphNode(
                node_id=nid,
                label=str(n.get("label", "")),
                node_type=str(n.get("type", "")),
                properties=props if isinstance(props, dict) else {},
            )

        # 解析边
        raw_edges = payload.get("edges", [])
        for e in raw_edges:
            if not isinstance(e, dict):
                continue
            src = str(e.get("source", "")).strip()
            dst = str(e.get("target", "")).strip()
            rel = str(e.get("relation", e.get("type", ""))).strip()
            if not src or not dst or not rel:
                continue
            ep = e.get("properties", {})
            if isinstance(ep, str):
                try:
                    ep = json.loads(ep)
                except Exception:
                    ep = {}
            edge = GraphEdge(source=src, target=dst, rel_type=rel, properties=ep if isinstance(ep, dict) else {})
            self.edges.append(edge)

        self._rebuild_adjacency()
        self._loaded_source = f"json:{json_path.name}"
        logger.info(f"[KG-Engine] 从JSON加载完成: {len(self.nodes)} nodes, {len(self.edges)} edges")
        return True

    def _import_nodes_csv(self, csv_path: Path) -> int:
        """从标准 nodes.csv 导入节点."""
        import csv
        count = 0
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nid = str(row.get("id", "")).strip()
                if not nid:
                    continue
                label = str(row.get("label", ""))
                props_str = row.get("properties_json", "{}")
                try:
                    props = json.loads(props_str) if props_str else {}
                except Exception:
                    props = {}
                # 如果 properties 中有 name/text 等，保留
                node_type = props.pop("node_type", label) if isinstance(props, dict) else label
                self.nodes[nid] = GraphNode(
                    node_id=nid,
                    label=label,
                    node_type=node_type,
                    properties=props if isinstance(props, dict) else {},
                )
                count += 1
        return count

    def _import_edges_csv(self, csv_path: Path) -> int:
        """从标准 edges.csv 导入边."""
        import csv
        count = 0
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = str(row.get("source", "")).strip()
                dst = str(row.get("target", "")).strip()
                etype = str(row.get("type", row.get("relation", ""))).strip()
                if not src or not dst or not etype:
                    continue
                props_str = row.get("properties_json", "{}")
                try:
                    props = json.loads(props_str) if props_str else {}
                except Exception:
                    props = {}
                # 统一为内部格式
                edge = GraphEdge(
                    source=src, target=dst, rel_type=etype,
                    properties=props if isinstance(props, dict) else {},
                )
                self.edges.append(edge)
                count += 1
        return count

    def _rebuild_adjacency(self) -> None:
        """根据 edges 列表重建邻接表."""
        self.adjacency = {}
        self.reverse_adjacency = {}
        for edge in self.edges:
            self.adjacency.setdefault(edge.source, []).append(edge)
            self.reverse_adjacency.setdefault(edge.target, []).append(edge)

    # ── 检索方法 ────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        node_types: Optional[List[str]] = None,
        relation_filter: Optional[List[str]] = None,
    ) -> List[GraphHit]:
        """
        多维度图谱检索.

        匹配策略（加权组合）:
          1. 边的关键词匹配 (weight=1.0 per token)
          2. evidence_snippet 文本包含 (weight=0.5)
          3. 节点 label/type 包含查询词 (weight=0.3)
          4. evidence_index 中的原始数据匹配 (weight=0.4)
        """
        query_text = (query or "").strip().lower()
        if not query_text:
            return []

        # 分词
        query_tokens = set(re.findall(r'[a-zA-Z\u4e00-\u9fff]{2,}', query_text))
        if not query_tokens:
            query_tokens = {query_text}

        scored: List[GraphHit] = []

        for edge in self.edges:
            # 过滤关系类型
            if relation_filter and edge.rel_type not in relation_filter:
                continue

            src_node = self.nodes.get(edge.source)
            dst_node = self.nodes.get(edge.target)

            # 过滤节点类型
            if node_types:
                src_ok = not node_types or (src_node and src_node.node_type in node_types)
                dst_ok = not node_types or (dst_node and dst_node.node_type in node_types)
                if not src_ok and not dst_ok:
                    continue

            score = 0.0
            reasons = []

            # 1. 边关键词精确匹配
            for kw in edge.keywords:
                if kw and kw in query_text:
                    score += 1.0
                    reasons.append(f"keyword:{kw}")

            # 2. evidence_snippet 文本包含
            snippet = edge.evidence_snippet.lower()
            if query_text and query_text in snippet:
                score += 0.5
                reasons.append("snippet_contains")

            # 3. 节点文本匹配
            if src_node:
                src_text = src_node.text_for_search
                for token in query_tokens:
                    if token in src_text:
                        score += 0.3
                        reasons.append(f"src_node:{token}")
            if dst_node:
                dst_text = dst_node.text_for_search
                for token in query_tokens:
                    if token in dst_text:
                        score += 0.3
                        reasons.append(f"dst_node:{token}")

            # 4. properties 全文模糊
            all_props_text = " ".join(
                str(v).lower() for v in edge.properties.values() if isinstance(v, str)
            )
            for token in query_tokens:
                if token in all_props_text:
                    score += 0.15
                    reasons.append(f"prop:{token}")

            if score <= 0:
                continue

            # 查找关联的 evidence_index
            ev_ref = self._find_evidence_ref(edge, src_node, dst_node)

            hit = GraphHit(
                score=round(score, 4),
                edge=edge,
                source_node=src_node,
                target_node=dst_node,
                graph_id=self.graph_id,
                graph_version=self.graph_version,
                match_reason=";".join(reasons[:4]),
                evidence_ref=ev_ref,
            )
            scored.append(hit)

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:max(1, top_k)]

    def _find_evidence_ref(
        self,
        edge: GraphEdge,
        src_node: Optional[GraphNode],
        dst_node: Optional[GraphNode],
    ) -> Optional[Dict[str, Any]]:
        """查找与该边关联的 evidence_index 条目."""
        if not self.evidence_index:
            return None

        # 通过节点 ID 或特征名匹配
        candidate_keys = []
        if src_node:
            candidate_keys.append(src_node.node_id)
            candidate_keys.append(src_node.display_label)
        if dst_node:
            candidate_keys.append(dst_node.node_id)
            candidate_keys.append(dst_node.display_label)

        for key in candidate_keys:
            for ev_id, ev_data in self.evidence_index.items():
                raw = ev_data.get("raw", {})
                if not isinstance(raw, dict):
                    raw = {}

                # 检查 raw 字段中是否有相关内容
                raw_text = json.dumps(raw, ensure_ascii=False).lower()
                key_lower = key.lower()

                # 宽松匹配：key 的核心词出现在 raw 中
                key_terms = re.findall(r'[\u4e00-\u9fff_a-zA-Z]+', key_lower)
                matches = sum(1 for t in key_terms if len(t) >= 2 and t in raw_text)

                if matches >= 1 or key_lower in raw_text:
                    return {"evidence_id": ev_id, **ev_data}

        return None

    # ── 因果链检索 ──────────────────────────────────────

    def find_causal_chains(
        self,
        source_concept: str,
        target_concept: str,
        max_depth: int = 3,
    ) -> List[CausalChain]:
        """
        在图中查找两个概念之间的因果链路.

        用途: 当 LLM 声明"A导致B"时，验证图中是否存在支持或矛盾的链路。
        """
        source_concept = source_concept.strip().lower()
        target_concept = target_concept.strip().lower()
        if not source_concept or not target_concept:
            return []

        # 定位起点和终点候选节点
        source_candidates = self._find_nodes_by_text(source_concept)
        target_candidates = self._find_nodes_by_text(target_concept)

        chains: List[CausalChain] = []

        for src_node in source_candidates:
            for tgt_node in target_candidates:
                if src_node.node_id == tgt_node.node_id:
                    continue
                found = self._bfs_causal_path(src_node.node_id, tgt_node.node_id, max_depth)
                if found:
                    chains.append(found)

        chains.sort(key=lambda c: c.confidence, reverse=True)
        return chains[:5]

    def find_contradicting_relations(self, claim_text: str) -> List[CausalChain]:
        """
        检测声明文本是否与已知图关系矛盾.

        例如:
          声明: "渠道冗余不影响现金流安全"
          图中有: 渠道冗余 --protects--> 现金流跑道
          → 返回矛盾因果链
        """
        claim_lower = claim_text.strip().lower()
        if len(claim_lower) < 6:
            return []

        contradictions: List[CausalChain] = []

        # 反义词模式映射
        antonym_relation_map = [
            ("constrains", ["不影響", "无关", "没有关系", "不会影响", "无关联"]),
            ("supports", ["削弱", "阻碍", "损害", "不利"]),
            ("contradicts", ["一致", "相同", "吻合", "支持"]),
            ("protects", ["危害", "威胁", "破坏"]),
            ("requires", ["不需要", "无需", "可以没有"]),
            ("derived_from", ["独立于", "无关"]),
        ]

        for edge in self.edges:
            src_node = self.nodes.get(edge.source)
            dst_node = self.nodes.get(edge.target)
            if not src_node or not dst_node:
                continue

            src_label = src_node.display_label.lower()
            dst_label = dst_node.display_label.lower()

            # 检查声明是否同时提到边的两端节点
            if src_label not in claim_lower and dst_label not in claim_lower:
                continue

            # 检查反义词
            rel = edge.rel_type
            for mapped_rel, antonyms in antonym_relation_map:
                if rel != mapped_rel:
                    continue
                for antonym in antonyms:
                    if antonym in claim_lower:
                        chain = CausalChain(
                            path=[src_node, dst_node],
                            edges=[edge],
                            chain_text=(
                                f"已知关系: {src_node.display_label} --[{rel}]--> {dst_node.display_label}, "
                                f"但声明暗示'{antonym}'"
                            ),
                            confidence=0.8,
                        )
                        contradictions.append(chain)

        return contradictions

    def _find_nodes_by_text(self, text: str) -> List[TextNode]:
        """通过文本模糊匹配查找节点."""
        text = text.strip().lower()
        terms = set(re.findall(r'[\u4e00-\u9fff_a-zA-Z]{2,}', text))
        matched: List[Tuple[GraphNode, float]] = []

        for node in self.nodes.values():
            node_text = node.text_for_search
            score = 0.0
            matched_terms = 0
            for term in terms:
                if term in node_text:
                    score += 1.0
                    matched_terms += 1
            # 完全匹配 node ID 或 display_label
            if text in node.node_id.lower() or text in node.display_label.lower():
                score += 2.0
            if score > 0:
                matched.append((node, score))

        matched.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matched[:5]]

    def _bfs_causal_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int,
    ) -> Optional[CausalChain]:
        """BFS 查找两个节点间的因果路径."""
        from collections import deque

        queue: deque = deque()
        queue.append((start_id, [], []))

        visited: Set[str] = {start_id}

        while queue:
            curr_id, path_nodes, path_edges = queue.popleft()

            if len(path_edges) > max_depth:
                continue

            if curr_id == end_id and path_edges:
                node_objs = [self.nodes[nid] for nid in ([start_id] + path_nodes)]
                edge_objs = path_edges
                labels = " → ".join(n.display_label for n in node_objs if n)
                rels = " → ".join(e.rel_type for e in edge_objs)
                return CausalChain(
                    path=node_objs,
                    edges=edge_objs,
                    chain_text=f"{labels} ({rels})",
                    confidence=1.0 - 0.1 * len(path_edges),  # 路径越长置信度越低
                )

            for edge in self.adjacency.get(curr_id, []):
                next_id = edge.target
                if next_id in visited:
                    continue
                visited.add(next_id)
                queue.append((
                    next_id,
                    path_nodes + [next_id],
                    path_edges + [edge],
                ))

        return None

    # ── Evidence 种子生成 ───────────────────────────────

    def generate_evidence_seeds(self, max_seeds: int = 20) -> List[Dict[str, Any]]:
        """
        从图谱数据生成 AccuracyGate.EvidenceStore 所需的种子证据.

        每个 seed 格式:
          {
            "evidence_id": str,
            "content": str,
            "source_type": str,  // "knowledge_graph"
            "source_name": str,
            "expiration_days": int,
            "metadata": {...},
          }
        """
        seeds: List[Dict[str, Any]] = []
        seen_content: Set[str] = set()

        # 1. 从 evidence_index 生成
        for ev_id, ev_data in list(self.evidence_index.items())[:max_seeds // 2]:
            content = self._format_evidence_content(ev_data)
            if content in seen_content:
                continue
            seen_content.add(content)
            seeds.append({
                "evidence_id": f"EV-KG-{ev_id}",
                "content": content,
                "source_type": "knowledge_graph",
                "source_name": f"graph_index:{ev_data.get('source', 'unknown')}",
                "expiration_days": 180,
                "metadata": {"evidence_id_original": ev_id, **{k: v for k, v in ev_data.items() if k != "raw"}},
            })

        # 2. 从边生成（带 evidence_snippet 的优先）
        edges_with_snippets = [
            e for e in self.edges
            if e.evidence_snippet and len(e.evidence_snippet) > 10
        ]
        edges_with_snippets.sort(key=lambda e: len(e.evidence_snippet), reverse=True)

        for edge in edges_with_snippets[:max_seeds // 2]:
            src_node = self.nodes.get(edge.source)
            dst_node = self.nodes.get(edge.target)
            src_label = src_node.display_label if src_node else edge.source
            dst_label = dst_node.display_label if dst_node else edge.target

            content = f"[{edge.rel_type}] {src_label} → {dst_label}: {edge.evidence_snippet}"
            if content in seen_content:
                continue
            seen_content.add(content)

            seeds.append({
                "evidence_id": f"EV-KG-EDGE-{hash(edge.source + edge.target + edge.rel_type) & 0xFFFFFF:06x}",
                "content": content,
                "source_type": "knowledge_graph",
                "source_name": f"{self.graph_id}:{edge.rel_type}",
                "expiration_days": 180,
                "metadata": {
                    "source_node": src_label,
                    "target_node": dst_label,
                    "relation": edge.rel_type,
                },
            })

        return seeds[:max_seeds]

    def _format_evidence_content(self, ev_data: Dict[str, Any]) -> str:
        """将 evidence_index 条目格式化为可读文本."""
        raw = ev_data.get("raw", {})
        if isinstance(raw, dict):
            parts = []
            for k, v in raw.items():
                parts.append(f"{k}={v}")
            return f"[{ev_data.get('source', 'kg')}] " + "; ".join(parts)
        return str(raw) if raw else json.dumps(ev_data, ensure_ascii=False)

    # ── Prompt 上下文生成 ──────────────────────────────

    def format_hits_for_prompt(self, hits: List[GraphHit], max_hits: int = 3) -> str:
        """将图谱命中格式化为可注入 Prompt 的文本块."""
        if not hits:
            return ""

        hits = hits[:max_hits]
        lines = ["【知识图谱参考 — 历史数据驱动的风险因果规律】"]
        for i, hit in enumerate(hits, 1):
            src_lbl = hit.source_node.display_label if hit.source_node else hit.edge.source
            dst_lbl = hit.target_node.display_label if hit.target_node else hit.edge.target
            snippet = hit.edge.evidence_snippet
            reason = hit.match_reason

            line = (
                f"  {i}. [{hit.edge.rel_type}] {src_lbl} → {dst_lbl}\n"
                f"     规律: {snippet}"
            )
            if hit.evidence_ref:
                ref_src = hit.evidence_ref.get("source", "")
                if ref_src:
                    line += f"\n     来源: {ref_src}"
            lines.append(line)

        return "\n".join(lines)

    def get_similar_project_context(
        self,
        category: Optional[str] = None,
        country: Optional[str] = None,
        goal_range: Optional[Tuple[float, float]] = None,
        top_k: int = 3,
    ) -> str:
        """
        根据项目属性查找相似历史项目，返回结构化上下文.

        用于在推理前注入「同类项目的历史风险分布」作为基准锚点。
        """
        project_nodes = [
            n for n in self.nodes.values()
            if n.node_type in ("Project", "project") or n.label == "Project"
        ]

        if not project_nodes:
            return ""

        scored: List[Tuple[GraphNode, float]] = []

        for node in project_nodes:
            props = node.properties
            s = 0.0

            if category:
                cat = str(props.get("main_category", props.get("category", "")))
                if category.lower() in cat.lower() or cat.lower() in (category or "").lower():
                    s += 2.0

            if country:
                cnt = str(props.get("country", ""))
                if country.lower() in cnt.lower():
                    s += 1.0

            if goal_range:
                try:
                    g = float(props.get("goal_usd", props.get("goal", 0)) or 0)
                    if goal_range[0] <= g <= goal_range[1]:
                        s += 1.0
                except (ValueError, TypeError):
                    pass

            if s > 0:
                scored.append((node, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return ""

        lines = ["【相似历史项目风险参照 — 来自知识图谱】"]
        for i, (node, score) in enumerate(scored[:top_k], 1):
            props = node.properties
            name = props.get("name", node.node_id)
            cat = props.get("main_category", props.get("category", "-"))
            country_val = props.get("country", "-")
            goal = props.get("goal_usd", props.get("goal", "-"))

            # 找关联的 RiskFactor
            risk_factors = []
            for edge in self.adjacency.get(node.node_id, []):
                rf_node = self.nodes.get(edge.target)
                if rf_node and rf_node.node_type in ("RiskFactor", "risk_factor"):
                    rf_prop = rf_node.properties
                    risk_factors.append(
                        f"{rf_prop.get('feature', '?')}={rf_prop.get('value', '?')}"
                    )

            rf_str = ", ".join(risk_factors[:4]) if risk_factors else "(暂无)"

            lines.append(
                f"  {i}. {name}\n"
                f"     类别={cat}, 国家={country_val}, 目标金额={goal}\n"
                f"     风险因子: {rf_str}"
            )

        return "\n".join(lines)

    # ── 状态查询 ────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return len(self.nodes) > 0 or len(self.edges) > 0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "version": self.graph_version,
            "loaded_source": self._loaded_source,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "evidence_index_size": len(self.evidence_index),
            "node_types": list(set(n.node_type for n in self.nodes.values())),
            "relation_types": list(set(e.rel_type for e in self.edges)),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraphEngine(id={self.graph_id}, "
            f"nodes={len(self.nodes)}, edges={len(self.edges)}, "
            f"ev_idx={len(self.evidence_index)})"
        )


# ════════════════════════════════════════════
# 向后兼容接口 — 保持旧代码不中断
# ════════════════════════════════════════════

_engine_instance: Optional[KnowledgeGraphEngine] = None


def _get_engine() -> KnowledgeGraphEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = KnowledgeGraphEngine()
        _engine_instance.load()
    return _engine_instance


def load_knowledge_graph(graph_path: Optional[str] = None) -> Dict[str, Any]:
    """向后兼容: 返回旧格式的图数据字典."""
    engine = _get_engine()
    if graph_path:
        engine.load(graph_path=graph_path)
    return {
        "graph_id": engine.graph_id,
        "version": engine.graph_version,
        "nodes": [
            {"id": n.node_id, "label": n.label, "type": n.node_type, **n.properties}
            for n in engine.nodes.values()
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "relation": e.rel_type,
                "keywords": e.keywords,
                "evidence_snippet": e.evidence_snippet,
                **{"properties": e.properties},
            }
            for e in engine.edges
        ],
    }


def retrieve_knowledge_graph_hits(
    query: str,
    top_k: int = 3,
    graph_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """向后兼容: 返回旧格式的命中列表."""
    engine = _get_engine()
    if graph_path:
        engine.load(graph_path=graph_path)
    hits = engine.search(query=query, top_k=top_k)
    return [h.to_dict() for h in hits]


def summarize_knowledge_graph_hits(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """向后兼容: 汇总命中结果."""
    relations: List[str] = []
    sources_labels: List[str] = []
    for hit in hits:
        edge = hit.get("edge", {}) if isinstance(hit, dict) else {}
        if not isinstance(edge, dict):
            continue
        relation = str(edge.get("relation", "")).strip()
        sn = hit.get("source_node", {})
        source_label = str(sn.get("label", "")) if isinstance(sn, dict) else ""
        if relation:
            relations.append(relation)
        if source_label:
            sources_labels.append(source_label)

    uniq_rel = []
    for r in relations:
        if r not in uniq_rel:
            uniq_rel.append(r)

    uniq_src = []
    for s in sources_labels:
        if s not in uniq_src:
            uniq_src.append(s)

    return {
        "hit_count": len(hits),
        "top_relations": uniq_rel[:5],
        "top_sources": uniq_src[:5],
    }


# 新增快捷函数供外部直接使用引擎


def get_kg_engine(force_reload: bool = False) -> KnowledgeGraphEngine:
    """获取全局 KG 引擎实例（懒加载+缓存）."""
    global _engine_instance
    if force_reload or _engine_instance is None:
        _engine_instance = KnowledgeGraphEngine()
        _engine_instance.load()
    return _engine_instance
