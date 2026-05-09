#!/usr/bin/env python
# coding: utf-8

"""
M7 WebRetriever v2 — 可选 L2 外部搜索引擎客户端
=================================================

设计原则:
- 完全可降级: API 不可用时返回空列表，不阻塞主流程
- 多引擎备选: Serper (主力) → Bing (备用) → Mock (测试)
- 结果结构化: 每个 result 都带 URL、标题、摘要、时间戳、新鲜度评分
  
嵌入位置:
  被 m7_inference_runner.py 在 SearchArbiter 决定 need_local_evidence=True 时调用，
  搜索结果作为结构化证据注入 User Prompt。

注意：这是 L2（补充层），不是必须的。
    L1（LLM 内置联网）是主力，L2 仅用于交叉验证。
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

logger = logging.getLogger("m7_web_retriever")


@dataclass
class WebEvidence:
    """结构化网络证据"""
    evidence_id: str           # SHA1 哈希 ID
    url: str
    title: str
    snippet: str
    publish_date: Optional[str] = None
    freshness_score: float = 0.5  # 0-1, 新鲜度评分
    source_reliability: str = "medium"  # high / medium / low
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_format(self) -> str:
        """格式化为注入 Prompt 的证据文本"""
        date_str = f" | {self.publish_date}" if self.publish_date else ""
        return (
            f"[WEB-{self.evidence_id}] {self.title}{date_str}\n"
            f"  来源: {self.url}\n"
            f"  摘要: {self.snippet[:300]}\n"
            f"  可靠性: {self.source_reliability.upper()}"
        )


# ─────────── 引擎实现 ───────────

class _SearchEngineBase:
    """搜索引擎基接口"""

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _MockEngine(_SearchEngineBase):
    """Mock 引擎 — 用于测试和降级"""

    def __init__(self):
        self._mock_results = [
            {
                "title": "[Mock] 相关行业数据报告",
                "url": "https://example.com/mock-report",
                "snippet": "这是一个模拟的搜索结果，用于验证系统在无真实搜索API时的降级行为。",
                "date": datetime.now().strftime("%Y-%m-%d"),
            },
        ]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"[WebRetriever/Mock] 返回模拟结果 for query='{query}'")
        return self._mock_results[:top_k]


class _SerperEngine(_SearchEngineBase):
    """Serper.dev (Google Search) 引擎"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://google.serper.dev/search"

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        payload = json.dumps({"q": query, "num": top_k}).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=payload,
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                organic = data.get("organic", [])
                results = []
                for item in organic[:top_k]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "date": item.get("date", ""),
                    })
                return results
        except Exception as e:
            logger.warning(f"[WebRetriever/Serper] 搜索失败: {e}")
            return []


# ─────────── 主类 ───────────

class WebRetriever:
    """
    L2 外部搜索引擎客户端
    
    用法:
        retriever = WebRetriever()
        results = retriever.search("AI教育 创业风险", top_k=3)
        # 如果没有配置 API key，自动降级到 Mock（返回空列表或模拟数据）
        
        # 反证搜索
        counter_results = retriever.search_counter_evidence(
            claim="iPhone 17 尚未发布",
            exclude_urls={"https://example.com/wrong-source"},
        )
    """

    def __init__(self, config_path: Optional[str] = None):
        self._engine: Optional[_SearchEngineBase] = None
        self._engine_name = "none"
        self._init_engine(config_path)

    def _init_engine(self, config_path: Optional[str]) -> None:
        """按优先级初始化引擎: Serper → Mock"""
        # 尝试从环境变量获取 API Key
        serper_key = os.getenv("SERPER_API_KEY", "")

        # 尝试从配置文件获取
        if not serper_key and config_path:
            try:
                cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
                serper_key = cfg.get("serper_api_key", "")
            except Exception:
                pass

        if serper_key and serper_key != "your-api-key":
            self._engine = _SerperEngine(serper_key)
            self._engine_name = "serper"
            logger.info("[WebRetriever] 使用 Serper.dev 引擎")
        else:
            # 无可用引擎 → 降级为空（不使用 Mock 避免污染结果）
            self._engine = None
            self._engine_name = "disabled"
            logger.info("[WebRetriever] 无搜索API配置，L2搜索已禁用（仅使用L1联网）")

    @property
    def is_available(self) -> bool:
        return self._engine is not None

    def _compute_freshness_score(self, publish_date_str: Optional[str]) -> float:
        """基于发布时间计算新鲜度评分"""
        if not publish_date_str:
            return 0.5
        
        now = datetime.now(timezone.utc)
        try:
            # 尝试解析常见日期格式
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
                try:
                    pub_dt = datetime.strptime(publish_date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.5
            
            age_days = (now.replace(tzinfo=None) - pub_dt).days
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.85
            elif age_days <= 30:
                return 0.65
            elif age_days <= 365:
                return 0.4
            else:
                return 0.2
        except (ValueError, TypeError):
            return 0.5

    def _classify_reliability(self, url: str) -> str:
        """基于 URL 判断来源可靠性"""
        url_lower = url.lower()
        
        # 高可信来源
        high_trust_domains = [
            ".gov.cn", ".gov", ".edu.cn", ".edu",
            "who.is", "apple.com", "microsoft.com", "google.com",
            "baike.baidu.com", "zh.wikipedia.org", "wikipedia.org",
            "stats.gov.cn", "reuters.com", "bloomberg.com",
            "techcrunch.com", "theverge.com", "arstechnica.com",
        ]
        if any(d in url_lower for d in high_trust_domains):
            return "high"

        # 低可信/可能为广告
        ad_indicators = ["ad.", "/ad/", "sponsored", "promotion", "affiliate"]
        if any(ind in url_lower for ind in ad_indicators):
            return "low"

        return "medium"

    def _make_evidence_id(self, url: str, snippet: str) -> str:
        """生成唯一证据 ID"""
        raw = f"{url}|{snippet}".encode("utf-8", errors="ignore")
        digest = hashlib.sha1(raw).hexdigest()[:8]
        return digest

    def search(
        self,
        query: str,
        top_k: int = 5,
        time_filter: Optional[str] = None,
    ) -> List[WebEvidence]:
        """
        执行搜索并返回结构化证据
        
        Args:
            query: 搜索查询
            top_k: 返回结果数上限
            time_filter: 时间范围过滤 ("past_day"/"past_week"/"past_month"/"past_year")
        
        Returns:
            List[WebEvidence]，搜索失败时返回空列表（永不阻塞主流程）
        """
        if not self._engine or not query.strip():
            return []

        try:
            raw_results = self._engine.search(query.strip(), top_k=top_k)
            
            evidence_list: List[WebEvidence] = []
            for r in raw_results:
                ev_id = self._make_evidence_id(r.get("url", ""), r.get("snippet", ""))
                evidence_list.append(WebEvidence(
                    evidence_id=ev_id,
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    publish_date=r.get("date"),
                    freshness_score=self._compute_freshness_score(r.get("date")),
                    source_reliability=self._classify_reliability(r.get("url", "")),
                    raw=r,
                ))

            logger.debug(f"[WebRetriever] 搜索 '{query}' → {len(evidence_list)} 条结果 (engine={self._engine_name})")
            return evidence_list

        except Exception as e:
            logger.warning(f"[WebRetriever] 搜索异常（已降级）: {e}")
            return []

    def search_counter_evidence(
        self,
        claim: str,
        exclude_urls: Optional[set] = None,
    ) -> List[WebEvidence]:
        """
        反证定向搜索 — 针对给定声明主动搜索反面证据
        
        Args:
            claim: 待验证的声明文本
            exclude_urls: 要排除的URL集合（原始证据来源）
        
        Returns:
            与原始来源不同的新证据列表
        """
        exclude = exclude_urls or set()

        # 生成反证查询词
        counter_queries = self._generate_counter_queries(claim)
        
        all_results: List[WebEvidence] = []
        seen_urls: set = set()
        
        for q in counter_queries:
            results = self.search(q, top_k=3)
            for ev in results:
                if ev.url not in exclude and ev.url not in seen_urls:
                    all_results.append(ev)
                    seen_urls.add(ev.url)

        if all_results:
            logger.debug(f"[WebRetriever] 反证搜索 '{claim[:50]}...' → {len(all_results)} 条反证")
        return all_results

    @staticmethod
    def _generate_counter_queries(claim: str) -> List[str]:
        """
        根据声明生成反证查询词
        
        策略：
          - 给声明加上"真的吗""是否""最新"等反问前缀
          - 提取关键词 + "2024 2025 2026" 追加年份限定
        """
        queries = [claim]

        # 反问式
        for prefix in [f"{claim} 是否属实", f"{claim} 最新情况", f"{claim} 真的吗"]:
            queries.append(prefix)

        # 关键词+年份式
        words = claim.split(" ")[:6]
        year_query = " ".join(words) + " 2025 2026"
        queries.append(year_query)

        return list(dict.fromkeys(queries))[:4]  # 去重保序，最多4条

    @staticmethod
    def format_evidence_for_prompt(evidence_list: List[WebEvidence], max_items: int = 5) -> str:
        """
        将网络证据列表格式化为注入 Prompt 的文本块
        
        这个方法被 m7_inference_runner.py 调用来构建增强 User Prompt。
        """
        if not evidence_list:
            return ""

        # 按新鲜度排序
        sorted_ev = sorted(evidence_list, key=lambda e: e.freshness_score, reverse=True)[:max_items]

        lines = [
            "以下是经互联网检索得到的最新参考资料，请确保你的回答与这些资料一致：\n",
        ]
        for idx, ev in enumerate(sorted_ev, 1):
            lines.append(f"--- 参考资料 [{idx}] ---")
            lines.append(ev.to_prompt_format())
            lines.append("")

        return "\n".join(lines)
