#!/usr/bin/env python
# coding: utf-8

"""
M7 FreshnessDetector — 知识时效性信号检测器
================================================

核心职责：检测用户输入中隐含或显式的「需要最新知识」的信号，
为 SearchArbiter 提供时效性评分和决策依据。

设计原则：
- 不是硬编码具体例子（如 iPhone 17、高铁抢票），而是识别
  「答案可能随时间变化」的通用语言模式和语义类别。
- 所有模式列表可从 search_config.json 扩展，支持运行时热更新。

检测层次：
  L0  无时效性（稳态知识）          → score ≈ 0.1
  L1  显式时间指涉                  → score 0.9-1.0
  L2  隐式时效依赖（产品/价格/状态）→ score 0.6-0.8
  L3  快速变化领域                  → score 0.5-0.7
  L4  事实验证需求                  → score 0.4-0.6
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("m7_freshness_detector")


# ─────────── 默认配置（可被 JSON 覆盖） ───────────

# L1: 显式时间表达式（用户明确提到时间点）
DEFAULT_EXPLICIT_TIME_PATTERNS = [
    r"(?:现在|目前|当前|今天|昨天|今天|今晚|今晨)\b",
    r"\b(?:20\d{2})\s*[年\-/]\d{1,2}(?:\s*[\-月]\s*\d{1,2})?\b",  # 具体日期 2026年 / 2026-04
    r"(?:最近|近期|近来|最近有没有|这阵子|如今|眼下|现阶段|当前阶段)\b",
    r"(?:最新(?:版)?|当前版本|现版本|新版|更新(?:后)?|已升级)\b",
    r"(?:已经[发上]?了?|已[发布上市]?)\b",
]

# L2: 隐式时效依赖模式（答案随时间变化但用户未明说）
DEFAULT_IMPLICIT_TEMPORAL_PATTERNS = {
    "product_version": {
        "pattern": r"\b[A-Za-z]{2,}\s*\d{1,4}[sSpPrRoOMmAaUuXx]*(?:\s*(?:Pro|Max|Plus|Mini|Air|Ultra|SE))?(\b|[^\w])",
        "description": "产品型号/版本号（答案取决于当前市售版本）",
        "weight": 0.85,
        "examples": ["iPhone 17", "GPT-5", "Android 16"],
    },
    "current_status": {
        "pattern": r"(?:现任|目前|当前| incumbent |现在的?)\s*(CEO?|总裁|负责人|经理|部长|局长|市长|省长|主席|创始人|队长|教练)",
        "description": "当前状态/在任角色",
        "weight": 0.75,
        "examples": ["现任苹果 CEO", "目前谁在管"],
    },
    "changing_metric": {
        "pattern": r"(?:售价|价格|多少钱|票价|房价|工资|薪资|营收|市值|股价|汇率|利率|票房|涨了?|跌了?|增长率|同比|环比|多少[个人把张只])",
        "description": "数量/度量值（可能随时间变化）",
        "weight": 0.80,
        "examples": ["iPhone 售价多少", "比特币涨了没"],
    },
    "fact_check": {
        "pattern": r"(?:是否|真的吗|假的吧|听说|传闻|确认.*是不是|到底|真的假的|骗人|谣言|属实吗|确有其事)",
        "description": "事实核查/真伪判断",
        "weight": 0.65,
        "examples": ["某公司破产了吗是真的吗"],
    },
    "release_event": {
        "pattern": r"(?:发布|上市|开售|推出|更新|上线|发售|预售|首发|开演|开播|上映|落地|实施)",
        "description": "发布/上市/更新/落地事件",
        "weight": 0.70,
        "examples": ["什么时候发布的", "何时上线"],
    },
    "policy_change": {
        "pattern": r"(?:新政策|新规定|新法规|出台|已经改|变了没有|还[允能行]?吗|放开|收紧|限制|禁令|解禁|调整了?)",
        "description": "政策/法规变更类问题",
        "weight": 0.72,
        "examples": ["新政策出台了没", "规定变了没有"],
    },
    "ranking_trend": {
        "pattern": r"(?:排行|榜|第[一二三四五]\d*名|最[好差大小多少高低]|热门|火[爆不]?|流[行不]?|趋势|风向|风口)",
        "description": "排名/榜单/趋势/现状类",
        "weight": 0.60,
        "examples": ["现在什么最火", "排行第几"],
    },
    "service_status": {
        "pattern": r"(?:营业|开门|关门|倒闭|破产|停业|在营|运营中|正常营业|打烊|下班|人多不多|拥挤|排队|还有座|满座)",
        "description": "运营/服务实时状态",
        "weight": 0.78,
        "examples": ["现在人多吗", "还在营业吗"],
    },
}

# L3: 高频变化的领域关键词（命中时提升时效性分数）
DEFAULT_VOLATILE_DOMAINS = {
    "finance": {"keywords": ["股票", "基金", "比特币", "加密货币", "金价", "油价", "汇率", "利率", "CPI", "GDP", "加息", "降息"], "base_score": 0.55},
    "tech_products": {"keywords": ["iphone", "ipad", "android", "华为", "小米", "特斯拉", "英伟达", "OpenAI", "GPT", "Claude", "Gemini", "Llama", "DeepSeek"], "base_score": 0.55},
    "transportation": {"keywords": ["高铁", "航班", "机票", "地铁", "公交", "堵车", "限行", "高速", "油价", "路况"], "base_score": 0.60},
    "entertainment": {"keywords": ["票房", "热播", "综艺", "剧集", "电影", "音乐节", "演唱会", "赛事", "世界杯", "奥运", "综艺"], "base_score": 0.50},
    "social_public": {"keywords": ["疫情", "签证", "入境", "出境", "社保", "公积金", "税率", "补贴", "放假", "调休", "节假日"], "base_score": 0.50},
    "daily_life": {"keywords": ["天气", "温度", "PM2.5", "空气质量", "紫外线", "穿衣", "降水"], "base_score": 0.70},  # 天气类天然高时效
}


@dataclass
class FreshnessResult:
    """时效性检测结果"""
    temporal_score: float              # 0.0 - 1.0 综合时效评分
    signals: List[Dict[str, Any]]      # 匹配到的所有信号详情
    suggested_time_range: Optional[str] = None   # 建议的搜索时间范围
    extracted_entities: List[str] = field(default_factory=list)  # 提取出的关键实体


class FreshnessDetector:
    """
    时效性信号检测器
    
    用法:
        detector = FreshnessDetector()
        result = detector.detect("iPhone 17现在售价多少")
        # result.temporal_score ≈ 0.95
        # result.signals 包含 L1"现在" + L2"product_version" + L2"changing_metric"
    """

    def __init__(self, config_path: Optional[str] = None):
        self._explicit_patterns = list(DEFAULT_EXPLICIT_TIME_PATTERNS)
        self._implicit_patterns = dict(DEFAULT_IMPLICIT_TEMPORAL_PATTERNS)
        self._volatile_domains = dict(DEFAULT_VOLATILE_DOMAINS)
        
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """从 JSON 配置文件加载/覆盖默认模式"""
        try:
            p = Path(config_path)
            if not p.exists():
                return
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "explicit_time_patterns" in cfg:
                self._explicit_patterns = cfg["explicit_time_patterns"]
            if "implicit_temporal_patterns" in cfg:
                self._implicit_patterns.update(cfg["implicit_temporal_patterns"])
            if "volatile_domains" in cfg:
                self._volatile_domains.update(cfg["volatile_domains"])
            logger.info(f"[FreshnessDetector] 已加载配置: {config_path}")
        except Exception as e:
            logger.warning(f"[FreshnessDetector] 加载配置失败: {e}")

    def detect(self, user_input: str) -> FreshnessResult:
        """
        综合检测输入的时效性需求强度
        
        Returns:
            FreshnessResult(temporal_score, signals, suggested_time_range, extracted_entities)
        """
        if not user_input or not user_input.strip():
            return FreshnessResult(0.0, [])

        text = user_input.strip()
        signals: List[Dict[str, Any]] = []
        score_components: List[Tuple[float, str]] = []  # (score_weight, source)

        # ── L1: 显式时间指涉 (最高权重) ──
        explicit_hits = []
        for pat in self._explicit_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                hit_str = m if isinstance(m, str) else str(m)
                if hit_str.strip():
                    explicit_hits.append(hit_str)

        if explicit_hits:
            l1_score = min(1.0, 0.85 + 0.05 * len(explicit_hits))
            score_components.append((l1_score * 0.35, "L1_explicit_time"))
            signals.append({
                "level": "L1", "type": "explicit_time",
                "matches": explicit_hits[:3], "score_contribution": round(l1_score * 0.35, 3),
            })

        # ── L2: 隐式时效依赖模式 ──
        l2_max = 0.0
        l2_entities: List[str] = []
        for pattern_name, pattern_cfg in self._implicit_patterns.items():
            if isinstance(pattern_cfg, dict):
                pat_str = pattern_cfg.get("pattern", "")
                weight = pattern_cfg.get("weight", 0.7)
            else:
                pat_str = pattern_cfg
                weight = 0.7
            
            matches = re.findall(pat_str, text, re.IGNORECASE)
            if matches:
                l2_max = max(l2_max, weight)
                matched_text = matches[0] if isinstance(matches[0], str) else str(matches[0])
                l2_entities.append(f"{pattern_name}:{matched_text}")
                signals.append({
                    "level": "L2", "type": pattern_name,
                    "match": matched_text, "weight": weight,
                    "score_contribution": round(weight * 0.30, 3),
                })

        if l2_max > 0:
            score_components.append((l2_max * 0.30, "L2_implicit_temporal"))

        # ── L3: 高频变化领域关键词 ──
        l3_max = 0.0
        text_lower = text.lower()
        for domain_name, domain_cfg in self._volatile_domains.items():
            if isinstance(domain_cfg, dict):
                keywords = domain_cfg.get("keywords", [])
                base = domain_cfg.get("base_score", 0.5)
            else:
                keywords = domain_cfg if isinstance(domain_cfg, list) else []
                base = 0.5
            
            for kw in keywords:
                if kw.lower() in text_lower:
                    l3_max = max(l3_max, base)
                    break

        if l3_max > 0:
            score_components.append((l3_max * 0.20, "L3_volatile_domain"))
            signals.append({
                "level": "L3", "type": "volatile_domain",
                "max_domain_score": l3_max,
                "score_contribution": round(l3_max * 0.20, 3),
            })

        # ── L4: 事实验证标记（低权重的加分项） ──
        has_fact_check = any(s.get("type") == "fact_check" for s in signals)
        if not has_fact_check and any(kw in text for kw in ["吗", "?", "？"]):
            # 问句但没有明确时效信号 → 微量加分
            question_boost = 0.05
            score_components.append((question_boost, "L4_question_form"))

        # ── 综合评分 ──
        final_score = sum(sc for sc, _ in score_components)
        final_score = min(1.0, max(0.05, final_score))  # clamp [0.05, 1.0]

        # ── 提取实体（用于搜索提示生成） ──
        entities = list(set(explicit_hits + l2_entities))

        # ── 时间范围建议 ──
        time_range = self._infer_time_range(text, signals)

        return FreshnessResult(
            temporal_score=round(final_score, 4),
            signals=signals,
            suggested_time_range=time_range,
            extracted_entities=entities,
        )

    @staticmethod
    def _infer_time_range(text: str, signals: List[Dict]) -> Optional[str]:
        """根据匹配到的信号推断建议的搜索时间范围"""
        for sig in signals:
            level = sig.get("level", "")
            match_text = ""
            if "matches" in sig:
                match_text = " ".join(sig["matches"][:2])
            elif "match" in sig:
                match_text = sig["match"]

            # 明确年份 → 以该年为中心前后一年
            year_match = re.search(r'(?:20\d{2})', text)
            if year_match:
                year = year_match.group(1)
                return f"{int(year)-1}-{int(year)+1}"

            # 近期类词 → 过去一个月到一年
            if any(w in match_text for w in ["最近", "近期", "近来", "这阵子", "如今"]):
                return "past_year"

            # 现在/今天 → 过去一周到一月
            if any(w in match_text for w in ["现在", "目前", "当前", "今天"]):
                return "past_month"

            # 产品版本类 → 过去两年（覆盖发布周期）
            if sig.get("type") in ("product_version",):
                return "past_2years"

            # 运营状态/天气 → 实时（过去一天）
            if sig.get("type") in ("service_status",) or "天气" in text:
                return "past_day"

        # 默认无特殊限制
        return None

    @classmethod
    def extract_temporal_entities(cls, user_input: str) -> List[str]:
        """快捷方法：提取文本中的时效相关实体"""
        detector = cls()
        result = detector.detect(user_input)
        return result.extracted_entities
