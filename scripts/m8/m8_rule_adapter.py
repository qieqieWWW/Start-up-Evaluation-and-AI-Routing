#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import re
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mas_blackboard.blackboard import SharedBlackboard

# ===================== 核心配置 =====================
# 高失败品类：历史数据中失败率Top3的品类，需重点加权
HIGH_FAIL_CATEGORIES = ['Journalism', 'Food', 'Crafts']
# 高风险阈值（基于AUC最优切点确定）
HIGH_RISK_THRESHOLD = 5.5
# 精确率补偿阈值：边缘样本降级临界值
PRECISION_COMPENSATE_THRESHOLD = 6.3

# 特征权重
FEATURE_WEIGHTS = {
    'goal_ratio': 0.36,
    'time_penalty': 0.30,
    'category_risk': 0.19,
    'country_factor': 0.09,
    'urgency_score': 0.05,
    'combined_risk': 0.95
}

# 场景约束阈值
SCENARIO_THRESHOLDS = {
    'goal_ratio': 1.8,
    'time_penalty': 2.4,
    'category_risk': 0.32,
    'country_factor': 0.26,
    'urgency_score': 0.10
}

# 特征加权系数（业务规则：目标倍率/周期惩罚权重系数7，品类/国家风险6，紧迫感1）
FEATURE_COEFFICIENTS = {
    'goal_ratio': 7,
    'time_penalty': 7,
    'category_risk': 6,
    'country_factor': 6,
    'urgency_score': 1
}

# 组合风险值融合比例（80%归一化加权值 + 20%原始值，平衡稳定性和灵敏度）
NORMALIZED_RATIO = 0.8
RAW_SUM_RATIO = 0.2

# 默认值（预留字段，适配脚本传参完整性）
DEFAULT_VALUES = {
    'goal_ratio': 0.0,
    'time_penalty': 0.0,
    'category_risk': 0.0,
    'country_factor': 0.5,
    'urgency_score': 0.0,
    'main_category': 'OTHER',
    'duration_days': 30.0,
    'goal_usd': 0.0,
    'actual_funding_usd': 0.0,
    'planned_duration_days': 30.0,
    'country': 'US',
    'user_input': ''  # 新增：用户输入文本
}

# ===================== 文本风险分析配置 =====================
# 文本风险关键词表已移除：由上游 LLM 负责文本风险识别与说明（保留空变量以兼容引用）
CRITICAL_RISK_KEYWORDS = []

# 高风险关键词（已移除，本地不再匹配）
HIGH_RISK_KEYWORDS = []

# 中等风险关键词（已移除，本地不再匹配）
MEDIUM_RISK_KEYWORDS = []


def analyze_text_risk(user_input: str) -> Tuple[float, List[str], List[str]]:
    """
    分析user_input文本中的风险关键词
    
    Args:
        user_input: 用户输入的文本内容
        
    Returns:
        (risk_bonus, matched_critical, matched_high): 
        - risk_bonus: 需要添加到combined_risk的分数
        - matched_critical: 匹配的极高风险关键词
        - matched_high: 匹配的高风险关键词
    """
    if not user_input:
        return 0.0, [], []
    
    risk_bonus = 0.0
    matched_critical = []
    matched_high = []
    
    text_lower = user_input.lower()
    
    # 检查极高风险关键词（每个+4，最多+8）
    critical_count = 0
    for kw in CRITICAL_RISK_KEYWORDS:
        if kw.lower() in text_lower:
            matched_critical.append(kw)
            if critical_count < 2:  # 最多算2个
                risk_bonus += 4.0
            critical_count += 1
    
    # 检查高风险关键词（每个+1，最多+6）
    high_count = 0
    for kw in HIGH_RISK_KEYWORDS:
        if kw.lower() in text_lower:
            matched_high.append(kw)
            if high_count < 6:  # 最多算6个不同类别
                risk_bonus += 1.0
            high_count += 1
    
    # 检查中等风险关键词（整体+0.5，不重复计数）
    has_medium_risk = False
    for kw in MEDIUM_RISK_KEYWORDS:
        if kw.lower() in text_lower:
            has_medium_risk = True
            break
    if has_medium_risk:
        risk_bonus += 0.5
    
    return risk_bonus, matched_critical, matched_high


# ===================== 工具函数 =====================
def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0.01 else default

def _validate_numeric(value: float, field: str, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    数值字段合法性校验，避免异常值（负数/无穷大）影响计算
    :param value: 待校验值
    :param field: 字段名（仅日志用）
    :param min_val: 最小值限制
    :param max_val: 最大值限制
    :return: 校验后的值
    """
    if not isinstance(value, (int, float)):
        return min_val
    return max(min_val, min(value, max_val))

# ===================== 核心函数（适配M3调用脚本） =====================
def judge_project_risk_m8(project_data: Dict[str, Any], verbose: bool = True) -> Tuple[str, List[str], Dict[str, float]]:
    """
    完全适配M3调用脚本的M8风险判定函数
    :param project_data: 项目数据字典（脚本要求的全字段）
    :param verbose: 是否打印详细信息（脚本要求的入参）
    :return: (risk_level, reasons, intermediate)  完全匹配脚本返回格式
             - risk_level: 风险等级字符串
             - reasons: 判定原因列表
             - intermediate: 包含6个核心特征的字典（脚本M3调用必需）
    """
    # Early exit: if upstream extractor flagged insufficient data, return diagnostic
    if project_data.get("_insufficient_data"):
        return ("信息不足", ["无足够有效信息，无法判断"], {})

    # 1. 填充默认值，确保兼容脚本传入的所有字段
    project_data = {**DEFAULT_VALUES, **project_data}
    reasons = []
    intermediate = {}  # 核心：必须包含脚本需要的6个特征字段
    is_high_fail = project_data['main_category'] in HIGH_FAIL_CATEGORIES

    # detect which fields were filled by extractor defaults; if present, treat them as 'missing' for independent judgments
    evidence_obj = project_data.get('evidence') if isinstance(project_data.get('evidence'), dict) else {}
    filled_fields = evidence_obj.get('filled_from_defaults', []) if isinstance(evidence_obj, dict) else []

    # 1.5 文本风险分析（新增：分析user_input中的风险关键词）
    user_input = project_data.get('user_input', '')
    text_risk_bonus, matched_critical, matched_high = analyze_text_risk(user_input)

    # 如果有匹配的极高风险关键词，添加到原因列表
    if matched_critical:
        reasons.append(f"【文本风险】检测到极高风险关键词：{', '.join(matched_critical)}")
    if matched_high:
        reasons.append(f"【文本风险】检测到高风险关键词：{', '.join(matched_high[:5])}{'...' if len(matched_high) > 5 else ''}")

    # 2. 提取所有必要字段并做合法性校验
    goal_ratio = _validate_numeric(project_data['goal_ratio'], 'goal_ratio')
    time_penalty = _validate_numeric(project_data['time_penalty'], 'time_penalty')
    category_risk = _validate_numeric(project_data['category_risk'], 'category_risk')
    country_factor = _validate_numeric(project_data['country_factor'], 'country_factor')
    urgency_score = _validate_numeric(project_data['urgency_score'], 'urgency_score')

    # 3. 加权计算组合风险值
    high_fail_weight = 1.8 if is_high_fail else 0.8
    goal_ratio_weighted = goal_ratio * FEATURE_WEIGHTS['goal_ratio'] * FEATURE_COEFFICIENTS['goal_ratio'] * high_fail_weight
    time_penalty_weighted = time_penalty * FEATURE_WEIGHTS['time_penalty'] * FEATURE_COEFFICIENTS['time_penalty'] * high_fail_weight
    category_risk_weighted = category_risk * FEATURE_WEIGHTS['category_risk'] * FEATURE_COEFFICIENTS['category_risk'] * high_fail_weight
    country_factor_weighted = country_factor * FEATURE_WEIGHTS['country_factor'] * FEATURE_COEFFICIENTS['country_factor'] * high_fail_weight
    urgency_score_weighted = urgency_score * FEATURE_WEIGHTS['urgency_score'] * FEATURE_COEFFICIENTS['urgency_score']

    # 如果 extractor 使用了填充值或回退值（filled_fields），将这些字段排除在组合风险计算之外：
    # - 对应原始数值置 0（避免进入 raw_sum）
    # - 对应加权分量置 0（避免进入 weighted_sum）
    # - 从 total_weight 中剔除其权重以避免归一化时稀释有效特征
    if isinstance(filled_fields, list) and filled_fields:
        if 'goal_ratio' in filled_fields:
            goal_ratio = 0.0
            goal_ratio_weighted = 0.0
        if 'time_penalty' in filled_fields:
            time_penalty = 0.0
            time_penalty_weighted = 0.0
        if 'category_risk' in filled_fields:
            category_risk = 0.0
            category_risk_weighted = 0.0
        if 'country_factor' in filled_fields:
            country_factor = 0.0
            country_factor_weighted = 0.0
        if 'urgency_score' in filled_fields:
            urgency_score = 0.0
            urgency_score_weighted = 0.0

    # 4. 组合风险值计算（重构权重求和逻辑，显式定义基础特征）
    BASE_FEATURES = ['goal_ratio', 'time_penalty', 'category_risk', 'country_factor', 'urgency_score']
    # 在计算总权重时排除被填充的字段权重
    effective_features = [f for f in BASE_FEATURES if f not in (filled_fields or [])]
    total_weight = sum(FEATURE_WEIGHTS[feat] for feat in effective_features) if effective_features else 0.0

    weighted_sum = goal_ratio_weighted + time_penalty_weighted + category_risk_weighted + country_factor_weighted + urgency_score_weighted
    normalized_weighted = weighted_sum / total_weight if total_weight > 0 else 0.0
    raw_sum = goal_ratio + time_penalty + category_risk + country_factor + urgency_score
    combined_risk = (normalized_weighted * NORMALIZED_RATIO + raw_sum * RAW_SUM_RATIO) * FEATURE_WEIGHTS['combined_risk']

    # 3.5 应用文本风险加成（新增）
    if text_risk_bonus > 0:
        combined_risk += text_risk_bonus

    # 5. 最终平衡版校验规则
    core_triggered = 0
    # Only add per-feature reasons if that feature was NOT filled_from_defaults
    if 'goal_ratio' not in filled_fields and goal_ratio > SCENARIO_THRESHOLDS['goal_ratio']:
        core_triggered += 1
        reasons.append(f"目标倍率过高（{goal_ratio:.2f} > 阈值{SCENARIO_THRESHOLDS['goal_ratio']:.2f}）")
    if 'time_penalty' not in filled_fields and time_penalty > SCENARIO_THRESHOLDS['time_penalty']:
        core_triggered += 1
        reasons.append(f"周期惩罚过高（{time_penalty:.2f} > 阈值{SCENARIO_THRESHOLDS['time_penalty']:.2f}）")
    if 'category_risk' not in filled_fields and category_risk > SCENARIO_THRESHOLDS['category_risk']:
        core_triggered += 1
        reasons.append(f"品类风险过高（{category_risk:.2%} > 阈值{SCENARIO_THRESHOLDS['category_risk']:.2f}）")
    if 'country_factor' not in filled_fields and country_factor > SCENARIO_THRESHOLDS['country_factor']:
        reasons.append(f"国家风险过高（{country_factor:.2f} > 阈值{SCENARIO_THRESHOLDS['country_factor']:.2f}）")
    if 'urgency_score' not in filled_fields and urgency_score > SCENARIO_THRESHOLDS['urgency_score']:
        reasons.append(f"紧迫感风险过高（{urgency_score:.2f} > 阈值{SCENARIO_THRESHOLDS['urgency_score']:.2f}）")

    # 6. 边缘样本校验（平衡精确率和召回率）
    if not is_high_fail:
        if (HIGH_RISK_THRESHOLD <= combined_risk < PRECISION_COMPENSATE_THRESHOLD) and core_triggered < 2:
            combined_risk = 4.0
            reasons.append(f"边缘高风险样本，核心特征触发不足（{core_triggered}个），降级为中风险")

    # 7. 风险等级判定
    if combined_risk >= 8.0:
        risk_level = '风险很高'
    elif combined_risk >= HIGH_RISK_THRESHOLD:
        risk_level = '风险较高'
    elif combined_risk >= 4.5:
        risk_level = '风险中等'
    elif combined_risk >= 3.5:
        risk_level = '风险较低'
    else:
        risk_level = '风险很低'

    # 8. 填充intermediate字典（核心！M3调用必需的6个字段）
    # For fields filled by extractor, set intermediate value to None to indicate missing independent judgment
    intermediate['goal_ratio'] = None if 'goal_ratio' in filled_fields else round(goal_ratio, 4)
    intermediate['time_penalty'] = None if 'time_penalty' in filled_fields else round(time_penalty, 4)
    intermediate['category_risk'] = None if 'category_risk' in filled_fields else round(category_risk, 4)
    intermediate['combined_risk'] = round(combined_risk, 4)
    intermediate['country_factor'] = None if 'country_factor' in filled_fields else round(country_factor, 4)
    intermediate['urgency_score'] = None if 'urgency_score' in filled_fields else round(urgency_score, 4)

    # 9. 整理原因（兼容verbose参数）
    if not reasons:
        reasons.append(f"无高风险因素，组合风险值：{combined_risk:.2f}")
    else:
        reasons.insert(0, f"核心风险等级：{risk_level}（组合风险值：{combined_risk:.2f}）")

    if verbose:
        print(f"[M8判定结果] 风险等级：{risk_level} | 组合风险值：{combined_risk:.2f}")

    return risk_level, reasons, intermediate

def _calc_confidence(risk_level: str) -> float:
    """根据风险等级返回对应置信度（供 Blackboard 写入）。"""
    mapping = {
        "风险很高": 0.90,
        "风险较高": 0.82,
        "风险中等": 0.76,
        "风险较低": 0.72,
        "风险很低": 0.65,
    }
    return mapping.get(risk_level, 0.5)


def _map_to_experts(risk_level: str) -> List[str]:
    """将中文风险等级映射为建议专家列表（仅供参考，M11 最终决策）。"""
    normalized = {
        "风险很高": "extreme_high",
        "风险较高": "high",
        "风险中等": "medium",
        "风险较低": "low",
        "风险很低": "low",
    }.get(risk_level, "medium")

    mapping = {
        "extreme_high": ["risk_guardian", "finance_advisor"],
        "high": ["risk_guardian", "finance_advisor"],
        "medium": ["finance_advisor", "ops_executor"],
        "low": ["growth_strategist", "ops_executor"],
    }
    return mapping.get(normalized, ["risk_guardian"])


def _map_to_rationale(risk_level: str) -> str:
    """返回对应建议理由文本，供 M11 参考。"""
    normalized = {
        "风险很高": "extreme_high",
        "风险较高": "high",
        "风险中等": "medium",
        "风险较低": "low",
        "风险很低": "low",
    }.get(risk_level, "medium")

    rationale_map = {
        "extreme_high": "极高风险优先控制下行与现金流安全。",
        "high": "高风险优先风控与财务兜底。",
        "medium": "中风险采取财务稳态加运营修复。",
        "low": "低风险转向增长与运营放大。",
    }
    return rationale_map.get(normalized, "请结合更多上下文判断最终路由。")


def judge_and_write_to_blackboard(project_data: Dict[str, Any], blackboard: "SharedBlackboard", include_thresholds: bool = True) -> Dict[str, Any]:
    """主入口：计算 M8 风险信号并写入 SharedBlackboard。

    - 不修改 judge_project_risk_m8 的内部逻辑；仅封装其返回并写入 blackboard。
    - 写入 zone 名称固定为 "m8_risk"，tags 为 ["risk","m8"], agent_id 为 "m8_rule_engine"。
    - include_thresholds 为 True 时，附加子特征阈值评估与简短建议到 payload。
    - 返回原始计算结果以便独立测试调用。
    """
    # 调用原有计算函数（保持行为不变）
    risk_level, risk_reasons, intermediate = judge_project_risk_m8(project_data, verbose=False)

    confidence = _calc_confidence(risk_level)
    suggestion = {
        "suggested_experts": _map_to_experts(risk_level),
        "suggested_rationale": _map_to_rationale(risk_level),
    }

    payload = {
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "intermediate": intermediate,
        "confidence": confidence,
        "suggestion": suggestion,
    }

    # 可选：附加按阈值划分的小项评估与简短建议
    if include_thresholds:
        try:
            assessment = _build_subfeature_assessment(intermediate)
            brief_advices = _map_subfeature_brief_advice(assessment)
            payload["subfeature_assessment"] = assessment
            payload["subfeature_suggestions"] = brief_advices
        except Exception:
            # 不阻断主流程，记录占位字段
            payload["subfeature_assessment"] = {}
            payload["subfeature_suggestions"] = {}

    # 写入 Blackboard（遵循约定：zone="m8_risk"）
    try:
        # Delay import to avoid circular imports at module import time
        from mas_blackboard.blackboard import SharedBlackboard as _SB  # type: ignore
    except Exception:
        _SB = None

    # Basic runtime check: ensure provided blackboard has write method
    if not hasattr(blackboard, "write"):
        raise ValueError("Provided blackboard does not implement write(zone, content, ...) method")

    # include evidence_refs param for compatibility
    try:
        blackboard.write(zone="m8_risk", content=payload, tags=["risk", "m8"], evidence_refs=None, agent_id="m8_rule_engine")
    except TypeError:
        # fallback if blackboard.write doesn't accept evidence_refs
        blackboard.write(zone="m8_risk", content=payload, tags=["risk", "m8"], agent_id="m8_rule_engine")

    # 返回原始计算结果，供单元测试或兼容旧调用
    return {"risk_level": risk_level, "risk_reasons": risk_reasons, "intermediate": intermediate}

# ===================== 附加阈值配置（由外部分析得出，可作为可选输出） =====================
THRESHOLDS = {
    "goal_ratio": {"low_upper": 0.5, "medium_upper": 1.862572},
    "time_penalty": {"binary_threshold": 1.718282},
    "category_risk": {"low_upper": 0.259151, "medium_upper": 0.328249},
    "country_factor": {"binary_threshold": 0.319879},
    "urgency_score": {"low_upper": 0.225806, "medium_upper": 0.233333},
    "combined_risk": {"high_threshold": 3.844452},
}


def _label_by_threshold(value: float, cfg: Dict[str, float]) -> str:
    """根据阈值配置将数值标记为 'low'|'medium'|'high' 或 'unknown'。"""
    if value is None:
        return "unknown"
    if "binary_threshold" in cfg:
        return "low" if value <= cfg["binary_threshold"] else "high"
    low_u = cfg.get("low_upper")
    med_u = cfg.get("medium_upper")
    if low_u is not None and med_u is not None:
        if value <= low_u:
            return "low"
        if value <= med_u:
            return "medium"
        return "high"
    return "unknown"


def _build_subfeature_assessment(intermediate: Dict[str, float]) -> Dict[str, Any]:
    """构建每个子特征的评估结构。"""
    assessment: Dict[str, Any] = {}
    for feat in ["goal_ratio", "time_penalty", "category_risk", "country_factor", "urgency_score"]:
        val = intermediate.get(feat)
        cfg = THRESHOLDS.get(feat, {})
        label = _label_by_threshold(val, cfg)
        assessment[feat] = {"value": val, "label": label, "thresholds": cfg}

    combined = intermediate.get("combined_risk")
    high_th = THRESHOLDS.get("combined_risk", {}).get("high_threshold")
    assessment["combined_risk_flag"] = {"value": combined, "is_over_high": (combined is not None and high_th is not None and combined > high_th), "threshold": high_th}
    return assessment


def _map_subfeature_brief_advice(assessment: Dict[str, Any]) -> Dict[str, str]:
    """为每个子特征生成简短建议，按 label 映射到一句话建议。"""
    label_to_advice = {
        "goal_ratio": {
            "low": "融资压力低，可按计划推进。",
            "medium": "关注融资节奏，准备替代计划。",
            "high": "融资压力高，请优先评估成本削减与短期融资。",
            "unknown": "请补充融资相关数据以评估。",
        },
        "time_penalty": {
            "low": "进度健康，按里程碑推进。",
            "high": "存在逾期风险，需重排计划并增加资源。",
            "unknown": "时间信息缺失，无法评估逾期风险。",
        },
        "category_risk": {
            "low": "行业风险低，可优先增长投入。",
            "medium": "行业存在中等风险，需关注合规/市场波动。",
            "high": "行业风险高，建议尽快咨询专业风控或合规。",
            "unknown": "行业信息不足，建议补充行业说明。",
        },
        "country_factor": {
            "low": "地区风险低，监管环境友好或稳定。",
            "high": "地区风险高，注意跨境合规与支付路径。",
            "unknown": "国家风险信息不明。",
        },
        "urgency_score": {
            "low": "紧迫度低，优先级可调整。",
            "medium": "有一定紧迫性，请关注短期现金与交付。",
            "high": "紧迫度高，应立即采取缓解措施。",
            "unknown": "紧迫度数据缺失。",
        },
        "combined_risk_flag": {
            "over_high": "总体风险超过高风险阈值，建议优先风控。",
            "normal": "总体风险未超过高风险阈值。",
        },
    }

    advices: Dict[str, str] = {}
    for k, v in assessment.items():
        if k == "combined_risk_flag":
            advices[k] = label_to_advice["combined_risk_flag"]["over_high"] if v.get("is_over_high") else label_to_advice["combined_risk_flag"]["normal"]
            continue
        label = v.get("label", "unknown")
        adv = label_to_advice.get(k, {}).get(label, "无具体建议，请补充信息。")
        advices[k] = adv
    return advices

# ===================== 测试入口（验证接口兼容性） =====================
if __name__ == "__main__":
    # 模拟M3调用脚本传入的测试数据
    test_project = {
        "main_category": "Food",
        "goal_ratio": 1.6877,
        "time_penalty": 1.7183,
        "category_risk": 0.5300,
        "country_factor": 0.3271,
        "urgency_score": 0.2333,
        "duration_days": 30.0,
        "goal_usd": 5000.0,
        "actual_funding_usd": 2500.0,
        "planned_duration_days": 30.0,
        "country": "US"
    }
    
    # 调用函数（完全匹配脚本格式）
    risk_level, reasons, intermediate = judge_project_risk_m8(test_project, verbose=True)
    
    # 打印验证
    print("\n=== 接口兼容性验证 ===")
    print(f"风险等级：{risk_level}")
    print("判定原因：")
    for r in reasons:
        print(f"- {r}")
    print("\nM3调用必需的6个特征：")
    for k, v in intermediate.items():
        print(f"- {k}: {v}")


# In[ ]:



