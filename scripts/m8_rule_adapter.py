#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
from typing import Dict, List, Tuple, Any

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
    'country': 'US'
}

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
    # 1. 填充默认值，确保兼容脚本传入的所有字段
    project_data = {**DEFAULT_VALUES, **project_data}
    reasons = []
    intermediate = {}  # 核心：必须包含脚本需要的6个特征字段
    is_high_fail = project_data['main_category'] in HIGH_FAIL_CATEGORIES
    
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
    
    # 4. 组合风险值计算（重构权重求和逻辑，显式定义基础特征）
    BASE_FEATURES = ['goal_ratio', 'time_penalty', 'category_risk', 'country_factor', 'urgency_score']
    total_weight = sum(FEATURE_WEIGHTS[feat] for feat in BASE_FEATURES)
    
    weighted_sum = goal_ratio_weighted + time_penalty_weighted + category_risk_weighted + country_factor_weighted + urgency_score_weighted
    normalized_weighted = weighted_sum / total_weight if total_weight > 0 else 0.0
    raw_sum = goal_ratio + time_penalty + category_risk + country_factor + urgency_score
    combined_risk = (normalized_weighted * NORMALIZED_RATIO + raw_sum * RAW_SUM_RATIO) * FEATURE_WEIGHTS['combined_risk']
    
    # 5. 最终平衡版校验规则
    core_triggered = 0
    if goal_ratio > SCENARIO_THRESHOLDS['goal_ratio']:
        core_triggered += 1
        reasons.append(f"目标倍率过高（{goal_ratio:.2f} > 阈值{SCENARIO_THRESHOLDS['goal_ratio']:.2f}）")
    if time_penalty > SCENARIO_THRESHOLDS['time_penalty']:
        core_triggered += 1
        reasons.append(f"周期惩罚过高（{time_penalty:.2f} > 阈值{SCENARIO_THRESHOLDS['time_penalty']:.2f}）")
    if category_risk > SCENARIO_THRESHOLDS['category_risk']:
        core_triggered += 1
        reasons.append(f"品类风险过高（{category_risk:.2%} > 阈值{SCENARIO_THRESHOLDS['category_risk']:.2f}）")
    if country_factor > SCENARIO_THRESHOLDS['country_factor']:
        reasons.append(f"国家风险过高（{country_factor:.2f} > 阈值{SCENARIO_THRESHOLDS['country_factor']:.2f}）")
    if urgency_score > SCENARIO_THRESHOLDS['urgency_score']:
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
    intermediate['goal_ratio'] = round(goal_ratio, 4)
    intermediate['time_penalty'] = round(time_penalty, 4)
    intermediate['category_risk'] = round(category_risk, 4)
    intermediate['combined_risk'] = round(combined_risk, 4)
    intermediate['country_factor'] = round(country_factor, 4)
    intermediate['urgency_score'] = round(urgency_score, 4)
    
    # 9. 整理原因（兼容verbose参数）
    if not reasons:
        reasons.append(f"无高风险因素，组合风险值：{combined_risk:.2f}")
    else:
        reasons.insert(0, f"核心风险等级：{risk_level}（组合风险值：{combined_risk:.2f}）")
    
    if verbose:
        print(f"[M8判定结果] 风险等级：{risk_level} | 组合风险值：{combined_risk:.2f}")
    
    return risk_level, reasons, intermediate

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




