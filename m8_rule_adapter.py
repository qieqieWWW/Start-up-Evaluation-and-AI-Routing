#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from math import inf
from typing import Dict, List, Tuple, Optional

# ===================== 众筹项目失败风险规则 =====================
# 扩展：失败/取消/暂停=高风险，成功=低风险；M8新增：风险分层（极高/高/中/低）、异常值处理、可配置化

# 基准配置（继承M4并扩展）
CATEGORY_MEDIAN_GOAL = {
    'Art': 1850.0, 'Comics': 1875.25, 'Crafts': 1674.33, 'Dance': 3629.57, 'Design': 3342.18,
    'Fashion': 3871.49, 'Film & Video': 5000.0, 'Food': 10000.0, 'Games': 1172.98, 'Journalism': 5000.0,
    'Music': 4105.41, 'OTHER': 3601.03, 'Photography': 3691.76, 'Publishing': 2529.01,
    'Technology': 7740.21, 'Theater': 3500.0
}
CATEGORY_RISK_RATE = {
    'Art': 0.16809196691434178, 'Comics': 0.07819502395302118, 'Crafts': 0.5431529411764706,
    'Dance': 0.12048453385247675, 'Design': 0.2067414072872105, 'Fashion': 0.31211692597831214,
    'Film & Video': 0.21642284367545195, 'Food': 0.5278857262892553, 'Games': 0.17434334154591388,
    'Journalism': 0.6559679037111334, 'Music': 0.14314443015857317, 'OTHER': 0.13010007698229406,
    'Photography': 0.3409767718880286, 'Publishing': 0.2390272373540856, 'Technology': 0.35256436411114284,
    'Theater': 0.19440175631174533
}
# M4基础阈值 + M8新增风险分层阈值
OPTIMAL_THRESHOLDS = {
    'goal_ratio': 1.7081081081081082, 'time_penalty': 2.6692966676192444,
    'category_risk': 0.31211692597831214, 'combined_risk': 4.776823483116145,
    'country_factor': 0.28695652173913044, 'urgency_score': 0.11666666666666667,
    'fund_sim': inf, 'time_ratio': inf
}
# M8新增：风险等级阈值（基于combined_risk分层）
RISK_LEVEL_THRESHOLDS = {
    'extreme_high': 8.0,    # 极高风险
    'high': 4.776823483116145,  # 高风险（复用M4的combined_risk阈值）
    'medium': 2.0,          # 中风险
    'low': 0.0              # 低风险
}
FEATURE_AUC = {
    'goal_ratio': 0.7114371030897235, 'time_penalty': 0.6683061536423809,
    'category_risk': 0.6878606367589315, 'combined_risk': 0.7337647469640597,
    'country_factor': 0.5730287965048341, 'urgency_score': 0.3316938463576192,
    'fund_sim': 0.5, 'time_ratio': 0.5
}
# M8新增：默认值配置（处理缺失/异常数据）
DEFAULT_VALUES = {
    'median_goal': 5000.0,
    'country_factor': 0.5,
    'actual_funding_usd': 0.0,
    'planned_duration_days': 30.0,  # 行业通用默认周期
    'duration_days': 30.0
}


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """M8新增：安全除法，避免除零错误"""
    if denominator <= 0.01:  # 阈值兼容M4的max(..., 0.01)逻辑
        return default
    return numerator / denominator


def _get_risk_level(combined_risk: float) -> str:
    """M8新增：根据组合风险值判定风险等级"""
    if combined_risk >= RISK_LEVEL_THRESHOLDS['extreme_high']:
        return '极高风险'
    elif combined_risk >= RISK_LEVEL_THRESHOLDS['high']:
        return '高风险'
    elif combined_risk >= RISK_LEVEL_THRESHOLDS['medium']:
        return '中风险'
    else:
        return '低风险'


def judge_project_risk_m8(project_data: Dict[str, float], verbose: bool = True) -> Tuple[str, List[str], Dict[str, float]]:
    """
    M8版本风险评估函数
    :param project_data: 项目数据字典，包含goal_usd/duration_days/main_category等字段
    :param verbose: 是否输出详细计算日志
    :return: 风险等级(str)、风险原因(List[str])、中间计算值(Dict[str, float])
    """
    risk_reasons = []
    intermediate_values = {}  # M8新增：存储中间计算值，便于排查
    project_data = {**DEFAULT_VALUES, **project_data}  # 填充默认值，处理缺失数据

    # ========== 1. 目标倍率风险（继承M4，增强鲁棒性） ==========
    median_goal = CATEGORY_MEDIAN_GOAL.get(project_data['main_category'], DEFAULT_VALUES['median_goal'])
    goal_ratio_val = _safe_divide(project_data['goal_usd'], median_goal)
    intermediate_values['goal_ratio'] = goal_ratio_val
    if goal_ratio_val > OPTIMAL_THRESHOLDS['goal_ratio']:
        risk_reasons.append(f"目标倍率过高（{goal_ratio_val:.2f} > 阈值{OPTIMAL_THRESHOLDS['goal_ratio']:.2f}）")
        if verbose:
            print(f"[计算日志] 品类{project_data['main_category']}基准目标{median_goal}$，项目目标{project_data['goal_usd']}$，倍率{goal_ratio_val:.2f}")

    # ========== 2. 周期风险（继承M4，异常值限制） ==========
    duration_days = max(project_data['duration_days'], 1)  # 避免天数为0/负数
    time_penalty_val = np.exp(duration_days / 30) - 1
    intermediate_values['time_penalty'] = time_penalty_val
    if time_penalty_val > OPTIMAL_THRESHOLDS['time_penalty']:
        risk_reasons.append(f"周期风险过高（惩罚指数{time_penalty_val:.2f} > 阈值{OPTIMAL_THRESHOLDS['time_penalty']:.2f}）")
        if verbose:
            print(f"[计算日志] 项目周期{duration_days}天，周期惩罚指数{time_penalty_val:.2f}")

    # ========== 4. 品类风险（继承M4，补充百分比格式化） ==========
    cat_risk_val = CATEGORY_RISK_RATE.get(project_data['main_category'], 0.5)
    intermediate_values['category_risk'] = cat_risk_val
    if cat_risk_val > OPTIMAL_THRESHOLDS['category_risk']:
        risk_reasons.append(f"品类风险过高（高风险率{cat_risk_val:.2%} > 阈值{OPTIMAL_THRESHOLDS['category_risk']:.2%}）")

    # ========== 3. 组合风险（继承M4，作为核心分层依据） ==========
    combined_risk_val = goal_ratio_val * time_penalty_val
    intermediate_values['combined_risk'] = combined_risk_val
    if combined_risk_val > OPTIMAL_THRESHOLDS['combined_risk']:
        risk_reasons.append(f"组合风险过高（{combined_risk_val:.2f} > 阈值{OPTIMAL_THRESHOLDS['combined_risk']:.2f}）")

    
    # ========== 5. 国家风险（M4基础上补充默认值） ==========
    country_factor_val = project_data.get('country_factor', DEFAULT_VALUES['country_factor'])  # 变量名+默认值key统一
    intermediate_values['country_factor'] = country_factor_val  # 中间结果key统一
    if country_factor_val > OPTIMAL_THRESHOLDS['country_factor']:
        risk_reasons.append(f"国家风险过高（高风险率{country_factor_val:.2%} > 阈值{OPTIMAL_THRESHOLDS['country_factor']:.2%}）")  # 变量名统一

    # ========== 6. 紧迫感风险（M4基础上限制分母） ==========
    urgency_score_val = _safe_divide(7, duration_days)
    intermediate_values['urgency_score'] = urgency_score_val
    if urgency_score_val > OPTIMAL_THRESHOLDS['urgency_score']:
        risk_reasons.append(f"紧迫感风险过高（分数{urgency_score_val:.2f} > 阈值{OPTIMAL_THRESHOLDS['urgency_score']:.2f}）")

    # ========== M8核心新增：风险等级判定 ==========
    risk_level = _get_risk_level(combined_risk_val)
    if not risk_reasons:
        risk_reasons.append(f"无高风险因素，风险等级：{risk_level}")
    else:
        risk_reasons.insert(0, f"核心风险等级：{risk_level}（组合风险值：{combined_risk_val:.2f}）")

    return risk_level, risk_reasons, intermediate_values


# In[ ]:




