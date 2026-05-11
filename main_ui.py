#!/usr/bin/env python
# coding: utf-8

"""
项目评估与AI路由 - 统一前端入口
================================
Streamlit Web界面，整合全部模块：
- 文字输入 / 数据文件上传 双通道
- M2清洗 → M4特征 → M8风险 → M7路由 → M9决策 → M10监控 → M12 OOD → M16实验
- 统一输出完整JSON分析结果

架构:
  用户输入 → parse_input() → project_data + user_query
    → M8风险判定 → M7专家路由 → M7 LLM推理 + Blender
    → M9终极引擎 → M10性能监控 → M12 OOD测试
    → 输出JSON + 可视化

作者：架构集成
日期：2026-05-03
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

# ==========================================
# 路径配置
# ==========================================
PROJECT_ROOT = Path(__file__).parent.absolute()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
M7_DIR = SCRIPTS_DIR / "m7"
CONFIG_DIR = PROJECT_ROOT / "config"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "Kickstarter_Clean"
LOGS_DIR = PROJECT_ROOT / "logs"

for p in [str(SCRIPTS_DIR), str(M7_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ==========================================
# 模块延迟导入
# ==========================================
def _import_m8():
    from m8_rule_adapter import judge_project_risk_m8, analyze_text_risk
    return judge_project_risk_m8, analyze_text_risk


def _import_m7_router():
    from m7_router import route_experts
    return route_experts


def _import_m7_inference():
    from m7_inference_runner import run_expert_llm_inference, run_expert_llm_inference_with_blender
    return run_expert_llm_inference, run_expert_llm_inference_with_blender


def _import_m7_context():
    from m7_context_analyzer import build_layer1_context, build_layer2_context, build_layer3_context, build_layer4_context
    return build_layer1_context, build_layer2_context, build_layer3_context, build_layer4_context


def _import_m9():
    from m9 import M9UltimateRiskEngine
    return M9UltimateRiskEngine


def _import_m5():
    from M5_AutoTest_Suite import M5AutomationTestEngine
    return M5AutomationTestEngine


def _import_m6():
    from M6状态管理与日志系统 import EnvStateManager, EnvState, RuleLibrary, RouteDispatcher, StaticValidator
    return EnvStateManager, EnvState, RuleLibrary, RouteDispatcher, StaticValidator


def _import_m10():
    from M10_PerformanceMonitor import M10_PerformanceMonitor
    return M10_PerformanceMonitor


def _import_m12():
    from M12环境增强与OOD测试 import OODScenarioGenerator, ResilienceEvaluator, OODConfig
    return OODScenarioGenerator, ResilienceEvaluator, OODConfig


def _import_m15():
    from m15 import run_ablation_experiments
    return run_ablation_experiments


def _import_m2():
    # M2 是脚本式运行，用 subprocess 调用
    return str(SCRIPTS_DIR / "data_process.py")


def _import_m4():
    return str(SCRIPTS_DIR / "M4代码.py")


def _import_m3():
    from M3仿真环境基座模块版 import StartupEnv, M3Config
    return StartupEnv, M3Config


def _import_mas_blackboard():
    from mas_blackboard.workflow import AdaptiveTieredWorkflow
    from mas_blackboard.classifier import ComplexityClassifier
    return AdaptiveTieredWorkflow, ComplexityClassifier


def _import_m7_visualization():
    try:
        from m7_visualization import save_m7_visualizations
        return save_m7_visualizations
    except ImportError:
        return None


def _import_m7_expert_pool():
    from m7_expert_pool import get_expert_map
    return get_expert_map


# ==========================================
# CSV行 → project_data 转换
# ==========================================

# Kickstarter CSV 列名 → 项目分析特征 映射
CATEGORY_RISK_MAP = {
    "technology": 0.3, "design": 0.25, "games": 0.35, "film & video": 0.45,
    "music": 0.4, "publishing": 0.5, "food": 0.65, "crafts": 0.7,
    "journalism": 0.75, "fashion": 0.45, "art": 0.4, "comics": 0.45,
    "photography": 0.4, "theater": 0.35, "dance": 0.35,
}

COUNTRY_FACTOR_MAP = {
    "us": 0.3, "gb": 0.35, "ca": 0.35, "au": 0.4, "de": 0.4,
    "fr": 0.45, "nl": 0.4, "se": 0.4, "dk": 0.4, "no": 0.4,
    "it": 0.5, "es": 0.55, "jp": 0.35, "kr": 0.4, "cn": 0.6,
}


def row_to_project_data(row: pd.Series) -> Dict[str, Any]:
    """将一行 Kickstarter CSV 数据转换为项目分析所需的 project_data"""
    goal = float(row.get("goal", 0) or 0)
    pledged = float(row.get("pledged", 0) or row.get("converted_pledged_amount", 0) or 0)
    backers = int(row.get("backers_count", 0) or row.get("backers", 0) or 0)
    state = str(row.get("state", "unknown")).lower().strip()
    country = str(row.get("country", "US")).lower().strip()
    category = str(row.get("main_category", row.get("category", "Technology"))).lower().strip()
    duration_days = int(row.get("duration_days", 30) or 30)

    goal_ratio = round(goal / max(pledged, 1), 4) if pledged > 0 else round(goal / 5000, 4)
    time_penalty = round(max(0, (60 - duration_days) / 30.0), 4) if duration_days < 60 else 0.0
    category_risk = CATEGORY_RISK_MAP.get(category, 0.5)
    country_factor = COUNTRY_FACTOR_MAP.get(country, 0.5)

    success_prob = min(1.0, backers / 500.0) if backers > 0 else 0.1
    urgency_score = round(max(0, 1.0 - success_prob) * 0.3, 4)

    combined_risk = round(goal_ratio * 0.36 + time_penalty * 0.30 + category_risk * 0.19 + country_factor * 0.09 + urgency_score * 0.05, 4)

    return {
        "goal_usd": goal,
        "pledged_usd": pledged,
        "backers_count": backers,
        "state": state,
        "duration_days": duration_days,
        "main_category": category.title(),
        "country": country.upper(),
        "goal_ratio": goal_ratio,
        "time_penalty": time_penalty,
        "category_risk": category_risk,
        "country_factor": country_factor,
        "urgency_score": urgency_score,
        "combined_risk": combined_risk,
        "actual_funding_usd": pledged,
        "planned_duration_days": duration_days,
        "project_name": str(row.get("name", "Unknown"))[:100],
    }


def text_to_project_data(text: str) -> Tuple[Dict[str, Any], str]:
    """
    将文字输入解析为 project_data + user_query。
    支持结构化格式: GoalUSD:xxx/Country:xxx/Category:xxx/DurationDays:xxx
    也支持自然语言描述。
    """
    text = text.strip()
    project_data = {}
    user_query = text

    # 尝试解析 key:value 格式
    import re
    kv_pattern = re.compile(r'(\w+)\s*[:=]\s*([^/=]+)')
    matches = kv_pattern.findall(text)
    if matches and len(matches) >= 2:
        kv_map = {k.lower().strip(): v.strip() for k, v in matches}
        project_data = _kv_to_project_data(kv_map)
        if project_data.get("_parsed"):
            user_query = text

    # 如果没有解析出足够信息，使用 M8 的文本分析生成默认值
    if not project_data or not project_data.get("_parsed"):
        project_data = _infer_project_data_from_text(text)

    project_data.pop("_parsed", None)
    return project_data, user_query


def _kv_to_project_data(kv: Dict[str, str]) -> Dict[str, Any]:
    """将键值对映射转为 project_data（支持中英文键名）"""
    result = {"_parsed": False}
    try:
        # 中文键名 → 英文键名 映射
        cn_key = {k.lower().strip(): v for k, v in kv.items()}
        cn_map = {
            "目标金额": "goalusd", "目标": "goalusd", "goalusd": "goalusd", "goal": "goalusd",
            "已筹集": "pledged", "已筹": "pledged", "pledged": "pledged", "funding": "pledged",
            "品类": "category", "类别": "category", "category": "category", "maincategory": "category",
            "国家": "country", "country": "country",
            "持续时间": "durationdays", "周期": "durationdays", "durationdays": "durationdays", "duration": "durationdays",
            "支持者": "backers", "backers": "backers",
        }
        eng_kv = {}
        for cn, eng in cn_map.items():
            if cn in cn_key:
                eng_kv[eng] = cn_key[cn]

        if "goalusd" in eng_kv:
            result["goal_usd"] = float(eng_kv["goalusd"].replace(",", "").replace("$", ""))
        if "country" in eng_kv:
            c = eng_kv["country"].lower().strip()
            result["country"] = c.upper()
            result["country_factor"] = COUNTRY_FACTOR_MAP.get(c, 0.5)
        if "category" in eng_kv:
            cat = eng_kv["category"].lower().strip()
            result["main_category"] = cat.title()
            result["category_risk"] = CATEGORY_RISK_MAP.get(cat, 0.5)
        if "durationdays" in eng_kv:
            result["duration_days"] = int(float(eng_kv["durationdays"]))
        if "pledged" in eng_kv:
            result["pledged_usd"] = float(eng_kv["pledged"].replace(",", "").replace("$", ""))
            result["actual_funding_usd"] = result["pledged_usd"]
        if "backers" in eng_kv:
            result["backers_count"] = int(float(eng_kv["backers"]))

        goal = result.get("goal_usd", 0)
        pledged = result.get("pledged_usd", result.get("actual_funding_usd", 0))
        if goal > 0:
            result["goal_ratio"] = round(goal / max(pledged, 1), 4)

        dur = result.get("duration_days", 30)
        result["time_penalty"] = round(max(0, (60 - dur) / 30.0), 4) if dur < 60 else 0.0
        result["urgency_score"] = result.get("urgency_score", 0.3)
        result["combined_risk"] = round(
            result.get("goal_ratio", 1.0) * 0.36
            + result.get("time_penalty", 0.0) * 0.30
            + result.get("category_risk", 0.5) * 0.19
            + result.get("country_factor", 0.5) * 0.09
            + result.get("urgency_score", 0.3) * 0.05, 4
        )
        result["planned_duration_days"] = result.get("duration_days", 30)

        if "goal_usd" in result and result["goal_usd"] > 0:
            result["_parsed"] = True
    except Exception:
        pass
    return result


def _infer_project_data_from_text(text: str) -> Dict[str, Any]:
    """从自然语言文本推断 project_data 默认值"""
    import re
    text_lower = text.lower()

    # 尝试从结构化中文格式提取关键字段: 目标金额/已筹集/品类/国家/持续时间
    goal_usd = 15000.0
    pledged_usd = 0.0

    # 尝试提取"目标金额: XXXX"格式
    goal_match = re.search(r'目标金额[：:]\s*([\d,]+)', text)
    if goal_match:
        goal_usd = float(goal_match.group(1).replace(",", ""))
    else:
        # 优先找与"融"相关的金额（"想融 X 万"、"融资 X 万"、"目标 X 万"）
        funding_match = re.search(r'(?:融|融资|募资)[资]?\s*(\d[\d,]*(?:\.\d+)?)\s*(万|w|k|m|美元|usd|\$|元)', text_lower)
        if funding_match:
            val = float(funding_match.group(1).replace(",", ""))
            unit = funding_match.group(2)
            goal_usd = val * (10000 if unit in ("万", "w") else 1000 if unit in ("k",) else 1000000 if unit in ("m",) else 1)
        else:
            # 尝试提取通用金额（带单位）
            money_match = re.search(r'(\d[\d,]*(?:\.\d+)?)\s*(万|w|k|m|美元|usd|\$|元)', text_lower)
            if money_match:
                val = float(money_match.group(1).replace(",", ""))
                unit = money_match.group(2)
                if unit in ("万", "w"):
                    goal_usd = val * 10000
                elif unit in ("k",):
                    goal_usd = val * 1000
                elif unit in ("m",):
                    goal_usd = val * 1000000
                else:
                    goal_usd = val

    # 提取"已筹集: XXXX"或类似表述（投了/融了/已融/已筹）
    pledged_match = re.search(r'(?:已筹[集]?|已融|投了|friend[：:])\s*[：:]?\s*(\d[\d,]*(?:\.\d+)?)\s*(万|w|k|m|美元|usd|\$|元)', text_lower)
    if pledged_match:
        val = float(pledged_match.group(1).replace(",", ""))
        unit = pledged_match.group(2)
        pledged_usd = val * (10000 if unit in ("万", "w") else 1000 if unit in ("k",) else 1)

    # 推断品类
    main_category = "Technology"
    category_risk = 0.3
    for cat, risk in CATEGORY_RISK_MAP.items():
        if cat in text_lower:
            main_category = cat.title()
            category_risk = risk
            break

    # 推断国家（优先匹配完整国家名/城市名，再匹配 country code）
    country = "US"
    country_factor = 0.3
    city_country_map = {
        "北京": "cn", "上海": "cn", "深圳": "cn", "广州": "cn", "杭州": "cn",
        "纽约": "us", "san francisco": "us", "silicon valley": "us",
        "伦敦": "gb", "london": "gb", "柏林": "de", "berlin": "de",
        "东京": "jp", "tokyo": "jp",
    }
    for city, cc in city_country_map.items():
        if city in text_lower:
            country = cc.upper()
            country_factor = COUNTRY_FACTOR_MAP.get(cc, 0.5)
            break
    else:
        # 用单词边界匹配 country code，避免 "base"→"se" 这种误匹配
        for c, f in COUNTRY_FACTOR_MAP.items():
            if re.search(r'\b' + re.escape(c) + r'\b', text_lower):
                country = c.upper()
                country_factor = f
                break

    # 推断周期
    duration_days = 60
    dur_match = re.search(r'(\d+)\s*(天|day|days|周|week|月|month)', text_lower)
    if dur_match:
        val = int(dur_match.group(1))
        unit = dur_match.group(2)
        if unit in ("周", "week"):
            duration_days = val * 7
        elif unit in ("月", "month"):
            duration_days = val * 30
        else:
            duration_days = val

    goal_ratio = round(goal_usd / max(pledged_usd, 5000), 4) if pledged_usd > 0 else round(goal_usd / 5000, 4)
    time_penalty = round(max(0, (60 - duration_days) / 30.0), 4) if duration_days < 60 else 0.0

    # 用 M8 文本风险分析辅助
    try:
        _, analyze_text_risk = _import_m8()
        risk_bonus, _, _ = analyze_text_risk(text)
    except Exception:
        risk_bonus = 0.0

    # 提取支持者数量
    backers_count = 0
    backers_match = re.search(r'支持者[：:]\s*(\d+)', text)
    if backers_match:
        backers_count = int(backers_match.group(1))

    urgency_score = round(max(0, 1.0 - min(1.0, backers_count / max(500, 1))) * 0.3, 4)
    combined_risk = round(goal_ratio * 0.36 + time_penalty * 0.30 + category_risk * 0.19 + country_factor * 0.09 + urgency_score * 0.05 + risk_bonus, 4)

    return {
        "goal_usd": goal_usd,
        "pledged_usd": pledged_usd,
        "actual_funding_usd": pledged_usd,
        "backers_count": backers_count,
        "duration_days": duration_days,
        "main_category": main_category,
        "country": country,
        "country_factor": country_factor,
        "goal_ratio": goal_ratio,
        "time_penalty": time_penalty,
        "category_risk": category_risk,
        "urgency_score": 0.3,
        "combined_risk": combined_risk,
        "actual_funding_usd": goal_usd * 0.3,
        "planned_duration_days": duration_days,
    }


# ==========================================
# 核心分析流水线
# ==========================================

def run_analysis_pipeline(
    project_data: Dict[str, Any],
    user_query: str,
    api_key: str = "",
    enable_m9: bool = True,
    enable_m10: bool = True,
    enable_m12: bool = False,
    enable_m5: bool = False,
    enable_mas_blackboard: bool = True,
    enable_blender: bool = True,
    enable_m15: bool = False,
    user_id: str = "web_user",
) -> Dict[str, Any]:
    """
    完整分析流水线，串联所有模块
    返回统一 JSON 结果
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "project_data": project_data,
        "modules": {},
    }

    # ========== Step 0: 初始化 SharedBlackboard ==========
    try:
        from scripts.mas_blackboard.blackboard import SharedBlackboard
        from scripts.mas_blackboard.models import BlackboardState

        bb_state = BlackboardState(session_id=f"session_{user_id}")
        shared_bb = SharedBlackboard(bb_state)
        results["modules"]["M11_路由决策"] = {"status": "Blackboard 已初始化"}
    except Exception as e:
        shared_bb = None
        results["modules"]["M11_路由决策"] = {"error": f"Blackboard 初始化失败: {e}"}
        # BB 初始化失败不等同于 M11 失败，允许继续

    # ========== Step 1: M8 风险规则判定 ==========
    st.session_state.pipeline_step = "M8 风险规则判定"
    try:
        judge_project_risk_m8, analyze_text_risk = _import_m8()
        project_data_with_text = {**project_data, "user_input": user_query}
        risk_level, risk_reasons, intermediate = judge_project_risk_m8(project_data_with_text, verbose=False)
        text_risk_bonus, matched_critical, matched_high = analyze_text_risk(user_query)

        results["modules"]["M8_风险判定"] = {
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "intermediate": intermediate,
            "text_risk": {
                "risk_bonus": text_risk_bonus,
                "critical_keywords": matched_critical,
                "high_keywords": matched_high,
            },
        }

        # M8 写入 Blackboard
        if shared_bb is not None:
            try:
                shared_bb.write(
                    zone="m8_risk",
                    content={
                        "risk_level": risk_level,
                        "risk_reasons": risk_reasons,
                        "intermediate": intermediate,
                        "text_risk_bonus": text_risk_bonus,
                    },
                    tags=["risk", "m8"],
                    agent_id="m8_rule_engine",
                )
            except Exception:
                pass  # BB 写入失败不阻断流程
    except Exception as e:
        results["modules"]["M8_风险判定"] = {"error": str(e)}
        risk_level = "风险中等"
        risk_reasons = [f"M8执行异常: {str(e)}"]
        intermediate = {}

    # ========== Step 2: M11 小模型路由（替代原 M7 专家路由） ==========
    st.session_state.pipeline_step = "M11 小模型路由"
    route_result = {"selected_experts": [], "confidence": 0, "route_reason": "M11 未执行"}
    try:
        from scripts.m11.m11_core import M11Router, SmallModelNotConfigured, SmallModelLoadFailed

        m11_router = M11Router(blackboard=shared_bb)
        route_result = m11_router.route(
            user_input=user_query,
            risk_level=risk_level,
            intermediate=intermediate,
            project_data=project_data,
            user_id=user_id,
        )
        results["modules"]["M11_路由决策"] = route_result
    except (SmallModelNotConfigured, SmallModelLoadFailed) as e:
        error_msg = f"⚠️ M11 小模型不可用：{e}。请配置模型后再使用本系统。"
        st.error(error_msg)
        results["modules"]["M11_路由决策"] = {"error": error_msg}
        return results
    except Exception as e:
        error_msg = f"M11 路由失败: {e}"
        st.error(error_msg)
        results["modules"]["M11_路由决策"] = {"error": error_msg}
        return results
    except Exception as e:
        results["modules"]["M7_专家路由"] = {"error": str(e)}
        route_result = {"selected_experts": [], "confidence": 0, "route_reason": str(e)}

    # ========== Step 3: M7 LLM推理 + Blender（数据通过 Blackboard 自动读取） ==========
    st.session_state.pipeline_step = "M7 LLM推理"
    llm_outputs = []
    blender_result = {}
    if api_key and os.environ.get("DEEPSEEK_API_KEY", api_key):
        try:
            if enable_blender:
                _, run_with_blender = _import_m7_inference()
                blender_result = run_with_blender(
                    user_input=user_query,
                    user_id=user_id,
                    blackboard=shared_bb,
                    top_k=2,
                    model="deepseek-chat",
                    use_llm_fuser=True,
                )
                results["modules"]["M7_Blender融合"] = blender_result
            else:
                run_inference, _ = _import_m7_inference()
                llm_outputs = run_inference(
                    user_input=user_query,
                    user_id=user_id,
                    blackboard=shared_bb,
                    top_k=2,
                    model="deepseek-chat",
                )
                results["modules"]["M7_LLM推理"] = llm_outputs
        except Exception as e:
            results["modules"]["M7_LLM推理"] = {"error": str(e), "note": "DEEPSEEK_API_KEY可能无效"}
    else:
        results["modules"]["M7_LLM推理"] = {"note": "未设置DEEPSEEK_API_KEY，跳过LLM推理。M7路由结果仍可用。"}

    # ========== Step 4: M7 可视化 ==========
    try:
        save_m7_visualizations = _import_m7_visualization()
        if save_m7_visualizations:
            vis_dir = str(OUTPUT_DIR / "m7_visualization")
            visual_paths = save_m7_visualizations(route_result, vis_dir)
            results["modules"]["M7_可视化"] = visual_paths
    except Exception as e:
        results["modules"]["M7_可视化"] = {"note": f"可视化跳过: {str(e)}"}

    # ========== Step 5: M6 状态管理（记录当前分析状态） ==========
    st.session_state.pipeline_step = "M6 状态管理"
    try:
        EnvStateManager, EnvState, RuleLibrary, RouteDispatcher, _ = _import_m6()
        import logging as _logging
        _m6_logger = _logging.getLogger(f"M6.StateManager.web_pipeline_{id(project_data)}")
        _m6_logger.handlers.clear()
        _m6_logger.propagate = False
        try:
            _m6_fh = _logging.FileHandler(str(LOGS_DIR / "m6_web.log"), encoding="utf-8")
            _m6_fh.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
            _m6_logger.addHandler(_m6_fh)
        except Exception:
            pass
        state_manager = EnvStateManager(project_id="web_pipeline", logger=_m6_logger)
        init_state = EnvState(project_id="web_pipeline", observation=project_data)
        state_manager.on_reset(init_state.to_dict())
        step_state = EnvState(project_id="web_pipeline", action="analysis_completed", reward=1.0)
        state_manager.on_step(step_state.to_dict())
        results["modules"]["M6_状态管理"] = {
            "trace": state_manager.get_trace(),
            "latest_state": state_manager.latest_state,
        }
        _m6_logger.handlers.clear()
    except Exception as e:
        results["modules"]["M6_状态管理"] = {"error": str(e)}

    # ========== Step 6: M9 终极引擎 ==========
    st.session_state.pipeline_step = "M9 终极引擎"
    if enable_m9:
        try:
            M9UltimateRiskEngine = _import_m9()
            if api_key:
                os.environ["DEEPSEEK_API_KEY"] = api_key
            import logging as _logging
            _m9_logger = _logging.getLogger(f"M9_OOD_web_{id(project_data)}")
            _m9_logger.handlers.clear()
            _m9_logger.propagate = False
            try:
                _m9_fh = _logging.FileHandler(str(LOGS_DIR / "m9_web.log"), encoding="utf-8")
                _m9_fh.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
                _m9_logger.addHandler(_m9_fh)
            except Exception:
                pass
            engine = M9UltimateRiskEngine(api_key=api_key if api_key else None, verbose=False)
            # 替换内部 ood_logger 避免与 Streamlit 的 StreamHandler 冲突
            engine.ood_logger = _m9_logger
            m9_result = engine.run_full_decision(
                user_query=user_query,
                project_data=project_data,
                user_id=user_id,
                enable_ood_test=False,
                enable_perf_monitor=False,
                external_m8_result={
                    "risk_level": risk_level,
                    "risk_reasons": risk_reasons,
                    "feature_values": intermediate,
                },
                external_m7_result={
                    "selected_experts": [e.get("name", "") for e in route_result.get("selected_experts", [])],
                    "route_reason": route_result.get("route_reason", ""),
                    "confidence": route_result.get("confidence", 0.8),
                    "routing_scores": route_result.get("routing_scores", {}),
                    "normalized_risk_level": route_result.get("normalized_risk_level", ""),
                    "intent_analysis": route_result.get("intent_result", {}),
                },
            )
            results["modules"]["M9_终极引擎"] = m9_result
            _m9_logger.handlers.clear()
        except Exception as e:
            results["modules"]["M9_终极引擎"] = {"error": str(e)}
    else:
        results["modules"]["M9_终极引擎"] = {"note": "M9未启用"}

    # ========== Step 7: M10 性能监控 ==========
    st.session_state.pipeline_step = "M10 性能监控"
    if enable_m10:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            M10_PerformanceMonitor = _import_m10()
            monitor = M10_PerformanceMonitor(
                project_id="web_pipeline",
                output_dir=str(OUTPUT_DIR / "performance_reports"),
                log_dir=str(LOGS_DIR),
            )
            log_files = list(LOGS_DIR.glob("*.log")) if LOGS_DIR.exists() else []
            if log_files:
                log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                report = monitor.monitor_log_stream(str(log_files[0]))
                if report:
                    results["modules"]["M10_性能监控"] = {
                        "report_id": report.report_id,
                        "total_cost": report.total_cost,
                        "token_cost": report.total_token_cost,
                        "bottleneck_count": len(report.bottlenecks),
                    }
                else:
                    results["modules"]["M10_性能监控"] = {"status": "no_report_generated"}
            else:
                results["modules"]["M10_性能监控"] = {"status": "no_log_file"}
        except Exception as e:
            results["modules"]["M10_性能监控"] = {"error": str(e)}

    # ========== Step 8: M12 OOD测试 ==========
    st.session_state.pipeline_step = "M12 OOD测试"
    if enable_m12:
        try:
            OODScenarioGenerator, ResilienceEvaluator, OODConfig = _import_m12()
            import logging as _logging
            _m12_logger = _logging.getLogger(f"m12_web_{id(project_data)}")
            _m12_logger.handlers.clear()
            _m12_logger.propagate = False
            try:
                _m12_fh = _logging.FileHandler(str(LOGS_DIR / "m12_web.log"), encoding="utf-8")
                _m12_fh.setFormatter(_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
                _m12_logger.addHandler(_m12_fh)
            except Exception:
                pass
            config = OODConfig()
            generator = OODScenarioGenerator(config=config, logger=_m12_logger)
            evaluator = ResilienceEvaluator(config=config, logger=_m12_logger)
            scenario = generator.generate_scenario_from_project(project_data, difficulty="medium")
            results["modules"]["M12_OOD测试"] = {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "difficulty": scenario.difficulty_level,
                "black_swan_events": [
                    {"name": e.name, "type": e.event_type.value, "severity": e.severity}
                    for e in scenario.black_swan_events
                ],
                "expected_outcome": scenario.expected_outcome,
            }
            _m12_logger.handlers.clear()
        except Exception as e:
            results["modules"]["M12_OOD测试"] = {"error": str(e)}

    # ========== Step 9: MAS Blackboard 黑板架构 ==========
    st.session_state.pipeline_step = "MAS Blackboard 黑板架构"
    if enable_mas_blackboard:
        try:
            import logging as _logging
            _bb_loggers = ["mas_blackboard.workflow", "mas_blackboard.classifier",
                           "mas_blackboard.agents", "mas_blackboard.blackboard"]
            for _ln in _bb_loggers:
                _l = _logging.getLogger(_ln)
                _l.handlers.clear()
                _l.propagate = False
            # 调试：打印 env var 实际值
            sm_val = os.environ.get("USE_REAL_SMALL_MODEL", "NOT_SET")
            print(f"[M15-DEBUG] 即将创建 AdaptiveTieredWorkflow, USE_REAL_SMALL_MODEL={sm_val}")
            AdaptiveTieredWorkflow, _ = _import_mas_blackboard()
            workflow = AdaptiveTieredWorkflow(blackboard=shared_bb)
            bb_result = workflow.run(user_input=user_query)
            results["modules"]["MAS_黑板架构"] = {
                "routing_decision": bb_result.decision.model_dump() if hasattr(bb_result.decision, "model_dump") else str(bb_result.decision),
                "final_report": bb_result.final_report,
                "status": bb_result.blackboard_state.global_state.status if hasattr(bb_result.blackboard_state, "global_state") else "unknown",
            }
        except Exception as e:
            results["modules"]["MAS_黑板架构"] = {"error": str(e)}

    # ========== Step 10: M5 自动化测试（可选） ==========
    st.session_state.pipeline_step = "M5 自动化测试"
    if enable_m5:
        try:
            import logging as _logging
            _m5_logger = _logging.getLogger(f"M5_web_{id(project_data)}")
            _m5_logger.handlers.clear()
            _m5_logger.propagate = False
            M5AutomationTestEngine = _import_m5()
            cleaned_csv = OUTPUT_DIR / "kickstarter_cleaned.csv"
            engine = M5AutomationTestEngine(
                project_id="web_pipeline",
                output_dir=str(OUTPUT_DIR / "test_reports"),
                data_file=str(cleaned_csv) if cleaned_csv.exists() else None,
            )
            report = engine.run_full_pipeline(use_synthetic=True, parallel=True)
            results["modules"]["M5_自动化测试"] = report.to_dict()
        except Exception as e:
            results["modules"]["M5_自动化测试"] = {"error": str(e)}

    # ========== Step 11: M15 消融实验 ==========
    st.session_state.pipeline_step = "M15 消融实验"
    if enable_m15:
        try:
            run_ablation_experiments = _import_m15()
            m15_result = run_ablation_experiments(
                project_data=project_data,
                user_query=user_query,
                api_key=api_key,
                enable_blender=enable_blender,
                user_id=user_id,
                m8_result=results["modules"].get("M8_风险判定"),
                m7_result=results["modules"].get("M7_专家路由"),
            )
            results["modules"]["M15_消融实验"] = m15_result
        except Exception as e:
            results["modules"]["M15_消融实验"] = {"error": str(e)}

    # ========== 生成最终总结 ==========
    results["final_summary"] = _build_final_summary(results)

    return results


def _build_final_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """从各模块结果中提取关键信息，生成最终摘要"""
    summary = {
        "risk_level": "未知",
        "risk_reasons": [],
        "selected_experts": [],
        "route_reason": "",
        "confidence": 0.0,
        "actions": [],
        "alerts": [],
        "fused_summary": "",
    }

    m8 = results.get("modules", {}).get("M8_风险判定", {})
    if "risk_level" in m8:
        summary["risk_level"] = m8["risk_level"]
        summary["risk_reasons"] = m8.get("risk_reasons", [])

    m7_route = results.get("modules", {}).get("M7_专家路由", {})
    if "selected_experts" in m7_route:
        summary["selected_experts"] = [
            {"name": e.get("name", ""), "role": e.get("role", "")}
            for e in m7_route["selected_experts"]
        ]
        summary["route_reason"] = m7_route.get("route_reason", "")
        summary["confidence"] = m7_route.get("confidence", 0.0)

    # 从Blender融合结果提取 actions/alerts
    m7_blend = results.get("modules", {}).get("M7_Blender融合", {})
    fused = m7_blend.get("fused_result", {})
    if isinstance(fused, dict):
        summary["fused_summary"] = fused.get("fused_risk_summary", "")
        summary["actions"] = fused.get("fused_actions", [])
        summary["alerts"] = fused.get("fused_alerts", [])

    # 从M9提取
    m9 = results.get("modules", {}).get("M9_终极引擎", {})
    if isinstance(m9, dict) and m9.get("code") == 0:
        final = m9.get("final", {})
        if isinstance(final, dict):
            if not summary["fused_summary"] and final.get("fused_risk_summary"):
                summary["fused_summary"] = final["fused_risk_summary"]
            if not summary["actions"] and final.get("fused_actions"):
                summary["actions"] = final["fused_actions"]
            if not summary["alerts"] and final.get("fused_alerts"):
                summary["alerts"] = final["fused_alerts"]

    return summary


# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(page_title="项目评估与AI路由系统", page_icon="🧭", layout="wide")


def _init_session_state():
    """初始化 session state"""
    defaults = {
        "messages": [],
        "analysis_results": [],
        "current_result": None,
        "pipeline_step": "",
        "analysis_running": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_backend_status(api_key: str):
    """渲染后端状态面板"""
    st.divider()
    st.subheader("🔌 后端状态")

    # 1) 大模型 API (DeepSeek)
    if api_key and len(api_key) > 10:
        st.success("✅ DeepSeek API: 已配置")
    elif api_key:
        st.warning("⚠️ DeepSeek API: 密钥太短，可能无效")
    else:
        st.info("💡 DeepSeek API: 未配置，将使用规则模式")

    # 2) 小模型 (Qwen3-1.7B)
    use_real_model = os.environ.get("USE_REAL_SMALL_MODEL", "false").lower() == "true"

    # 检查依赖包
    deps_ok = False
    missing_deps = []
    for mod_name in ["torch", "transformers", "peft"]:
        try:
            __import__(mod_name)
        except ImportError:
            missing_deps.append(mod_name)
    deps_ok = len(missing_deps) == 0

    try:
        from model_asset_manager import inspect_small_model_assets
        sm_status = inspect_small_model_assets()
        files_ready = sm_status.get("ready", False)

        if use_real_model:
            if not deps_ok:
                st.error(f"❌ 小模型: 缺少依赖包 {', '.join(missing_deps)}，无法加载")
            elif files_ready:
                st.success("✅ 小模型 Qwen3-1.7B: 已启用（文件就绪 + 依赖完整）")
            else:
                st.warning("⚠️ 小模型 Qwen3-1.7B: 已启用但文件缺失（将回退规则引擎）")
        else:
            if files_ready and deps_ok:
                st.info("💡 小模型 Qwen3-1.7B: 文件就绪，但未启用。请勾选侧边栏「小模型推理模式」")
            elif files_ready and not deps_ok:
                st.warning(f"⚠️ 小模型文件就绪，但缺少依赖包: {', '.join(missing_deps)}")
            else:
                missing = sm_status.get("missing", [])
                deps_info = f" + 缺少依赖包: {', '.join(missing_deps)}" if not deps_ok else ""
                st.info(f"💡 小模型 Qwen3-1.7B: 未就绪（{len(missing)} 项文件缺失{deps_info}）")
    except Exception:
        st.info("💡 小模型检查: 模块不可用")


def _render_sidebar() -> Dict[str, Any]:
    """渲染侧边栏，返回配置"""
    with st.sidebar:
        st.header("⚙️ 分析配置")

        api_key = st.text_input(
            "DeepSeek API Key",
            value=os.environ.get("DEEPSEEK_API_KEY", ""),
            type="password",
            help="设置后可启用M7 LLM推理、M9终极引擎等模块。留空则使用规则引擎模式。",
        )

        st.divider()
        st.subheader("模块开关")

        enable_m9 = st.checkbox("M9 终极引擎", value=True, help="整合M7+M8+M10+M12的完整决策")
        enable_m10 = st.checkbox("M10 性能监控", value=False, help="监控Token消耗和瓶颈")
        enable_m12 = st.checkbox("M12 OOD高压测试", value=False, help="模拟黑天鹅事件")
        enable_m5 = st.checkbox("M5 自动化测试", value=False, help="运行批量测试套件")
        enable_mas_blackboard = st.checkbox("MAS 黑板架构", value=True, help="多Agent黑板协作路由")
        enable_blender = st.checkbox("M7 Blender融合", value=True, help="PairRanker+GenFuser多专家融合")
        enable_m15 = st.checkbox("M15 消融实验", value=False, help="启动消融实验：对比路由策略/LLM推理/组件完整度对结果的影响")

        st.divider()
        st.subheader("推理模式")

        use_small_model = st.checkbox(
            "小模型推理模式",
            value=os.environ.get("USE_REAL_SMALL_MODEL", "false").lower() == "true",
            help="勾选后尝试加载 Qwen3-1.7B + LoRA 进行复杂度分类；失败时自动回退规则引擎。",
        )
        os.environ["USE_REAL_SMALL_MODEL"] = "true" if use_small_model else "false"

        st.divider()
        st.subheader("数据集工具")

        if st.button("🔄 运行M2数据清洗", use_container_width=True):
            with st.spinner("正在运行M2数据清洗..."):
                import subprocess
                try:
                    m2_script = _import_m2()
                    result = subprocess.run(
                        [sys.executable, m2_script],
                        capture_output=True, text=True, timeout=600,
                        cwd=str(SCRIPTS_DIR),
                    )
                    if result.returncode == 0:
                        st.success("M2数据清洗完成！")
                    else:
                        st.error(f"M2清洗失败: {result.stderr[:500]}")
                except Exception as e:
                    st.error(f"M2执行异常: {str(e)}")

        if st.button("📊 运行M4特征工程", use_container_width=True):
            with st.spinner("正在运行M4特征工程..."):
                import subprocess
                try:
                    m4_script = _import_m4()
                    result = subprocess.run(
                        [sys.executable, m4_script],
                        capture_output=True, text=True, timeout=1800,
                        cwd=str(SCRIPTS_DIR),
                    )
                    if result.returncode == 0:
                        st.success("M4特征工程完成！")
                    else:
                        st.error(f"M4失败: {result.stderr[:500]}")
                except Exception as e:
                    st.error(f"M4执行异常: {str(e)}")

        st.divider()
        st.subheader("关于")
        st.caption("项目评估与AI路由系统 v2.0")
        st.caption("整合 M2→M4→M8→M7→M9→M10→M12→M15→M16 全模块")
        st.caption("支持文字输入和数据文件分析")

        # 后端状态检查
        _render_backend_status(api_key)

        # 调试：显示当前 env var 实际值
        debug_sm = os.environ.get("USE_REAL_SMALL_MODEL", "未设置")
        st.caption(f"DEBUG: USE_REAL_SMALL_MODEL={debug_sm}")

    return {
        "api_key": api_key,
        "enable_m9": enable_m9,
        "enable_m10": enable_m10,
        "enable_m12": enable_m12,
        "enable_m5": enable_m5,
        "enable_mas_blackboard": enable_mas_blackboard,
        "enable_blender": enable_blender,
        "enable_m15": enable_m15,
    }


def _render_text_input() -> Optional[str]:
    """渲染文字输入区域"""
    st.subheader("💬 文字描述输入")
    st.caption("输入项目描述，系统将通过M8文本分析+M7路由+M9决策进行完整分析")
    st.caption("支持格式: `GoalUSD:15000/Country:US/Category:Technology/DurationDays:60` 或自然语言")

    examples = [
        "GoalUSD:15000/Country:US/Category:Technology/DurationDays:60",
        "我们做跨境SaaS，增长看起来不错，但我担心知识产权、合规风险和现金流断裂",
        "项目做AI医疗诊断，融资目标500万，周期3个月，团队缺少技术骨干，现金流只够2个月",
        "这是一个食品类众筹项目，目标2万美元，位于中国，周期45天",
    ]

    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(f"示例{i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state.text_input_value = example

    default_text = st.session_state.get("text_input_value", "")
    user_input = st.text_area(
        "项目描述",
        value=default_text,
        height=100,
        placeholder="输入项目描述，例如：GoalUSD/Country/Category/DurationDays... 或自然语言描述项目情况",
        key="text_input_area",
    )

    return user_input.strip() if user_input else None


def _render_file_input() -> Optional[Tuple[pd.DataFrame, str]]:
    """渲染文件上传区域，返回 (DataFrame, filename) 或 None"""
    st.subheader("📁 数据文件输入")
    st.caption("上传CSV/Excel数据文件（与datasets中Kickstarter数据格式一致），逐行分析")

    uploaded_file = st.file_uploader(
        "选择数据文件",
        type=["csv", "xlsx", "xls"],
        help="支持与项目datasets中Kickstarter数据格式一致的CSV文件",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"已加载文件: {uploaded_file.name} ({len(df)} 行)")
            with st.expander("数据预览", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"列名: {list(df.columns)}")

            # 选择分析行
            max_rows = min(len(df), 100)
            row_range = st.slider(
                "选择分析行范围",
                min_value=1, max_value=max_rows,
                value=(1, min(5, max_rows)),
                key="row_range_slider",
            )

            return df, uploaded_file.name
        except Exception as e:
            st.error(f"文件加载失败: {str(e)}")
    return None


def _render_analysis_result(result: Dict[str, Any]):
    """渲染分析结果"""
    if not result:
        st.info("暂无分析结果，请先进行输入分析。")
        return

    summary = result.get("final_summary", {})

    # ===== 顶部关键指标 =====
    st.subheader("📋 分析摘要")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_level = summary.get("risk_level", "未知")
        risk_color = {
            "风险很高": "🔴", "风险较高": "🟠", "风险中等": "🟡", "风险较低": "🟢", "风险很低": "🟢"
        }.get(risk_level, "⚪")
        st.metric("风险等级", f"{risk_color} {risk_level}")
    with col2:
        confidence = summary.get("confidence", 0)
        st.metric("路由置信度", f"{confidence:.2%}" if confidence <= 1 else f"{confidence}")
    with col3:
        experts = summary.get("selected_experts", [])
        expert_names = [e.get("role", e.get("name", "")) for e in experts]
        st.metric("选中专家", ", ".join(expert_names) if expert_names else "无")
    with col4:
        actions_count = len(summary.get("actions", []))
        st.metric("建议动作数", actions_count)

    # ===== 风险原因 =====
    risk_reasons = summary.get("risk_reasons", [])
    if risk_reasons:
        with st.expander("⚠️ 风险原因", expanded=True):
            for i, reason in enumerate(risk_reasons, 1):
                st.markdown(f"**{i}.** {reason}")

    # ===== 路由理由 =====
    route_reason = summary.get("route_reason", "")
    if route_reason:
        with st.expander("🧭 路由理由", expanded=True):
            st.markdown(route_reason)

    # ===== 建议动作 =====
    actions = summary.get("actions", [])
    if actions:
        with st.expander("✅ 建议动作", expanded=True):
            for i, action in enumerate(actions, 1):
                if isinstance(action, dict):
                    priority = action.get("priority", "-")
                    title = action.get("title", "未命名")
                    owner = action.get("owner", "未指定")
                    due = action.get("due_days", "-")
                    impact = action.get("expected_impact", "")
                    st.markdown(f"**[{priority}] {title}**")
                    st.caption(f"负责人: {owner} | 期限: {due}天 | 影响: {impact}")
                else:
                    st.markdown(f"**{i}.** {action}")

    # ===== 风险提醒 =====
    alerts = summary.get("alerts", [])
    if alerts:
        with st.expander("🚨 风险提醒", expanded=True):
            for i, alert in enumerate(alerts, 1):
                st.markdown(f"**{i}.** {alert}")

    # ===== 融合摘要 =====
    fused = summary.get("fused_summary", "")
    if fused:
        with st.expander("📝 融合摘要", expanded=True):
            st.markdown(fused)

    # ===== 各模块详细结果 =====
    st.divider()
    st.subheader("🔧 各模块详细结果")

    modules = result.get("modules", {})
    module_tabs = st.tabs(list(modules.keys())) if modules else []

    for tab, (module_name, module_data) in zip(module_tabs, modules.items()):
        with tab:
            if isinstance(module_data, dict) and "error" in module_data:
                st.error(f"模块执行异常: {module_data['error']}")
            _render_module_detail(module_name, module_data)


def _render_module_detail(module_name: str, data: Any):
    """渲染单个模块的详细结果"""
    if isinstance(data, dict):
        # 特殊处理：M7路由结果中的selected_experts
        if "selected_experts" in data and isinstance(data["selected_experts"], list):
            for expert in data["selected_experts"]:
                if isinstance(expert, dict):
                    st.markdown(f"**{expert.get('role', '')}** ({expert.get('name', '')})")
                    st.caption(expert.get("system_prompt", "")[:200])

        # 特殊处理：M8 intermediate
        if "intermediate" in data and isinstance(data["intermediate"], dict):
            st.markdown("**核心特征值:**")
            for k, v in data["intermediate"].items():
                st.markdown(f"- `{k}`: {v}")

        # 特殊处理：text_risk
        if "text_risk" in data and isinstance(data["text_risk"], dict):
            tr = data["text_risk"]
            st.markdown(f"**文本风险加成:** {tr.get('risk_bonus', 0)}")
            if tr.get("critical_keywords"):
                st.markdown(f"**极高风险关键词:** {', '.join(tr['critical_keywords'])}")
            if tr.get("high_keywords"):
                st.markdown(f"**高风险关键词:** {', '.join(tr['high_keywords'][:10])}")

        # 特殊处理：M7 Blender融合
        if "fused_result" in data and isinstance(data["fused_result"], dict):
            fused = data["fused_result"]
            st.markdown(f"**融合方法:** {fused.get('fusion_method', 'unknown')}")
            st.markdown(f"**融合置信度:** {fused.get('fusion_confidence', 0)}")

        # 特殊处理：M15 消融实验
        if module_name == "M15_消融实验" and isinstance(data, dict):
            if "error" in data:
                st.error(f"M15 消融实验执行异常: {data['error']}")
            elif "summary_text" in data:
                st.markdown("### 📊 消融实验结果概览")
                st.markdown(data["summary_text"])

                reports = data.get("reports", {})
                if reports:
                    dim_tabs = st.tabs(list(reports.keys()))
                    dim_labels = {
                        "routing_strategy": "路由策略",
                        "llm_reasoning": "LLM 推理",
                        "component_integrity": "组件完整度",
                    }
                    for tab, (dim_key, report) in zip(dim_tabs, reports.items()):
                        with tab:
                            dim_cn = dim_labels.get(dim_key, dim_key)
                            st.subheader(f"🧪 {dim_cn}")

                            cs = report.get("comparison_summary", {})
                            findings = cs.get("key_finding", "")
                            if findings:
                                st.info(f"**关键发现:** {findings}")

                            comparisons = cs.get("comparisons", [])
                            if comparisons:
                                # 表格显示
                                table_data = []
                                for c in comparisons:
                                    row = {
                                        "条件": c.get("label", ""),
                                        "置信度": c.get("confidence", 0),
                                        "置信度变化": c.get("confidence_diff", 0),
                                        "耗时(ms)": c.get("execution_time_ms", 0),
                                        "专家数": c.get("expert_count", 0),
                                        "Action数": c.get("action_count", 0),
                                    }
                                    if "m8_risk_score" in c:
                                        row["风险分值"] = c.get("m8_risk_score", 0)
                                    table_data.append(row)
                                st.table(table_data)

                            # 对比详情
                            for c in comparisons:
                                label = c.get("label", "")
                                experts = c.get("selected_experts", [])
                                error = c.get("error", "")
                                with st.expander(f"📌 {label}", expanded=False):
                                    if experts:
                                        st.markdown(f"**选中专家:** {', '.join(experts)}")
                                    if error:
                                        st.error(f"执行错误: {error}")

        # 通用JSON展示
        with st.expander("原始JSON", expanded=False):
            st.json(_safe_serialize(data), expanded=False)
    else:
        st.write(data)


def _safe_serialize(obj: Any) -> Any:
    """安全序列化，处理非JSON类型"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return _safe_serialize(obj.model_dump())
    if hasattr(obj, "to_dict"):
        return _safe_serialize(obj.to_dict())
    return str(obj)


def _render_history():
    """渲染历史分析记录"""
    results = st.session_state.get("analysis_results", [])
    if not results:
        return

    st.divider()
    st.subheader("📜 分析历史")

    for i, result in enumerate(reversed(results)):
        idx = len(results) - i
        summary = result.get("final_summary", {})
        risk = summary.get("risk_level", "未知")
        ts = result.get("timestamp", "未知时间")
        query = result.get("user_query", "")[:80]

        with st.expander(f"#{idx} [{risk}] {ts[:19]} - {query}", expanded=False):
            if st.button(f"查看详情 #{idx}", key=f"view_history_{idx}"):
                st.session_state.current_result = result
                st.rerun()


def main():
    """主界面"""
    _init_session_state()
    config = _render_sidebar()

    st.title("🧭 项目评估与AI路由系统")
    st.caption("整合 M2→M4→M8→M7→M9→M10→M12→M16 + MAS黑板架构 | 支持文字输入和数据文件分析 | 输出完整JSON")

    # ===== 输入模式选择 =====
    input_mode = st.radio(
        "选择输入方式",
        options=["💬 文字输入", "📁 数据文件上传", "🔄 两者同时"],
        horizontal=True,
        key="input_mode",
    )

    text_input = None
    file_data = None

    if input_mode in ["💬 文字输入", "🔄 两者同时"]:
        text_input = _render_text_input()

    if input_mode in ["📁 数据文件上传", "🔄 两者同时"]:
        file_data = _render_file_input()

    # ===== 分析按钮 =====
    st.divider()

    col_left, col_right = st.columns([1, 3])

    with col_left:
        analyze_button = st.button(
            "🚀 开始分析",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("analysis_running", False),
        )

    with col_right:
        if st.session_state.get("analysis_running"):
            st.info(f"⏳ 正在执行: {st.session_state.get('pipeline_step', '...')}")

    # ===== 执行分析 =====
    if analyze_button:
        if not text_input and not file_data:
            st.warning("请先输入项目描述或上传数据文件")
        else:
            st.session_state.analysis_running = True

            # 设置API Key
            if config["api_key"]:
                os.environ["DEEPSEEK_API_KEY"] = config["api_key"]

            progress = st.progress(0, text="准备分析...")

            try:
                all_results = []

                # 处理文字输入
                if text_input:
                    progress.progress(10, text="解析文字输入...")
                    project_data, user_query = text_to_project_data(text_input)

                    progress.progress(20, text="执行分析流水线...")
                    result = run_analysis_pipeline(
                        project_data=project_data,
                        user_query=user_query,
                        api_key=config["api_key"],
                        enable_m9=config["enable_m9"],
                        enable_m10=config["enable_m10"],
                        enable_m12=config["enable_m12"],
                        enable_m5=config["enable_m5"],
                        enable_mas_blackboard=config["enable_mas_blackboard"],
                        enable_blender=config["enable_blender"],
                        enable_m15=config["enable_m15"],
                    )
                    all_results.append(result)

                # 处理文件输入
                if file_data:
                    df, filename = file_data
                    row_start, row_end = st.session_state.get("row_range_slider", (1, 5))

                    progress.progress(30, text=f"分析数据文件: {filename}")

                    for idx in range(row_start - 1, min(row_end, len(df))):
                        progress.progress(
                            30 + int(60 * (idx - row_start + 1) / max(1, row_end - row_start)),
                            text=f"分析第 {idx + 1} 行...",
                        )
                        row = df.iloc[idx]
                        project_data = row_to_project_data(row)
                        user_query = f"分析项目: {project_data.get('project_name', 'Unknown')}, " \
                                     f"品类={project_data.get('main_category', '')}, " \
                                     f"目标=${project_data.get('goal_usd', 0):,.0f}, " \
                                     f"国家={project_data.get('country', '')}"

                        result = run_analysis_pipeline(
                            project_data=project_data,
                            user_query=user_query,
                            api_key=config["api_key"],
                            enable_m9=config["enable_m9"],
                            enable_m10=config["enable_m10"],
                            enable_m12=config["enable_m12"],
                            enable_m5=False,  # 批量不跑M5
                            enable_mas_blackboard=config["enable_mas_blackboard"],
                            enable_blender=config["enable_blender"],
                            enable_m15=config["enable_m15"],
                            user_id=f"file_row_{idx}",
                        )
                        all_results.append(result)

                # 保存结果
                if all_results:
                    st.session_state.current_result = all_results[-1]
                    st.session_state.analysis_results.extend(all_results)

                    # 保存JSON到文件
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    json_path = OUTPUT_DIR / f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(_safe_serialize(all_results), f, ensure_ascii=False, indent=2)

                    progress.progress(100, text="分析完成！")

            except Exception as e:
                st.error(f"分析执行异常: {str(e)}")
                with st.expander("异常详情"):
                    st.code(traceback.format_exc())
            finally:
                st.session_state.analysis_running = False

    # ===== 展示结果 =====
    st.divider()

    if st.session_state.current_result:
        _render_analysis_result(st.session_state.current_result)

        # 下载JSON按钮
        json_str = json.dumps(_safe_serialize(st.session_state.current_result), ensure_ascii=False, indent=2)
        st.download_button(
            "📥 下载完整JSON结果",
            data=json_str,
            file_name=f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ===== 历史记录 =====
    _render_history()


if __name__ == "__main__":
    main()
