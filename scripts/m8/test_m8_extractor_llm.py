import json
from pathlib import Path
import sys
import pytest
import os
import itertools

# ensure scripts dir on path
HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from m8_text_extractor import extract_via_llm, DeepseekClient, SimpleLLMClient

CORE_FIELDS = ["goal_ratio", "time_penalty", "category_risk", "country_factor", "urgency_score", "combined_risk"]

# sample values for present fields
SAMPLE_VALUES = {
    "goal_ratio": 0.5,
    "time_penalty": 1.2,
    "category_risk": 0.2,
    "country_factor": 0.31987929,
    "urgency_score": 0.1,
    "combined_risk": 0.6,
}


def make_clear_responder(missing):
    def responder(prompt):
        obj = {}
        for k in CORE_FIELDS:
            if k not in missing:
                obj[k] = SAMPLE_VALUES[k]
        obj["evidence"] = {"prompt": prompt, "raw_llm_response": "clear", "method": "llm"}
        obj["confidence"] = 0.9
        return json.dumps(obj, ensure_ascii=False)
    return responder


def make_unclear_responder(missing):
    # return keys present but missing ones set to null
    def responder(prompt):
        obj = {}
        for k in CORE_FIELDS:
            obj[k] = None if k in missing else SAMPLE_VALUES[k]
        obj["evidence"] = {"prompt": prompt, "raw_llm_response": "unclear", "method": "llm"}
        obj["confidence"] = 0.1
        return json.dumps(obj, ensure_ascii=False)
    return responder


def run_case(missing, responder_factory, proj):
    responder = responder_factory(missing)
    client = SimpleLLMClient(responder=responder)
    pd = extract_via_llm(proj, llm_client=client)
    return pd


def check_expected(pd, missing):
    # if more than 3 missing -> extractor should mark insufficient (threshold per extractor)
    if len(missing) > 3:
        assert pd.get("_insufficient_data") is True or pd.get("evidence", {}).get("raw_llm_response", "").find("insufficient_data") != -1
    else:
        # should have filled_from_defaults in evidence when any missing
        if len(missing) > 0:
            ev = pd.get("evidence", {})
            assert isinstance(ev, dict)
            filled = ev.get("filled_from_defaults")
            assert isinstance(filled, list)
            # filled set should contain at least the missing items
            for m in missing:
                assert m in filled
        # check values for filled fields equal to sample or fill values (extractor uses its own fill values)
        for k in CORE_FIELDS:
            assert k in pd


def test_user_scenarios():
    """Run the user-specified scenarios (clear vs fuzzy variants) and assert behavior."""
    scenarios = []

    # Scenario 1: all fields present
    scenarios.append({
        "name": "all_present",
        "missing": [],
        "proj_clear": {"user_input": "我有一个美国艺术品类项目，项目启动15天，融资进度刚好一半，项目目标金额42000美元。", "country": "US", "main_category": "Art", "duration_days": 15, "goal_usd": 42000},
        "proj_fuzzy": {"user_input": "我在美国做了一个艺术类的创业项目，做了半个月，资金融了一半，目标金额差不多四万美元", "country": "US", "main_category": "Art"},
    })

    # Scenario 2: single missing fields (6 cases)
    scenarios += [
        {"name": "miss_goal_ratio", "missing": ["goal_ratio"], "proj_clear": {"user_input": "我有一个加拿大食品品类项目，启动15天，融资了50%。", "country": "CA", "main_category": "Food", "duration_days": 15, "goal_usd": 42000}, "proj_fuzzy": {"user_input": "我在加拿大做食品项目，做了半个月，融资一半。", "country": "CA", "main_category": "Food"}},
        {"name": "miss_time_penalty", "missing": ["time_penalty"], "proj_clear": {"user_input": "我有一个英国设计品类项目，融资50%，项目目标金额42000美元。", "country": "GB", "main_category": "Design", "goal_usd": 42000}, "proj_fuzzy": {"user_input": "我在英国做设计项目，融资一半，目标金额差不多四万美元。", "country": "GB", "main_category": "Design"}},
        {"name": "miss_category_risk", "missing": ["category_risk"], "proj_clear": {"user_input": "我有一个澳大利亚项目，启动15天，融资50%，项目时长15天，项目目标金额42000美元。", "country": "AU", "duration_days": 15, "goal_usd": 42000}, "proj_fuzzy": {"user_input": "我在澳大利亚项目，做了半个月，融资一半，目标金额差不多四万美元。", "country": "AU"}},
        {"name": "miss_combined_risk", "missing": ["combined_risk"], "proj_clear": {"user_input": "我有一个美国科技品类项目，启动15天，融资50%，项目时长15天。", "country": "US", "main_category": "Technology", "duration_days": 15}, "proj_fuzzy": {"user_input": "我在美国做科技项目，做了半个月，融资一半。", "country": "US", "main_category": "Technology"}},
        {"name": "miss_country_factor", "missing": ["country_factor"], "proj_clear": {"user_input": "我有一个音乐品类项目，启动15天，融资50%，项目时长15天，项目目标金额42000美元。", "main_category": "Music", "duration_days": 15, "goal_usd": 42000}, "proj_fuzzy": {"user_input": "我在做音乐项目，做了半个月，融资一半，目标金额差不多四万美元。"}},
        {"name": "miss_urgency_score", "missing": ["urgency_score"], "proj_clear": {"user_input": "我有一个法国时尚品类项目，融资50%，目标金额42000美元。", "country": "FR", "main_category": "Fashion", "goal_usd": 42000}, "proj_fuzzy": {"user_input": "我在法国做时尚项目，融资一半，目标金额差不多四万美元。", "country": "FR", "main_category": "Fashion"}},
    ]

    # Scenario 3: two missing combinations (3 typical)
    scenarios += [
        {"name": "miss_goal_time", "missing": ["goal_ratio", "time_penalty"], "proj_clear": {"user_input": "我有一个日本游戏品类项目，融资50%。", "country": "JP", "main_category": "Games"}, "proj_fuzzy": {"user_input": "我在日本做游戏项目，做了半个月，融资一半。", "country": "JP", "main_category": "Games"}},
        {"name": "miss_category_country", "missing": ["category_risk", "country_factor"], "proj_clear": {"user_input": "我有一个项目，启动15天，融资50%，目标金额42000美元。", "duration_days": 15, "goal_usd": 42000}, "proj_fuzzy": {"user_input": "我在项目，做了半个月，融资一半，目标金额差不多四万美元。"}},
        {"name": "miss_combined_urgency", "missing": ["combined_risk", "urgency_score"], "proj_clear": {"user_input": "我有一个西班牙戏剧品类项目。", "country": "ES", "main_category": "Theater"}, "proj_fuzzy": {"user_input": "我在西班牙做戏剧项目。", "country": "ES", "main_category": "Theater"}},
    ]

    # Scenario 4: three missing combinations
    scenarios += [
        {"name": "miss_goal_time_combined", "missing": ["goal_ratio", "time_penalty", "combined_risk"], "proj_clear": {"user_input": "我有一个新西兰手工品类项目，融资50%。", "country": "NZ", "main_category": "Crafts"}, "proj_fuzzy": {"user_input": "我在新西兰做手工项目，融资一半。", "country": "NZ", "main_category": "Crafts"}},
        {"name": "miss_category_country_urgency", "missing": ["category_risk", "country_factor", "urgency_score"], "proj_clear": {"user_input": "我有一个项目，融资50%，项目时长15天，目标金额42000美元。", "duration_days": 15, "goal_usd": 42000}, "proj_fuzzy": {"user_input": "我项目，做了半个月，融资一半，目标金额差不多四万美元。"}},
    ]

    # Scenario 5: four missing (-> insufficient per threshold)
    scenarios += [
        {"name": "only_goal_ratio", "missing": ["time_penalty", "category_risk", "country_factor", "urgency_score"], "proj_clear": {"user_input": "我有一个项目，融资50%，目标金额42000美元。", "goal_usd": 42000}, "proj_fuzzy": {"user_input": "我项目，融资一半，目标金额差不多四万美元。"}},
        {"name": "only_country_factor", "missing": ["goal_ratio", "time_penalty", "category_risk", "urgency_score"], "proj_clear": {"user_input": "我有一个卢森堡项目。", "country": "LU"}, "proj_fuzzy": {"user_input": "我在卢森堡做项目。", "country": "LU"}},
    ]

    # Scenario 6: five missing (insufficient)
    scenarios += [
        {"name": "only_urgency", "missing": ["goal_ratio", "time_penalty", "category_risk", "country_factor", "combined_risk"], "proj_clear": {"user_input": "我有一个项目，启动15天。", "duration_days": 15}, "proj_fuzzy": {"user_input": "我项目，做了半个月。"}},
        {"name": "only_category_risk", "missing": ["goal_ratio", "time_penalty", "country_factor", "urgency_score", "combined_risk"], "proj_clear": {"user_input": "我有一个游戏品类项目。", "main_category": "Games"}, "proj_fuzzy": {"user_input": "我在做游戏项目。", "main_category": "Games"}},
    ]

    # Scenario 7: all missing
    scenarios.append({"name": "all_missing", "missing": CORE_FIELDS.copy(), "proj_clear": {"user_input": "资金断链，资产被冻结。"}, "proj_fuzzy": {"user_input": "资金断链，资产被冻结。"}})

    for s in scenarios:
        # clear
        pd_clear = run_case(s["missing"], make_clear_responder, s["proj_clear"])
        check_expected(pd_clear, s["missing"]) 

        # fuzzy
        pd_fuzzy = run_case(s["missing"], make_unclear_responder, s["proj_fuzzy"])
        check_expected(pd_fuzzy, s["missing"])


# keep existing live tests (they will still be skipped if DEEPSEEK_API_KEY not set)

@pytest.mark.skipif('DEEPSEEK_API_KEY' not in os.environ, reason='DEEPSEEK_API_KEY not set; live integration test skipped')
def test_deepseek_live_detailed():
    """Live integration: call Deepseek with detailed input. Requires DEEPSEEK_API_KEY env var."""
    api_key = os.environ['DEEPSEEK_API_KEY']
    api_url = os.environ.get('DEEPSEEK_API_URL', None)
    client = DeepseekClient(api_key=api_key, api_url=api_url) if api_url else DeepseekClient(api_key=api_key)

    proj = {
        "user_input": "我们计划发起科技类众筹，目标15000美元，周期60天，已筹3100美元，在美国。",
        "main_category": "Technology",
        "goal_usd": 15000
    }
    pd = extract_via_llm(proj, llm_client=client)
    print("[LIVE] detailed - extracted:", json.dumps(pd, ensure_ascii=False))
    # basic assertions
    assert isinstance(pd, dict)
    for k in ["goal_ratio", "time_penalty", "combined_risk", "evidence"]:
        assert k in pd
    assert pd["evidence"]["method"] == "llm"


@pytest.mark.skipif('DEEPSEEK_API_KEY' not in os.environ, reason='DEEPSEEK_API_KEY not set; live integration test skipped')
def test_deepseek_live_highrisk():
    """Live integration: call Deepseek with a high-risk-sounding input."""
    api_key = os.environ['DEEPSEEK_API_KEY']
    api_url = os.environ.get('DEEPSEEK_API_URL', None)
    client = DeepseekClient(api_key=api_key, api_url=api_url) if api_url else DeepseekClient(api_key=api_key)

    proj = {"user_input": "项目资金紧张，进度滞后，账户收款出现问题，很难按预期完成。"}
    pd = extract_via_llm(proj, llm_client=client)
    print("[LIVE] highrisk - extracted:", json.dumps(pd, ensure_ascii=False))
    assert isinstance(pd, dict)
    assert "evidence" in pd
    assert pd["evidence"]["method"] == "llm"


@pytest.mark.skipif('DEEPSEEK_API_KEY' not in os.environ, reason='DEEPSEEK_API_KEY not set; live integration test skipped')
def test_deepseek_live_varied_inputs():
    """Run 4 varied live scenarios through Deepseek and print/examine outputs."""
    api_key = os.environ['DEEPSEEK_API_KEY']
    api_url = os.environ.get('DEEPSEEK_API_URL', None)
    client = DeepseekClient(api_key=api_key, api_url=api_url) if api_url else DeepseekClient(api_key=api_key)

    cases = [
        {
            "name": "detailed",
            "proj": {
                "user_input": "我们计划发起科技类众筹，目标15000美元，周期60天，已筹3100美元，在美国。",
                "main_category": "Technology",
                "goal_usd": 15000
            }
        },
        {
            "name": "minimal",
            "proj": {"user_input": "项目正在准备中，一切按计划进行。"}
        },
        {
            "name": "implicit",
            "proj": {"user_input": "我们的筹款目标比行业常见水平高一些，项目持续时间会比较长，目前推进速度一般。"}
        },
        {
            "name": "high_risk",
            "proj": {"user_input": "项目资金紧张，进度滞后，账户收款出现问题，很难按预期完成。"}
        },
    ]

    for c in cases:
        pd = extract_via_llm(c["proj"], llm_client=client)
        print(f"[LIVE VARIED] {c['name']} - extracted:", json.dumps(pd, ensure_ascii=False))
        assert isinstance(pd, dict)
        # evidence must exist for audit
        assert "evidence" in pd and isinstance(pd["evidence"], dict)
        # combined_risk may be None for minimal case, but ensure key is present
        assert "combined_risk" in pd


if __name__ == '__main__':
    pytest.main([str(Path(__file__))])
