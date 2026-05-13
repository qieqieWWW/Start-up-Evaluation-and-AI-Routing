import json
import math
import os
import requests
from typing import Dict, Any, Optional

from m8_rule_adapter import DEFAULT_VALUES


DEFAULT_CLIP = {
    "goal_ratio": (0.0, 50.0),
    "duration_days": (1.0, 180.0),
    "category_risk": (0.0, 1.0),
    "country_factor": (0.0, 1.0),
    "urgency_score": (0.0, 1.0),
    "combined_risk": (0.0, None),
}

CATEGORY_RISK_RATE = {
    'Art': 0.16809196691434178,
    'Comics': 0.07819502395302118,
    'Crafts': 0.5431529411764706,
    'Dance': 0.12048453385247675,
    'Design': 0.2067414072872105,
    'Fashion': 0.31211692597831214,
    'Film & Video': 0.21642284367545195,
    'Food': 0.5278857262892553,
    'Games': 0.17434334154591388,
    'Journalism': 0.6559679037111334,
    'Music': 0.14314443015857317,
    'OTHER': 0.13010007698229406,
    'Photography': 0.3409767718880286,
    'Publishing': 0.2390272373540856,
    'Technology': 0.35256436411114284,
    'Theater': 0.19440175631174533,
}


def _clip(val: Optional[float], lo: Optional[float], hi: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _build_prompt(user_input: str, context: Dict[str, Any]) -> str:
    # Chinese prompt per spec
    prompt = (
        "请仅基于下面的用户输入和可选上下文抽取并计算所需的风险特征。\n"
        "只输出一个单行的 JSON 对象，且严格遵守字段名与类型：\n"
        "{\n  \"goal_ratio\": number|null,\n  \"time_penalty\": number|null,\n  \"category_risk\": number|null,\n  \"country_factor\": number|null,\n  \"urgency_score\": number|null,\n  \"combined_risk\": number|null,\n  \"evidence\": {\"prompt\": string, \"raw_llm_response\": string, \"method\": \"llm\"},\n  \"confidence\": number|null\n}\n"
        "计算说明：\n"
        "- goal_ratio = project_goal_usd / category_median_goal_usd，并裁剪到 [0, 50]；若不知道 category_median_goal_usd，请合理估计并在 evidence 中说明。\n"
        "- time_penalty = exp(duration_days / 30) - 1（duration_days 需限定在 1..180 范围内）；若不知道 duration_days，请估计并在 evidence 中说明。\n"
        "- category_risk = 该品类内高风险项目比例的估计值（0..1）。\n"
        "- country_factor = 该国家内高风险项目比例的估计值（0..1）。\n"
        "- urgency_score = 若 duration_days <= 7 则为 1.0，否则为 7.0 / duration_days。\n"
        "- combined_risk = goal_ratio * time_penalty。\n"
        "若必须猜测某一字段，请将该字段设为 null 并在 evidence 中说明原因。\n"
    )

    prompt = prompt + f"用户输入（user_input）：\"{user_input}\"\n"
    if context:
        prompt = prompt + f"已知上下文（可选）： {json.dumps(context, ensure_ascii=False)}\n"
    prompt = prompt + "请严格只输出 JSON，不要包含额外文字或注释。"
    return prompt


class SimpleLLMClient:
    """示例接口包装类，实际集成时替换为真实 client 实现。"""

    def __init__(self, responder=None):
        # responder: Callable[[str], str]
        self.responder = responder

    def generate(self, prompt: str) -> str:
        if self.responder:
            return self.responder(prompt)
        # default safe fallback
        return json.dumps({
            "goal_ratio": None,
            "time_penalty": None,
            "category_risk": None,
            "country_factor": None,
            "urgency_score": None,
            "combined_risk": None,
            "evidence": {"prompt": prompt, "raw_llm_response": "", "method": "llm"},
            "confidence": None,
        }, ensure_ascii=False)


class DeepseekClient:
    """Minimal Deepseek API client implementing generate(prompt)->str.

    Adjust api_url and payload keys per your Deepseek documentation.
    """
    def __init__(self, api_key: str, api_url: str = None, model: Optional[str] = None, timeout: int = 15):
        self.api_key = api_key
        self.api_url = api_url or os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Chat-style payload required by Deepseek: messages list
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            # include other model kwargs if needed by Deepseek
            "model": self.model,
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            # try to parse body
            try:
                data = resp.json()
            except Exception:
                data = resp.text

            if resp.status_code >= 400:
                return json.dumps({"status": resp.status_code, "body": data}, ensure_ascii=False)

            # Try common response shapes
            if isinstance(data, dict):
                # Chat completions: choices -> choice -> message -> content
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    if isinstance(choice, dict):
                        # OpenAI-like chat: choice.message.content
                        msg = choice.get("message") or choice.get("delta")
                        if isinstance(msg, dict) and msg.get("content"):
                            return msg.get("content")
                        # fallback to text field
                        return choice.get("text") or json.dumps(data, ensure_ascii=False)
                # legacy top-level text/result
                if "text" in data:
                    return data["text"]
                if "result" in data:
                    return data["result"]

            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            # return error as json-string to be captured as raw response
            return json.dumps({"error": str(e)})


def extract_via_llm(project_data: Dict[str, Any], llm_client: Optional[Any] = None, strict: bool = True) -> Dict[str, Any]:
    """Call LLM to extract and compute final features, validate and clip results.

    Returns a new project_data dict enriched with the six features plus evidence/confidence.
    """
    pd = {**DEFAULT_VALUES, **(project_data or {})}
    user_input = pd.get("user_input", "") or ""
    context = {
        "main_category": pd.get("main_category"),
        "country": pd.get("country"),
        "goal_usd": pd.get("goal_usd"),
    }

    if llm_client is None:
        llm_client = SimpleLLMClient()

    prompt = _build_prompt(user_input, context)
    # robust call: retry with stricter instruction if initial response not parseable
    raw_attempts = []
    parsed = None
    max_attempts = 3
    strict_suffix = "\n请严格只输出单行合法 JSON 对象，不要任何注释或额外文字。若无法输出 JSON，请输出 {\"error\":\"cannot_output_json\"}."
    for attempt in range(max_attempts):
        try:
            call_prompt = prompt if attempt == 0 else prompt + strict_suffix
            raw = llm_client.generate(call_prompt)
        except Exception as e:
            raw = json.dumps({"error": str(e)})
        raw_attempts.append(raw)

        # try direct json parse
        try:
            candidate = json.loads(raw)
            if isinstance(candidate, dict):
                parsed = candidate
                break
        except Exception:
            # try to extract json substring
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                candidate = raw[start:end]
                candidate_parsed = json.loads(candidate)
                if isinstance(candidate_parsed, dict):
                    parsed = candidate_parsed
                    break
            except Exception:
                parsed = None

    # if still not parsed, aggregate raw attempts for evidence and let existing non-dict handler proceed
    if not isinstance(parsed, dict):
        # combine attempts into a single raw string to aid debugging/auditing
        if len(raw_attempts) == 1:
            raw = raw_attempts[0]
        else:
            raw = json.dumps({"attempts": raw_attempts}, ensure_ascii=False)

    if not isinstance(parsed, dict):
        # parsing failed
        pd.update({
            "goal_ratio": pd.get("goal_ratio", DEFAULT_VALUES.get("goal_ratio")),
            "time_penalty": pd.get("time_penalty", DEFAULT_VALUES.get("time_penalty")),
            "category_risk": pd.get("category_risk", DEFAULT_VALUES.get("category_risk")),
            "country_factor": pd.get("country_factor", DEFAULT_VALUES.get("country_factor")),
            "urgency_score": pd.get("urgency_score", DEFAULT_VALUES.get("urgency_score")),
            "combined_risk": pd.get("combined_risk", DEFAULT_VALUES.get("combined_risk")),
            "evidence": {"prompt": prompt, "raw_llm_response": raw, "method": "llm"},
            "confidence": None,
        })
        return pd

    # extract fields safely
    goal_ratio = _clip(parsed.get("goal_ratio"), *DEFAULT_CLIP["goal_ratio"])
    time_penalty = parsed.get("time_penalty")
    if time_penalty is not None:
        time_penalty = _clip(time_penalty, 0.0, None)
    category_risk = _clip(parsed.get("category_risk"), *DEFAULT_CLIP["category_risk"])
    country_factor = _clip(parsed.get("country_factor"), *DEFAULT_CLIP["country_factor"])
    urgency_score = _clip(parsed.get("urgency_score"), *DEFAULT_CLIP["urgency_score"])
    combined_risk = parsed.get("combined_risk")
    if combined_risk is not None:
        combined_risk = _clip(combined_risk, 0.0, None)

    confidence = parsed.get("confidence")

    # capture parsed evidence raw if provided
    parsed_evidence = parsed.get("evidence", {}) if isinstance(parsed.get("evidence"), dict) else {}
    parsed_raw_llm = parsed_evidence.get("raw_llm_response", raw)

    # if time_penalty missing but duration available in project_data, compute
    if time_penalty is None:
        # try to infer duration from project_data
        dur = pd.get("duration_days") or pd.get("planned_duration_days")
        try:
            dur = float(dur)
        except Exception:
            dur = None
        if dur is not None:
            dur = _clip(dur, *DEFAULT_CLIP["duration_days"])
            time_penalty = math.exp(dur / 30.0) - 1.0
        else:
            time_penalty = None

    # if combined_risk missing but goal_ratio and time_penalty present compute
    if combined_risk is None and goal_ratio is not None and time_penalty is not None:
        combined_risk = goal_ratio * time_penalty

    # consistency check: if both provided, enforce computed value
    if goal_ratio is not None and time_penalty is not None and combined_risk is not None:
        recomputed = goal_ratio * time_penalty
        if recomputed > 0:
            diff = abs(recomputed - combined_risk) / recomputed
            if diff > 0.10:
                # take recomputed as truth, but record discrepancy
                parsed_evidence_local = dict(parsed_evidence)
                parsed_evidence_local["discrepancy"] = {"reported": combined_risk, "recomputed": recomputed}
                combined_risk = recomputed
                evidence = {"prompt": prompt, "raw_llm_response": parsed_raw_llm, "method": "llm", **parsed_evidence_local}
            else:
                evidence = {"prompt": prompt, "raw_llm_response": parsed_raw_llm, "method": "llm"}
        else:
            evidence = {"prompt": prompt, "raw_llm_response": parsed_raw_llm, "method": "llm"}
    else:
        evidence = {"prompt": prompt, "raw_llm_response": parsed_raw_llm, "method": "llm"}

    # final clipping & defaults fallback
    if goal_ratio is None:
        goal_ratio = pd.get("goal_ratio", DEFAULT_VALUES.get("goal_ratio"))
    if time_penalty is None:
        time_penalty = pd.get("time_penalty", DEFAULT_VALUES.get("time_penalty"))
    if category_risk is None:
        category_risk = pd.get("category_risk", DEFAULT_VALUES.get("category_risk"))
    if country_factor is None:
        country_factor = pd.get("country_factor", DEFAULT_VALUES.get("country_factor"))
    if urgency_score is None:
        urgency_score = pd.get("urgency_score", DEFAULT_VALUES.get("urgency_score"))
    if combined_risk is None:
        combined_risk = pd.get("combined_risk", DEFAULT_VALUES.get("combined_risk"))

    # write back into pd
    pd.update({
        "goal_ratio": round(float(goal_ratio), 6) if goal_ratio is not None else None,
        "time_penalty": round(float(time_penalty), 6) if time_penalty is not None else None,
        "category_risk": round(float(category_risk), 6) if category_risk is not None else None,
        "country_factor": round(float(country_factor), 6) if country_factor is not None else None,
        "urgency_score": round(float(urgency_score), 6) if urgency_score is not None else None,
        "combined_risk": round(float(combined_risk), 6) if combined_risk is not None else None,
        "evidence": evidence,
        "confidence": float(confidence) if confidence is not None else None,
    })

    # Try to forward the enriched project_data to m9_rule_adapter if available
    try:
        from m9_rule_adapter import receive_m8_output  # type: ignore
        try:
            receive_m8_output(pd)
        except Exception:
            # don't block main flow if m9 processing fails
            pass
    except Exception:
        # m9 not available; ignore
        pass

    # Also run local M8 rule evaluation and attach its result for downstream consumers
    try:
        from m8_rule_adapter import judge_project_risk_m8  # type: ignore
        try:
            res = judge_project_risk_m8(pd, verbose=False)
            pd["_m8_evaluation"] = {"risk_level": res[0], "reasons": res[1], "intermediate": res[2]}
        except Exception:
            pd["_m8_evaluation"] = {}
    except Exception:
        # if m8 adapter not importable, ignore
        pd["_m8_evaluation"] = {}

    return pd
