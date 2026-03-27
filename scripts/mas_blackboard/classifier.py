from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List

from .models import RoutingDecision


@dataclass
class MockSmallLLM:
    """Simulates a small model (e.g., Qwen-7B) for cheap routing decisions."""

    def score_complexity(self, user_input: str) -> float:
        text = user_input or ""
        text_lower = text.lower()
        score = 2.0
        length_factor = min(4.0, len(text) / 180.0)
        score += length_factor

        complexity_signals = ["知识产权", "现金流", "合规", "架构", "跨境", "融资", "多域", "争议"]
        score += 0.8 * sum(1 for token in complexity_signals if token in text)

        complexity_signals_en = [
            "compliance",
            "regulatory",
            "cross-border",
            "intellectual property",
            "healthcare",
            "medical",
            "fundraising",
            "architecture",
            "multi-country",
            "platform",
            "risk",
            "legal",
        ]
        score += 0.7 * sum(1 for token in complexity_signals_en if token in text_lower)

        if re.search(r"(\d+\s*(万|w|k|m|million|billion))", text_lower):
            score += 0.8

        if any(token in text for token in ["同时", "但是", "不过", "一方面"]):
            score += 1.0
        if any(token in text_lower for token in ["however", "while", "although", "meanwhile"]):
            score += 1.0

        # 结构化众筹信号（来自训练样本文本模式）
        goal_match = re.search(r"goalusd\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text_lower)
        if goal_match:
            goal = float(goal_match.group(1))
            if goal >= 50000:
                score += 1.5
            elif goal >= 15000:
                score += 1.0
            elif goal >= 5000:
                score += 0.5

        duration_match = re.search(r"durationdays\s*[:=]\s*(\d+)", text_lower)
        if duration_match:
            duration = int(duration_match.group(1))
            if duration <= 21:
                score += 0.8
            elif duration <= 30:
                score += 0.4

        country_match = re.search(r"country\s*[:=]\s*([a-z]{2})", text_lower)
        if country_match and country_match.group(1) not in {"us", "ca", "gb"}:
            score += 0.6

        if any(token in text_lower for token in ["no technical team", "without technical team", "无技术团队"]):
            score += 1.2

        # 复合高风险模式：跨境 + 医疗 + 缺少技术团队
        has_cross_border = any(token in text_lower for token in ["cross-border", "跨境"])
        has_medical = any(token in text_lower for token in ["medical", "healthcare", "医疗", "健康"])
        has_no_tech = any(token in text_lower for token in ["no technical team", "without technical team", "无技术团队"])
        if has_cross_border and has_medical:
            score += 1.0
        if has_cross_border and has_medical and has_no_tech:
            score += 1.4

        return max(0.0, min(10.0, round(score, 2)))


class ComplexityClassifier:
    _ALLOWED_AGENTS = {"general_agent", "legal_agent", "finance_agent", "tech_agent"}

    def __init__(self) -> None:
        self.rule_model = MockSmallLLM()
        self.model = None
        self.tokenizer = None
        self.use_real_model = os.getenv("USE_REAL_SMALL_MODEL", "false").lower() == "true"
        if self.use_real_model:
            self._load_real_model()
        else:
            print("INFO 路由器：使用规则引擎（USE_REAL_SMALL_MODEL=false）")

    def _load_real_model(self) -> None:
        """安全加载 Qwen3-1.7B + LoRA（8G 显存优化版）"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch

            def _select_runtime_backend() -> tuple[str, Any, str]:
                if torch.cuda.is_available():
                    return "cuda", torch.float16, "CUDA"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps", torch.float16, "Apple Silicon MPS"
                return "cpu", torch.float32, "CPU"

            base_path = os.getenv("QWEN3_BASE_PATH", "models/Qwen3-1.7B")
            adapter_path = os.getenv("ROUTER_ADAPTER_PATH", "scripts/training/output/adapter")
            resolved_adapter_path = self._resolve_adapter_path(adapter_path)

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path,
                trust_remote_code=True,
                use_fast=False,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            runtime_device, runtime_dtype, runtime_backend = _select_runtime_backend()

            model_load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if runtime_backend == "CUDA":
                model_load_kwargs["device_map"] = "auto"
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    dtype=runtime_dtype,
                    **model_load_kwargs,
                )
            except TypeError as te:
                if "unexpected keyword argument 'dtype'" not in str(te):
                    raise
                print("WARN 当前 transformers 不支持 dtype，自动回退 torch_dtype")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    torch_dtype=runtime_dtype,
                    **model_load_kwargs,
                )
            self.model = PeftModel.from_pretrained(base_model, resolved_adapter_path)

            if runtime_backend != "CUDA":
                self.model.to(runtime_device)
            self.model.eval()

            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                print(
                    f"SUCCESS 小模型加载成功 | 显存占用: {allocated_gb:.2f}GB | Adapter: {resolved_adapter_path}"
                )
                if allocated_gb > 6.5:
                    print(f"WARN 显存占用偏高 ({allocated_gb:.2f}GB)，建议监控")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    mps_allocated_gb = torch.mps.current_allocated_memory() / 1e9
                    print(
                        f"SUCCESS 小模型加载成功 | 后端: {runtime_backend} | 显存占用: {mps_allocated_gb:.2f}GB | Adapter: {resolved_adapter_path}"
                    )
                except Exception:
                    print(
                        f"SUCCESS 小模型加载成功 | 后端: {runtime_backend} | Adapter: {resolved_adapter_path}"
                    )
            else:
                print(f"WARN 未检测到硬件加速后端，当前为 {runtime_backend} 模式")

        except Exception as e:
            print(f"ERROR 小模型加载失败: {str(e)} -> 回退规则引擎")
            self.use_real_model = False

    def _resolve_adapter_path(self, configured_path: str) -> str:
        """优先返回可解析的 adapter 目录，规避损坏的 adapter_config.json。"""

        def _is_valid_adapter_dir(path_str: str) -> bool:
            cfg = Path(path_str) / "adapter_config.json"
            if not cfg.exists():
                return False
            try:
                with cfg.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return isinstance(data, dict) and "target_modules" in data
            except Exception:
                return False

        candidates = [
            configured_path,
            str(Path(configured_path) / "adapter_model"),
            str(Path(configured_path) / "checkpoint-102468"),
            str(Path(configured_path) / "checkpoint-102000"),
        ]

        for candidate in candidates:
            if _is_valid_adapter_dir(candidate):
                if candidate != configured_path:
                    print(
                        f"WARN 检测到主适配器配置异常，自动切换到可用目录: {candidate}"
                    )
                return candidate

        raise ValueError(f"未找到可用 LoRA 适配器目录: {configured_path}")

    def _default_agents_for_tier(self, tier: str) -> List[str]:
        if tier == "L1":
            return ["general_agent"]
        if tier == "L2":
            return ["legal_agent", "finance_agent", "tech_agent"]
        return ["legal_agent", "finance_agent", "tech_agent", "general_agent"]

    def _normalize_agents(self, agents: Any, tier: str) -> List[str]:
        if not isinstance(agents, list):
            agents = [agents] if agents else []

        alias_map = {
            "general": "general_agent",
            "generalist": "general_agent",
            "legal": "legal_agent",
            "compliance": "legal_agent",
            "finance": "finance_agent",
            "financial": "finance_agent",
            "tech": "tech_agent",
            "engineering": "tech_agent",
            "engineer": "tech_agent",
            "healthcare_agents": "legal_agent",
            "medical_assessment": "legal_agent",
        }

        normalized: List[str] = []
        for item in agents:
            key = str(item).strip().lower()
            key = alias_map.get(key, key)
            if key in self._ALLOWED_AGENTS and key not in normalized:
                normalized.append(key)

        if not normalized:
            normalized = self._default_agents_for_tier(tier)

        return normalized

    def _tier_from_score(self, score: float) -> str:
        if score <= 3.5:
            return "L1"
        if score <= 6.8:
            return "L2"
        return "L3"

    def _tier_rank(self, tier: str) -> int:
        return {"L1": 1, "L2": 2, "L3": 3}[tier]

    def _coerce_tier(self, value: Any, user_input: str) -> str:
        tier = str(value).upper().strip() if value is not None else ""
        if tier in {"L1", "L2", "L3"}:
            return tier
        score = self.rule_model.score_complexity(user_input)
        return self._tier_from_score(score)

    def _calibrate_tier(self, model_tier: str, user_input: str) -> str:
        """融合规则复杂度，降低高风险项目被低估的概率。"""
        score = self.rule_model.score_complexity(user_input)
        rule_tier = self._tier_from_score(score)

        # 轻量防过冲：模型判 L3 但规则分明显不足时，回调到 L2。
        if model_tier == "L3" and score < 5.8:
            return "L2"

        # 只做向上校准：当规则分数明显更高时，提升 tier。
        if self._tier_rank(rule_tier) > self._tier_rank(model_tier):
            if score >= 7.2:
                return "L3"
            if score >= 5.0:
                return "L2"

        return model_tier

    def _canonicalize_router_output(
        self,
        result: Dict[str, Any],
        user_input: str,
        source_path: str,
    ) -> Dict[str, Any]:
        tier = self._coerce_tier(result.get("tier"), user_input)
        tier = self._calibrate_tier(tier, user_input)
        parallelism_by_tier = {"L1": 1, "L2": 2, "L3": 3}

        sub_type = result.get("sub_type", result.get("type", "unknown"))
        sub_type = str(sub_type).strip().lower().replace(" ", "_")
        if not sub_type:
            sub_type = "unknown"

        confidence = result.get("confidence_score", result.get("confidence", 0.55))
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.55
        if confidence > 1:
            confidence = confidence / 100.0
        confidence = max(0.05, min(0.99, round(confidence, 4)))

        # 为了输出语义稳定，parallelism 由 tier 统一决定。
        parallelism = parallelism_by_tier[tier]

        agents_input = result.get("suggested_agents", result.get("recommended_agents", []))
        suggested_agents = self._normalize_agents(agents_input, tier)

        reason = result.get("reason")
        if not reason:
            reason = f"normalized-from-{source_path}"

        return {
            "tier": tier,
            "sub_type": sub_type,
            "suggested_agents": suggested_agents,
            "parallelism": parallelism,
            "confidence_score": confidence,
            "reason": str(reason),
            "_path": source_path,
        }

    def predict(self, user_input: str) -> Dict[str, Any]:
        if self.use_real_model and hasattr(self, "model") and self.model is not None:
            try:
                import torch

                prompt = (
                    "<|im_start|>system\n"
                    "你是一个创业项目路由决策器。\n"
                    "你必须只输出一个 JSON 对象，不要输出任何解释文本或 Markdown。\n"
                    "不要输出 <think> 标签或任何思维链内容。\n"
                    "JSON 必须包含字段: tier, sub_type, suggested_agents, parallelism, confidence_score, reason。\n"
                    "tier 只能是 L1/L2/L3；suggested_agents 只能包含 general_agent/legal_agent/finance_agent/tech_agent。\n"
                    "confidence_score 范围是 0~1。\n"
                    "<|im_end|>\n"
                    f"<|im_start|>user\n{user_input}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

                model_device = next(self.model.parameters()).device
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)

                self.model.generation_config.do_sample = False
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
                self.model.generation_config.top_k = None

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

                if "<|im_start|>assistant" in full_text:
                    json_str = full_text.split("<|im_start|>assistant")[-1].strip()
                    json_str = json_str.replace("<|im_end|>", "").strip()
                else:
                    json_str = full_text[len(prompt) :].strip()

                parsed = self._parse_and_validate(json_str, user_input)
                parsed["_raw_model_output"] = json.dumps(
                    {
                        "tier": parsed.get("tier"),
                        "sub_type": parsed.get("sub_type"),
                        "suggested_agents": parsed.get("suggested_agents"),
                        "parallelism": parsed.get("parallelism"),
                        "confidence_score": parsed.get("confidence_score"),
                        "reason": parsed.get("reason"),
                    },
                    ensure_ascii=False,
                )
                parsed["_path"] = parsed.get("_path", "model")
                return parsed

            except Exception as e:
                print(f"WARN 小模型推理异常: {str(e)[:100]} -> 回退规则引擎")

        result = self._rule_based_predict(user_input)
        result["_path"] = "rule_fallback"
        return result

    def _rule_based_predict(self, user_input: str) -> Dict[str, Any]:
        score = self.rule_model.score_complexity(user_input)

        if score <= 3.5:
            tier = "L1"
            agents = ["general_agent"]
            parallelism = 1
        elif score <= 6.8:
            tier = "L2"
            agents = ["legal_agent", "finance_agent", "tech_agent"]
            parallelism = 2
        else:
            tier = "L3"
            agents = ["legal_agent", "finance_agent", "tech_agent", "general_agent"]
            parallelism = 3

        return {
            "tier": tier,
            "sub_type": "rule_based",
            "suggested_agents": agents,
            "parallelism": parallelism,
            "confidence_score": round(score / 10.0, 4),
            "reason": f"rule-based complexity score={score}",
            "_path": "rule",
        }

    def _normalize_to_router_schema(self, result: Dict[str, Any], user_input: str) -> Dict[str, Any] | None:
        """把非标准输出尽量映射为路由标准 schema。"""

        # 常见错配 schema：type/branch/manager/engine/additional
        if "tier" not in result and ("branch" in result or "type" in result):
            branch = str(result.get("branch", "")).upper()
            branch_to_tier = {
                "ELITE": "L3",
                "ADVANCED": "L3",
                "STANDARD": "L2",
                "BASIC": "L1",
            }
            tier = branch_to_tier.get(branch)
            if tier is None:
                score = self.rule_model.score_complexity(user_input)
                tier = self._tier_from_score(score)

            suggested_agents: List[str] = []
            manager = str(result.get("manager", "")).lower()
            engine = str(result.get("engine", "")).lower()
            type_name = str(result.get("type", "")).lower()

            if "legal" in manager or "legal" in engine or "compliance" in type_name:
                suggested_agents.append("legal_agent")
            if "finance" in manager or "finance" in engine:
                suggested_agents.append("finance_agent")
            if "tech" in manager or "engineer" in manager or "gpt" in engine:
                suggested_agents.append("tech_agent")
            if not suggested_agents:
                suggested_agents = ["general_agent"]

            parallelism = {"L1": 1, "L2": 2, "L3": 3}[tier]
            confidence = result.get("confidence_score", result.get("confidence", 0.55))
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.55
            if confidence > 1:
                confidence = confidence / 100.0
            confidence = max(0.05, min(0.99, round(confidence, 4)))

            mapped = {
                "tier": tier,
                "sub_type": str(result.get("type", "mapped_external_schema")).lower(),
                "suggested_agents": suggested_agents,
                "parallelism": parallelism,
                "confidence_score": confidence,
                "reason": "mapped from external model schema",
                "_path": "model_mapped",
            }
            return self._canonicalize_router_output(mapped, user_input, "model_mapped")

        # 常见错配 schema：decision/reason
        if "tier" not in result and "decision" in result:
            score = self.rule_model.score_complexity(user_input)
            tier = self._tier_from_score(score)
            mapped = {
                "tier": tier,
                "sub_type": "mapped_decision_only",
                "suggested_agents": ["general_agent"] if tier == "L1" else ["legal_agent", "finance_agent", "tech_agent"],
                "parallelism": {"L1": 1, "L2": 2, "L3": 3}[tier],
                "confidence_score": round(max(0.2, min(0.8, score / 10.0)), 4),
                "reason": "mapped from decision-only schema",
                "_path": "model_mapped",
            }
            return self._canonicalize_router_output(mapped, user_input, "model_mapped")

        return None

    def _parse_and_validate(self, raw_output: str, user_input: str) -> Dict[str, Any]:
        """严格校验 JSON + 字段补全 + 异常修复"""

        try:
            raw_output = re.sub(r"```json\s*|\s*```", "", raw_output)
            raw_output = raw_output.replace("'", '"')

            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                raw_output = match.group()
            else:
                fallback = self._rule_based_predict(user_input)
                fallback["reason"] = "model returned non-json output; fallback to rule"
                return self._canonicalize_router_output(fallback, user_input, "model_non_json_fallback")

            result = json.loads(raw_output)

            normalized = self._normalize_to_router_schema(result, user_input)
            if normalized is not None:
                return normalized

            required = ["tier", "sub_type", "suggested_agents", "parallelism", "confidence_score"]
            missing = [k for k in required if k not in result]
            if missing:
                raise ValueError(f"缺失关键字段: {missing}")

            if "reason" not in result:
                result["reason"] = "small-model prediction"
            return self._canonicalize_router_output(result, user_input, "model")

        except Exception as e:
            print(f"WARN JSON 解析失败: {str(e)} -> 触发规则引擎 fallback")
            return self._rule_based_predict(user_input)

    def classify(self, user_input: str) -> RoutingDecision:
        result = self.predict(user_input)
        confidence = float(result.get("confidence_score", 0.0))
        if confidence > 1:
            confidence = confidence / 100.0
        score = round(max(0.0, min(10.0, confidence * 10.0)), 2)

        return RoutingDecision(
            complexity_score=score,
            tier=result["tier"],
            recommended_agents=list(result.get("suggested_agents", [])),
            reason=str(result.get("reason", f"Complexity score={score}")),
        )
