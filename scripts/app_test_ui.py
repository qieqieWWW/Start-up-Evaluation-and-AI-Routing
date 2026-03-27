from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from classifier import ComplexityClassifier
from model_asset_manager import auto_place_assets_from_project_root, inspect_small_model_assets


st.set_page_config(page_title="AI Routing Test UI", page_icon="🧭", layout="wide")


@st.cache_resource
def get_classifier(use_real_model: bool) -> ComplexityClassifier:
    os.environ["USE_REAL_SMALL_MODEL"] = "true" if use_real_model else "false"
    return ComplexityClassifier()


def _build_step_logs(user_text: str, result: Dict[str, Any]) -> List[str]:
    now = datetime.now().strftime("%H:%M:%S")
    logs: List[str] = []
    logs.append(f"[{now}] 输入接收: {user_text[:120]}")
    logs.append("[step-1] 调用路由分类器")

    path = str(result.get("_path", "unknown"))
    if path.startswith("model"):
        logs.append("[step-2] 执行小模型推理路径")
    elif path.startswith("rule"):
        logs.append("[step-2] 执行规则推理路径")
    else:
        logs.append(f"[step-2] 执行混合路径: {path}")

    logs.append("[step-3] 标准化输出 schema")
    logs.append("[step-4] 返回结构化评级结果")
    return logs


def _append_chat(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def _safe_json_str(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _render_structured_panel(result: Dict[str, Any]) -> None:
    st.subheader("结构化项目评级")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Tier", str(result.get("tier", "N/A")))
    with c2:
        conf = float(result.get("confidence_score", 0.0) or 0.0)
        st.metric("Confidence", f"{conf:.2f}")
    with c3:
        st.metric("Parallelism", str(result.get("parallelism", "N/A")))

    st.write("Sub Type:", str(result.get("sub_type", "unknown")))
    st.write("Suggested Agents:", ", ".join(result.get("suggested_agents", [])))
    st.write("Reason:", str(result.get("reason", "")))

    with st.expander("完整结构化 JSON", expanded=False):
        st.code(_safe_json_str(result), language="json")


def _render_log_panel(result: Dict[str, Any], step_logs: List[str]) -> None:
    st.subheader("控制台日志")

    with st.expander("模型原始输出", expanded=True):
        raw_model_output = result.get("_raw_model_output", "(无原始输出，当前可能是规则路径)")
        st.code(str(raw_model_output), language="json")

    with st.expander("系统步骤", expanded=True):
        st.code("\n".join(step_logs), language="text")


def main() -> None:
    st.title("AI 路由测试界面")
    st.caption("聊天 + 日志 + 结构化评级（最小可用版本）")

    if "asset_status" not in st.session_state:
        st.session_state.asset_status = None
    if "asset_actions" not in st.session_state:
        st.session_state.asset_actions = []
    if "asset_warnings" not in st.session_state:
        st.session_state.asset_warnings = []

    # Real-time status based on classifier default paths (or env-overridden paths).
    current_asset_status = inspect_small_model_assets()
    st.session_state.asset_status = current_asset_status

    with st.sidebar:
        st.subheader("推理设置")
        mode = st.radio(
            "推理模式",
            options=["规则引擎", "小模型"],
            index=0,
            help="小模型模式会尝试加载 Qwen + LoRA；失败时会自动回退规则路径。",
        )
        use_real_model = mode == "小模型"

        if use_real_model:
            if current_asset_status.get("ready"):
                st.success("已检测到小模型文件，可启用小模型模式。")
            else:
                st.warning("当前指定目录下没有可用小模型文件，已准备回退到规则引擎。")
                st.caption("默认基座目录")
                st.code(str(current_asset_status.get("base_path", "")), language="text")
                st.caption("默认适配器目录")
                st.code(str(current_asset_status.get("adapter_path", "")), language="text")

                if st.button("归位（从项目根目录）", use_container_width=True):
                    result = auto_place_assets_from_project_root()
                    st.session_state.asset_status = result.get("status")
                    st.session_state.asset_actions = result.get("actions", [])
                    st.session_state.asset_warnings = result.get("warnings", [])
                    get_classifier.clear()
                    st.rerun()

        effective_use_real_model = use_real_model and bool(current_asset_status.get("ready"))

        if st.button("应用模式并重建分类器", use_container_width=True):
            get_classifier.clear()
            st.session_state.last_result = None
            st.session_state.last_logs = []
            st.rerun()

        st.divider()
        st.subheader("小模型资产自检")
        st.caption("用于团队成员在本地下载模型后快速校验/归位。")

        if st.button("检测小模型模式可加载性", use_container_width=True):
            st.session_state.asset_status = inspect_small_model_assets()

        if st.button("自动归位根目录模型文件夹", use_container_width=True):
            result = auto_place_assets_from_project_root()
            st.session_state.asset_status = result.get("status")
            st.session_state.asset_actions = result.get("actions", [])
            st.session_state.asset_warnings = result.get("warnings", [])
            get_classifier.clear()

        status = st.session_state.get("asset_status")
        if status:
            if status.get("ready"):
                st.success("小模型模式文件检查通过。")
            else:
                st.error("当前文件不满足小模型模式加载要求。")

            st.write("默认基座路径:")
            st.code(str(status.get("base_path", "")), language="text")
            st.write("默认适配器路径:")
            st.code(str(status.get("adapter_path", "")), language="text")

            missing = status.get("missing", [])
            if missing:
                st.warning(
                    "\n".join(missing)
                    + "\n\n请先将下载后的基座模型文件夹与适配器文件夹放到项目根目录，"
                    "再点击“自动归位根目录模型文件夹”。"
                )

        for item in st.session_state.get("asset_actions", []):
            st.info(item)
        for item in st.session_state.get("asset_warnings", []):
            st.warning(item)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_logs" not in st.session_state:
        st.session_state.last_logs = []

    clf = get_classifier(effective_use_real_model)
    runtime_mode = "小模型" if clf.use_real_model else "规则引擎"
    if use_real_model and not current_asset_status.get("ready"):
        st.info("你选择了小模型模式，但当前目录缺少模型文件，系统已自动回退规则引擎。")
    st.info(f"当前运行模式: {runtime_mode} (USE_REAL_SMALL_MODEL={str(clf.use_real_model).lower()})")

    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("用户与评估系统聊天")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("输入项目描述，例如: GoalUSD/Country/Category/DurationDays...")
        if user_input:
            _append_chat("user", user_input)

            result = clf.predict(user_input)
            tier = result.get("tier", "N/A")
            conf = result.get("confidence_score", 0.0)
            reason = result.get("reason", "")

            assistant_reply = (
                f"评级结果: **{tier}**  |  置信度: **{conf}**\n\n"
                f"建议专家: {', '.join(result.get('suggested_agents', []))}\n\n"
                f"原因: {reason}"
            )
            _append_chat("assistant", assistant_reply)

            st.session_state.last_result = result
            st.session_state.last_logs = _build_step_logs(user_input, result)
            st.rerun()

    with right:
        result = st.session_state.last_result
        if result is None:
            st.info("先在左侧输入一条项目描述，右侧将显示结构化评级和日志。")
        else:
            _render_structured_panel(result)
            st.divider()
            _render_log_panel(result, st.session_state.last_logs)


if __name__ == "__main__":
    main()
