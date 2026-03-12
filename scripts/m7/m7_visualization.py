from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import patches


EXPERT_DISPLAY_NAME = {
    "risk_guardian": "Risk Guardian",
    "finance_advisor": "Finance Advisor",
    "growth_strategist": "Growth Strategist",
    "ops_executor": "Operations Executor",
}


ROUTE_REASON_EN = {
    "extreme_high": "Prioritize downside control and cash-flow safety for extreme-high risk.",
    "high": "Prioritize risk control and financial safeguards for high risk.",
    "medium": "Use financial stabilization and operations recovery for medium risk.",
    "low": "Shift focus to growth and operational scaling for low risk.",
}


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _plot_expert_selection(route_result: Dict[str, object], save_path: Path) -> None:
    selected_experts: List[Dict[str, str]] = route_result["selected_experts"]  # type: ignore[assignment]
    roles = [EXPERT_DISPLAY_NAME.get(expert["name"], expert["name"]) for expert in selected_experts]
    scores = [1 for _ in selected_experts]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(roles, scores, color=["#3498db", "#2ecc71", "#f39c12", "#9b59b6"][: len(roles)])
    ax.set_title("M7 Selected Experts", fontsize=12)
    ax.set_ylabel("Selected")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1])

    for index, expert in enumerate(selected_experts):
        ax.text(index, 1.03, expert["name"], ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_route_flow(route_result: Dict[str, object], save_path: Path) -> None:
    normalized_risk = str(route_result["normalized_risk_level"])
    selected_experts: List[Dict[str, str]] = route_result["selected_experts"]  # type: ignore[assignment]
    expert_roles = " + ".join(EXPERT_DISPLAY_NAME.get(expert["name"], expert["name"]) for expert in selected_experts)
    route_reason = ROUTE_REASON_EN.get(normalized_risk, str(route_result["route_reason"]))

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    node_style = dict(boxstyle="round,pad=0.35", facecolor="#f5f7fa", edgecolor="#34495e", linewidth=1.2)

    ax.text(1.3, 2.2, f"Input\nRisk={normalized_risk}", ha="center", va="center", fontsize=10, bbox=node_style)
    ax.text(4.3, 2.2, "M7 Router\nRule Engine", ha="center", va="center", fontsize=10, bbox=node_style)
    ax.text(7.6, 2.2, f"Experts\n{expert_roles}", ha="center", va="center", fontsize=10, bbox=node_style)
    ax.text(10.1, 2.2, "Output\nRoute Decision", ha="center", va="center", fontsize=10, bbox=node_style)

    arrow_style = dict(arrowstyle="->", linewidth=1.3, color="#2c3e50")
    ax.annotate("", xy=(3.2, 2.2), xytext=(2.1, 2.2), arrowprops=arrow_style)
    ax.annotate("", xy=(6.5, 2.2), xytext=(5.4, 2.2), arrowprops=arrow_style)
    ax.annotate("", xy=(9.2, 2.2), xytext=(8.7, 2.2), arrowprops=arrow_style)

    note_box = patches.FancyBboxPatch(
        (2.0, 0.45),
        7.9,
        0.9,
        boxstyle="round,pad=0.25",
        facecolor="#ecf0f1",
        edgecolor="#95a5a6",
        linewidth=1.0,
    )
    ax.add_patch(note_box)
    ax.text(5.95, 0.9, f"Reason: {route_reason}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_m7_visualizations(route_result: Dict[str, object], save_dir: str) -> Dict[str, str]:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    risk_tag = _safe_filename(str(route_result.get("normalized_risk_level", "unknown")))
    bar_path = output_dir / f"m7_expert_selection_{risk_tag}.png"
    flow_path = output_dir / f"m7_route_flow_{risk_tag}.png"

    _plot_expert_selection(route_result, bar_path)
    _plot_route_flow(route_result, flow_path)

    return {
        "expert_selection_chart": str(bar_path),
        "route_flow_chart": str(flow_path),
    }
