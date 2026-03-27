from __future__ import annotations

import shutil
import os
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_REL = "models/Qwen3-1.7B"
DEFAULT_ADAPTER_REL = "scripts/training/output/adapter"

BASE_REQUIRED_FILES = ["config.json", "tokenizer.json"]


def _resolve_effective_paths() -> Tuple[Path, Path]:
    base_rel = Path(
        os.getenv("QWEN3_BASE_PATH", DEFAULT_BASE_REL)
    )
    adapter_rel = Path(
        os.getenv("ROUTER_ADAPTER_PATH", DEFAULT_ADAPTER_REL)
    )

    return PROJECT_ROOT / base_rel, PROJECT_ROOT / adapter_rel


def _is_valid_base_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / req).exists() for req in BASE_REQUIRED_FILES)


def _is_valid_adapter_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    candidates = [
        path,
        path / "adapter_model",
        path / "checkpoint-102468",
        path / "checkpoint-102000",
    ]

    for candidate in candidates:
        cfg = candidate / "adapter_config.json"
        if cfg.exists():
            return True
    return False


def inspect_small_model_assets() -> Dict[str, object]:
    base_path, adapter_path = _resolve_effective_paths()

    base_ok = _is_valid_base_dir(base_path)
    adapter_ok = _is_valid_adapter_dir(adapter_path)

    missing: List[str] = []
    if not base_ok:
        missing.append(f"基座模型目录不可用: {base_path}")
    if not adapter_ok:
        missing.append(f"LoRA 适配器目录不可用: {adapter_path}")

    return {
        "ready": base_ok and adapter_ok,
        "base_path": str(base_path),
        "adapter_path": str(adapter_path),
        "base_ok": base_ok,
        "adapter_ok": adapter_ok,
        "missing": missing,
    }


def _root_drop_candidates() -> Tuple[List[Path], List[Path]]:
    base_sources = [
        PROJECT_ROOT / "Qwen3-1.7B",
        PROJECT_ROOT / "qwen3-1.7b",
    ]

    adapter_sources = [
        PROJECT_ROOT / "adapter",
        PROJECT_ROOT / "adapter_model",
        PROJECT_ROOT / "lora_adapter",
        PROJECT_ROOT / "router_adapter",
    ]

    return base_sources, adapter_sources


def _pick_first_existing(candidates: List[Path]) -> Path | None:
    for item in candidates:
        if item.exists() and item.is_dir():
            return item
    return None


def auto_place_assets_from_project_root() -> Dict[str, object]:
    base_dst, adapter_dst = _resolve_effective_paths()
    base_sources, adapter_sources = _root_drop_candidates()

    base_src = _pick_first_existing(base_sources)
    adapter_src = _pick_first_existing(adapter_sources)

    actions: List[str] = []
    warnings: List[str] = []

    if base_src is None and not _is_valid_base_dir(base_dst):
        warnings.append(
            "未在项目根目录找到基座模型文件夹（候选名: Qwen3-1.7B / qwen3-1.7b）。"
        )
    if adapter_src is None and not _is_valid_adapter_dir(adapter_dst):
        warnings.append(
            "未在项目根目录找到适配器文件夹（候选名: adapter / adapter_model / lora_adapter / router_adapter）。"
        )

    if base_src is not None and base_src.resolve() != base_dst.resolve() and not base_dst.exists():
        base_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(base_src), str(base_dst))
        actions.append(f"已归位基座模型: {base_src} -> {base_dst}")

    if adapter_src is not None and adapter_src.resolve() != adapter_dst.resolve() and not adapter_dst.exists():
        adapter_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(adapter_src), str(adapter_dst))
        actions.append(f"已归位 LoRA 适配器: {adapter_src} -> {adapter_dst}")

    # If destination exists already, do not overwrite; just warn.
    if base_src is not None and base_dst.exists() and base_src.resolve() != base_dst.resolve():
        if not any("基座模型" in act for act in actions):
            warnings.append(f"目标已存在，跳过基座模型归位: {base_dst}")

    if adapter_src is not None and adapter_dst.exists() and adapter_src.resolve() != adapter_dst.resolve():
        if not any("LoRA 适配器" in act for act in actions):
            warnings.append(f"目标已存在，跳过适配器归位: {adapter_dst}")

    status = inspect_small_model_assets()
    return {
        "actions": actions,
        "warnings": warnings,
        "status": status,
    }


def cli_main() -> int:
    print("[INFO] 检查小模型资产状态...")
    before = inspect_small_model_assets()
    print(f"[INFO] 基座目录: {before['base_path']} | 可用={before['base_ok']}")
    print(f"[INFO] 适配器目录: {before['adapter_path']} | 可用={before['adapter_ok']}")

    if before["ready"]:
        print("[OK] 当前已满足小模型模式加载要求，无需归位。")
        return 0

    print("[INFO] 尝试从项目根目录自动归位...")
    result = auto_place_assets_from_project_root()

    for act in result["actions"]:
        print(f"[ACTION] {act}")
    for warn in result["warnings"]:
        print(f"[WARN] {warn}")

    after = result["status"]
    if after["ready"]:
        print("[OK] 归位完成，已满足小模型模式加载要求。")
        return 0

    print("[ERROR] 归位后仍不满足要求，请检查基座模型与 LoRA 目录内容。")
    return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
