"""
M11 小模型加载与单例管理。

使用 Qwen3-1.7B base model（不含 LoRA adapter）做文本生成，
与 classifier.py 的 PEFT 分类实例区分开。
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("m11_model")

_MODEL_INSTANCE = None
_TOKENIZER_INSTANCE = None


def _select_runtime_backend() -> str:
    """选择运行时后端：CUDA > MPS > CPU"""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_model_path() -> Path:
    """解析 Qwen3 模型路径，优先 QWEN3_BASE_PATH 环境变量"""
    env_path = os.getenv("QWEN3_BASE_PATH", "")
    if env_path:
        return Path(env_path)
    # 默认路径：项目根目录下的 models/Qwen3-1.7B
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "models" / "Qwen3-1.7B"


def _load_small_model() -> Tuple[object, object]:
    """加载 Qwen3-1.7B base model（不含 LoRA）。

    Returns:
        (model, tokenizer) 元组

    Raises:
        SmallModelNotConfigured: USE_REAL_SMALL_MODEL 未开启
        SmallModelLoadFailed: 模型文件缺失或加载失败
    """
    from .m11_core import SmallModelNotConfigured, SmallModelLoadFailed

    use_real = os.getenv("USE_REAL_SMALL_MODEL", "false").lower()
    if use_real != "true":
        raise SmallModelNotConfigured(
            "小模型未启用。请设置 USE_REAL_SMALL_MODEL=true 并配置模型路径"
        )

    model_path = _resolve_model_path()
    if not model_path.exists():
        raise SmallModelLoadFailed(
            f"模型文件未找到: {model_path}。"
            f"请下载 Qwen3-1.7B 到 {model_path} 或设置 QWEN3_BASE_PATH 环境变量"
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        backend = _select_runtime_backend()
        logger.info(f"M11 加载 Qwen3-1.7B from {model_path}, backend={backend}")

        # 1. 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. 加载 base model（不含 LoRA）
        device_map = "auto" if backend == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
            torch_dtype=torch.float16 if backend == "cuda" else None,
        )

        # 非 CUDA 后端手动移动到设备
        if backend != "cuda":
            model = model.to(backend)
        model.eval()

        logger.info("M11 模型加载成功")
        return model, tokenizer

    except SmallModelNotConfigured:
        raise
    except SmallModelLoadFailed:
        raise
    except Exception as e:
        raise SmallModelLoadFailed(
            f"模型加载失败: {e}。请检查模型文件完整性"
        ) from e


def get_small_model() -> Optional[Tuple[object, object]]:
    """获取 (model, tokenizer) 全局单例。

    第一次调用时加载，后续复用。
    加载失败则抛出 SmallModelNotConfigured 或 SmallModelLoadFailed。
    """
    global _MODEL_INSTANCE, _TOKENIZER_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE, _TOKENIZER_INSTANCE

    model, tokenizer = _load_small_model()
    _MODEL_INSTANCE = model
    _TOKENIZER_INSTANCE = tokenizer
    return model, tokenizer


def clear_model():
    """释放模型实例（测试用）。"""
    global _MODEL_INSTANCE, _TOKENIZER_INSTANCE
    _MODEL_INSTANCE = None
    _TOKENIZER_INSTANCE = None
    logger.info("M11 模型实例已释放")
