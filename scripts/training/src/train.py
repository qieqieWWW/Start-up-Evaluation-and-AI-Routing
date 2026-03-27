import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from utils import load_yaml, resolve_existing_path


def build_prompt(user_input: str, output_text: str) -> str:
    return (
        "<|im_start|>system\n路由决策器<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n{output_text}<|im_end|>"
    )


class GpuMemoryCallback(TrainerCallback):
    def __init__(self, max_gb: float = 7.8, print_interval: int = 100) -> None:
        self.max_gb = max_gb
        self.print_interval = print_interval

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available() or state.global_step == 0:
            return
        if state.global_step % self.print_interval != 0:
            return

        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(
            f"[GPU] step={state.global_step} "
            f"allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB"
        )

        if allocated_gb > self.max_gb:
            raise RuntimeError(
                f"GPU allocated memory {allocated_gb:.2f}GB exceeds safety limit {self.max_gb:.2f}GB"
            )


class EvalRiseEarlyStoppingCallback(TrainerCallback):
    """Stop when eval loss has risen for 3 consecutive evaluations."""

    def __init__(self) -> None:
        self.eval_losses = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return

        current = float(metrics["eval_loss"])
        self.eval_losses.append(current)

        if len(self.eval_losses) >= 4:
            a, b, c, d = self.eval_losses[-4:]
            if a < b < c < d:
                print(
                    "[EarlyStop] eval_loss increased in 3 consecutive evaluations: "
                    f"{a:.4f} -> {b:.4f} -> {c:.4f} -> {d:.4f}"
                )
                control.should_training_stop = True


def load_jsonl(path: Path) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def tokenize_batch(example: Dict[str, str], tokenizer, max_seq_length: int):
    prompt = build_prompt(example["input"], example["output"])
    tokenized = tokenizer(
        prompt,
        max_length=max_seq_length,
        truncation=True,
    )
    return tokenized


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def list_checkpoints(output_dir: Path):
    checkpoints = []
    if not output_dir.exists():
        return checkpoints

    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[-1])
            except ValueError:
                continue
            checkpoints.append((step, item))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def choose_training_mode(output_dir: Path) -> tuple[bool, Optional[str]]:
    """Return (overwrite_output_dir, resume_from_checkpoint)."""
    checkpoints = list_checkpoints(output_dir)

    print("\n" + "=" * 72)
    print("训练启动菜单")
    print("=" * 72)
    print("1. 从头开始训练")
    print("2. 从最新检查点继续训练")
    print("3. 从指定检查点继续训练")

    if checkpoints:
        print("\n检测到检查点:")
        for idx, (step, ckpt) in enumerate(checkpoints, start=1):
            print(f"  {idx}. checkpoint-{step} -> {ckpt}")
    else:
        print("\n未检测到检查点。")

    while True:
        try:
            choice = input("\n请选择 [1/2/3]: ").strip()
        except EOFError:
            print("\n[WARN] 未检测到交互输入，默认从头训练。")
            return True, None

        if choice == "1":
            print("\n[Mode] 从头开始训练")
            return True, None

        if choice == "2":
            if not checkpoints:
                print("[WARN] 当前没有可用检查点，请选择 1。")
                continue
            latest_step, latest_ckpt = checkpoints[-1]
            print(f"\n[Mode] 从最新检查点恢复: checkpoint-{latest_step}")
            return False, str(latest_ckpt)

        if choice == "3":
            if not checkpoints:
                print("[WARN] 当前没有可用检查点，请选择 1。")
                continue

            index_or_path = input("请输入检查点编号或完整路径: ").strip()

            # Number from listed checkpoints
            if index_or_path.isdigit():
                idx = int(index_or_path) - 1
                if 0 <= idx < len(checkpoints):
                    selected = checkpoints[idx][1]
                    print(f"\n[Mode] 从指定检查点恢复: {selected}")
                    return False, str(selected)
                print("[WARN] 编号超出范围，请重试。")
                continue

            # Direct checkpoint path
            selected_path = Path(index_or_path)
            if selected_path.exists() and selected_path.is_dir():
                print(f"\n[Mode] 从指定检查点恢复: {selected_path}")
                return False, str(selected_path)

            print("[WARN] 路径不存在或不是目录，请重试。")
            continue

        print("[WARN] 无效选择，请输入 1 / 2 / 3。")


def _is_torch_lt_26() -> bool:
    core = str(torch.__version__).split("+")[0]
    parts = core.split(".")
    nums = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums) < (2, 6, 0)


def maybe_disable_unsafe_resume_states(checkpoint_dir: Optional[str]) -> None:
    """
    transformers>=4.57 blocks torch.load on torch<2.6 due to CVE-2025-32434.
    To keep resume usable on torch 2.4, disable optimizer/scheduler state loading.
    """
    if not checkpoint_dir or not _is_torch_lt_26():
        return

    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        return

    state_files = [
        "optimizer.pt",
        "optimizer.bin",
        "scheduler.pt",
    ]
    disabled_any = False
    for name in state_files:
        p = ckpt / name
        if p.exists() and p.is_file():
            backup = ckpt / f"{name}.disabled_torch_lt_2_6"
            os.replace(str(p), str(backup))
            disabled_any = True

    if disabled_any:
        print(
            "[WARN] torch<2.6 detected. Disabled optimizer/scheduler checkpoint states "
            "for safe resume (model weights and trainer state will still resume)."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    print(f"[Env] python={os.sys.executable}")
    print(f"[Env] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, cuda_version={torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in current Python environment. "
            "Please run with a CUDA-enabled interpreter, for example: "
            "D:/software/anaconda/python.exe src/train.py"
        )

    cfg = load_yaml(args.config)

    train_batch_size = int(cfg["training"]["per_device_train_batch_size"])
    max_seq_length = int(cfg["training"]["max_seq_length"])
    use_fp16 = bool(cfg["training"].get("fp16", True))
    use_bf16 = bool(cfg["training"].get("bf16", False))
    if train_batch_size > 2:
        raise ValueError("per_device_train_batch_size must be <= 2 for this script's safety guard.")
    if max_seq_length > 512:
        raise ValueError("max_seq_length must be <= 512.")
    if use_fp16 and use_bf16:
        raise ValueError("fp16 and bf16 cannot both be true.")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        print("[WARN] bf16 requested but not supported on this GPU. Fallback to fp16.")
        use_bf16 = False
        use_fp16 = True
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    model_path = resolve_existing_path(cfg["model"]["model_name_or_path"], anchor_file=__file__)

    train_path = resolve_existing_path(
        str(Path(cfg["data"]["processed_dir"]) / "train.jsonl"), anchor_file=__file__
    )
    val_path = resolve_existing_path(
        str(Path(cfg["data"]["processed_dir"]) / "val.jsonl"), anchor_file=__file__
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    device_map_cfg = cfg["training"].get("device_map", "cuda:0")

    use_4bit = bool(cfg["training"].get("load_in_4bit", True))
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bool(cfg["training"].get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=str(cfg["training"].get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=model_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=model_dtype,
            trust_remote_code=True,
            device_map=device_map_cfg,
            quantization_config=quant_cfg,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=model_dtype,
            trust_remote_code=True,
            device_map=device_map_cfg,
        )

    model.config.use_cache = False
    if bool(cfg["training"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["lora_r"]),
        lora_alpha=int(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    train_ds = load_jsonl(train_path)
    val_ds = load_jsonl(val_path)

    train_ds = train_ds.map(
        lambda x: tokenize_batch(x, tokenizer, max_seq_length),
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda x: tokenize_batch(x, tokenizer, max_seq_length),
        remove_columns=val_ds.column_names,
    )

    adapter_root = Path(cfg["output"]["adapter_dir"])
    adapter_model_dir = adapter_root / "adapter_model"
    tensorboard_dir = Path(cfg["output"]["tensorboard_dir"])
    adapter_root.mkdir(parents=True, exist_ok=True)
    adapter_model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    report_to = cfg["training"].get("report_to", "tensorboard")
    if isinstance(report_to, str):
        report_to = [report_to]

    # Newer transformers prefers env-based tensorboard logging path.
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(tensorboard_dir))

    overwrite_output_dir, resume_from_checkpoint = choose_training_mode(adapter_root)

    max_grad_norm = float(cfg["training"]["max_grad_norm"])
    if use_fp16 and max_grad_norm > 0:
        print("[WARN] fp16 + grad clipping may trigger unscale errors on this stack. max_grad_norm -> 0.0")
        max_grad_norm = 0.0

    training_args_kwargs = dict(
        output_dir=str(adapter_root),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=int(cfg["training"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        num_train_epochs=float(cfg["training"]["num_train_epochs"]),
        max_grad_norm=max_grad_norm,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=bool(cfg["training"]["gradient_checkpointing"]),
        optim=cfg["training"]["optim"],
        logging_steps=int(cfg["training"]["logging_steps"]),
        save_steps=int(cfg["training"]["save_steps"]),
        eval_steps=int(cfg["training"]["eval_steps"]),
        save_strategy="steps",
        save_total_limit=int(cfg["training"]["save_total_limit"]),
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        report_to=report_to,
        run_name=cfg["training"].get("run_name", "qwen3-lora-8gb"),
        logging_nan_inf_filter=bool(cfg["training"]["logging_nan_inf_filter"]),
        dataloader_pin_memory=False,
        group_by_length=True,
    )

    # Only pass arguments supported by the current transformers version.
    ta_signature = inspect.signature(TrainingArguments.__init__)
    supported_params = set(ta_signature.parameters.keys())

    if "overwrite_output_dir" in supported_params:
        training_args_kwargs["overwrite_output_dir"] = overwrite_output_dir

    # Prefer warmup_steps on newer transformers; fallback to warmup_ratio for older versions.
    configured_warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    if configured_warmup_steps > 0 and "warmup_steps" in supported_params:
        training_args_kwargs["warmup_steps"] = configured_warmup_steps
    elif "warmup_ratio" in supported_params:
        training_args_kwargs["warmup_ratio"] = float(cfg["training"].get("warmup_ratio", 0.0))

    dropped_keys = [k for k in list(training_args_kwargs.keys()) if k not in supported_params]
    if dropped_keys:
        print(f"[WARN] TrainingArguments does not support keys: {dropped_keys}. They will be ignored.")

    training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if k in supported_params}

    if "evaluation_strategy" in ta_signature.parameters:
        training_args_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_signature.parameters:
        training_args_kwargs["eval_strategy"] = "steps"

    args_train = TrainingArguments(**training_args_kwargs)

    # bitsandbytes may be missing or CPU-only; fallback to stable torch optimizer.
    requested_optim = str(getattr(args_train, "optim", "")).lower()
    wants_8bit_optim = any(token in requested_optim for token in ["8bit", "paged_adamw"])
    if wants_8bit_optim:
        bnb_gpu_ok = False
        try:
            import bitsandbytes as bnb  # type: ignore

            if hasattr(bnb, "cuda_setup") and hasattr(bnb.cuda_setup, "main_check"):
                check = bnb.cuda_setup.main_check()
                bnb_gpu_ok = bool(getattr(check, "cuda_available", False))
            else:
                # Newer bitsandbytes may not expose cuda_setup in the same shape.
                bnb_gpu_ok = torch.cuda.is_available()
        except Exception:
            bnb_gpu_ok = False

        if not bnb_gpu_ok:
            print("[WARN] bitsandbytes GPU backend unavailable, fallback optim -> adamw_torch")
            args_train.optim = "adamw_torch"

    trainer_kwargs = dict(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            GpuMemoryCallback(max_gb=float(cfg["training"]["vram_limit_gb"])),
            EvalRiseEarlyStoppingCallback(),
        ],
    )

    trainer_signature = inspect.signature(Trainer.__init__)
    trainer_supported = set(trainer_signature.parameters.keys())
    if "processing_class" in trainer_supported:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_supported:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    if resume_from_checkpoint:
        maybe_disable_unsafe_resume_states(resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    model.save_pretrained(str(adapter_model_dir))
    tokenizer.save_pretrained(str(adapter_model_dir))

    peft_config = model.peft_config.get("default")
    if peft_config is not None:
        with open(adapter_root / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(peft_config.to_dict()), f, ensure_ascii=False, indent=2)

    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")


if __name__ == "__main__":
    main()
