# Qwen3 1.7B LoRA 训练流水线

## 目录说明

- `config/config.yaml`：训练与推理的完整配置
- `src/preprocess.py`：标签预处理与数据集划分（8:1:1）
- `src/train.py`：使用 FP16 + 梯度检查点 + 8-bit 优化器进行 LoRA 训练
- `src/evaluate.py`：计算 tier 准确率、JSON 解析成功率、L3 召回率
- `src/inference.py`：单样本推理，带 JSON 修复兜底
- `src/utils.py`：路径解析、模型加载、JSON 修复工具
- `data/`：生成的 train/val/test JSONL 数据
- `output/adapter/`：LoRA 适配器产物

## 5 分钟快速启动

```bash
cd "/home/qieqie/桌面/Start-up-Evaluation-and-AI-Routing/scripts/training" && conda activate qwen-train && rm -rf src/__pycache__ && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/qieqie/miniconda3/envs/qwen-train/bin/python src/train.py
```

说明：执行 `python src/train.py` 后会先出现启动菜单，可选择：
- 从头开始训练
- 从最新检查点继续训练
- 从指定检查点继续训练

## TensorBoard 训练图表

训练日志默认写入：`output/tensorboard/`。

训练完成后，在 `scripts/training` 目录执行：

```bash
tensorboard --logdir output/tensorboard --port 6006
```

浏览器访问：`http://localhost:6006`

可查看典型曲线：`loss`、`eval_loss`、`learning_rate`。

## 显存护栏与监控

训练时可并行执行以下命令监控显存：

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

脚本内置安全策略：
- `fp16=true`, `bf16=false`
- `per_device_train_batch_size=1`
- `gradient_checkpointing=true`
- `optim=paged_adamw_8bit`
- 每 100 步通过 callback 检查 `torch.cuda.memory_allocated()`
- 硬阈值：`vram_limit_gb=7.8`

## OOM 应急清单

1. 将 `max_seq_length` 从 `448` 下调到 `384`。
2. 增大 `gradient_accumulation_steps`，保持等效 batch size 稳定。
3. 适当增大 `save_steps` 与 `eval_steps`，减少峰值波动。
4. 确保没有其他 GPU 进程占用（`nvidia-smi`）。
5. 调用 `torch.cuda.empty_cache()` 后重启训练（建议直接新进程重启）。

## 与 `scripts/mas_blackboard/classifier.py` 集成

建议使用环境变量，便于无缝切换：

```bash
set ROUTER_MODEL_PATH=../models/Qwen3-1.7B
set ROUTER_LORA_PATH=scripts/training/output/adapter/adapter_model
```

推荐集成步骤：
1. 在 `ComplexityClassifier.__init__` 中调用 `scripts/training/src/utils.py` 里的 `load_qwen3_with_lora()`。
2. 在 `classify` 中使用同一套 Qwen3 prompt 模板组织输入。
3. 用 `repair_json_output()` 解析模型输出。
4. 将解析后的 JSON 字段（`tier`、`suggested_agents`）映射到 `RoutingDecision`。

## 输出格式约定

模型输出应为 JSON，包含以下键：
- `tier`
- `sub_type`
- `suggested_agents`
- `parallelism`
- `confidence_score`

## 显存占用预估

| 阶段 | 预估显存 |
|---|---|
| 预处理（token 长度检查） | ~1.0-2.0 GB |
| 训练（LoRA, fp16, gc, 8bit optimizer） | ~6.8-7.6 GB |
| 验证 | ~5.0-6.2 GB |
| 单样本推理 | ~4.6-5.8 GB |

