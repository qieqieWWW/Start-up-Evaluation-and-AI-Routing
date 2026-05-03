# 项目 HF 数据集清理与生成 — 快速说明

目标

- 将系统运行产生的原始日志 / analysis_result JSON 归一化、脱敏并生成为可用于 HuggingFace 的训练数据集（arrow 或 jsonl）。

仅包含的三份脚本（位于 scripts/M14）

1) scripts/M14/trajectory_pipeline.py

   - 功能：归一化与脱敏原始日志（.json/.jsonl/.log），生成标准化的 .jsonl 记录（字段包括 id, raw_content, raw_parsed, final, raw_hash, metadata）。
   - 用法示例：python3 scripts/M14/trajectory_pipeline.py --input-dir <run_dir> --output-dir <normalized_dir>
2) scripts/M14/prepare_hf_dataset.py

   - 功能：读取规范化的 jsonl 记录，进行二次脱敏、清洗（clean_jsonl_dir），并保存为 HuggingFace Dataset（arrow）或合并 jsonl。会重新计算 raw_hash 并标记匿名化字段。
   - 用法示例（生成 arrow + jsonl）：python3 scripts/M14/prepare_hf_dataset.py --jsonl-dir data/normalized_logs --out-local data/hf_dataset --test-size 0 --out-format both
   - 注意：此脚本会备份原始 jsonl（.bak）并跳过 malformed 行。
3) scripts/M14/auto_prepare.py

   - 功能：上层编排。优先直接 ingest analysis_report_*.json；否则调用 trajectory_pipeline 对多次运行归一化，再调用 prepare_hf_dataset 生成最终数据集。
   - 用法示例（常用）：
     - 如果已有 analysis_report JSON：
       python3 scripts/M14/auto_prepare.py --log-dir analysis_reports --out-local data/hf_dataset --test-size 0
     - 如果有多个 run 目录（logs/ 下）：
       python3 scripts/M14/auto_prepare.py --log-dir logs --normalized-dir data/normalized_logs --out-local data/hf_dataset --test-size 0

清理与脱敏要点

- clean_jsonl_dir: 移除注释行（// 或 #）、修复并备份原始文件（.bak），避免 malformed JSON 导致失败。该步骤由 prepare_hf_dataset 自动执行。
- 脱敏流程：trajectory_pipeline 与 prepare_hf_dataset 均会替换邮箱、电话、IP、UUID、银行卡号等为占位标记（例如 [EMAIL], [PHONE], [IP], [CARD]）。prepare_hf_dataset 在保存前会再次脱敏并重新计算 raw_hash。

如何检查与调试输出

- 合并 jsonl 输出路径：data/hf_dataset/jsonl 或者 data/hf_dataset/ 下的 arrow 存档。
- 快速预览第一条记录（Python）:
  python3 - <<'PY'
  import json, pathlib
  p=pathlib.Path('data/hf_dataset/jsonl')
  for f in sorted(p.glob('*.jsonl')):
  with f.open('r',encoding='utf-8') as fh:
  print(next(fh))
  break
  PY

常见问题与建议

- 标签（final）缺失：如果 final（fused_summary/actions）为空，建议用 modules.M7.intent_result.reasoning_summary 或 modules.M8.risk_reasons 作为弱标签，或进行人工标注。
- 样本量：文本生成任务需要大量有标签样本；样本少时可先做自监督或分类任务。
- 小样本运行：使用 --test-size 0 以避免切分导致训练集为空。
- 脱敏验证：上线前请对脱敏结果人工抽查，确保合规性。
