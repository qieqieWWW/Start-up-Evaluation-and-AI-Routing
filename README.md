本项目聚焦于使用Kickstarter平台的项目数据建立初创项目的成功率预测模型并输出决策建议，同时模型将由独特的路由机制进行切换，保证资源的合理充分利用。

在docs里查看仓库基本使用流程

由于github不允许过大文件上传，我们需要从https://webrobots.io/kickstarter-datasets/ 下载原始数据集放到项目目录（目前需选择2025-12-18的csv，或自助修改data_process.py中的dataset_folder_name字段为原始集文件夹），然后用scripts/data_process.py对其进行清洗

添加pre-commit规则，配置文件在config/pre-commit-config.yaml，首次使用需（pip）安装pre-commit工具

## 脚本迁移说明（2026-03）

根目录业务脚本已统一迁移到 `scripts/` 目录，当前建议入口：
- 全流程入口：`main.py`
- M8规则适配：`scripts/m8_rule_adapter.py`
- M8→M3联调：`scripts/M8+M3基础调用脚本.py`
- M5测试引擎：`scripts/M5代码.py`
- M6状态日志：`scripts/M6状态管理与日志系统.py`

## 全流程运行（M2→M4→M3→M7）

### 运行前条件

1. 数据集目录存在（任一位置）：
	- `Kickstarter_2025-12-18T03_20_24_296Z`
	- `datasets/Kickstarter_2025-12-18T03_20_24_296Z`
2. Python依赖已安装（最低包含）：`pandas`、`numpy`、`scikit-learn`、`gymnasium`。
3. 若需M7真实LLM推理：设置 `DEEPSEEK_API_KEY`。

### 运行命令

```bash
/Users/qieqieqie/Desktop/Start-up-Evaluation-and-AI-Routing/.venv/bin/python main.py
```

## M7接入真实DeepSeek API（Prompt工程）

已支持在 `scripts/m7/m7_demo.py` 中走真实LLM推理链路（M8风险判定 → M7专家路由 → DeepSeek专家输出）。

### 1）配置环境变量

```bash
export DEEPSEEK_API_KEY="你的deepseek_api_key"
# 可选：默认 deepseek-chat
export DEEPSEEK_MODEL="deepseek-chat"
```

### 2）运行

```bash
/usr/bin/python3 scripts/m7/m7_demo.py
```

说明：
- 若未配置 `DEEPSEEK_API_KEY`，脚本会自动跳过真实LLM调用，仅保留M8→M7路由演示。
- 若缺少 `matplotlib`，脚本会跳过M7可视化，但不影响核心路由与LLM调用。

### 3）团队协作（push到仓库给组员）

请不要把真实 API Key 提交到仓库。仓库只保留 `.env.example` 模板，每位组员在本地创建自己的 `.env`。

```bash
cp .env.example .env
# 然后编辑 .env，填入自己的 DEEPSEEK_API_KEY
```

仓库已在 `.gitignore` 忽略 `.env`，正常 `git add .` 不会提交本地密钥。
