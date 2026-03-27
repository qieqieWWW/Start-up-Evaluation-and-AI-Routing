# M7 第一层上下文窗口（L1）

新增了 `scripts/m7/m7_context_analyzer.py`，用于构建第一层上下文窗口：
- 来源：用户直接输入 + 用户上传文件片段
- 处理方式：不做语义加工，原样透传
- 用途：直接嵌入 prompt，让 LLM 优先响应用户当下问题

关键接口：
- `build_layer1_context(user_input, uploaded_snippets)`
- `render_layer1_for_prompt(layer1_context)`

最小演示：
```bash
/Users/qieqieqie/Desktop/Start-up-Evaluation-and-AI-Routing/.venv/bin/python scripts/m7/m7_layer1_context_demo.py
```

M7 推理链路已经支持可选参数：
- `run_expert_llm_inference(..., user_input="...", uploaded_snippets=[...])`

## M7 第二层会话上下文窗口（L2）

新增会话上下文层，支持两种策略：
- `sliding_window`：保留最近 N 轮对话摘要
- `summary_buffer`：历史摘要缓冲 + 最近 N 轮窗口摘要合并

用途：
- 保持对话连贯性
- 支持指代消解（例如“这个问题”“上次提到的预算”）

关键接口：
- `build_layer2_context(conversation_turns, max_turns, summary_buffer, strategy)`
- `render_layer2_for_prompt(layer2_context)`

最小演示：
```bash
/Users/qieqieqie/Desktop/Start-up-Evaluation-and-AI-Routing/.venv/bin/python scripts/m7/m7_layer2_context_demo.py
```

推理入口新增可选参数：
- `conversation_turns`
- `summary_buffer`
- `session_max_turns`
- `session_strategy`（`sliding_window` 或 `summary_buffer`）

## M7 第三层用户历史画像（L3）

新增用户画像层，来源于历史评估记录、偏好、行业标签，并通过本地向量检索（RAG 形式）召回相关记录。

实现位置：
- `scripts/m7/m7_profile_rag.py`：轻量向量检索器（可替换为真实向量数据库）
- `scripts/m7/m7_context_analyzer.py`：`build_layer3_context(...)` 与 `render_layer3_for_prompt(...)`

默认画像库：
- `config/m7_user_profile_records.json`

个性化路由信号（示例）：
- `dominant_risk_appetite`
- `top_preferences`
- `common_industry_tags`

最小演示：
```bash
/Users/qieqieqie/Desktop/Start-up-Evaluation-and-AI-Routing/.venv/bin/python scripts/m7/m7_layer3_context_demo.py
```

推理入口新增可选参数：
- `user_id`
- `profile_db_path`
- `profile_top_k`

画像自动写入（新增）：
- 当调用 `run_expert_llm_inference(...)` 且传入 `user_id` 时，默认会把本轮问询自动追加到画像库
- 可通过 `auto_profile_log=False` 关闭自动写入

新增接口：
- `append_profile_record(record, db_path=None)`
- `save_profile_records(records, db_path=None)`
- `infer_risk_appetite_from_text(text, fallback="")`

自动写入最小演示（无需 API Key）：
```bash
/Users/qieqieqie/Desktop/Start-up-Evaluation-and-AI-Routing/.venv/bin/python scripts/m7/m7_profile_autolog_demo.py
```

## 意图识别引擎（Intent Recognition Engine）

新增能力：将用户非结构化问题转换为结构化路由指令。

实现位置：
- `scripts/m7/m7_intent_engine.py`
- `config/m7_intent_library.json`

技术路径：
- Prompt Engineering（Few-Shot）输出结构化 JSON
- 语义相似度（Embedding风格）匹配意图库作为辅助证据
- 复杂输入输出 `reasoning_summary`（简明推理摘要）

输出字段示例：
- `primary_intent`
- `sub_intent`
- `urgency`
- `required_experts`
- `missing_info`
- `confidence_score`

## 历史轨迹追踪与融合（Historical Trajectory Integration）

新增能力：
- 会话内记忆（短期）：实体识别与代词解析、状态机维护
- 跨会话记忆（长期）：结合用户历史记录检索与画像摘要

实现位置：
- `scripts/m7/m7_trajectory_manager.py`

状态机示例：
- `Waiting_For_Financials` -> 用户提供财务信息 -> `Analyzing_Financials`
- 识别IP风险输入 -> `Analyzing_IP`

路由联动：
- `scripts/m7/m7_router.py` 现已将意图识别结果与轨迹状态用于动态路由打分。

最小演示：
```bash
/Users/qieqieqie/Desktop/Start-up-Evaluation-and-AI-Routing/.venv/bin/python scripts/m7/m7_intent_trajectory_demo.py
```

## LLM-Blender（PairRanker + GenFuser）

新增模块：
- `scripts/m7/m7_blender.py`

核心流程：
- PairRanker：对专家候选输出进行两两比较与全局排序
- GenFuser：融合排序后的候选，生成统一决策稿

推理入口（新增）：
- `run_expert_llm_inference_with_blender(...)`

说明：
- 若有 `DEEPSEEK_API_KEY`，可启用 LLM GenFuser
- 若无 API Key，自动回退规则融合（仍可离线演示）

离线演示（无需 API Key）：
```bash
/Users/qieqieqie/Desktop/Start-up-Evaluation-and-AI-Routing/.venv/bin/python scripts/m7/m7_blender_demo.py
```

## M7 第四层全局知识库（L4）

新增全局知识层，来源于行业标准、风险模型定义、历史成功案例，采用静态知识库检索，提供专业基准（Grounding）。

实现位置：
- `scripts/m7/m7_global_kb.py`：静态知识库检索器（关键词+轻量向量相似度）
- `scripts/m7/m7_context_analyzer.py`：`build_layer4_context(...)` 与 `render_layer4_for_prompt(...)`

默认知识库：
- `config/m7_global_knowledge_base.json`

最小演示：
```bash
/Users/qieqieqie/Desktop/Start-up-Evaluation-and-AI-Routing/.venv/bin/python scripts/m7/m7_layer4_context_demo.py
```

推理入口新增可选参数：
- `global_kb_path`
- `global_kb_top_k`

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

## 启用小模型路由

1. 复制配置：`cp .env.example .env`，然后将 `USE_REAL_SMALL_MODEL` 改为 `true`
2. 验证集成：`python scripts/verify_small_model.py`
3. 端到端测试：`python test_integration.py --input "跨境医疗项目..."`

## UI 启动与使用（Mac / Windows）

测试界面脚本位置：`scripts/app_test_ui.py`

界面包含三个模块：
- 聊天窗口（用户与评估系统）
- 控制台日志（原始输出与步骤日志）
- 结构化评级面板（tier、置信度、专家建议等）

### 1）环境准备

- 需要安装 `streamlit`
- 建议使用你当前项目环境（例如 Airouting）

Mac（conda）示例：

```bash
conda activate Airouting
python -c "import streamlit; print(streamlit.__version__)"
```

Windows（conda）示例：

```bash
conda activate Airouting
python -c "import streamlit; print(streamlit.__version__)"
```

如果未安装：

```bash
conda install -n Airouting -c conda-forge -y streamlit
```

### 2）启动命令

Mac（项目根目录执行）：

```bash
streamlit run scripts/app_test_ui.py --server.headless true --server.port 8501
```

Windows（项目根目录执行）：

```bash
streamlit run scripts/app_test_ui.py --server.headless true --server.port 8501
```

启动后默认访问：
- `http://localhost:8501`

### 3）怎么用

1. 打开页面后，在左侧聊天框输入项目描述。
2. 右侧会同步展示结构化评级与日志。
3. 侧边栏可切换推理模式：`规则引擎` 或 `小模型`。
4. 切换模式后点击“应用模式并重建分类器”。

### 4）常见问题

- 页面显示规则路径、没有小模型：
	- 在侧边栏切到 `小模型`，并点击“应用模式并重建分类器”。
	- 若小模型加载失败，会自动回退规则引擎。

- 端口占用（8501）：
	- 可改端口，例如 `--server.port 8502`。

- 命令找不到 streamlit：
	- 说明当前终端不在目标环境，请先 `conda activate Airouting`。
