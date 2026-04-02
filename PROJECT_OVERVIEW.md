# 智能数据分析师 Agent — 项目文档

## 目录

1. [项目简介](#项目简介)
2. [目录结构](#目录结构)
3. [整体架构](#整体架构)
4. [模块详解](#模块详解)
5. [一次完整的数据流](#一次完整的数据流)

---

## 项目简介

本项目是一个基于 **LangGraph + Gradio** 的智能数据分析 Agent。

用户上传 CSV / Excel / JSON 数据文件后，用自然语言提问（例如"哪个地区销售额最高？"），Agent 会自主决定调用哪些工具完成分析，并生成交互式可视化图表，最终以"思考过程 + 结论 + 图表"的形式展示结果。

**技术栈：**

| 层次 | 使用技术 |
|------|----------|
| 前端界面 | Gradio 4.x |
| Agent 编排 | LangGraph（`create_react_agent` + `StateGraph`） |
| 大语言模型 | OpenAI `gpt-4o-mini`（通过 LangChain） |
| 数据处理 | Pandas 2.x |
| 可视化 | Plotly（交互图）/ Matplotlib（静态图兜底） |
| 代码执行 | subprocess 子进程沙箱 |

---

## 目录结构

```
data_analyst_agent/
├── .env                          # 环境变量（API Key、超时等配置）
├── requirements.txt              # Python 依赖列表
├── PROJECT_OVERVIEW.md           # 本文档
└── src/
    ├── app.py                    # Gradio 前端入口，负责界面和流式输出
    ├── config.py                 # 统一配置中心（从 .env 加载）
    ├── llm.py                    # LLM 单例（ChatOpenAI 实例）
    │
    ├── agent/
    │   ├── graph.py              # LangGraph 图：节点编排 + 路由逻辑
    │   ├── state.py              # AgentState 类型定义（对话上下文）
    │   ├── tools_definition.py   # Agent 可调用的 5 个工具
    │   └── nodes/
    │       ├── router.py         # 意图路由（ANALYZE / FALLBACK）
    │       └── fallback.py       # 非分析请求的兜底回复
    │
    └── tools/
        ├── sandbox.py            # 子进程代码沙箱执行器
        └── dataframe_utils.py    # 文件读取、校验、摘要生成
```

---

## 整体架构

下图描述了用户发出一条消息后，系统内部的处理路径：

```
用户在 Gradio 界面输入问题
          │
          ▼
┌─────────────────────────────────────────────┐
│              app.py（Gradio 前端）            │
│                                             │
│  上传文件 → validate_file() + 生成摘要       │
│  提问     → 调用 agent_graph.stream()        │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│         LangGraph 外层图（graph.py）          │
│                                             │
│   intent_router                             │
│       ├─ 无文件 ──────────► fallback_node   │
│       ├─ 闲聊无关 ─────────► fallback_node  │
│       └─ 数据分析请求 ──────► analyst_node   │
└──────────────────────┬──────────────────────┘
                       │ analyst_node 内部
                       ▼
┌─────────────────────────────────────────────┐
│       ReAct Agent（create_react_agent）       │
│                                             │
│  Agent 自主循环，按需调用以下工具：            │
│  ① inspect_dataframe  了解数据结构            │
│  ② python_repl        执行分析计算            │
│  ③ recommend_chart    决策图表类型            │
│  ④ create_chart       生成可视化图表          │
│  ⑤ filter_data        按条件筛选数据          │
└──────────────────────┬──────────────────────┘
                       │ python_repl / create_chart
                       ▼
┌─────────────────────────────────────────────┐
│          子进程沙箱（sandbox.py）              │
│  LLM 生成的 Python 代码在此隔离执行            │
│  • 超时保护（默认 30 秒）                     │
│  • df 变量预注入，LLM 无法访问任意路径         │
│  • 使用 sys.executable 保证虚拟环境一致性      │
└─────────────────────────────────────────────┘
```

**关键设计：外层图只负责路由，内层 ReAct 负责推理**

- 外层图（`graph.py`）：只做一件事——判断是否进入分析流程。
- 内层 ReAct Agent：拿到问题后，自主决定调哪些工具、调多少次、按什么顺序，直到得出结论。

---

## 模块详解

### `src/config.py` — 配置中心

从 `.env` 文件加载所有配置，项目里任何地方需要配置都从这里取，不直接读 `os.environ`。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥（必须设置） | 无 |
| `OPENAI_MODEL` | 使用的模型名称 | `gpt-4o-mini` |
| `CHART_OUTPUT_PATH` | 图表默认输出路径 | `tmp/chart_output.png` |
| `CODE_EXEC_TIMEOUT` | 子进程执行超时（秒） | `30` |

---

### `src/agent/state.py` — 对话状态

定义了贯穿整个对话过程的"共享数据"，每次调用 Agent 都会读写这里：

```python
class AgentState(TypedDict):
    messages:          list        # 完整对话历史（LangGraph 自动追加）
    file_path:         Optional[str]   # 已上传文件的磁盘路径
    dataframe_info:    Optional[str]   # 数据摘要文字（注入 prompt）
    chart_output_path: Optional[str]   # 本 session 的图表存储路径
    final_answer:      Optional[str]   # Agent 的最终文字结论
```

每个用户 session 有独立的 `thread_id`（UUID），不同用户的 state 互不干扰。

---

### `src/agent/nodes/router.py` — 意图路由

每次用户提问，先经过这里判断要不要进入分析流程：

- **未上传文件**：直接返回 `FALLBACK`，不消耗 token
- **已上传文件**：调用 LLM 判断，若是数据分析相关则返回 `ANALYZE`，否则 `FALLBACK`

---

### `src/agent/nodes/fallback.py` — 兜底回复

处理两种情况：未上传文件、与数据分析无关的闲聊。直接返回固定提示文字，**不调用 LLM**，节省费用。

---

### `src/agent/graph.py` — 图编排

把所有节点连接成一张图，定义执行顺序：

```
[入口] intent_router
           ├─ ANALYZE  → analyst_node → [结束]
           └─ FALLBACK → fallback_node → [结束]
```

`analyst_node` 内部使用 `create_react_agent` 构建 ReAct 循环。每次调用会生成一个随机 `thread_id` 传给内层 Agent，确保多用户并发时互不污染。同时把文件路径和图表路径都注入到系统提示词中，让 LLM 知道数据在哪里、图表要存到哪里。

---

### `src/agent/tools_definition.py` — 工具集（核心）

定义了 Agent 可以调用的 5 个工具：

#### 工具 ① `inspect_dataframe(file_path)`
了解数据结构。返回列名、数据类型、空值情况、数值列的统计量（min/max/mean/std）、类别列的样本值，以及前 3 行预览。

这是 Agent 拿到新数据后**必须第一步调用**的工具。

#### 工具 ② `python_repl(code, file_path)`
执行分析代码。在子进程沙箱中运行，`df` 变量已预加载，代码必须用 `print()` 输出结果。**只做计算，不画图。**

#### 工具 ③ `recommend_chart(file_path, analysis_result, user_question)`
决定用什么图表。采用**规则引擎优先、LLM 兜底**的混合策略：

| 规则 | 触发条件 | 推荐图表 |
|------|----------|----------|
| 1 | 数据含时间列 + 数值列 | 折线图 |
| 2 | 仅两列数值，无类别列 | 散点图 |
| 3 | ≥3 列数值且有强相关（r>0.7） | 热力图 |
| 4 | 仅一列数值，无类别列 | 直方图 |
| 5a | 类别列（≤7种）+ 用户问占比 | 饼图 |
| 5b | 类别列（>10种）+ 数值列 | 横向柱状图 |
| 5c | 其余类别+数值组合 | 柱状图 |
| 兜底 | 以上均不符合 | 交给 LLM 判断 |

返回结果包含：推荐图表类型、使用的可视化库、x/y 轴列名、推荐理由、Plotly 配置。

#### 工具 ④ `create_chart(chart_code, file_path, chart_output_path)`
执行画图代码，代码中必须创建名为 `fig` 的 Plotly Figure 对象。图表保存为 HTML 文件（交互式）。

#### 工具 ⑤ `filter_data(file_path, conditions)`
用 pandas `.query()` 语法筛选数据子集，返回匹配行数、统计量和前 5 行预览。

---

### `src/tools/sandbox.py` — 代码沙箱

工具 ② 和工具 ④ 的底层执行引擎，负责安全地运行 LLM 生成的 Python 代码。

**安全机制：**

| 机制 | 作用 |
|------|------|
| 子进程隔离 | 代码崩溃不影响主进程 |
| 超时保护 | 默认 30 秒，防止死循环 |
| df 预注入 | 数据由外部加载后注入脚本，LLM 无法指定任意文件路径 |
| `sys.executable` | 保证使用当前虚拟环境的 Python，依赖包不会找不到 |
| 非交互后端 | Matplotlib 强制使用 `Agg`，无 GUI 依赖 |

执行的完整脚本结构：
```
① 环境初始化（warnings、pandas）
② 可视化库导入（plotly 或 matplotlib，按需加载）
③ 数据预加载（df = pd.read_csv / read_excel / read_json）
④ LLM 生成的代码
⑤ 图表自动保存代码
```

---

### `src/tools/dataframe_utils.py` — 数据工具

| 函数 | 功能 |
|------|------|
| `validate_file(path)` | 校验文件：存在性 → 格式 → 大小（≤50MB）→ 可读性 → 非空 |
| `load_dataframe(path)` | 统一入口，自动识别 CSV / Excel / JSON 并读取 |
| `generate_df_summary(path)` | 生成注入 Agent Prompt 的文字摘要（最多展示 50 列） |

**CSV 读取**自动按顺序尝试编码：`utf-8` → `gbk` → `utf-8-sig` → `latin-1`，兼容中文 Excel 导出的 CSV。

`generate_df_summary()` 摘要内容包含：文件格式、大小、行列数、每列的类型和统计信息、前 3 行预览。当列数超过 50 时自动截断并提示，防止超出 LLM 上下文窗口。

---

### `src/app.py` — Gradio 前端

**界面布局：**
- 左栏：文件上传 + 问题输入 + 操作按钮 + 使用说明（折叠）
- 右栏：Agent 思考过程（Markdown）+ 交互图（Plotly HTML）+ 静态图（PNG 兜底）

**会话管理：**
每个用户在页面加载时分配一个独立 `thread_id`（UUID），图表文件以 `chart_{thread_id}.html` 命名存储，点击"新会话"按钮后重置全部状态并生成新 `thread_id`，实现多用户并发互不干扰。

**流式输出解析（三类消息）：**

| 消息类型 | 含义 | 展示内容 |
|----------|------|----------|
| `AIMessage`（含 tool_calls） | Agent 决定调用某工具 | 工具名 + 关键参数（代码高亮） |
| `ToolMessage` | 工具返回结果 | 执行结果或图表 |
| `AIMessage`（无 tool_calls） | Agent 的最终结论 | 文字分析结论 |

---

## 一次完整的数据流

以"展示各月销售趋势"为例，完整路径如下：

```
① 用户上传 sales.csv
   └─ validate_file()       检查格式/大小/可读性
   └─ generate_df_summary() 生成摘要 → 存入 df_summary_state

② 用户输入"展示各月销售趋势"
   └─ HumanMessage 加入 messages
   └─ agent_graph.stream() 启动

③ intent_router
   └─ 有文件 + 分析相关 → ANALYZE

④ analyst_node（ReAct 循环）
   ├─ [Thought] 需要先了解数据结构
   ├─ [Action]  inspect_dataframe → 发现"月份"列和"销售额"列
   ├─ [Thought] 有时间列，需要计算各月汇总
   ├─ [Action]  python_repl → 按月求和，print 输出结果
   ├─ [Thought] 需要画图展示趋势
   ├─ [Action]  recommend_chart → 规则1命中：时间列+数值列 → 折线图
   ├─ [Action]  create_chart → 生成 Plotly 折线图，保存为 HTML
   └─ [Final]   输出文字结论："1月销售额最高，达到..."

⑤ 前端解析输出
   ├─ 思考过程 → 右栏 Markdown（实时流式展示）
   ├─ HTML 图   → 右栏 Plotly 交互图
   └─ PNG 图    → 右栏静态图（Plotly 失败时兜底）
```
