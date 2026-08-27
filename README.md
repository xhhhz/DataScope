# DataScope

基于 LangGraph 的智能数据分析 Agent。

上传 CSV / Excel / JSON 文件，用自然语言提问，Agent 自主规划分析步骤、调用工具执行代码，最终给出**思考过程 + 结论 + 交互式图表**。

```
你：各地区销售额总和是多少？画个柱状图

Agent：
  ● inspect_dataframe   检查数据结构
  ● python_repl         执行分析代码
  ● recommend_chart     决策图表类型
  ● create_chart        生成图表

  结论：华东 27,500 元遥遥领先，占总销售额 60.5%……
```

---

## 快速开始

**环境要求**：Python 3.10+（开发环境为 3.12）

```bash
git clone git@github.com:xhhhz/data_analyst_agent.git
cd data_analyst_agent
```

创建虚拟环境并安装依赖：

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

在项目根目录创建 `.env`：

```bash
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
CHART_OUTPUT_PATH=tmp/chart.png
CODE_EXEC_TIMEOUT=30
```

启动：

```bash
python -m src.app
```

浏览器打开 http://127.0.0.1:7860 即可使用。

> **必须从项目根目录以模块方式启动**（`python -m src.app`），
> 代码内部使用 `from src.xxx import` 的绝对导入，直接 `python src/app.py` 会报 `ModuleNotFoundError`。

---

## 配置说明

所有配置集中在 `.env`，由 `src/config.py` 统一加载，其他模块不直接读 `os.environ`。

| 变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | 模型 API 密钥 |
| `OPENAI_BASE_URL` | | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 任何 **OpenAI 兼容**端点都可以 |
| `OPENAI_MODEL` | | `qwen-max` | 模型名 |
| `CHART_OUTPUT_PATH` | | — | 图表默认输出路径 |
| `CODE_EXEC_TIMEOUT` | | `30` | 沙箱单次执行超时（秒） |
| `LANGCHAIN_TRACING_V2` | | — | 设为 `true` 开启 LangSmith 追踪 |

因为走的是 OpenAI 兼容协议，换模型只需改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL` 两项，代码无需改动。

> 若网络无法访问 `api.smith.langchain.com`，请把 `LANGCHAIN_TRACING_V2` 留空或设为 `false`，
> 否则每次调用都会尝试上报追踪数据并等待超时。

---

## 架构

采用**双层图**结构：外层负责意图路由，内层是自主编排的 ReAct Agent。

```
用户提问
   │
   ▼
┌──────────────────────────────────────┐
│  外层 LangGraph（src/agent/graph.py）  │
│                                      │
│   intent_router                      │
│       ├── 非分析请求 ──► fallback     │
│       └── 数据分析   ──► analyst      │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│  内层 ReAct Agent                     │
│  （create_react_agent）               │
│                                      │
│  自主循环：Reason → Act → Observe     │
│  按需调用 5 个工具，次数不固定          │
└───────────────┬──────────────────────┘
                │
                ▼
        思考过程 + 结论 + 图表
```

这样拆分的好处是：闲聊、无文件等场景在外层就被拦下，不必加载内层 Agent 那份包含全部工具描述的 System Prompt。

### 目录结构

```
src/
├── app.py                    # Gradio 前端入口，流式展示 Agent 过程
├── config.py                 # 配置中心（从 .env 加载）
├── llm.py                    # LLM 单例
├── agent/
│   ├── graph.py              # 外层图：节点编排 + 路由
│   ├── state.py              # AgentState 定义
│   ├── tools_definition.py   # 5 个工具 + 图表决策引擎
│   └── nodes/
│       ├── router.py         # 意图路由
│       └── fallback.py       # 非分析请求兜底
└── tools/
    ├── sandbox.py            # 子进程代码沙箱
    └── dataframe_utils.py    # 文件读取、校验、摘要
```

---

## Agent 工具

Agent 自主决定调用哪些、调用几次，顺序不是硬编码的。

| 工具 | 作用 |
|---|---|
| `inspect_dataframe` | 查看列名、类型、空值、统计量、前 3 行预览 |
| `python_repl` | 在沙箱中执行分析代码，数据预加载为 `df`，用 `print()` 输出 |
| `recommend_chart` | 决策图表类型，返回类型、列名、Plotly 配置 |
| `create_chart` | 执行画图代码，生成 Plotly 交互图 |
| `filter_data` | 按 pandas query 语法筛选数据子集 |

### 图表决策：规则引擎优先，LLM 兜底

`recommend_chart` 不直接问 LLM，而是先走确定性规则：

| 数据特征 | 推荐 |
|---|---|
| 含时间列且有数值列 | 折线图 |
| 两列数值、无类别列 | 散点图 |
| 三列以上数值且存在强相关 | 热力图 |
| 单列数值 | 直方图 |
| 类别 + 数值，且问的是占比（类别 ≤ 7） | 饼图 |
| 类别 + 数值，类别 > 10 | 横向柱状图 |
| 类别 + 数值，其他情况 | 柱状图 |
| 以上都不匹配 | 交给 LLM 判断 |

规则命中时结果是确定的：同样的数据和问题永远得到同样的图表类型，且不额外消耗一次 LLM 调用。

---

## 代码执行沙箱

LLM 生成的 Python 代码在**独立子进程**中执行（`src/tools/sandbox.py`），做了三件事：

- **崩溃隔离** —— 子进程异常不会影响主进程
- **超时控制** —— 超过 `CODE_EXEC_TIMEOUT` 强制终止，防止死循环
- **数据预注入** —— 沙箱自己加载数据并赋值给 `df`，LLM 生成的代码里没有机会指定任意文件路径

### ⚠️ 安全边界

**当前沙箱不是操作系统级隔离。** 子进程与主进程共用同一个操作系统用户，理论上可以访问文件系统和网络。它防的是"代码写错导致崩溃/卡死"，不是"代码存心作恶"。

**请勿直接暴露到公网。** 若要在不可信环境中运行，应把 `sandbox.py` 里的 `subprocess.run` 替换为容器化执行，例如：

```python
subprocess.run([
    "docker", "run", "--rm",
    "--network=none",       # 断网
    "--memory=256m",        # 限内存
    "--read-only",          # 只读文件系统
    "python:3.12-slim",
    "python", "/sandbox/script.py",
], ...)
```

或接入 E2B 等专为 AI Agent 设计的云沙箱服务。改动集中在一个函数内。

---

## 支持的数据与问题

**文件格式**：CSV、Excel（`.xlsx` / `.xls`）、JSON，单文件 ≤ 50 MB

**问题类型**：

- 统计查询 —— 总量、均值、排名
- 趋势分析 —— 时间序列变化
- 对比分析 —— 类别间差异
- 分布分析 —— 数据分布形态
- 相关分析 —— 变量间关系

---

## 已知限制

- 沙箱不具备操作系统级隔离（见上文「安全边界」）
- 对话历史存于内存（`MemorySaver`），进程重启即丢失
- 单次分析耗时主要来自 LLM 多轮往返，数据量大小影响很小
- 尚无自动化评估流水线，工具调用准确率与图表推荐准确率均未做定量测试

---

## 更多文档

架构细节、模块职责与完整数据流见 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)。
