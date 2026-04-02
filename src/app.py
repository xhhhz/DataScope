"""
Gradio 前端（路径一适配版）。

与旧版的主要区别：
  1. 不再管理 Agent 内部状态（generated_code/retry_count 等字段消失）
  2. stream() 展示的是 Agent 的工具调用过程（ReAct 循环），而非固定节点
  3. 图表通过 gr.Plot 展示 Plotly Figure 对象
  4. 移除静态图（PNG）兜底，仅保留交互图（Plotly）
  5. 修复了旧版 df_summary 断链 Bug
"""
import os
import re
import uuid
from typing import Optional

import gradio as gr
import plotly.io as pio
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent.graph import agent_graph
from src.tools.dataframe_utils import validate_file, generate_df_summary


# ── 文件上传处理 ──────────────────────────────────────────────────────────

def process_upload(file) -> tuple[str, str, str]:
    """
    处理文件上传事件。

    返回三个值，分别写入：
      file_path_state  —— 文件路径（传给 Agent）
      df_summary_state —— 数据摘要（传给 Agent，注入 prompt）
      file_status      —— 展示给用户的状态文字
    """
    if file is None:
        return "", "", "⚠️ 未检测到文件，请重新上传"

    ok, err_msg = validate_file(file.name)
    if not ok:
        return "", "", f"❌ {err_msg}"

    try:
        summary = generate_df_summary(file.name)
        status  = f"✅ 文件加载成功\n\n```\n{summary}\n```"
        return file.name, summary, status
    except Exception as e:
        return "", "", f"❌ 文件解析失败：{e}"


# ── 核心对话函数 ──────────────────────────────────────────────────────────

def chat(
    user_message: str,
    file_path:    str,
    df_summary:   str,
    thread_id:    str,
    progress=gr.Progress(),
) -> tuple[str, Optional[go.Figure]]:
    """
    接收用户输入，调用 Agent，返回思考过程和图表。

    返回两个值：
      thought_output —— Agent 思考过程（Markdown 文字）
      plotly_output  —— Plotly Figure 对象（或 None）
    """
    if not user_message.strip():
        return "⚠️ 请输入问题", None

    progress(0, desc="🚀 启动 Agent...")

    # 每个 session 独立的图表路径，防止多用户并发时互相覆盖
    os.makedirs("tmp", exist_ok=True)
    chart_base       = f"tmp/chart_{thread_id}"
    html_output_path = f"{chart_base}.html"
    json_output_path = f"{chart_base}.json"

    # 清理上次遗留的图表文件
    for path in (html_output_path, json_output_path):
        if os.path.exists(path):
            os.remove(path)

    config = {"configurable": {"thread_id": thread_id}}

    state_input = {
        "messages":          [HumanMessage(content=user_message)],
        "file_path":         file_path or None,
        "dataframe_info":    df_summary or None,
        "chart_output_path": f"{chart_base}.png",
    }

    thought_logs: list[str]           = []
    final_chart:  Optional[go.Figure] = None

    progress(0.2, desc="🔍 Agent 分析中...")

    # ── 流式处理 Agent 输出 ──────────────────────────────────
    for chunk in agent_graph.stream(state_input, config=config):
        for node_name, node_state in chunk.items():
            logs, chart = _process_node_output(
                node_name,
                node_state,
                json_output_path,
            )
            thought_logs.extend(logs)
            if chart is not None:
                final_chart = chart

    progress(0.9, desc="📊 整理结果...")

    if not thought_logs:
        thought_logs = ["Agent 未产生任何输出，请重试"]

    # 兜底：流式输出未能检测到图表时，主动检查 JSON 文件是否已生成
    if final_chart is None and os.path.exists(json_output_path):
        final_chart = _load_plotly_figure(json_output_path)

    progress(1.0, desc="✅ 分析完成")

    return "\n\n---\n\n".join(thought_logs), final_chart


def _process_node_output(
    node_name:        str,
    node_state:       dict,
    json_output_path: str,
) -> tuple[list[str], Optional[go.Figure]]:
    """
    解析单个节点的输出，返回 (日志列表, Plotly Figure)。
    """
    logs:  list[str]           = []
    chart: Optional[go.Figure] = None

    # LangGraph 流式输出中某些节点的 node_state 可能为 None，跳过避免报错
    if node_state is None:
        return [], None

    if node_name == "intent_router":
        intent = node_state.get("intent", "unknown")
        emoji  = {"ANALYZE": "🔍", "FALLBACK": "⛔"}.get(intent, "❓")
        logs.append(f"{emoji} **意图识别**：`{intent}`")

    elif node_name == "analyst":
        messages = node_state.get("messages", [])
        for msg in messages:
            log_entries, msg_chart = _parse_agent_message(msg, json_output_path)
            logs.extend(log_entries)
            if msg_chart is not None:
                chart = msg_chart

    elif node_name == "fallback":
        answer = node_state.get("final_answer", "")
        logs.append(f"💬 **提示**\n\n{answer}")

    return logs, chart


def _parse_agent_message(
    msg,
    json_output_path: str,
) -> tuple[list[str], Optional[go.Figure]]:
    """
    解析 ReAct 循环中的单条消息。

    消息类型：
      AIMessage（含 tool_calls）—— Agent 决定调用某工具（Thought + Action）
      ToolMessage               —— 工具返回结果（Observation）
      AIMessage（无 tool_calls）—— Agent 最终结论（Final Answer）
    """
    logs:  list[str]           = []
    chart: Optional[go.Figure] = None

    if isinstance(msg, AIMessage):
        tool_calls = getattr(msg, "tool_calls", []) or []

        if tool_calls:
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                logs.append(_format_tool_call(tool_name, tool_args))
        else:
            # 最终结论：过滤模型插入的 markdown 图片语法，避免显示破损图标
            content = msg.content
            if content and content.strip():
                content = re.sub(r'!\[.*?\]\(.*?\)', '', content).strip()
                logs.append(f"📊 **最终结论**\n\n{content}")

    elif isinstance(msg, ToolMessage):
        tool_name = getattr(msg, "name", "tool")
        content   = msg.content or ""
        log, msg_chart = _format_tool_result(tool_name, content, json_output_path)
        logs.append(log)
        if msg_chart is not None:
            chart = msg_chart

    return logs, chart


def _format_tool_call(tool_name: str, tool_args: dict) -> str:
    """格式化工具调用日志（Thought + Action）。"""
    icons = {
        "inspect_dataframe": "🔎",
        "python_repl":       "🤔",
        "recommend_chart":   "🎨",
        "create_chart":      "📈",
        "filter_data":       "🔬",
    }
    labels = {
        "inspect_dataframe": "检查数据结构",
        "python_repl":       "执行分析代码",
        "recommend_chart":   "决策图表类型",
        "create_chart":      "生成图表",
        "filter_data":       "筛选数据",
    }
    icon  = icons.get(tool_name, "🔧")
    label = labels.get(tool_name, f"调用工具 {tool_name}")

    if tool_name == "python_repl":
        code = tool_args.get("code", "")
        return f"{icon} **Action：{label}**\n```python\n{code}\n```"
    elif tool_name == "create_chart":
        code = tool_args.get("chart_code", "")
        return f"{icon} **Action：{label}**\n```python\n{code}\n```"
    elif tool_name == "recommend_chart":
        question = tool_args.get("user_question", "")
        return f"{icon} **Action：{label}**\n> 分析问题：{question}"
    elif tool_name == "filter_data":
        conditions = tool_args.get("conditions", "")
        return f"{icon} **Action：{label}**\n> 筛选条件：`{conditions}`"
    else:
        return f"{icon} **Action：{label}**"


def _format_tool_result(
    tool_name:        str,
    content:          str,
    json_output_path: str,
) -> tuple[str, Optional[go.Figure]]:
    """格式化工具返回结果日志（Observation），图表生成时同时返回 Figure 对象。"""
    chart: Optional[go.Figure] = None

    if tool_name == "inspect_dataframe":
        log = f"✅ **Observation：数据结构**\n```json\n{content[:800]}\n```"

    elif tool_name == "python_repl":
        if content.startswith("执行失败"):
            log = f"⚠️ **Observation：执行出错**\n```\n{content}\n```"
        else:
            log = f"✅ **Observation：执行结果**\n```\n{content[:600]}\n```"

    elif tool_name == "recommend_chart":
        log = f"🎨 **Observation：图表推荐**\n```json\n{content}\n```"

    elif tool_name == "create_chart":
        if "成功" in content:
            log = f"✅ **Observation：图表已生成**"
            if os.path.exists(json_output_path):
                chart = _load_plotly_figure(json_output_path)
        else:
            log = f"⚠️ **Observation：图表生成失败**\n```\n{content}\n```"

    elif tool_name == "filter_data":
        log = f"✅ **Observation：筛选结果**\n```json\n{content[:600]}\n```"

    else:
        log = f"✅ **Observation**\n```\n{content[:400]}\n```"

    return log, chart


def _load_plotly_figure(json_path: str) -> Optional[go.Figure]:
    """从沙箱保存的 JSON 文件还原 Plotly Figure 对象，供 gr.Plot 直接展示。"""
    try:
        with open(json_path, encoding="utf-8") as f:
            content = f.read()
        return pio.from_json(content)
    except Exception as e:
        print(f"图表 JSON 加载失败：{e}")
        return None


# ── Gradio UI ─────────────────────────────────────────────────────────────

def _new_session():
    """重置会话状态，返回所有需要清空的组件值。"""
    return (
        str(uuid.uuid4()),  # 新 thread_id
        "",                 # file_path_state
        "",                 # df_summary_state
        "请上传文件...",     # file_status
        "*等待输入...*",     # thought_output
        None,               # plotly_output
    )


with gr.Blocks(title="智能数据分析师", theme=gr.themes.Soft()) as demo:

    # ── Session 级状态 ──────────────────────────────────────
    thread_id_state  = gr.State(lambda: str(uuid.uuid4()))
    file_path_state  = gr.State("")
    df_summary_state = gr.State("")

    # ── 页面标题 ────────────────────────────────────────────
    gr.Markdown("# 🤖 智能数据分析师 Agent")
    gr.Markdown(
        "上传数据文件（CSV / Excel / JSON，≤50MB），"
        "用自然语言提问，Agent 自动分析并选择最优图表。"
    )

    with gr.Row():

        # ── 左栏：输入区 ─────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            file_input = gr.File(
                label="📂 上传数据文件",
                file_types=[".csv", ".xlsx", ".xls", ".json"],
            )
            file_status = gr.Markdown("请上传文件...")

            user_input = gr.Textbox(
                label="💬 你的问题",
                placeholder=(
                    "例如：\n"
                    "· 哪个产品的销售额最高？\n"
                    "· 展示各月销售趋势\n"
                    "· 各地区销售占比是多少？"
                ),
                lines=4,
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 开始分析", variant="primary", scale=3)
                clear_btn  = gr.Button("🔄 新会话", scale=1)

            with gr.Accordion("📖 使用说明", open=False):
                gr.Markdown("""
**Agent 工具列表：**
- 🔎 `inspect_dataframe`：自动检查数据结构
- 🤔 `python_repl`：执行分析计算
- 🎨 `recommend_chart`：智能推荐图表类型
- 📈 `create_chart`：生成可视化图表
- 🔬 `filter_data`：按条件筛选数据

**支持的问题类型：**
- 统计查询：总量、均值、排名
- 趋势分析：时间序列变化
- 对比分析：类别间差异
- 分布分析：数据分布形态
- 相关分析：变量间关系
""")

        # ── 右栏：输出区 ─────────────────────────────────────
        with gr.Column(scale=2, min_width=500):

            thought_output = gr.Markdown(
                label="🧠 Agent 思考过程",
                value="*等待输入...*",
            )

            plotly_output = gr.Plot(label="📈 交互式图表")

    # ── 事件绑定 ──────────────────────────────────────────────

    file_input.upload(
        fn=process_upload,
        inputs=[file_input],
        outputs=[file_path_state, df_summary_state, file_status],
    )

    submit_btn.click(
        fn=lambda: ("⏳ **Agent 正在分析中，请稍候...**", None),
        inputs=[],
        outputs=[thought_output, plotly_output],
    ).then(
        fn=chat,
        inputs=[user_input, file_path_state, df_summary_state, thread_id_state],
        outputs=[thought_output, plotly_output],
    )

    user_input.submit(
        fn=lambda: ("⏳ **Agent 正在分析中，请稍候...**", None),
        inputs=[],
        outputs=[thought_output, plotly_output],
    ).then(
        fn=chat,
        inputs=[user_input, file_path_state, df_summary_state, thread_id_state],
        outputs=[thought_output, plotly_output],
    )

    clear_btn.click(
        fn=_new_session,
        inputs=[],
        outputs=[
            thread_id_state,
            file_path_state,
            df_summary_state,
            file_status,
            thought_output,
            plotly_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
