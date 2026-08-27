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

# 必须在任何 HTTP 客户端库初始化之前设置，确保 Gradio 健康检查绕过系统代理
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,0.0.0.0")

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

def _file_meta(path: str, summary: str) -> str:
    """
    从摘要首行提取「行 · 列 · 大小」，用于已上传态的等宽副标题。
    摘要首行形如：格式：CSV | 大小：2.3 MB | Shape：1204 行 × 8 列
    """
    rows = re.search(r"Shape：([\d,]+)\s*行", summary)
    cols = re.search(r"×\s*([\d,]+)\s*列", summary)
    size = re.search(r"大小：([\d.]+\s*\w+)", summary)

    parts = []
    if rows:
        parts.append(f"{int(rows.group(1).replace(',', '')):,} 行")
    if cols:
        parts.append(f"{cols.group(1)} 列")
    if size:
        parts.append(size.group(1))

    if parts:
        return " · ".join(parts)
    # 兜底：摘要格式变了也不至于空着
    return f"{os.path.getsize(path) / 1024 / 1024:.1f} MB"


def process_upload(file):
    """
    处理文件上传事件。

    返回五个值，分别写入：
      file_path_state  —— 文件路径（传给 Agent）
      df_summary_state —— 完整数据摘要（传给 Agent，注入 prompt）
      file_input       —— 上传区显隐（成功后收起虚线拖放区）
      file_info        —— 已上传信息条的 HTML
      replace_btn      —— 「更换」按钮显隐
    """
    def _fail(msg: str):
        # 失败时保持拖放区可见，错误信息就地显示在信息条上
        return (
            "", "",
            gr.update(visible=True),
            gr.update(visible=True, value=f'<div id="ds-fileinfo">{msg}</div>'),
            gr.update(visible=False),
        )

    if file is None:
        return _fail("⚠️ 未检测到文件，请重新上传")

    # gr.File 传带 .name 的对象；UploadButton(type="filepath") 传字符串路径。
    # 两种都接住，避免以后换组件时再踩一次。
    path = file if isinstance(file, str) else getattr(file, "name", "")

    ok, err_msg = validate_file(path)
    if not ok:
        return _fail(f"❌ {err_msg}")

    try:
        summary = generate_df_summary(path)
        info    = _fileinfo_html(os.path.basename(path), _file_meta(path, summary))
        return (
            path, summary,
            gr.update(visible=False),
            gr.update(visible=True, value=info),
            gr.update(visible=True),
        )
    except Exception as e:
        return _fail(f"❌ 文件解析失败：{e}")


# ── 核心对话函数 ──────────────────────────────────────────────────────────

def _message_text(content) -> str:
    """
    把 Chatbot 消息的 content 统一成纯文本。

    经前端回传后 content 不一定是字符串：Gradio 允许它是分段列表
    （文本片段 / 文件 / 组件混排），直接当字符串用会抛
    AttributeError: 'list' object has no attribute 'strip'。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(_message_text(part) for part in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("value") or "")
    return str(getattr(content, "text", "") or "")


def _last_user_text(history: list) -> str:
    """从对话历史里取回最后一条用户提问的文本。"""
    for msg in reversed(history or []):
        if isinstance(msg, dict):
            role, content = msg.get("role"), msg.get("content", "")
        else:                       # Gradio 也可能回传 Message 对象
            role, content = getattr(msg, "role", None), getattr(msg, "content", "")
        if role == "user":
            return _message_text(content)
    return ""


def _failure_text(err: Exception, done_logs: list) -> str:
    """
    把异常翻译成用户能看懂、并且知道下一步该干什么的提示。
    已经跑完的步骤保留在前面，方便判断卡在了哪一环。
    """
    name = type(err).__name__
    msg  = str(err)

    if "Connection" in name or "connection" in msg.lower() or "timed out" in msg.lower():
        tip = (
            "**连接模型服务失败**\n\n"
            "没能连上 API 端点，通常是网络波动或服务端暂时不可用。"
            "请点「开始分析」重试一次；如果连续失败，检查 `.env` 里的 "
            "`OPENAI_BASE_URL` 是否可访问。"
        )
    elif "AuthenticationError" in name or "401" in msg or "api key" in msg.lower():
        tip = "**API 密钥无效**\n\n请检查 `.env` 里的 `OPENAI_API_KEY`。"
    elif "RateLimit" in name or "429" in msg:
        tip = "**触发限流**\n\n请求过于频繁，稍等片刻再重试。"
    else:
        tip = f"**分析中断**\n\n`{name}`：{msg[:300]}"

    if done_logs:
        return "\n\n".join(done_logs) + "\n\n---\n\n" + tip
    return tip


def append_user_message(user_message: str, history: list):
    """
    提交后的第一步：立刻把用户提问上屏并清空输入框。

    这样做不只是为了体验 —— 空状态占位文字和 gr.Progress 的进度浮层都居中
    绝对定位，同时可见时会叠印成一团。先上屏一条消息，空状态消失，进度条
    就不会再和它抢同一块位置。
    """
    history = list(history or [])
    if not user_message.strip():
        return history, user_message          # 空输入：不清空，保留用户已敲的内容
    return history + [{"role": "user", "content": user_message}], ""


def chat(
    file_path:    str,
    df_summary:   str,
    thread_id:    str,
    history:      list,
    progress=gr.Progress(),
):
    """
    调用 Agent 跑完整分析，把结果追加进历史。

    用户提问已由 append_user_message 上屏，这里从 history 末尾取回，
    因此不再接收 user_message 参数（此时输入框已被清空）。

    返回三个值：
      history       —— 累积的对话消息列表（喂给 gr.Chatbot）
      plotly_output —— Plotly Figure 对象（或 None）
      aside_col     —— 图表栏显隐：没出图就整栏不占位

    注意：history 每次都返回新列表，不原地 append —— Gradio 的 State 靠
    对象引用判断是否更新，原地修改会导致界面不刷新。
    """
    history = list(history or [])

    user_message = _last_user_text(history)

    if not user_message.strip():
        return history, None, gr.update()

    # 没有数据文件就别去跑 Agent —— 原来会一路崩到 Gradio，
    # 界面上只显示一个没有任何说明的红色「错误」
    if not file_path:
        history = history + [{
            "role": "assistant",
            "content": "请先在上方上传数据文件（CSV / Excel / JSON），再开始分析。",
        }]
        return history, None, gr.update()

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
    # 任何一步抛异常（最常见的是 LLM 端点连接失败）都不能让整个调用崩掉：
    # 崩了 Gradio 只会在界面上显示一个没有任何说明的红色「错误」，
    # 用户既不知道原因也不知道能不能重试。
    try:
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
    except Exception as e:
        history = history + [{
            "role": "assistant",
            "content": _failure_text(e, thought_logs),
        }]
        return history, final_chart, gr.update(visible=final_chart is not None)

    progress(0.9, desc="📊 整理结果...")

    if not thought_logs:
        thought_logs = ["Agent 未产生任何输出，请重试"]

    # 兜底：流式输出未能检测到图表时，主动检查 JSON 文件是否已生成
    if final_chart is None and os.path.exists(json_output_path):
        final_chart = _load_plotly_figure(json_output_path)

    progress(1.0, desc="✅ 分析完成")

    # 用户提问已在 append_user_message 里上屏，这里只追加 Agent 的推理过程
    history = history + [
        {"role": "assistant", "content": "\n\n".join(thought_logs)},
    ]

    # 图表栏按需展开：没出图就保持隐藏，避免整栏空白占位
    return history, final_chart, gr.update(visible=final_chart is not None)


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
        logs.append(_step("intent_router", f"意图识别：{intent}",
                          err=(intent == "FALLBACK")))

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
                logs.append(f"**结论**\n\n{content}")

    elif isinstance(msg, ToolMessage):
        tool_name = getattr(msg, "name", "tool")
        content   = msg.content or ""
        log, msg_chart = _format_tool_result(tool_name, content, json_output_path)
        logs.append(log)
        if msg_chart is not None:
            chart = msg_chart

    return logs, chart


_TOOL_LABELS = {
    "inspect_dataframe": "检查数据结构",
    "python_repl":       "执行分析代码",
    "recommend_chart":   "决策图表类型",
    "create_chart":      "生成图表",
    "filter_data":       "筛选数据",
}


def _step(tool_name: str, desc: str, err: bool = False) -> str:
    """
    渲染运行流的一个步骤：状态点 + 工具名 + 描述。

    用 HTML 而非 markdown：状态点需要 CSS 上色。已实测 gr.Chatbot 的
    sanitize_html 会保留 class（只清洗 data-* 属性），所以样式能生效。

    注：规范要求的「耗时」未实现 —— analyst_node 用 agent.invoke() 一次性
    返回全部消息，拿不到单个工具的起止时间。需要改成 stream + 计时才能补上。
    """
    dot = '<span class="ds-dot is-err"></span>' if err else '<span class="ds-dot"></span>'
    return (
        f'<div class="ds-step">{dot}'
        f'<span class="ds-step-name">{tool_name}</span>'
        f'<span class="ds-step-desc">{desc}</span></div>'
    )


def _format_tool_call(tool_name: str, tool_args: dict) -> str:
    """格式化工具调用（Action）为运行流的一个步骤。"""
    label = _TOOL_LABELS.get(tool_name, f"调用工具 {tool_name}")
    head  = _step(tool_name, label)

    if tool_name == "python_repl":
        code = tool_args.get("code", "")
        return f"{head}\n\n```python\n{code}\n```"
    elif tool_name == "create_chart":
        code = tool_args.get("chart_code", "")
        return f"{head}\n\n```python\n{code}\n```"
    elif tool_name == "recommend_chart":
        question = tool_args.get("user_question", "")
        return f"{head}\n\n> 分析问题：{question}"
    elif tool_name == "filter_data":
        conditions = tool_args.get("conditions", "")
        return f"{head}\n\n> 筛选条件：`{conditions}`"
    else:
        return head


def _format_tool_result(
    tool_name:        str,
    content:          str,
    json_output_path: str,
) -> tuple[str, Optional[go.Figure]]:
    """格式化工具返回结果日志（Observation），图表生成时同时返回 Figure 对象。"""
    chart: Optional[go.Figure] = None

    if tool_name == "inspect_dataframe":
        log = f"```json\n{content[:800]}\n```"

    elif tool_name == "python_repl":
        if content.startswith("执行失败"):
            log = _step(tool_name, "执行出错", err=True) + f"\n\n```\n{content}\n```"
        else:
            log = f"```\n{content[:600]}\n```"

    elif tool_name == "recommend_chart":
        log = f"```json\n{content}\n```"

    elif tool_name == "create_chart":
        if "成功" in content:
            log = _step(tool_name, "图表已生成")
            if os.path.exists(json_output_path):
                chart = _load_plotly_figure(json_output_path)
        else:
            log = _step(tool_name, "图表生成失败", err=True) + f"\n\n```\n{content}\n```"

    elif tool_name == "filter_data":
        log = f"```json\n{content[:600]}\n```"

    else:
        log = f"```\n{content[:400]}\n```"

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

# 设计规范来源：DataScope-修正prompt-保留导航版.md
# 注意：Gradio 6 起 css / theme / head / js 参数均在 launch() 传入，
#      写在 Blocks() 里会被静默忽略。
CUSTOM_CSS = """
/* ═══ 设计 token ═══ */
:root, .light {
    --ds-teal:        #0E7C86;   /* 主色 */
    --ds-teal-soft:   #E8F3F4;   /* 选中/浅青底 */
    --ds-border:      #E3E8EA;
    --ds-text:        #1B2A32;
    --ds-text-sub:    #5B6B76;
    --ds-text-weak:   #93A2AB;
    --ds-hover:       #F0F3F4;
    --ds-bg-soft:     #FAFBFC;
    --ds-white:       #FFFFFF;
    --ds-r-sm:        6px;
    --ds-r:           8px;
    --ds-nav-w:       200px;
    --ds-nav-w-min:   56px;
    --ds-bar-h:       56px;
    --ds-aside-w:     400px;
}
.dark {
    --ds-teal:        #4FD1C5;
    --ds-teal-soft:   #12343B;
    --ds-border:      #24313A;
    --ds-text:        #E6EDF1;
    --ds-text-sub:    #9FB0BA;
    --ds-text-weak:   #6B7C87;
    --ds-hover:       #1A242C;
    --ds-bg-soft:     #131B21;
    --ds-white:       #0F171C;
}

/* ═══ 全局：锁死一屏，只有运行流和图表栏内部滚动 ═══ */
html, body { height: 100%; overflow: hidden; }
body, .gradio-container {
    font-family: 'Roboto Flex', system-ui, -apple-system, sans-serif !important;
    background: var(--ds-white) !important;
    color: var(--ds-text) !important;
}
.gradio-container {
    max-width: none !important; padding: 0 !important;
    height: 100vh !important; overflow: hidden !important;
}
/* Gradio 的 .app 包装层自带 max-width:1280px + margin:0 80px + padding:0 32px，
   会把通栏布局压窄；.gradio-container 上设 max-width 命中不到它 */
.gradio-container .app {
    max-width: none !important; width: 100% !important;
    margin: 0 !important; padding: 0 !important;
}
.gradio-container > div,
.gradio-container > div > div,
.gradio-container > div > div > div,
.gradio-container > div > div > div > div { height: 100% !important; min-height: 0 !important; }
footer { display: none !important; }

.ds-icon { font-size: 18px; line-height: 1; }

/* ═══ 骨架 ═══ */
#ds-shell {
    gap: 0 !important; align-items: stretch !important; flex-wrap: nowrap !important;
    height: 100vh !important; overflow: hidden !important;
}

/* ── 导航栏 200px，可折叠到 56px ── */
#ds-sidebar {
    flex: 0 0 var(--ds-nav-w) !important;
    max-width: var(--ds-nav-w) !important;
    background: var(--ds-white) !important;
    border-right: 1px solid var(--ds-border) !important;
    padding: 0 !important; gap: 0 !important;
    height: 100% !important; overflow: hidden !important;
    transition: flex-basis .18s ease, max-width .18s ease;
}
#ds-sidebar > * { padding-left: 0 !important; padding-right: 0 !important; }

/* 折叠态：由 JS 给 body 加 .ds-nav-collapsed，或窗口 <1280px 自动触发 */
#ds-shell.ds-nav-collapsed #ds-sidebar {
    flex-basis: var(--ds-nav-w-min) !important;
    max-width: var(--ds-nav-w-min) !important;
}
#ds-shell.ds-nav-collapsed .ds-nav-label,
#ds-shell.ds-nav-collapsed .ds-brand-name,
#ds-shell.ds-nav-collapsed #ds-new-analysis .ds-btn-text { display: none !important; }
#ds-shell.ds-nav-collapsed .ds-brand,
#ds-shell.ds-nav-collapsed .ds-nav-item,
#ds-shell.ds-nav-collapsed #ds-new-analysis { justify-content: center !important; }
#ds-shell.ds-nav-collapsed .ds-nav { padding: 8px !important; }
#ds-shell.ds-nav-collapsed #ds-new-analysis { margin: 8px !important; padding: 0 !important; }
/* 窗口 <1280px 自动折叠。选择器必须带 #ds-shell 提权：Gradio 生成的
   作用域副本 `.gradio-container ... .contain #ds-sidebar` 权重 (1,3,0)，
   会压过这里裸写的 `#ds-sidebar` (0,1,0)，媒体查询将完全失效。 */
@media (max-width: 1279px) {
    #ds-shell #ds-sidebar {
        flex-basis: var(--ds-nav-w-min) !important;
        max-width: var(--ds-nav-w-min) !important;
    }
    #ds-shell .ds-nav-label,
    #ds-shell .ds-brand-name,
    #ds-shell #ds-new-analysis .ds-btn-text { display: none !important; }
    #ds-shell .ds-brand,
    #ds-shell .ds-nav-item,
    #ds-shell #ds-new-analysis { justify-content: center !important; }
    #ds-shell .ds-nav { padding: 8px !important; }
    #ds-shell #ds-new-analysis { margin: 8px !important; padding: 0 !important; }
}

/* 品牌区：高 56px，与右侧顶栏对齐；图标用线条青色，不要深色方块底 */
.ds-brand {
    display: flex; align-items: center; gap: 10px;
    height: var(--ds-bar-h); padding: 0 12px;
    border-bottom: 1px solid var(--ds-border);
    box-sizing: border-box;
}
.ds-brand .ds-svg {
    font-size: 24px; color: var(--ds-teal); flex: 0 0 24px;
}
.ds-brand-name {
    font-size: 18px; font-weight: 500; line-height: 1.2;
    color: var(--ds-text); margin: 0; white-space: nowrap;
}

/* 「+ 新建分析」：导航栏顶部，白底青字描边，高 36px */
#ds-new-analysis {
    display: flex !important; align-items: center !important; gap: 8px !important;
    height: 36px !important; min-height: 36px !important;
    margin: 12px !important; padding: 0 12px !important;
    width: calc(100% - 24px) !important;
    background: var(--ds-white) !important;
    color: var(--ds-teal) !important;
    border: 1px solid var(--ds-border) !important;
    border-radius: var(--ds-r) !important;
    font-size: 14px !important; font-weight: 500 !important;
    box-shadow: none !important;
}
#ds-new-analysis:hover { background: var(--ds-hover) !important; border-color: var(--ds-teal) !important; }

.ds-nav { padding: 0 12px; display: flex; flex-direction: column; gap: 2px; }
.ds-nav-item {
    display: flex; align-items: center; gap: 10px;
    height: 40px; padding: 0 12px; border-radius: var(--ds-r);
    font-size: 14px; color: var(--ds-text-sub);
    cursor: pointer; box-sizing: border-box;
}
.ds-nav-item .ds-svg { color: var(--ds-text-weak); flex: 0 0 18px; }
.ds-nav-item:hover { background: var(--ds-hover); }
.ds-nav-item.is-active { background: var(--ds-teal-soft); color: var(--ds-teal); }
.ds-nav-item.is-active .ds-svg { color: var(--ds-teal); }

/* ═══ 内容区 ═══ */
/* flex-wrap 必须显式设为 nowrap：Gradio 的 .column 默认 flex-wrap:wrap，
   在 column 方向上一旦高度不够，装不下的子项会「另起一列」从顶部重新排，
   表现为顶栏正常但整个主区消失（被甩到屏幕右侧不可见处）。
   窗口高 ≥900px 时不会触发，容易漏测 —— 1280x718 下必现。 */
#ds-content {
    flex: 1 1 auto !important; min-width: 560px !important;
    padding: 0 !important; gap: 0 !important;
    height: 100% !important;
    display: flex !important; flex-direction: column !important;
    flex-wrap: nowrap !important;
    overflow: hidden !important;
}

/* 顶栏 56px，与品牌区等高 */
#ds-topbar {
    display: flex; align-items: center; justify-content: space-between;
    height: var(--ds-bar-h); padding: 0 20px;
    background: var(--ds-white);
    border-bottom: 1px solid var(--ds-border);
    flex: 0 0 var(--ds-bar-h);
    box-sizing: border-box;
}
.ds-topbar-title {
    font-size: 18px; font-weight: 500; line-height: 1.2;
    color: var(--ds-text); margin: 0;
}
.ds-topbar-sub {
    font-size: 12px; line-height: 1.3;
    color: var(--ds-text-sub); margin: 2px 0 0 0;
}
.ds-topbar-actions { display: flex; align-items: center; gap: 4px; }
.ds-icon-btn {
    width: 32px; height: 32px; border-radius: var(--ds-r-sm);
    display: flex; align-items: center; justify-content: center;
    color: var(--ds-text-sub); cursor: pointer; background: transparent;
    border: none;
}
.ds-icon-btn:hover { background: var(--ds-hover); color: var(--ds-teal); }

/* ═══ 主区 ═══ */
#ds-main {
    flex: 1 1 auto !important; min-height: 0 !important;
    gap: 0 !important; padding: 0 !important;
    align-items: stretch !important; flex-wrap: nowrap !important;
    overflow: hidden !important;
}
#ds-chat {
    flex: 1 1 auto !important; min-width: 0 !important; min-height: 0 !important;
    display: flex !important; flex-direction: column !important;
    flex-wrap: nowrap !important;   /* 同 #ds-content，防止高度不足时换列 */
    padding: 0 !important; gap: 0 !important;
    overflow: hidden !important;
    background: var(--ds-white) !important;
}

/* ── 数据源条 56px：虚线拖放区 ── */
#ds-datasource {
    flex: 0 0 auto !important;
    /* 信息条是 flex:1，若允许换行会把「更换」按钮挤到下一行 */
    flex-wrap: nowrap !important;
    align-items: center !important;
    padding: 12px 20px !important;
    gap: 0 !important;
    background: var(--ds-white) !important;
}
/* gr.File 的内层拖放区最小高 240px，必须逐层压平才能收进 56px 一条 */
#ds-upload {
    height: var(--ds-bar-h) !important; min-height: 0 !important;
    border: 2px dashed var(--ds-border) !important;
    border-radius: var(--ds-r) !important;
    background: var(--ds-bg-soft) !important;
    padding: 0 !important; overflow: hidden !important;
}
#ds-upload:hover { border-color: var(--ds-teal) !important; background: var(--ds-teal-soft) !important; }
#ds-upload .wrap {
    min-height: 0 !important; height: 100% !important;
    padding: 0 !important; background: transparent !important;
    flex-direction: row !important; align-items: center !important;
    justify-content: center !important; gap: 8px !important;
}
#ds-upload button.center { min-height: 0 !important; height: 100% !important; }
#ds-upload .icon-wrap { height: 16px !important; width: 16px !important; margin: 0 !important; }
#ds-upload .icon-wrap svg { width: 16px !important; height: 16px !important; }
/* Gradio 内置文案「将文件拖放到此处 / - 或 - / 点击上传」是裸文本节点，
   不在任何元素里，选择器选不中 —— 用 font-size:0 折叠掉，
   再在伪元素上恢复字号，注入规范要求的文案 */
#ds-upload .wrap { font-size: 0 !important; }
#ds-upload .wrap .or { display: none !important; }
#ds-upload .wrap::after {
    content: "拖放文件到这里，或点击上传";
    font-size: 13px; color: var(--ds-text-sub); white-space: nowrap;
}
#ds-upload .wrap::before {
    content: "CSV · Excel · JSON，最大 50MB";
    font-size: 12px; color: var(--ds-text-weak); white-space: nowrap;
    order: 3;
}
/* 已上传态：实心条 */
#ds-fileinfo {
    display: flex !important; align-items: center !important; gap: 10px !important;
    height: var(--ds-bar-h); padding: 0 14px;
    border: 1px solid var(--ds-border); border-radius: var(--ds-r);
    background: var(--ds-white); box-sizing: border-box;
}
#ds-fileinfo .ds-file-name { font-size: 13px; font-weight: 500; color: var(--ds-text); }
#ds-fileinfo .ds-file-meta {
    font-size: 12px; color: var(--ds-text-weak);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
#ds-fileinfo .ds-svg { color: var(--ds-teal); flex: 0 0 18px; }
/* 信息条撑满可用宽度，「更换」紧贴其右侧。
   原来给按钮加 margin-left:auto，会把它推到整行最右端、离信息条很远 */
#ds-fileinfo-wrap { flex: 1 1 auto !important; min-width: 0 !important; }
#ds-fileinfo { width: 100%; }
#ds-replace {
    flex: 0 0 auto !important;
    margin-left: 12px !important;
    background: transparent !important; border: none !important;
    color: var(--ds-teal) !important; box-shadow: none !important;
    font-size: 13px !important; padding: 0 !important;
    width: auto !important; min-width: 0 !important; height: auto !important;
}
#ds-replace:hover { text-decoration: underline; }

/* ── 运行流 ── */
#ds-thought {
    flex: 1 1 auto !important; min-height: 0 !important;
    overflow-y: auto !important;
    padding: 16px 20px !important;
    background: var(--ds-white) !important;
    border: none !important; box-shadow: none !important;
}
#ds-thought > .wrap, #ds-thought .bubble-wrap {
    height: 100% !important; max-height: none !important; background: transparent !important;
}
/* 空状态：主区内水平垂直双向居中 */
#ds-thought .placeholder-content, #ds-thought .placeholder {
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    height: 100% !important; text-align: center !important;
    border: none !important; background: transparent !important;
}
#ds-thought .message {
    border-radius: var(--ds-r) !important;
    border: 1px solid var(--ds-border) !important;
    box-shadow: none !important;
    font-size: 14px !important;
}
#ds-thought .user .message {
    background: var(--ds-teal-soft) !important;
    color: var(--ds-text) !important;
    border-color: transparent !important;
}
#ds-thought .bot .message { background: var(--ds-white) !important; }
#ds-thought pre { border-radius: var(--ds-r-sm) !important; font-size: 12px !important; }
#ds-thought::-webkit-scrollbar, #ds-thought .bubble-wrap::-webkit-scrollbar { width: 8px; }
#ds-thought::-webkit-scrollbar-thumb, #ds-thought .bubble-wrap::-webkit-scrollbar-thumb {
    background: var(--ds-border); border-radius: 4px;
}

/* 运行流时间线：状态点 + 工具名 */
.ds-step { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.ds-dot {
    width: 8px; height: 8px; border-radius: 50%; flex: 0 0 8px;
    background: var(--ds-teal);
}
.ds-dot.is-err { background: #C0392B; }
.ds-step-name {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px; color: var(--ds-teal);
}
.ds-step-desc { font-size: 13px; color: var(--ds-text-sub); }

/* ── 底部输入区 ── */
#ds-composer {
    flex: 0 0 auto !important;
    flex-wrap: nowrap !important;   /* 同上，防止 chips/输入框/按钮被拆到另一列 */
    padding: 12px 20px 16px 20px !important;
    gap: 8px !important;
    background: var(--ds-white) !important;
    border-top: 1px solid var(--ds-border) !important;
}
#ds-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.ds-chip {
    background: var(--ds-teal-soft); color: var(--ds-teal);
    font-size: 12px; padding: 4px 10px; border-radius: var(--ds-r-sm);
    border: none; white-space: nowrap;
}
#ds-question textarea {
    min-height: 64px !important; max-height: 96px !important;
    border-radius: var(--ds-r) !important;
    border-color: var(--ds-border) !important;
    font-size: 14px !important;
}
#ds-question textarea:focus { border-color: var(--ds-teal) !important; box-shadow: none !important; }
/* 两个按钮等宽等高 40px，gap 12px，位于输入框下方 */
#ds-btns { gap: 12px !important; flex-wrap: nowrap !important; width: 100% !important; }
#ds-btns > * { flex: 1 1 0 !important; }
#ds-submit, #ds-reset {
    height: 40px !important; min-height: 40px !important;
    border-radius: var(--ds-r) !important;
    font-size: 14px !important; font-weight: 500 !important;
    width: 100% !important; box-shadow: none !important;
}
#ds-submit {
    background: var(--ds-teal) !important; color: #fff !important;
    border: 1px solid var(--ds-teal) !important;
}
#ds-submit:hover { filter: brightness(1.06); }
#ds-reset {
    background: var(--ds-white) !important; color: var(--ds-text-sub) !important;
    border: 1px solid var(--ds-border) !important;
}
#ds-reset:hover { background: var(--ds-hover) !important; }

/* ── 图表栏 400px，无图时整栏隐藏 ── */
#ds-aside {
    flex: 0 0 var(--ds-aside-w) !important;
    max-width: var(--ds-aside-w) !important;
    min-height: 0 !important;
    display: flex !important; flex-direction: column !important;
    flex-wrap: nowrap !important;   /* 同上 */
    padding: 16px !important; gap: 10px !important;
    background: var(--ds-bg-soft) !important;
    border-left: 1px solid var(--ds-border) !important;
    overflow-y: auto !important;
}
.ds-aside-title {
    font-size: 13px !important; font-weight: 500 !important;
    color: var(--ds-text-sub) !important;
    background: transparent !important; border: none !important;
    padding: 0 !important; margin: 0 !important; min-height: 0 !important;
}
.ds-aside-title p { margin: 0 !important; }
/* 图表只保留一层容器，去掉 Gradio 默认的内层空白边框盒子 */
#ds-chart {
    flex: 1 1 auto !important; min-height: 240px !important;
    background: var(--ds-white) !important;
    border: 1px solid var(--ds-border) !important;
    border-radius: var(--ds-r) !important;
    padding: 8px !important; box-shadow: none !important;
}
#ds-chart .block, #ds-chart .form {
    border: none !important; background: transparent !important;
    box-shadow: none !important; padding: 0 !important;
}
"""

# launch(head=...) 注入：字体必须在 <head> 加载，CSS 里 @import 会被 Gradio 剥离
CUSTOM_HEAD = """
<link href="https://fonts.googleapis.com" rel="preconnect"/>
<link crossorigin href="https://fonts.gstatic.com" rel="preconnect"/>
<link href="https://fonts.googleapis.com/css2?family=Roboto+Flex:opsz,wght@8..144,400..700&display=swap" rel="stylesheet"/>
<script>
/* 导航栏折叠：状态存 localStorage，刷新后保持。
   注意：不能用 launch(js=...) —— 实测那个函数不会被执行，
   按钮会变成点击无反应的死控件。写在 head 里的 script 才可靠。 */
(function () {
    var KEY = 'ds-nav-collapsed';
    /* 状态类必须挂在 #ds-shell 而不是 body：Gradio 6 会把自定义 CSS 复制一份
       加上 `.gradio-container ... .contain ` 前缀，body 选择器加了前缀后永远
       匹配不到，同时那个高权重副本会压过原始规则，导致折叠完全失效。 */
    function shell() { return document.querySelector('#ds-shell'); }

    function restore() {
        var el = shell();
        if (!el) { setTimeout(restore, 100); return; }   // Gradio 异步挂载，等它出现
        if (localStorage.getItem(KEY) === '1') el.classList.add(KEY);
    }
    restore();

    // 事件委托：Gradio 会重绘 DOM，绑在具体元素上会失效
    document.addEventListener('click', function (e) {
        if (!e.target.closest || !e.target.closest('#ds-nav-toggle')) return;
        var el = shell();
        if (!el) return;
        var on = el.classList.toggle(KEY);
        localStorage.setItem(KEY, on ? '1' : '0');
    });
})();
</script>
"""

# 导航配置集中在一处，做完一个功能就把 enabled 改成 True。
# 规范硬要求：界面上不允许出现点击无反应的控件，所以只渲染 enabled 的项。
NAV_ITEMS = [
    {"id": "workbench", "icon": "activity",  "label": "分析工作台", "enabled": True},
    {"id": "history",   "icon": "history",   "label": "历史会话",   "enabled": False},
    {"id": "files",     "icon": "folder",    "label": "文件管理",   "enabled": False},
    {"id": "templates", "icon": "file-text", "label": "分析模板",   "enabled": False},
    {"id": "insights",  "icon": "bar-chart", "label": "洞察收藏",   "enabled": False},
]
NAV_VISIBLE = [i for i in NAV_ITEMS if i["enabled"]]

# 内联 SVG 图标（Lucide 线条风格，与规范里的图标命名一致）。
# 不用 Material Symbols 图标字体：它依赖 fonts.googleapis.com，实测该域名
# 响应约 3 秒且时通时不通，加载失败时连字会退化成 "monitoring" 这样的原始
# 文字直接显示在界面上。内联 SVG 无网络依赖，永远可用。
_ICON_PATHS = {
    "activity":  '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "history":   '<path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/>',
    "folder":    '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v5h5"/><path d="M8 13h8"/><path d="M8 17h8"/>',
    "bar-chart": '<path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/>',
    "panel":     '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/>',
}


def _svg(name: str, size: int = 18) -> str:
    """渲染一个内联 SVG 图标；stroke 用 currentColor，颜色交给 CSS 控制。"""
    return (
        f'<svg class="ds-svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f'{_ICON_PATHS.get(name, "")}</svg>'
    )


def _nav_html() -> str:
    rows = []
    for i, item in enumerate(NAV_VISIBLE):
        cls = "ds-nav-item is-active" if i == 0 else "ds-nav-item"
        rows.append(
            f'<div class="{cls}" title="{item["label"]}">'
            f'{_svg(item["icon"], 18)}'
            f'<span class="ds-nav-label">{item["label"]}</span></div>'
        )
    return "\n".join(rows)


BRAND_HTML = f"""
<div class="ds-brand">
  {_svg("activity", 24)}
  <p class="ds-brand-name">DataScope</p>
</div>
"""

NAV_HTML = f'<div class="ds-nav">{_nav_html()}</div>'

TOPBAR_HTML = f"""
<div id="ds-topbar">
  <div>
    <p class="ds-topbar-title">分析工作台</p>
    <p class="ds-topbar-sub">上传数据，用自然语言提问</p>
  </div>
  <div class="ds-topbar-actions">
    <button class="ds-icon-btn" id="ds-nav-toggle" title="折叠/展开导航栏">
      {_svg("panel", 18)}
    </button>
  </div>
</div>
"""

CHIPS_HTML = """
<div id="ds-chips">
  <span class="ds-chip">哪个产品的销售额最高？</span>
  <span class="ds-chip">展示各月销售趋势</span>
  <span class="ds-chip">各地区销售占比是多少？</span>
</div>
"""


def _fileinfo_html(name: str, meta: str) -> str:
    return (
        '<div id="ds-fileinfo">'
        + _svg("file-text", 18)
        + f'<span class="ds-file-name">{name}</span>'
        + f'<span class="ds-file-meta">{meta}</span>'
        + '</div>'
    )


def _new_session():
    """重置会话，返回所有需要清空的组件值。"""
    return (
        str(uuid.uuid4()),                       # thread_id
        "",                                      # file_path_state
        "",                                      # df_summary_state
        gr.update(visible=True),                 # 上传区（恢复虚线拖放态）
        gr.update(visible=False, value=""),      # 已上传信息条
        gr.update(visible=False),                # 「更换」按钮
        [],                                      # 对话历史
        None,                                    # 图表
        gr.update(visible=False),                # 图表栏整体隐藏
    )


with gr.Blocks(title="DataScope · 智能数据分析 Agent") as demo:

    # ── Session 级状态 ──────────────────────────────────────
    thread_id_state  = gr.State(lambda: str(uuid.uuid4()))
    file_path_state  = gr.State("")
    df_summary_state = gr.State("")

    with gr.Row(equal_height=False, elem_id="ds-shell"):

        # ── 导航栏 ──────────────────────────────────────────
        with gr.Column(elem_id="ds-sidebar", scale=0, min_width=56):
            gr.HTML(BRAND_HTML)
            new_btn = gr.Button("＋ 新建分析", elem_id="ds-new-analysis")
            gr.HTML(NAV_HTML)

        # ── 内容区 ──────────────────────────────────────────
        with gr.Column(elem_id="ds-content"):

            gr.HTML(TOPBAR_HTML)

            with gr.Row(equal_height=False, elem_id="ds-main"):

                # ── 中间：对话式工作流 ──────────────────────
                with gr.Column(elem_id="ds-chat"):

                    with gr.Row(elem_id="ds-datasource"):
                        file_input = gr.File(
                            file_types=[".csv", ".xlsx", ".xls", ".json"],
                            show_label=False,
                            height=56,
                            elem_id="ds-upload",
                        )
                        file_info    = gr.HTML(visible=False, elem_id="ds-fileinfo-wrap")
                        replace_btn  = gr.Button("更换", visible=False, elem_id="ds-replace")

                    chatbot = gr.Chatbot(
                        value=[],
                        show_label=False,
                        elem_id="ds-thought",
                        render_markdown=True,
                        placeholder=(
                            "### 开始你的第一次分析\n"
                            "上传数据并提问，分析过程和图表会显示在这里"
                        ),
                    )

                    with gr.Column(elem_id="ds-composer"):
                        gr.HTML(CHIPS_HTML)
                        user_input = gr.Textbox(
                            show_label=False,
                            placeholder="用自然语言描述你想分析什么",
                            lines=2,
                            max_lines=4,
                            elem_id="ds-question",
                        )
                        with gr.Row(elem_id="ds-btns"):
                            submit_btn = gr.Button("开始分析", elem_id="ds-submit")
                            reset_btn  = gr.Button("新会话",  elem_id="ds-reset")

                # ── 右侧：图表栏，无图时整栏不占位 ──────────
                with gr.Column(elem_id="ds-aside", visible=False) as aside_col:
                    gr.Markdown("交互式图表", elem_classes="ds-aside-title")
                    plotly_output = gr.Plot(show_label=False, elem_id="ds-chart")

    # ── 事件绑定 ──────────────────────────────────────────────

    file_input.upload(
        fn=process_upload,
        inputs=[file_input],
        outputs=[
            file_path_state, df_summary_state,
            file_input, file_info, replace_btn,
        ],
    )

    # 「更换」：切回虚线拖放态
    replace_btn.click(
        fn=lambda: (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ),
        inputs=[],
        outputs=[file_input, file_info, replace_btn],
    )

    # 提交入口有两个（按钮 / 回车），共用同一套处理链。
    # 两步：先把提问上屏（不显示进度），再跑 Agent。
    # 进度条限定只在对话区显示且用 minimal 样式 —— 默认的 full 会在每个
    # outputs 组件上各盖一层居中浮层，输入框上也来一条，页面上会同时出现两条。
    for trigger in (submit_btn.click, user_input.submit):
        trigger(
            fn=append_user_message,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input],
            show_progress="hidden",
        ).then(
            fn=chat,
            inputs=[
                file_path_state, df_summary_state,
                thread_id_state, chatbot,
            ],
            outputs=[chatbot, plotly_output, aside_col],
            show_progress="minimal",
            show_progress_on=[chatbot],
        )

    # 「＋ 新建分析」和「新会话」是同一个动作
    for btn in (new_btn, reset_btn):
        btn.click(
            fn=_new_session,
            inputs=[],
            outputs=[
                thread_id_state, file_path_state, df_summary_state,
                file_input, file_info, replace_btn,
                chatbot, plotly_output, aside_col,
            ],
        )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        # Gradio 6 起 theme / css / head / js 均在 launch() 传入
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="teal",
            neutral_hue="slate",
            radius_size="sm",
        ),
        css=CUSTOM_CSS,
        head=CUSTOM_HEAD,   # 字体 + 导航折叠脚本
    )
