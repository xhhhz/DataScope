"""
路径一：单 Agent + 工具调用。
用 create_react_agent 替代手动编排的多节点图。
Router 只做一件事：判断是否进入 Agent 循环。
"""
import uuid  # 修复问题1：用于生成内层 Agent 的唯一 thread_id，避免并发污染

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.agent.nodes.router import intent_router, route_by_intent
from src.agent.nodes.fallback import fallback_node
from src.agent.tools_definition import ANALYST_TOOLS
from src.llm import llm
from src.config import config  # 修复问题4：用于获取 CHART_OUTPUT_PATH 默认值

_AGENT_SYSTEM_PROMPT = """\
你是一个专业的数据分析师 Agent。

你有以下工具，按推荐顺序使用：

① inspect_dataframe
   拿到新数据时第一步必须调用，了解列名和数据类型。

② python_repl
   执行计算和分析，必须用 print() 输出结果。
   只做计算，不画图。

③ recommend_chart
   决定画图之前必须调用，不允许自己猜图表类型。
   根据返回的 recommended_chart 和 plotly_template 字段来写画图代码。

④ create_chart
   根据 recommend_chart 的推荐结果生成图表。
   代码中必须创建名为 fig 的 plotly Figure 对象。

⑤ filter_data
   需要聚焦某个数据子集时使用。

判断是否需要画图：
  需要   ← 用户明确要求 / 数据有趋势和对比关系 / 图比文字更清晰
  不需要 ← 用户只问单个数值 / 是否类问题 / 简单计数

当前数据文件路径：{file_path}
图表输出路径：{chart_output_path}
"""
# 修复问题4：在提示词中加入 chart_output_path，让 LLM 知道图表应保存到哪里，避免 session 隔离失效


def _make_analyst_agent(file_path: str, chart_output_path: str):
    """根据当前文件路径和图表输出路径创建 ReAct Agent 实例。"""
    system = SystemMessage(
        content=_AGENT_SYSTEM_PROMPT.format(
            file_path=file_path or "未上传",
            chart_output_path=chart_output_path,  # 修复问题4：注入图表路径
        )
    )
    return create_react_agent(
        model=llm,
        tools=ANALYST_TOOLS,
        prompt=system,  # 修复：LangGraph 0.6.x 将参数名改为 prompt（旧版为 state_modifier / messages_modifier）
    )


def analyst_node(state: AgentState) -> dict:
    """
    调用 ReAct Agent 处理分析请求。
    Agent 内部自主决定调用哪些工具、调用几次。
    """
    agent  = _make_analyst_agent(
        file_path=state.get("file_path", ""),
        chart_output_path=state.get("chart_output_path", config.CHART_OUTPUT_PATH),  # 修复问题4：传入 session 级图表路径
    )
    # 修复问题1：每次调用生成唯一 thread_id，避免多用户并发时内层 Agent 记忆互相污染
    inner_config = {"configurable": {"thread_id": f"inner_{uuid.uuid4().hex}"}}
    result = agent.invoke({"messages": state["messages"]}, inner_config)

    # 返回本轮新产生的全部消息（工具调用 AIMessage + 工具结果 ToolMessage + 最终回答）。
    # 原先只返回 result["messages"][-1]，导致界面完全拿不到工具调用过程，
    # app.py 里解析 tool_calls / ToolMessage 的分支从来不会被触发。
    new_messages = result["messages"][len(state["messages"]):]

    last_message = result["messages"][-1]
    return {
        "messages":     new_messages or [last_message],
        "final_answer": last_message.content,
    }


def build_graph():
    """
    组装外层图（Router → Agent / Fallback）。

    外层图只负责路由，真正的分析逻辑在 analyst_node 内的 ReAct 循环里。
    """
    builder = StateGraph(AgentState)

    builder.add_node("intent_router", intent_router)
    builder.add_node("analyst",       analyst_node)
    builder.add_node("fallback",      fallback_node)

    builder.set_entry_point("intent_router")

    builder.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "ANALYZE":  "analyst",
            "FALLBACK": "fallback",
        },
    )

    builder.add_edge("analyst",  END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=MemorySaver())


agent_graph = build_graph()