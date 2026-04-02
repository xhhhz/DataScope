"""
路径一的 router 职责简化：
旧版需要区分 analyze/clarify/chitchat 三种意图。
路径一中 Agent 自己能处理 analyze 内部的所有决策，
router 只需要判断：是否是数据分析相关请求。
"""
from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.state import AgentState
from src.llm import llm

_SYSTEM_PROMPT = """\
判断用户输入是否与数据分析相关。
只返回以下两个词之一，不要输出任何其他内容：

ANALYZE  —— 用户想分析数据、查询数据、生成图表
FALLBACK —— 与数据分析完全无关的闲聊，或用户还未上传文件就提问
"""


def intent_router(state: AgentState) -> dict:
    """
    判断请求类型，结果写入 state["intent"]。
    """
    if not state.get("file_path"):
        return {"intent": "FALLBACK"}

    last_message = state["messages"][-1]

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=last_message.content),
    ])

    intent = response.content.strip().upper()
    if intent not in ("ANALYZE", "FALLBACK"):
        intent = "ANALYZE"

    return {"intent": intent}


def route_by_intent(state: AgentState) -> str:
    """
    条件边函数，返回值对应 graph.py 中 add_conditional_edges 的 key。
    """
    return state.get("intent", "ANALYZE")