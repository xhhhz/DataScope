"""
处理无法分析的情况：
- 用户还未上传文件
- 与数据分析完全无关的闲聊
不调用 LLM，直接返回固定提示，节省 token。
"""
from langchain_core.messages import AIMessage
from src.agent.state import AgentState


def fallback_node(state: AgentState) -> dict:
    has_file = bool(state.get("file_path"))

    if not has_file:
        message = (
            "请先上传数据文件（支持 CSV / Excel / JSON），"
            "然后告诉我你想分析什么 😊"
        )
    else:
        message = (
            "我是专注于数据分析的 Agent，"
            "请上传数据文件并告诉我你想分析的问题。"
        )

    return {
        "final_answer": message,
        "messages":     [AIMessage(content=message)],
    }