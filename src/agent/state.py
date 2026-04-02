"""
路径一的 AgentState。
相比旧版大幅简化：code/executor 相关字段全部删除，
Agent 的中间状态由 LangGraph 内部的 ReAct 循环管理。
"""
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 对话历史，add_messages 确保多轮对话追加而非覆盖
    messages: Annotated[list, add_messages]

    # 数据上下文，由 app.py 在用户上传文件后写入
    file_path:      Optional[str]
    dataframe_info: Optional[str]

    # 每个 session 独立的图表输出路径，防止多用户并发覆盖
    chart_output_path: Optional[str]

    # 最终输出
    final_answer: Optional[str]