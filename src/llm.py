"""
LLM 单例。
所有节点都从这里拿同一个实例，避免重复初始化，
也方便将来统一切换模型（比如换成 Claude）。
"""
from langchain_openai import ChatOpenAI
from src.config import config

llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    temperature=0,
)