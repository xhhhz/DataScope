"""
统一配置加载。
所有模块从这里取配置，不直接读 os.environ。
这样改配置只改一处，也方便测试时 mock。
"""
import os
from dotenv import load_dotenv

load_dotenv()  # 读取 .env 文件

class Config:
    OPENAI_API_KEY: str    = os.environ["OPENAI_API_KEY"]
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL")
    OPENAI_MODEL: str      = os.getenv("OPENAI_MODEL")
    CHART_OUTPUT_PATH: str = os.getenv("CHART_OUTPUT_PATH")
    # 修复问题6：删除从未使用的 MAX_RETRY_COUNT，避免误导
    CODE_EXEC_TIMEOUT: int = int(os.getenv("CODE_EXEC_TIMEOUT", "30"))


config = Config()