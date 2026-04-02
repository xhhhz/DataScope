"""
数据文件的读取、校验与摘要生成。

支持格式：CSV / Excel(.xlsx .xls) / JSON
被以下模块调用：
  - src/agent/tools_definition.py 中的 inspect_dataframe 工具
  - src/app.py 中的 process_upload 函数
"""
import os
import re
import json
import numpy as np
import pandas as pd


# ── 常量配置 ──────────────────────────────────────────────────────────────

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}

# 文件大小上限：50 MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# 摘要中展示的预览行数
SUMMARY_PREVIEW_ROWS = 3

# 类别列展示的样本数量
CATEGORY_SAMPLE_COUNT = 5

# 修复问题8：摘要最多展示的列数，超出时截断并提示，防止超出 LLM 上下文窗口
MAX_COLS_IN_SUMMARY = 50


# ── 对外暴露的主要函数 ────────────────────────────────────────────────────

def validate_file(file_path: str) -> tuple[bool, str]:
    """
    校验上传文件是否合法。

    检查顺序：存在性 → 扩展名 → 大小 → 可读性 → 非空
    返回：(是否合法, 错误信息)，合法时错误信息为空字符串。
    """
    # 1. 文件是否存在
    if not os.path.exists(file_path):
        return False, "文件不存在，请重新上传"

    # 2. 扩展名是否支持
    ext = _get_extension(file_path)
    if ext not in SUPPORTED_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"不支持的文件类型 `{ext}`，目前支持：{supported}"

    # 3. 文件大小
    size_bytes = os.path.getsize(file_path)
    if size_bytes == 0:
        return False, "文件为空，请检查后重新上传"
    if size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = size_bytes / 1024 / 1024
        return False, f"文件过大（{size_mb:.1f} MB），上限为 50 MB"

    # 4. 是否可以正常读取
    try:
        df = load_dataframe(file_path)
    except Exception as e:
        return False, f"文件读取失败，请检查格式是否正确：{e}"

    # 5. 是否为空表
    if df.empty:
        return False, "文件中没有数据行，请检查内容"
    if len(df.columns) == 0:
        return False, "文件中没有列，请检查格式"

    return True, ""


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    根据扩展名自动选择读取方式，返回 DataFrame。
    是所有模块读取数据的统一入口，避免各处重复实现读取逻辑。
    """
    ext = _get_extension(file_path)

    if ext == ".csv":
        return _read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return _read_excel(file_path)
    elif ext == ".json":
        return _read_json(file_path)
    else:
        raise ValueError(f"不支持的文件类型：{ext}")


def generate_df_summary(file_path: str) -> str:
    """
    生成注入 Agent prompt 的数据摘要字符串。

    摘要包含：
      - 文件基本信息（格式、大小、行列数）
      - 每列的类型、空值数、统计量或样本值
      - 前 N 行预览

    设计原则：
      - 数值列给统计量（min/max/mean），帮助 LLM 理解数据范围
      - 类别列给样本值和唯一数，帮助 LLM 理解可用的筛选值
      - 控制总长度，避免超出 LLM 上下文窗口
    """
    df       = load_dataframe(file_path)
    ext      = _get_extension(file_path).upper().lstrip(".")
    size_mb  = os.path.getsize(file_path) / 1024 / 1024

    # 修复问题8：列数过多时只展示前 MAX_COLS_IN_SUMMARY 列，避免摘要超出 LLM 上下文窗口
    cols_to_show = list(df.columns[:MAX_COLS_IN_SUMMARY])
    truncated    = len(df.columns) > MAX_COLS_IN_SUMMARY

    col_lines = []
    for col in cols_to_show:
        line = _describe_column(df[col], col)
        col_lines.append(line)

    if truncated:
        col_lines.append(
            f"  ... （共 {len(df.columns)} 列，仅展示前 {MAX_COLS_IN_SUMMARY} 列，其余列可通过工具进一步查询）"
        )

    col_info = "\n".join(col_lines)
    preview  = df.head(SUMMARY_PREVIEW_ROWS).to_string(index=False)

    return (
        f"格式：{ext} | 大小：{size_mb:.1f} MB | "
        f"Shape：{df.shape[0]} 行 × {df.shape[1]} 列\n\n"
        f"列信息：\n{col_info}\n\n"
        f"前 {SUMMARY_PREVIEW_ROWS} 行预览：\n{preview}"
    )


# ── 内部辅助函数 ──────────────────────────────────────────────────────────

def _get_extension(file_path: str) -> str:
    """提取并统一小写扩展名。"""
    return os.path.splitext(file_path)[1].lower()


def _read_csv(file_path: str) -> pd.DataFrame:
    """
    读取 CSV，自动处理编码。
    UTF-8 失败后回退 GBK，兼容中文 Excel 导出的 CSV。
    """
    for encoding in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    # 最后兜底：让 pandas 自行推断
    return pd.read_csv(file_path, encoding="latin-1")


def _read_excel(file_path: str) -> pd.DataFrame:
    """
    读取 Excel 第一个 Sheet。
    使用 openpyxl 引擎处理 .xlsx，xlrd 处理旧版 .xls。
    """
    ext = _get_extension(file_path)
    engine = "openpyxl" if ext == ".xlsx" else "xlrd"
    try:
        return pd.read_excel(file_path, sheet_name=0, engine=engine)
    except Exception:
        # engine 推断失败时让 pandas 自动选择
        return pd.read_excel(file_path, sheet_name=0)


def _read_json(file_path: str) -> pd.DataFrame:
    """
    读取 JSON，支持两种常见格式：
      records : [{"a":1,"b":2}, {"a":3,"b":4}]   ← 最常见
      columns : {"a":[1,3], "b":[2,4]}
    """
    with open(file_path, encoding="utf-8") as f:
        raw = json.load(f)

    # 判断顶层结构
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    elif isinstance(raw, dict):
        # 检查是否是 {列名: [值列表]} 的结构
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, list):
            return pd.DataFrame(raw)
        else:
            # 单条记录的 dict，包装成单行 DataFrame
            return pd.DataFrame([raw])
    else:
        raise ValueError("JSON 格式不支持，请使用 records 或 columns 格式")


def _describe_column(series: pd.Series, col_name: str) -> str:
    """
    为单列生成描述字符串。
    数值列：给统计量；类别列：给样本值和唯一数。
    """
    dtype    = series.dtype
    null_cnt = int(series.isna().sum())
    null_str = f"，空值 {null_cnt} 个" if null_cnt > 0 else ""

    if pd.api.types.is_numeric_dtype(dtype):
        valid = series.dropna()
        if len(valid) == 0:
            return f"  - {col_name} ({dtype})：全部为空"
        return (
            f"  - {col_name} ({dtype})"
            f"：min={valid.min():.4g}"
            f"，max={valid.max():.4g}"
            f"，mean={valid.mean():.4g}"
            f"，std={valid.std():.4g}"
            f"{null_str}"
        )

    elif pd.api.types.is_datetime64_any_dtype(dtype):
        valid = series.dropna()
        return (
            f"  - {col_name} (datetime)"
            f"：{valid.min()} ~ {valid.max()}"
            f"{null_str}"
        )

    else:
        # object / category / bool 等
        n_unique = int(series.nunique())
        samples  = series.dropna().unique()[:CATEGORY_SAMPLE_COUNT].tolist()
        sample_str = " / ".join(str(s) for s in samples)
        more = f"...等共 {n_unique} 种" if n_unique > CATEGORY_SAMPLE_COUNT else f"（共 {n_unique} 种）"
        return (
            f"  - {col_name} ({dtype})"
            f"：{sample_str} {more}"
            f"{null_str}"
        )