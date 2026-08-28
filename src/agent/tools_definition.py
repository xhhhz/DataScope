"""
Agent 可调用的全部工具。
工具是路径一的核心，替代了旧版的 code_gen / executor / formatter 节点。

工具列表：
  inspect_dataframe  —— 了解数据结构（通常第一步调用）
  python_repl        —— 执行分析代码
  recommend_chart    —— 决策最优图表类型（画图前必须调用）
  create_chart       —— 生成可视化图表
  filter_data        —— 按条件筛选数据子集
"""
import json
import re
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.config import config
from src.tools.sandbox import run_code_in_sandbox


# ── 工具1：检查数据结构 ───────────────────────────────────────────────────

@tool
def inspect_dataframe(file_path: str) -> str:
    """
    查看数据文件的完整结构信息。

    当你拿到一份新数据，或者不确定列名时，第一步必须调用此工具。
    返回：列名、数据类型、空值情况、数值列统计量、前3行预览。
    """
    try:
        df = _load_df(file_path)

        col_info = {}
        for col in df.columns:
            dtype  = str(df[col].dtype)
            nulls  = int(df[col].isna().sum())
            # 修复：列值含嵌套 dict/list（不可哈希）时，转为字符串再计算唯一数
            try:
                unique = int(df[col].nunique())
            except TypeError:
                unique = int(df[col].astype(str).nunique())

            if pd.api.types.is_numeric_dtype(df[col]):
                col_info[col] = {
                    "type":   dtype,
                    "nulls":  nulls,
                    "unique": unique,
                    "min":    float(df[col].min()),
                    "max":    float(df[col].max()),
                    "mean":   round(float(df[col].mean()), 4),
                    "std":    round(float(df[col].std()), 4),
                }
            else:
                # 修复：列值含嵌套 dict/list 时，转为字符串再取样本
                try:
                    samples = df[col].dropna().unique()[:5].tolist()
                except TypeError:
                    samples = df[col].astype(str).dropna().unique()[:5].tolist()
                # 修复：样本值超长时截断，防止嵌套结构序列化后撑爆上下文
                samples = [
                    str(s)[:100] + "..." if len(str(s)) > 100 else s
                    for s in samples
                ]
                col_info[col] = {
                    "type":    dtype,
                    "nulls":   nulls,
                    "unique":  unique,
                    "samples": samples,
                }

        result = {
            "shape":   list(df.shape),
            "columns": col_info,
            "preview": df.head(3).to_dict(orient="records"),
        }
        # 修复：最终输出超长时整体截断，防止超出模型上下文窗口
        output = json.dumps(result, ensure_ascii=False, indent=2)
        if len(output) > 8000:
            output = output[:8000] + "\n... (内容过长已截断，请通过 python_repl 工具进一步探索数据)"
        return output

    except Exception as e:
        return f"错误：{e}"


# ── 工具2：执行 Python 分析代码 ───────────────────────────────────────────

@tool
def python_repl(code: str, file_path: str) -> str:
    """
    执行 Python 代码进行数据分析和计算。

    规则：
    - 数据已预加载为变量 df，直接使用
    - 必须用 print() 输出结果，返回值即为 print 的内容
    - 此工具只做计算，不画图；画图请使用 create_chart 工具
    - 如果需要先聚合数据再画图，在这里计算好，把结果 print 出来

    示例代码：
        result = df.groupby('product')['sales'].sum().sort_values(ascending=False)
        print(result.to_string())
    """
    result = run_code_in_sandbox(
        code=code,
        file_path=file_path,
        chart_decision={"needs_chart": False},
        png_output_path=config.CHART_OUTPUT_PATH,
        html_output_path=config.CHART_OUTPUT_PATH.replace(".png", ".html"),
        timeout=config.CODE_EXEC_TIMEOUT,
    )

    if result.success:
        output = result.stdout.strip()
        return output if output else "代码执行成功，但没有 print 任何输出"
    else:
        # 截取最关键的错误信息（最后5行），避免太长
        error_lines = result.stderr.strip().splitlines()
        key_error   = "\n".join(error_lines[-5:])
        return f"执行失败：\n{key_error}"


# ── 工具3：图表类型决策 ───────────────────────────────────────────────────

@tool
def recommend_chart(
    file_path:       str,
    analysis_result: str,
    user_question:   str,
) -> str:
    """
    根据真实数据特征和分析结果，推荐最合适的图表类型。

    在调用 create_chart 之前，必须先调用此工具。
    不允许自己猜图表类型，必须基于此工具的返回结果画图。

    参数：
        file_path       : 数据文件路径
        analysis_result : 你用 python_repl 分析后得到的结果文本
        user_question   : 用户的原始问题

    返回包含以下字段的 JSON：
        recommended_chart : 图表类型
        viz_lib           : 可视化库（固定为 plotly）
        x_col / y_col     : 建议使用的列名
        reason            : 推荐理由
        plotly_template   : plotly 配置建议
    """
    try:
        df       = _load_df(file_path)
        features = _extract_data_features(df)
        result   = _decide_chart_type(features, analysis_result, user_question)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        fallback = {
            "recommended_chart": "bar",
            "viz_lib":           "plotly",
            "reason":            f"特征提取失败，使用默认柱状图。错误：{e}",
            "plotly_template":   {"template": "plotly_white"},
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)


# ── 工具4：生成可视化图表 ─────────────────────────────────────────────────

@tool
def create_chart(
    chart_code:        str,
    file_path:         str,
    chart_output_path: str,
) -> str:
    """
    执行画图代码，生成并保存可视化图表。

    调用前提：必须已经调用过 recommend_chart，根据其推荐结果写 chart_code。

    chart_code 规则：
    - 数据已预加载为变量 df
    - 必须创建名为 fig 的 plotly Figure 对象
    - 不要调用 fig.show()，图表会自动保存
    - 使用 plotly.express（import as px）或 plotly.graph_objects（import as go）

    示例代码：
        fig = px.bar(df, x='product', y='sales',
                     title='各产品销售额对比',
                     template='plotly_white',
                     text_auto=True)
    """
    html_path = chart_output_path.replace(".png", ".html")

    result = run_code_in_sandbox(
        code=chart_code,
        file_path=file_path,
        chart_decision={"needs_chart": True, "viz_lib": "plotly"},
        png_output_path=chart_output_path,
        html_output_path=html_path,
        timeout=config.CODE_EXEC_TIMEOUT,
    )

    if result.success and result.html_path:
        return f"图表生成成功，已保存至：{result.html_path}"
    elif result.success:
        return (
            "代码执行成功，但未检测到图表文件。"
            "请确认代码中创建了名为 fig 的 plotly Figure 对象。"
        )
    else:
        error_lines = result.stderr.strip().splitlines()
        key_error   = "\n".join(error_lines[-5:])
        return f"图表生成失败：\n{key_error}"


# ── 工具5：数据筛选 ───────────────────────────────────────────────────────

@tool
def filter_data(file_path: str, conditions: str) -> str:
    """
    按条件筛选数据，返回筛选结果的摘要。

    当用户想聚焦某个子集时使用，例如"只看华东地区的数据"。

    conditions 使用 pandas query 语法，示例：
        "region == '华东' and sales > 1000"
        "date >= '2024-01' and product in ['A', 'B']"

    返回：筛选后的行数、基本统计量、前5行预览。
    """
    try:
        df       = _load_df(file_path)
        filtered = df.query(conditions)

        if filtered.empty:
            return f"筛选条件 `{conditions}` 没有匹配到任何数据，请检查条件是否正确"

        result = {
            "matched_rows":     len(filtered),
            "total_rows":       len(df),
            "match_percentage": round(len(filtered) / len(df) * 100, 1),
            "summary":          filtered.describe().to_dict(),
            "preview":          filtered.head(5).to_dict(orient="records"),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"筛选失败：{e}\n请检查列名和条件语法是否正确"


# ── 工具集合（注册到 Agent）──────────────────────────────────────────────

ANALYST_TOOLS = [
    inspect_dataframe,
    python_repl,
    recommend_chart,
    create_chart,
    filter_data,
]


# ── 内部辅助函数（不暴露为工具）─────────────────────────────────────────

def _load_df(file_path: str) -> pd.DataFrame:
    """统一的数据加载入口，支持 CSV / Excel / JSON。"""
    from src.tools.dataframe_utils import load_dataframe
    return load_dataframe(file_path)


def _extract_data_features(df: pd.DataFrame) -> dict:
    """提取数据统计特征，作为图表决策的客观依据。"""
    numeric_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols   = df.select_dtypes(include=["object"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # 检测字符串列中的时间格式
    time_like_cols = []
    for col in object_cols:
        samples = df[col].dropna().head(5).astype(str).tolist()
        if _looks_like_datetime(samples):
            time_like_cols.append(col)

    # 数值列间的相关性
    strong_correlations = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                if abs(r) > 0.7:
                    strong_correlations.append({
                        "col1": corr.columns[i],
                        "col2": corr.columns[j],
                        "r":    round(float(r), 3),
                    })

    # 类别列的唯一值数量
    category_info = {
        col: {
            "n_unique": int(df[col].nunique()),
            "samples":  df[col].dropna().unique()[:5].tolist(),
        }
        for col in object_cols
    }

    return {
        "row_count":           len(df),
        "numeric_cols":        numeric_cols,
        "object_cols":         object_cols,
        "datetime_cols":       datetime_cols,
        "time_like_cols":      time_like_cols,
        "has_time_series":     len(datetime_cols) > 0 or len(time_like_cols) > 0,
        "category_info":       category_info,
        "strong_correlations": strong_correlations,
        "has_strong_corr":     len(strong_correlations) > 0,
    }


def _looks_like_datetime(samples: list) -> bool:
    """判断字符串样本是否符合常见日期格式。"""
    patterns = [
        r"\d{4}-\d{2}",
        r"\d{4}/\d{2}/\d{2}",
        r"\d{4}-\d{2}-\d{2}",
        r"Q[1-4]\s\d{4}",
        r"\d{4}年\d{1,2}月",
    ]
    matches = sum(
        1 for s in samples
        if any(re.search(p, str(s)) for p in patterns)
    )
    return matches >= len(samples) * 0.6


# 用户在提问里直接点名的图表类型。放在规则引擎最前面，优先级最高 ——
# 用户都明说"画个饼图"了，再由规则去猜是本末倒置。
_EXPLICIT_CHART = [
    ("pie",       ["饼图", "饼状图", "pie"]),
    ("scatter",   ["散点图", "scatter"]),
    ("histogram", ["直方图", "histogram"]),
    ("heatmap",   ["热力图", "热图", "heatmap"]),
    ("box",       ["箱线图", "盒须图", "boxplot", "box plot"]),
    ("bar",       ["柱状图", "条形图", "柱形图", "bar chart"]),
    ("line",      ["折线图", "曲线图", "线图", "line chart"]),
]

# 用户在问趋势/随时间变化，才有理由画折线图
_TREND_KEYWORDS = ["趋势", "变化", "走势", "增长", "逐月", "逐年", "随时间", "trend"]


def _explicit_chart_type(user_question: str) -> Optional[str]:
    """用户是否在提问里直接点名了图表类型。"""
    q = (user_question or "").lower()
    for ctype, keywords in _EXPLICIT_CHART:
        if any(kw.lower() in q for kw in keywords):
            return ctype
    return None


def _cols_in_result(cols: list, analysis_result: str) -> list:
    """
    分析结果文本里真正出现过的列名。

    图表要画的是 python_repl 聚合之后的结果（例如「地区 × 销售额」），
    而不是原始文件的全部列。原先直接拿原始 df 的特征做决策，只要文件里
    有日期列就永远推荐折线图，规则 2~5 变成死代码。
    """
    text = analysis_result or ""
    return [c for c in cols if c and str(c) in text]


def _decide_chart_type(
    features:        dict,
    analysis_result: str,
    user_question:   str,
) -> dict:
    """
    规则引擎 + LLM 混合决策图表类型。
    规则引擎处理有明确统计依据的场景，
    LLM 处理规则覆盖不到的模糊场景。
    """
    numeric_cols   = features["numeric_cols"]
    object_cols    = features["object_cols"]
    has_time       = features["has_time_series"]
    time_cols      = features["time_like_cols"] + features["datetime_cols"]
    category_info  = features["category_info"]

    # 日期列本身是字符串，会混进 object_cols；若不排除，
    # 规则5 会把「日期」当成类别列去画柱状图
    cat_cols = [c for c in object_cols if c not in time_cols]

    # 优先使用分析结果里真正出现过的列
    num_col = (_cols_in_result(numeric_cols, analysis_result) or numeric_cols or [None])[0]
    cat_col = (_cols_in_result(cat_cols, analysis_result) or cat_cols or [None])[0]

    rule_result = None

    # 规则0：用户直接点名了图表类型 → 直接采纳，不再猜
    explicit = _explicit_chart_type(user_question)
    if explicit:
        rule_result = {
            "recommended_chart": explicit,
            "x_col":  cat_col or (time_cols[0] if time_cols else None),
            "y_col":  num_col,
            "reason": f"用户在提问中明确要求 {explicit} 图",
        }
        if explicit == "pie":
            rule_result["names_col"]  = cat_col
            rule_result["values_col"] = num_col

    # 规则1：时间序列 → 折线图。
    # 必须同时满足「原始数据有时间列」和「本次分析确实涉及时间」——
    # 否则按地区聚合的结果也会被推荐成按日期的折线图。
    elif (
        has_time
        and len(numeric_cols) >= 1
        and (
            any(k in (user_question or "") for k in _TREND_KEYWORDS)
            or _cols_in_result(time_cols, analysis_result)
        )
    ):
        rule_result = {
            "recommended_chart": "line",
            "x_col":  time_cols[0] if time_cols else None,
            "y_col":  num_col,
            "reason": (
                f"分析涉及时间列 `{time_cols[0] if time_cols else ''}`，"
                "折线图最能体现趋势变化"
            ),
        }

    # 规则2：仅两列数值 → 散点图
    elif len(numeric_cols) == 2 and len(object_cols) == 0:
        rule_result = {
            "recommended_chart": "scatter",
            "x_col":  numeric_cols[0],
            "y_col":  numeric_cols[1],
            "reason": f"两列数值变量，散点图展示 `{numeric_cols[0]}` 与 `{numeric_cols[1]}` 的关系",
        }

    # 规则3：多数值列且有强相关 → 热力图
    elif len(numeric_cols) >= 3 and features["has_strong_corr"]:
        rule_result = {
            "recommended_chart": "heatmap",
            "cols":   numeric_cols,
            "reason": (
                f"存在 {len(features['strong_correlations'])} 对强相关变量，"
                "热力图直观展示多变量相关性"
            ),
        }

    # 规则4：单数值列 → 直方图
    elif len(numeric_cols) == 1 and len(object_cols) == 0:
        rule_result = {
            "recommended_chart": "histogram",
            "col":    numeric_cols[0],
            "reason": f"单数值列 `{numeric_cols[0]}`，直方图展示频率分布",
        }

    # 规则5：类别列 + 数值列 → 饼图 / 柱状图
    elif len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        n_unique = category_info.get(cat_col, {}).get("n_unique", 0)
        is_proportion = any(
            kw in user_question
            for kw in ["占比", "比例", "构成", "份额", "percent", "ratio"]
        )

        if is_proportion and n_unique <= 7:
            rule_result = {
                "recommended_chart": "pie",
                "names_col":  cat_col,
                "values_col": num_col,
                "reason":     f"用户询问占比，{n_unique} 种类别（<=7），饼图清晰展示构成",
            }
        elif n_unique > 10:
            rule_result = {
                "recommended_chart": "bar",
                "x_col":       cat_col,
                "y_col":       num_col,
                "orientation": "h",
                "reason":      f"类别数量 {n_unique} 种（>10），横向柱状图避免标签拥挤",
            }
        else:
            rule_result = {
                "recommended_chart": "bar",
                "x_col":  cat_col,
                "y_col":  num_col,
                "reason": (
                    f"类别列 `{cat_col}`（{n_unique} 种）"
                    f"对比数值 `{num_col}`，柱状图直观展示差异"
                ),
            }

    # 规则兜底：交给 LLM 判断
    if rule_result is None:
        rule_result = _llm_decide_chart(features, analysis_result, user_question)

    # 统一附加 viz_lib 和 plotly 配置
    rule_result["viz_lib"]          = "plotly"
    rule_result["plotly_template"]  = _get_plotly_config(
        rule_result["recommended_chart"]
    )
    return rule_result


def _llm_decide_chart(
    features:        dict,
    analysis_result: str,
    user_question:   str,
) -> dict:
    """规则引擎无法覆盖时，交给 LLM 决策。"""
    from src.llm import llm
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = f"""
你是数据可视化专家。根据以下信息推荐最合适的图表类型。

用户问题：{user_question}
分析结果：{analysis_result[:500]}
数值列：{features['numeric_cols']}
类别列：{features['object_cols']}
是否有时间序列：{features['has_time_series']}
数据行数：{features['row_count']}

可选类型：bar / line / pie / scatter / histogram / heatmap / treemap

只返回 JSON，格式如下，不要输出其他内容：
{{"recommended_chart": "类型", "reason": "一句话理由", "x_col": "列名或null", "y_col": "列名或null"}}
"""
    response = llm.invoke([
        SystemMessage(content="你是数据可视化专家，只返回 JSON，不要输出其他内容。"),
        HumanMessage(content=prompt),
    ])

    # 修复问题2：原来 match 为 None 时返回 {}，导致后续访问 rule_result["recommended_chart"] 触发 KeyError
    # 现在统一在 except 中兜底，确保始终返回含 recommended_chart 的字典
    try:
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON")
        return json.loads(match.group())
    except Exception:
        return {
            "recommended_chart": "bar",
            "reason":            "解析推荐结果失败，使用默认柱状图",
        }


def _get_plotly_config(chart_type: str) -> dict:
    """为每种图表类型返回 Plotly 推荐配置。"""
    base = {
        "template":       "plotly_white",
        "color_sequence": "px.colors.qualitative.Set2",
    }
    configs = {
        "bar": {
            **base,
            "text_auto": True,
            "barmode":   "group",
        },
        "line": {
            **base,
            "markers":    True,
            "line_shape": "spline",
        },
        "pie": {
            **base,
            "hole": 0.3,
        },
        "scatter": {
            **base,
            "trendline": "ols",
            "opacity":   0.7,
        },
        "histogram": {
            **base,
            "nbins":    30,
            "marginal": "box",
        },
        "heatmap": {
            "color_continuous_scale": "RdBu_r",
            "text_auto":              True,
        },
        "treemap": {
            **base,
        },
    }
    return configs.get(chart_type, base)