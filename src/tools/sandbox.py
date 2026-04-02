"""
沙箱代码执行器。

核心职责：用子进程安全执行 LLM 生成的 Python 代码。

安全策略：
  1. 子进程隔离   —— 崩溃/异常不影响主进程
  2. 超时保护     —— 防止死循环占用资源
  3. 数据预注入   —— df 变量由外部加载，LLM 无法指定任意文件路径
  4. 非交互后端   —— matplotlib 强制使用 Agg，避免 GUI 依赖

在路径一中，此文件被 tools_definition.py 中的
python_repl 和 create_chart 两个工具调用。

生产环境升级路径：
  将 subprocess 替换为 E2B / Docker exec / Modal，
  实现真正的进程级沙箱隔离。
"""
import os
import sys  # 修复问题3：用 sys.executable 替代硬编码的 "python"，确保使用当前虚拟环境的解释器
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Optional


# ── 数据类：封装执行结果 ──────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    success:    bool
    stdout:     str
    stderr:     str
    chart_path: Optional[str] = field(default=None)  # PNG 路径（matplotlib）
    html_path:  Optional[str] = field(default=None)  # HTML 路径（plotly）


# ── 主函数 ────────────────────────────────────────────────────────────────

def run_code_in_sandbox(
    code:             str,
    file_path:        str,
    chart_decision:   dict,
    png_output_path:  str,
    html_output_path: str,
    timeout:          int = 30,
) -> ExecutionResult:
    """
    在子进程中安全执行 Python 代码。

    参数：
        code             : 要执行的 Python 代码字符串
        file_path        : 数据文件路径，预注入为 df 变量
        chart_decision   : 图表决策字典，决定预导入哪个可视化库
                           {"needs_chart": bool, "viz_lib": "plotly"/"matplotlib"/"none"}
        png_output_path  : matplotlib 图表保存路径
        html_output_path : plotly 图表保存路径
        timeout          : 执行超时秒数

    返回：ExecutionResult
    """
    if not code or not code.strip():
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="代码为空，没有可执行的内容",
        )

    if not file_path or not os.path.exists(file_path):
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"数据文件不存在：{file_path}",
        )

    # 确保输出目录存在
    for path in (png_output_path, html_output_path):
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # 根据图表决策构建 preamble 和 save 代码段
    needs_chart = chart_decision.get("needs_chart", False)
    viz_lib     = chart_decision.get("viz_lib", "none")

    preamble  = _build_preamble(needs_chart, viz_lib)
    save_code = _build_save_code(needs_chart, viz_lib, png_output_path, html_output_path)

    # 从文件扩展名决定读取方式
    read_code = _build_read_code(file_path)

    full_script = textwrap.dedent(f"""
# ── 环境初始化 ──
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
{preamble}

# ── 数据预加载（LLM 代码直接使用 df 变量）──
{read_code}

# ── LLM 生成的代码 ──
{code}

# ── 图表自动保存 ──
{save_code}
""")

    # 写入临时文件并执行
    tmp_path = _write_temp_script(full_script)

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],  # 修复问题3：使用当前虚拟环境的 Python，避免找不到 plotly 等依赖
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode == 0:
            # 检查实际生成了哪些图表文件
            actual_html = html_output_path if os.path.exists(html_output_path) else None
            actual_png  = png_output_path  if os.path.exists(png_output_path)  else None

            return ExecutionResult(
                success=True,
                stdout=proc.stdout,
                stderr="",
                chart_path=actual_png,
                html_path=actual_html,
            )
        else:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=proc.stderr,
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=(
                f"代码执行超时（>{timeout}s），请简化分析逻辑。\n"
                "建议：避免对大数据集使用嵌套循环，改用 pandas 向量化操作。"
            ),
        )

    except FileNotFoundError:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=(
                "找不到 Python 解释器。\n"
                "请确认虚拟环境已激活，或 Python 已加入系统 PATH。"
            ),
        )

    finally:
        # 无论成功失败，都清理临时脚本文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── 内部辅助函数 ──────────────────────────────────────────────────────────

def _build_preamble(needs_chart: bool, viz_lib: str) -> str:
    """
    根据图表决策生成导入语句。
    不需要图表时不导入任何可视化库，减少子进程启动开销。
    """
    if not needs_chart or viz_lib == "none":
        return ""

    if viz_lib == "plotly":
        return (
            "import plotly.express as px\n"
            "import plotly.graph_objects as go"
        )

    if viz_lib == "matplotlib":
        return (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']\n"
            "plt.rcParams['axes.unicode_minus'] = False"
        )

    return ""


def _build_save_code(
    needs_chart:      bool,
    viz_lib:          str,
    png_output_path:  str,
    html_output_path: str,
) -> str:
    """
    生成图表保存代码段，追加在用户代码之后。

    设计思路：
      - plotly：检测变量名 fig，调用 write_html
      - matplotlib：检测是否有活跃 figure，调用 savefig
      - 两种情况都用 try/except 包裹，避免保存失败导致整个脚本报错
    """
    if not needs_chart or viz_lib == "none":
        return ""

    if viz_lib == "plotly":
        # 修复：额外保存 JSON 文件，供主进程通过 gr.Plot 直接展示 Plotly Figure 对象
        json_output_path = html_output_path.replace('.html', '.json')
        return textwrap.dedent(f"""
# plotly 图表自动保存
try:
    if 'fig' in vars() and fig is not None:  # 修复问题5：vars() 比 dir() 更准确地检查局部变量是否存在
        fig.write_html(r'{html_output_path}')
        fig.write_json(r'{json_output_path}')
except Exception as _e:
    print(f"图表保存失败: {{_e}}")
""")

    if viz_lib == "matplotlib":
        return textwrap.dedent(f"""
# matplotlib 图表自动保存
try:
    import matplotlib.pyplot as _plt
    if _plt.get_fignums():
        _plt.savefig(r'{png_output_path}', dpi=150, bbox_inches='tight')
        _plt.close('all')
except Exception as _e:
    print(f"图表保存失败: {{_e}}")
""")

    return ""


def _build_read_code(file_path: str) -> str:
    """
    根据文件扩展名生成数据读取代码。
    统一赋值给变量 df，LLM 生成的代码直接使用 df 即可。
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        # 自动尝试多种编码
        return textwrap.dedent(f"""
for _enc in ('utf-8', 'gbk', 'utf-8-sig', 'latin-1'):
    try:
        df = pd.read_csv(r'{file_path}', encoding=_enc)
        break
    except UnicodeDecodeError:
        continue
""")

    elif ext in (".xlsx", ".xls"):
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        return f"df = pd.read_excel(r'{file_path}', sheet_name=0, engine='{engine}')"

    elif ext == ".json":
        return textwrap.dedent(f"""
import json as _json
with open(r'{file_path}', encoding='utf-8') as _f:
    _raw = _json.load(_f)
if isinstance(_raw, list):
    df = pd.DataFrame(_raw)
elif isinstance(_raw, dict):
    _first = next(iter(_raw.values()), None)
    df = pd.DataFrame(_raw) if isinstance(_first, list) else pd.DataFrame([_raw])
else:
    raise ValueError("不支持的 JSON 格式")
""")

    else:
        return f"raise ValueError('不支持的文件格式：{ext}')"


def _write_temp_script(content: str) -> str:
    """
    将脚本内容写入临时文件，返回文件路径。
    使用系统临时目录，delete=False 是因为 Windows 不允许同时打开并执行。
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
        prefix="analyst_sandbox_",
    ) as f:
        f.write(content)
        return f.name