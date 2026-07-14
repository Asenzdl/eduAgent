#!/usr/bin/env python
"""
基于 sys.settrace 的简历审查管道逐行追踪 → 交互式 HTML 可视化。

原理：sys.settrace 是 Python 调试器（pdb）的底层机制，能在真实运行时
捕获每个函数的 call/line/return 事件，以及该行所有局部变量的快照。

用法: python scripts/trace_resume_pipeline.py
输出: resume_pipeline_trace.html（自包含，浏览器直接打开）
"""

import sys
import os
import json
import time
import types
import asyncio
import threading
from pathlib import Path

# ── 路径设置（必须在 import backend 之前）──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # 确保 .env.local 能被 pydantic-settings 找到

NODES_FILE = str(PROJECT_ROOT / "backend" / "agents" / "resume" / "nodes.py")
SCRIPT_FILE = str(Path(__file__).resolve())
NODES_NORM = os.path.normcase(os.path.normpath(NODES_FILE))
SCRIPT_NORM = os.path.normcase(os.path.normpath(SCRIPT_FILE))

# ── 管道节点元数据（用于流程图展示）──────────────────────────────
PIPELINE_META = [
    {"key": "extract_text",       "func": "extract_text_node",       "label": "③ PDF文本提取",
     "desc": "PyMuPDF逐页解析，双栏先左后右", "color": "#3b82f6"},
    {"key": "extract_structured", "func": "extract_structured_node", "label": "④ 结构化提取",
     "desc": "LLM Function Calling → ResumeStructured", "color": "#8b5cf6"},
    {"key": "run_six_dimensions", "func": "run_six_dimensions_node", "label": "⑤ 六维度并行评审",
     "desc": "6个协程并行评分，加权综合", "color": "#ec4899"},
    {"key": "diagnose_issues",    "func": "diagnose_issues_node",    "label": "⑥ 问题诊断",
     "desc": "Think前置推理 → IssueList", "color": "#f59e0b"},
    {"key": "generate_summary",   "func": "generate_summary_node",   "label": "⑦ 整体评价",
     "desc": "综合评分+问题 → ResumeSummary", "color": "#10b981"},
]
FUNC_TO_KEY = {m["func"]: m["key"] for m in PIPELINE_META}
# 包含 main 的完整映射：用于 line/call 事件时设置 _current_node
# 关键：不在 return 事件上重置！因为 async 函数 await 挂起时也会触发 return，
# 提前重置会导致线程池函数和并行协程被错误归到 "main"。
FUNC_TO_NODE = {**FUNC_TO_KEY, "main": "main"}

# ── 追踪存储 ──────────────────────────────────────────────────
_steps = []
_current_node = "main"
_step_counter = 0
_file_cache = {}

# 需要过滤掉的局部变量类型（函数、类、模块等无信息量）
_SKIP_TYPES = (types.FunctionType, types.MethodType, types.ModuleType, type)


def _check_target(filename):
    """缓存版：判断文件是否为目标文件。返回 (is_nodes, is_script)。"""
    if filename not in _file_cache:
        fn = os.path.normcase(os.path.normpath(filename))
        _file_cache[filename] = (fn == NODES_NORM, fn == SCRIPT_NORM)
    return _file_cache[filename]


def safe_repr(obj, depth=0):
    """递归地把任意 Python 对象转成 JSON 可序列化的值。"""
    if depth > 15:
        return "<…>"
    try:
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            return obj[:10000] + f"…(+{len(obj) - 10000}字)" if len(obj) > 10000 else obj
        if isinstance(obj, (list, tuple)):
            n = len(obj)
            items = [safe_repr(x, depth + 1) for x in obj[:15]]
            if n > 15:
                items.append(f"…(+{n - 15}项)")
            return items
        if isinstance(obj, dict):
            return {str(k): safe_repr(v, depth + 1) for k, v in list(obj.items())[:25]}
        if hasattr(obj, "model_dump"):
            return safe_repr(obj.model_dump(), depth + 1)
        if isinstance(obj, type) and hasattr(obj, "model_json_schema"):
            return {"__pydantic_model__": obj.__name__, "schema": safe_repr(obj.model_json_schema(), depth + 1)}
        r = repr(obj)
        return r[:200] + "…" if len(r) > 200 else r
    except Exception:
        return f"<{type(obj).__name__}>"


def capture_locals(frame):
    """捕获 frame.f_locals，过滤掉函数/类/模块等。"""
    try:
        return {
            k: safe_repr(v)
            for k, v in frame.f_locals.items()
            if not k.startswith("__") and not isinstance(v, _SKIP_TYPES)
        }
    except Exception:
        return {}


# ── trace 函数（sys.settrace 的核心）──────────────────────────
def trace_fn(frame, event, arg):
    global _current_node, _step_counter

    if event == "call":
        is_nodes, is_script = _check_target(frame.f_code.co_filename)
        func_name = frame.f_code.co_name
        # 只追踪 nodes.py 里的函数
        if not is_nodes:
            return None  # 返回 None → 不追踪该帧的行级事件
        # <module> 是 exec 的顶层帧：跳过它的 call 事件（行号为0无意义），
        # 但返回 trace_fn 以保留它的 line 事件（if __name__ / async def / asyncio.run）
        if func_name == "<module>":
            return trace_fn

        if func_name in FUNC_TO_NODE:
            _current_node = FUNC_TO_NODE[func_name]

        _steps.append({
            "idx": _step_counter,
            "event": "call",
            "func": func_name,
            "line": frame.f_lineno,
            "locals": capture_locals(frame),
            "pipeline_node": _current_node,
            "return_value": None,
        })
        _step_counter += 1
        return trace_fn  # 返回自身 → 继续追踪该帧的 line/return

    # line / return 事件只会在我们返回了 trace_fn 的帧里触发
    is_nodes, is_script = _check_target(frame.f_code.co_filename)
    if not is_nodes and not (is_script and frame.f_code.co_name == "main"):
        return trace_fn

    if event == "line":
        func_name = frame.f_code.co_name
        # 在 line 事件上更新 _current_node（覆盖 main 和各节点函数）
        # 这样 return 挂起后恢复时也能正确归类
        if func_name in FUNC_TO_NODE:
            _current_node = FUNC_TO_NODE[func_name]
        _steps.append({
            "idx": _step_counter,
            "event": "line",
            "func": func_name,
            "line": frame.f_lineno,
            "locals": capture_locals(frame),
            "pipeline_node": _current_node,
            "return_value": None,
        })
        _step_counter += 1
    elif event == "return":
        func_name = frame.f_code.co_name
        _steps.append({
            "idx": _step_counter,
            "event": "return",
            "func": func_name,
            "line": frame.f_lineno,
            "locals": capture_locals(frame),
            "pipeline_node": _current_node,
            "return_value": safe_repr(arg),
        })
        _step_counter += 1
        # 不在 return 上重置 _current_node！
        # async 函数 await 挂起时也会触发 return，提前重置会导致
        # 线程池函数(_sync_extract_text)和并行协程(review_one_dimension)被错误归到 "main"

    return trace_fn


# ── 入口 ──────────────────────────────────────────────────────
def run():
    import backend.agents.resume.nodes as nodes_mod

    # Monkey-patch _sync_extract_text：它跑在线程池里，需要单独设 trace
    _orig_sync = nodes_mod._sync_extract_text

    def _traced_sync(pdf_path):
        sys.settrace(trace_fn)  # 在线程池线程里启用追踪
        try:
            return _orig_sync(pdf_path)
        finally:
            sys.settrace(None)

    nodes_mod._sync_extract_text = _traced_sync

    # 读取 nodes.py 源码（用于 HTML 展示 + 提取 __main__ 块）
    with open(NODES_FILE, "r", encoding="utf-8") as f:
        source_lines = f.read().split("\n")

    # 定位 __main__ 块的起始行（0-indexed）
    main_start = None
    for i, line in enumerate(source_lines):
        if line.strip() == 'if __name__ == "__main__":':
            main_start = i
            break
    if main_start is None:
        raise RuntimeError("在 nodes.py 中找不到 if __name__ == '__main__' 块")

    # 用空行填充到 main_start 行，使编译后的行号与原文件一致
    padded = "\n" * main_start + "\n".join(source_lines[main_start:])

    # 启用全局追踪
    sys.settrace(trace_fn)
    threading.settrace(trace_fn)

    print("=" * 60)
    print("🚀 开始追踪简历审查管道（真实运行，含 LLM 调用）...")
    print("   PDF:", str(PROJECT_ROOT / "backend" / "agents" / "resume" / "王俊森.pdf"))
    print("   预计耗时 30-90 秒（6 路 LLM 并行评审）")
    print("=" * 60)

    start = time.time()
    # 直接执行 nodes.py 的 __main__ 块：
    #   - 编译时指定 co_filename = NODES_FILE，使 main() 的行号指向 nodes.py
    #   - 在 nodes_mod.__dict__ 中执行，保证 ResumeState / extract_text_node 等可用
    #   - 临时设 __name__='__main__' 让 if 条件成立
    old_name = nodes_mod.__dict__.get("__name__")
    nodes_mod.__dict__["__name__"] = "__main__"
    try:
        exec(compile(padded, NODES_FILE, "exec"), nodes_mod.__dict__)
    finally:
        if old_name is not None:
            nodes_mod.__dict__["__name__"] = old_name
    elapsed = time.time() - start

    # 关闭追踪
    sys.settrace(None)
    threading.settrace(None)

    print(f"\n✅ 追踪完成！")
    print(f"   总步骤数: {_step_counter}")
    print(f"   总耗时: {elapsed:.1f}s")

    # 统计各节点步骤数
    node_counts = {}
    for s in _steps:
        node_counts[s["pipeline_node"]] = node_counts.get(s["pipeline_node"], 0) + 1
    for node, count in node_counts.items():
        print(f"   {node}: {count} 步")

    # 生成 HTML
    generate_html(source_lines, elapsed)
    print(f"\n📄 HTML 已生成，浏览器打开查看:")
    print(f"   {PROJECT_ROOT / 'resume_pipeline_trace.html'}")


def generate_html(source_lines, elapsed):
    data = json.dumps(
        {
            "source_lines": source_lines,
            "pipeline_meta": PIPELINE_META,
            "steps": _steps,
            "elapsed": round(elapsed, 2),
        },
        ensure_ascii=False,
    )
    # 防止 JSON 中的 </script> 截断 HTML
    data = data.replace("</script>", "<\\/script>")

    html = HTML_TEMPLATE.replace("__TRACE_DATA__", data)

    out = PROJECT_ROOT / "resume_pipeline_trace.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)


# ══════════════════════════════════════════════════════════════
# HTML 模板
# ══════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>简历审查 Agent · 数据流逐行追踪</title>
<style>
:root{
  --bg:#f0f2f5;--card:#fff;--ink:#1a1a2e;--muted:#6b7280;--line:#e5e7eb;
  --brand:#3b82f6;--code-bg:#0d1117;--code-fg:#e6edf3;--code-ln:#636c76;
  --code-hl:rgba(59,130,246,.18);--shadow:0 2px 10px rgba(0,0,0,.06);--radius:10px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Microsoft YaHei","PingFang SC",-apple-system,sans-serif;background:var(--bg);color:var(--ink);height:100vh;overflow:hidden;display:flex;flex-direction:column}

/* ── Header ── */
.header{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:#fff;padding:10px 20px;flex-shrink:0}
.header h1{font-size:17px;font-weight:600}
.header .sub{font-size:12px;opacity:.8;margin-top:2px}

/* ── Pipeline flow ── */
.pipeline{display:flex;align-items:center;gap:2px;padding:8px 20px;background:var(--card);border-bottom:1px solid var(--line);flex-shrink:0;overflow-x:auto}
.pipe-node{display:flex;align-items:center;gap:4px;padding:5px 10px;border-radius:6px;font-size:12px;cursor:pointer;transition:all .2s;border:2px solid transparent;white-space:nowrap;font-weight:500}
.pipe-node:hover{transform:translateY(-1px)}
.pipe-node.active{box-shadow:0 0 0 2px currentColor;transform:scale(1.06)}
.pipe-node.done{opacity:.5}
.pipe-arrow{color:var(--muted);font-size:13px;margin:0 1px}

/* ── Toolbar ── */
.toolbar{display:flex;align-items:center;gap:8px;padding:7px 20px;background:var(--card);border-bottom:1px solid var(--line);flex-shrink:0}
.toolbar button{padding:5px 12px;border:1px solid var(--line);border-radius:5px;background:#fff;cursor:pointer;font-size:12px;transition:all .12s;white-space:nowrap}
.toolbar button:hover{background:#f3f4f6}
.toolbar button.primary{background:var(--brand);color:#fff;border-color:var(--brand)}
.toolbar button.primary:hover{background:#2563eb}
.progress-wrap{flex:1;height:5px;background:var(--line);border-radius:3px;overflow:hidden;min-width:100px}
.progress-bar{height:100%;background:var(--brand);transition:width .15s;width:0}
.step-info{font-size:12px;color:var(--muted);min-width:70px;text-align:right;white-space:nowrap}

/* ── Main layout ── */
.main{flex:1;display:flex;overflow:hidden}

/* ── Code panel ── */
.code-panel{flex:0 0 38%;background:var(--code-bg);overflow:auto;font-family:"JetBrains Mono","Consolas","Courier New",monospace;font-size:12.5px;line-height:1.55}
.code-line{display:flex;padding:0 8px;min-height:21px}
.code-line:hover{background:rgba(255,255,255,.03)}
.code-line .ln{color:var(--code-ln);text-align:right;width:38px;margin-right:10px;user-select:none;flex-shrink:0;font-size:11px}
.code-line .src{color:var(--code-fg);white-space:pre-wrap;word-break:break-word}
.code-line.current{background:var(--code-hl);border-left:3px solid var(--brand);padding-left:5px}
.code-line.current .ln{color:#58a6ff;font-weight:600}
/* 语法高亮 */
.kw{color:#ff7b72}
.st{color:#a5d6ff}
.cm{color:#8b949e;font-style:italic}
.nm{color:#79c0ff}
.fn{color:#d2a8ff}
.dc{color:#d2a8ff}

/* ── Info panel ── */
.info-panel{flex:1;overflow-y:auto;padding:12px;background:var(--bg)}
.info-card{background:var(--card);border-radius:var(--radius);padding:12px;margin-bottom:10px;box-shadow:var(--shadow)}
.info-card h3{font-size:12px;color:var(--muted);margin-bottom:7px;font-weight:500;display:flex;align-items:center;gap:4px}
.func-name{font-size:15px;font-weight:600;color:var(--ink);font-family:monospace}
.event-badge{display:inline-block;padding:1px 7px;border-radius:3px;font-size:10px;font-weight:600;text-transform:uppercase}
.ev-call{background:#dbeafe;color:#1e40af}
.ev-line{background:#d1fae5;color:#065f46}
.ev-return{background:#fef3c7;color:#92400e}
.line-content{margin-top:7px;font-size:11.5px;color:#8b949e;font-family:"JetBrains Mono",monospace;background:#f6f8fa;padding:5px 7px;border-radius:5px;border-left:2px solid var(--line);white-space:pre-wrap;word-break:break-word;max-height:60px;overflow:hidden}

/* ── Variable table ── */
.var-table{width:100%;border-collapse:collapse;font-size:13px}
.var-table td{padding:4px 8px;border-bottom:1px solid #f3f4f6;vertical-align:top}
.var-table tr:last-child td{border-bottom:none}
.var-key{color:var(--brand);font-weight:600;white-space:nowrap;width:1%;font-family:monospace;font-size:12px;cursor:pointer;transition:background .1s;border-radius:4px}
.var-key:hover{background:#eef2ff;text-decoration:underline}
.var-val{font-family:"JetBrains Mono",monospace;word-break:break-word;line-height:1.5;font-size:12.5px}
.var-val-cell{padding-right:4px}
.var-changed td{background:#fffbeb}
.var-new td{background:#ecfdf5}

/* ── Modal ── */
.modal-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:1000;align-items:center;justify-content:center}
.modal-backdrop.show{display:flex}
.modal-card{background:var(--card);border-radius:12px;width:80vw;max-width:1100px;max-height:80vh;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.3)}
.modal-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--line);flex-shrink:0}
.modal-title{font-family:monospace;font-size:14px;font-weight:600;color:var(--brand)}
.modal-actions{display:flex;gap:6px}
.modal-actions button{padding:4px 10px;border:1px solid var(--line);border-radius:5px;background:#fff;cursor:pointer;font-size:11px}
.modal-actions button:hover{background:#f3f4f6}
.modal-body{flex:1;overflow:auto;padding:14px 16px;font-family:"JetBrains Mono","Consolas",monospace;font-size:12.5px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.modal-footer{padding:6px 16px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);flex-shrink:0}

/* ── Value rendering ── */
.str{color:#059669}
.num{color:#d97706}
.bool{color:#7c3aed}
.null{color:var(--muted)}
.key{color:var(--brand)}
.clickable{cursor:pointer;color:#6b7280;user-select:none}
.clickable:hover{color:var(--brand);text-decoration:underline}
.hidden{display:none}
.indent{padding-left:14px;border-left:1px solid var(--line);margin-left:2px}

/* ── Return value ── */
.return-val{background:var(--code-bg);color:var(--code-fg);border-radius:6px;padding:8px;font-family:monospace;font-size:11.5px;white-space:pre-wrap;word-break:break-word;max-height:250px;overflow:auto}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#c1c1c1;border-radius:4px}
.code-panel::-webkit-scrollbar-thumb{background:#30363d}
</style>
</head>
<body>

<div class="header">
  <h1>📄 简历审查 Agent · 数据流逐行追踪</h1>
  <div class="sub" id="subInfo">基于 sys.settrace 真实运行捕获</div>
</div>

<div class="pipeline" id="pipeline"></div>

<div class="toolbar">
  <button class="primary" id="autoBtn">▶ 自动播放</button>
  <label style="font-size:11px;color:var(--muted)">间隔ms</label>
  <input type="number" id="speedInput" value="600" min="20" max="5000" step="50" style="width:60px;font-size:11px;padding:2px 4px;border:1px solid var(--line);border-radius:4px;text-align:center"/>
  <button id="prevBtn">◀ 上一步</button>
  <button id="nextBtn" title="下一步: 进入函数内部 (j)">下一步</button>
  <button id="overBtn" title="下一行: 跳过函数调用，停在本函数下一行 (l)">下一行</button>
  <button id="loopBtn" title="跳出多行循环: 跳过所有循环迭代 (n)">跳出多行循环</button>
  <button id="outBtn" title="跑完当前函数: 跑到return (L)">跑完当前函数</button>
  <button id="resetBtn">⟲ 重置</button>
  <div class="progress-wrap"><div class="progress-bar" id="progress"></div></div>
  <span class="step-info" id="stepInfo">0 / 0</span>
</div>

<div class="main">
  <div class="code-panel" id="codePanel"></div>
  <div class="info-panel">
    <div class="info-card">
      <h3>📌 当前步骤</h3>
      <div class="func-name" id="funcName">准备开始</div>
      <div style="margin-top:5px;display:flex;align-items:center;gap:6px;flex-wrap:wrap">
        <span class="event-badge" id="eventBadge">-</span>
        <span style="font-size:11px;color:var(--muted)">行 <b id="lineNo">-</b></span>
        <span style="font-size:11px;color:var(--muted)" id="nodeTag"></span>
      </div>
      <div class="line-content" id="lineContent">点击"下一步"开始逐行追踪</div>
    </div>
    <div class="info-card">
      <h3>🔑 局部变量 (<span id="varCount">0</span>)</h3>
      <table class="var-table" id="varTable"><tr><td style="color:var(--muted)">无</td></tr></table>
    </div>
    <div class="info-card" id="returnCard" style="display:none">
      <h3>↩ 返回值</h3>
      <div class="return-val" id="returnVal"></div>
    </div>
  </div>
</div>

<!-- 模态层：点击变量值弹出完整内容 -->
<div class="modal-backdrop" id="modalBackdrop" onclick="if(event.target===this)closeModal()">
  <div class="modal-card">
    <div class="modal-header">
      <span class="modal-title" id="modalTitle">-</span>
      <div class="modal-actions">
        <button onclick="copyModal()" title="复制到剪贴板">📋 复制</button>
        <button onclick="closeModal()" title="关闭 (Esc)">✕</button>
      </div>
    </div>
    <div class="modal-body" id="modalBody"></div>
    <div class="modal-footer" id="modalFooter"></div>
  </div>
</div>

<script id="trace-data" type="application/json">__TRACE_DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('trace-data').textContent);
const SOURCE = DATA.source_lines;
const META = DATA.pipeline_meta;
const NODE_MAP = {};
META.forEach(m => NODE_MAP[m.key] = m);

let current = 0;
let timer = null;
let _currentLocals = {};
let _currentReturn = null;
let _modalValues = []; // 存储长字符串的完整值，供"(点击查看完整)"引用
let _fullExpandVars = new Set(); // 点击变量名→弹出框→关闭后，该变量全展开
let _modalVarName = null; // 记录弹出框是由哪个变量名打开的

// 预转义源码
const ESC = SOURCE.map(s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'));

// ── 过滤 async 挂起/恢复事件 + 推导式内部事件，让执行流线性化 ──
// 1. async await 挂起 return + 恢复 call（同函数同行）→ 过滤
// 2. <lambda>/<genexpr>/<listcomp> 等推导式内部事件 → 过滤（保留 <module>）
function filterSteps(raw){
  const skip=new Set();
  // 过滤挂起/恢复对
  for(let i=0;i<raw.length;i++){
    if(raw[i].event!=='return')continue;
    for(let j=i+1;j<raw.length;j++){
      if(raw[j].func!==raw[i].func)continue;
      if(raw[j].event==='call'&&raw[j].line===raw[i].line){skip.add(i);skip.add(j);}
      break;
    }
  }
  // 过滤推导式内部（func 以 < 开头且不是 <module>）
  return raw.filter((s,i)=>!skip.has(i)&&(s.func!=='<module>'?!s.func.startsWith('<'):true));
}
const STEPS=filterSteps(DATA.steps);

// ── 初始化源码面板（只渲染一次）──
// 简易 Python 语法高亮（GitHub 暗色风格）
function highlightPython(line){
  if(!line) return ' ';
  return line.replace(
    /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(#.*$)|\b(def|class|async|await|for|while|if|elif|else|try|except|finally|return|yield|import|from|as|with|in|not|and|or|is|None|True|False|pass|break|continue|raise|lambda|global|nonlocal|del)\b|\b(\d+\.?\d*)\b|(@\w+)/g,
    (m,str,cm,kw,nm,dc)=>{
      if(str)return `<span class="st">${str}</span>`;
      if(cm)return `<span class="cm">${cm}</span>`;
      if(kw)return `<span class="kw">${kw}</span>`;
      if(nm)return `<span class="nm">${nm}</span>`;
      if(dc)return `<span class="dc">${dc}</span>`;
      return m;
    }
  );
}
function initCode(){
  let h='';
  ESC.forEach((line,i)=>{
    const ln=i+1;
    h+=`<div class="code-line" id="ln-${ln}"><span class="ln">${ln}</span><span class="src">${highlightPython(line)}</span></div>`;
  });
  document.getElementById('codePanel').innerHTML=h;
}

// ── 渲染管道流程 ──
function renderPipeline(){
  const el=document.getElementById('pipeline');
  const curNode = current>=0 ? STEPS[current].pipeline_node : null;
  // 找到当前步骤之前已完成的节点
  const doneNodes = new Set();
  if(current>=0){
    for(let i=0;i<=current;i++){
      const n=STEPS[i].pipeline_node;
      // 如果该节点的 return 事件已过，标记为 done
      if(n!=='main'){
        doneNodes.add(n);
      }
    }
    // 如果当前正在某节点内（还没 return），不算 done
    if(curNode && curNode!=='main') doneNodes.delete(curNode);
  }
  let h='';
  META.forEach((m,i)=>{
    const isActive = curNode===m.key;
    const isDone = doneNodes.has(m.key);
    const cls = isActive?'active':(isDone?'done':'');
    h+=`<div class="pipe-node ${cls}" style="border-color:${m.color}44;background:${m.color}11;color:${m.color}" onclick="jumpToNode('${m.key}')" title="${m.desc}">${m.label}</div>`;
    if(i<META.length-1) h+='<span class="pipe-arrow">→</span>';
  });
  // main 节点
  const isMainActive = curNode==='main';
  h+=`<span class="pipe-arrow">→</span><div class="pipe-node ${isMainActive?'active':''}" style="border-color:#6b728044;background:#6b728011;color:#6b7280" onclick="jumpToMain()">main</div>`;
  el.innerHTML=h;
}

function jumpToNode(key){
  const idx=STEPS.findIndex(s=>s.pipeline_node===key);
  if(idx>=0){current=idx;render();}
}
function jumpToMain(){
  const idx=STEPS.findIndex(s=>s.pipeline_node==='main');
  if(idx>=0){current=idx;render();}
}

// ── 更新源码高亮 ──
function updateCode(){
  const old=document.querySelector('.code-line.current');
  if(old) old.classList.remove('current');
  const ln = current>=0 ? STEPS[current].line : -1;
  if(ln>0){
    const el=document.getElementById('ln-'+ln);
    if(el){
      el.classList.add('current');
      el.scrollIntoView({block:'center',behavior:'smooth'});
    }
  }
}

// ── 渲染步骤信息 ──
function renderStep(){
  if(current<0){
    document.getElementById('funcName').textContent='准备开始';
    document.getElementById('eventBadge').textContent='-';
    document.getElementById('eventBadge').className='event-badge';
    document.getElementById('lineNo').textContent='-';
    document.getElementById('nodeTag').textContent='';
    document.getElementById('lineContent').textContent='点击"下一步"开始逐行追踪';
    document.getElementById('returnCard').style.display='none';
    return;
  }
  const s=STEPS[current];
  document.getElementById('funcName').textContent=s.func;
  const badge=document.getElementById('eventBadge');
  badge.textContent=s.event;
  badge.className='event-badge ev-'+s.event;
  document.getElementById('lineNo').textContent=s.line;
  const nm=NODE_MAP[s.pipeline_node];
  const tag=document.getElementById('nodeTag');
  if(nm){tag.textContent='· '+nm.label;tag.style.color=nm.color;}
  else if(s.pipeline_node==='main'){tag.textContent='· 主流程';tag.style.color='#6b7280';}
  else{tag.textContent='';}
  const lc=SOURCE[s.line-1]||'';
  document.getElementById('lineContent').textContent=lc.trim()||'(空行)';
  if(s.event==='return' && s.return_value!==null && s.return_value!==undefined){
    document.getElementById('returnCard').style.display='block';
    document.getElementById('returnVal').innerHTML=`<div class="var-val-cell"><span class="clickable" onclick="openModal('返回值',_currentReturn)" style="color:var(--brand);font-weight:600">📋 点击查看完整返回值</span><br>${renderVal(s.return_value,0,2,'__return__')}</div>`;
    _currentReturn=s.return_value;
  }else{
    document.getElementById('returnCard').style.display='none';
  }
}

// ── 渲染局部变量（含 diff 高亮）──
function renderVars(){
  const tbl=document.getElementById('varTable');
  if(current<0){
    tbl.innerHTML='<tr><td style="color:var(--muted)">无</td></tr>';
    document.getElementById('varCount').textContent='0';
    return;
  }
  const s=STEPS[current];
  const locals=s.locals||{};
  _currentLocals=locals;
  const keys=Object.keys(locals);
  document.getElementById('varCount').textContent=keys.length;
  // 找同函数的上一步 locals 做 diff
  let prev={};
  for(let i=current-1;i>=0;i--){
    if(STEPS[i].func===s.func){prev=STEPS[i].locals||{};break;}
  }
  let h='';
  keys.forEach(k=>{
    const val=locals[k];
    const wasInPrev=k in prev;
    const changed=wasInPrev && JSON.stringify(prev[k])!==JSON.stringify(val);
    const isNew=!wasInPrev;
    const cls=isNew?'var-new':(changed?'var-changed':'');
    h+=`<tr class="${cls}"><td class="var-key" onclick="openModal('${esc(k)}',_currentLocals['${esc(k)}'])">${esc(k)}</td><td class="var-val"><div class="var-val-cell">${renderVal(val,0,_fullExpandVars.has(k)?99:2,k)}</div></td></tr>`;
  });
  tbl.innerHTML=h||'<tr><td style="color:var(--muted)">无局部变量</td></tr>';
  // 自动滚动到第一个变化/新增的变量
  const changed=tbl.querySelector('.var-changed,.var-new');
  if(changed) changed.scrollIntoView({block:'center',behavior:'smooth'});
}

// ── 递归渲染值 ──
// maxExpandDepth: 展开到第几层（0=全折叠, 2=表格用, 99=模态层全展开）
// path: 变量路径（如 state.structured.projects[0].tech_stack），用作稳定 ID
//       用户手动折叠/展开的状态记在 _userExpandState[path] 里，切步后保持
let _userExpandState={};
const MAX_RENDER_DEPTH=12;
const MAX_MODAL_DEPTH=20;
function renderVal(val,depth=0,maxExpandDepth=0,path=''){
  const limit=maxExpandDepth>=99?MAX_MODAL_DEPTH:MAX_RENDER_DEPTH;
  if(depth>limit) return '<span style="color:var(--muted)">…</span>';
  if(val===null) return '<span class="null">null</span>';
  if(typeof val==='boolean') return `<span class="bool">${val}</span>`;
  if(typeof val==='number') return `<span class="num">${val}</span>`;
  if(typeof val==='string'){
    if(maxExpandDepth>=99||val.length<=500) return `<span class="str">"${esc(val)}"</span>`;
    const idx=_modalValues.length;
    _modalValues.push(val);
    return `<span class="str">"${esc(val.slice(0,500))}…"</span> <span class="clickable" style="color:var(--muted);font-size:11px" onclick="openModal('完整内容',_modalValues[${idx}])">(共${val.length}字，点击查看完整)</span>`;
  }
  if(Array.isArray(val)){
    if(val.length===0) return '<span style="color:var(--muted)">[]</span>';
    const id='nd-'+path;
    // 默认展开状态 vs 用户手动覆盖
    const def=depth<maxExpandDepth;
    const usr=_userExpandState[path];
    const expanded=usr!==undefined?usr:def;
    const items=val.map((v,i)=>`<div class="indent">${renderVal(v,depth+1,maxExpandDepth,path+'['+i+']')}</div>`).join('');
    const cls=expanded?'':'hidden';
    const arrow=expanded?'▾':'▸';
    return `<span class="clickable" onclick="tog('${esc(path)}')">[${val.length}项] ${arrow}</span><div id="${id}" class="${cls}">${items}</div>`;
  }
  if(typeof val==='object'){
    const keys=Object.keys(val);
    if(keys.length===0) return '<span style="color:var(--muted)">{}</span>';
    const id='nd-'+path;
    const def=depth<maxExpandDepth;
    const usr=_userExpandState[path];
    const expanded=usr!==undefined?usr:def;
    const items=keys.map(k=>`<div class="indent"><span class="key">${esc(k)}:</span> ${renderVal(val[k],depth+1,maxExpandDepth,path?path+'.'+k:k)}</div>`).join('');
    const cls=expanded?'':'hidden';
    const arrow=expanded?'▾':'▸';
    return `<span class="clickable" onclick="tog('${esc(path)}')">{${keys.length}字段} ${arrow}</span><div id="${id}" class="${cls}">${items}</div>`;
  }
  return esc(String(val));
}
function tog(path){
  const isTopLevel=!path.includes('.')&&!path.includes('[');
  if(isTopLevel){
    // 顶层变量切换：需要重新渲染（展开深度可能变化）
    const wasExpanded=_fullExpandVars.has(path)||_userExpandState[path]!==false;
    if(wasExpanded){
      // 收起：清除全展开标记 + 清除子路径状态
      _fullExpandVars.delete(path);
      _userExpandState[path]=false;
      Object.keys(_userExpandState).forEach(p=>{
        if(p!==path&&(p.startsWith(path+'.')||p.startsWith(path+'['))) delete _userExpandState[p];
      });
    }else{
      // 展开：不恢复全展开，用默认2级
      _userExpandState[path]=true;
      Object.keys(_userExpandState).forEach(p=>{
        if(p!==path&&(p.startsWith(path+'.')||p.startsWith(path+'['))) delete _userExpandState[p];
      });
    }
    render(); // 重新渲染变量表
  }else{
    // 子级切换：只切 CSS，即时响应
    const el=document.getElementById('nd-'+path);
    if(!el)return;
    el.classList.toggle('hidden');
    const isHidden=el.classList.contains('hidden');
    _userExpandState[path]=!isHidden;
    const trigger=el.previousElementSibling;
    if(trigger&&trigger.classList.contains('clickable')){
      trigger.textContent=trigger.textContent.replace(/[▾▸]/,isHidden?'▸':'▾');
    }
  }
}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// ── 模态层：点击变量值弹出完整内容 ──
function openModal(title,val){
  _modalVarName=title;
  document.getElementById('modalTitle').textContent=title;
  document.getElementById('modalBody').innerHTML=renderVal(val,0,99,'__modal__');
  // 类型与长度信息
  let info='';
  if(val===null) info='类型: null';
  else if(typeof val==='string') info=`类型: string · 长度: ${val.length} 字符`;
  else if(typeof val==='number') info=`类型: number`;
  else if(typeof val==='boolean') info='类型: boolean';
  else if(Array.isArray(val)) info=`类型: array · ${val.length} 项`;
  else if(typeof val==='object') info=`类型: object · ${Object.keys(val).length} 字段`;
  else info=`类型: ${typeof val}`;
  document.getElementById('modalFooter').textContent=info;
  document.getElementById('modalBackdrop').classList.add('show');
}
function closeModal(){
  document.getElementById('modalBackdrop').classList.remove('show');
  // 如果弹出框是由变量名打开的，标记该变量为全展开
  if(_modalVarName&&_currentLocals[_modalVarName]!==undefined){
    _fullExpandVars.add(_modalVarName);
    // 清除子路径状态，避免与全展开冲突
    Object.keys(_userExpandState).forEach(p=>{
      if(p!==_modalVarName&&(p.startsWith(_modalVarName+'.')||p.startsWith(_modalVarName+'['))) delete _userExpandState[p];
    });
    render();
  }
  _modalVarName=null;
}
function copyModal(){
  const text=document.getElementById('modalBody').textContent;
  navigator.clipboard.writeText(text).then(()=>{const b=document.querySelector('.modal-actions button');if(b){const o=b.textContent;b.textContent='✓ 已复制';setTimeout(()=>b.textContent=o,1500);}});
}

// ── 总渲染 ──
function render(){
  _modalValues=[]; // 清空，供本轮 renderVal 存储长字符串引用
  renderPipeline();
  updateCode();
  renderStep();
  renderVars();
  const pct=current<0?0:((current+1)/STEPS.length*100);
  document.getElementById('progress').style.width=pct+'%';
  document.getElementById('stepInfo').textContent=current<0?`0 / ${STEPS.length}`:`${current+1} / ${STEPS.length}`;
}

// ── 导航 ──
function next(){if(current<STEPS.length-1){current++;render();}else stopAuto();}
function prev(){if(current>0){current--;render();}}
function reset(){stopAuto();current=0;document.getElementById('codePanel').scrollTop=0;render();}
function stopAuto(){if(timer){clearInterval(timer);timer=null;}document.getElementById('autoBtn').textContent='▶ 自动播放';}
function auto(){
  if(timer){stopAuto();return;}
  if(current>=STEPS.length-1) current=0;
  document.getElementById('autoBtn').textContent='⏸ 停止';
  const speed=parseInt(document.getElementById('speedInput').value)||600;
  timer=setInterval(()=>{if(current>=STEPS.length-1){stopAuto();return;}next();},speed);
}

// ── 跳过: 跳过函数调用，停在本函数下一个不同行 ──
function stepOver(){
  if(current<0||current>=STEPS.length-1)return;
  // 如果当前在 return 事件上，说明已到函数末尾，直接走下一步进入调用者
  if(STEPS[current].event==='return'){next();return;}
  const curFunc=STEPS[current].func, curLine=STEPS[current].line;
  for(let i=current+1;i<STEPS.length;i++){
    const s=STEPS[i];
    if(s.func===curFunc && s.event==='line' && s.line!==curLine){
      current=i;render();return;
    }
    if(s.func===curFunc && s.event==='return'){
      let hasMoreLines=false;
      for(let j=i+1;j<STEPS.length;j++){
        if(STEPS[j].func===curFunc){hasMoreLines=(STEPS[j].event==='line'||STEPS[j].event==='call');break;}
      }
      if(!hasMoreLines){current=i;render();return;}
    }
  }
  current=STEPS.length-1;render();
}

// ── 跳出循环: 跳过所有循环迭代，停在循环后的第一行 ──
// 原理：循环内行号会回退（如 49→51→...→77→49），检测到回退后，
//       找到第一个超过已见最大行号的行，即为循环出口
function skipLoop(){
  if(current<0||current>=STEPS.length-1)return;
  const curFunc=STEPS[current].func;
  let maxLine=STEPS[current].line;
  let inLoop=false;
  for(let i=current+1;i<STEPS.length;i++){
    const s=STEPS[i];
    if(s.func!==curFunc||s.event!=='line')continue;
    if(s.line<maxLine){inLoop=true;continue;}
    if(s.line>maxLine){
      if(inLoop){current=i;render();return;}
      maxLine=s.line;
    }
  }
}

// ── 步出: 跑完当前函数，停在 return ──
function stepOut(){
  if(current<0||current>=STEPS.length-1)return;
  const curFunc=STEPS[current].func;
  for(let i=current+1;i<STEPS.length;i++){
    if(STEPS[i].func!==curFunc||STEPS[i].event!=='return')continue;
    let isSuspension=false;
    for(let j=i+1;j<STEPS.length;j++){
      if(STEPS[j].func===curFunc){isSuspension=(STEPS[j].event==='call');break;}
    }
    if(!isSuspension){current=i;render();return;}
  }
}

document.getElementById('nextBtn').addEventListener('click',next);
document.getElementById('prevBtn').addEventListener('click',prev);
document.getElementById('overBtn').addEventListener('click',stepOver);
document.getElementById('loopBtn').addEventListener('click',skipLoop);
document.getElementById('outBtn').addEventListener('click',stepOut);
document.getElementById('resetBtn').addEventListener('click',reset);
document.getElementById('autoBtn').addEventListener('click',auto);
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT')return;
  if(e.key==='Escape'){closeModal();return;}
  if(document.getElementById('modalBackdrop').classList.contains('show'))return;
  if(e.key==='j'||e.key==='ArrowRight'){e.preventDefault();next();}
  if(e.key==='l'){e.preventDefault();stepOver();}
  if(e.key==='n'){e.preventDefault();skipLoop();}
  if(e.key==='L'||(e.key==='l'&&e.shiftKey)){e.preventDefault();stepOut();}
  if(e.key==='k'||e.key==='ArrowLeft'){e.preventDefault();prev();}
  if(e.key===' '){e.preventDefault();auto();}
});

// ── 初始化 ──
document.getElementById('subInfo').textContent=
  `sys.settrace 真实运行捕获 · ${STEPS.length} 步 · 耗时 ${DATA.elapsed}s · j下一步 l下一行 n跳出循环 L跑完函数 · 空格自动播放 · 点击变量查看完整内容`;
initCode();
render();
</script>
</body>
</html>"""


if __name__ == "__main__":
    run()