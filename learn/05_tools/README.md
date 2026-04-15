# 第五关：工具（Tools）

工具是 Agent 的"手"——让 LLM 能执行真实操作（查数据库、调 API、计算等）。

## 三件套

### 1. @tool — 把函数变成工具

```python
from langchain_core.tools import tool

@tool
def get_dau(date: str) -> str:
    """获取指定日期的日活跃用户数。date 格式：YYYY-MM-DD"""
    # LLM 会读 docstring 来决定何时调用这个工具
    return f"{date} 的 DAU 是 52 万"
```

**两个关键点：**
- **docstring** 非常重要：LLM 靠它判断"什么时候该调这个工具"
- **参数类型注解** 很重要：LLM 据此生成正确格式的参数

### 2. bind_tools — 让 LLM 知道有哪些工具可用

```python
llm_with_tools = llm.bind_tools([get_dau, execute_sql])
```

调用后，LLM 可能返回 `AIMessage(tool_calls=[...])`，表示它想调工具。
**注意：** 这时 LLM 没有真正执行工具，只是"说"它要调哪个。

### 3. ToolNode — 自动执行工具并返回结果

```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([get_dau, execute_sql])
```

`ToolNode` 接收含 `tool_calls` 的 `AIMessage`，执行对应工具，
返回 `ToolMessage`（工具执行结果）。

## ReAct 循环（完整工具调用流程）

```
用户提问
    ↓
llm_node（LLM 思考：直接回答 or 调工具？）
    ↓
  有 tool_calls ──→ tool_node（执行工具）──→ llm_node（继续思考）
  无 tool_calls ──→ END（直接回复用户）
```

`tools_condition` 是 LangGraph 内置的路由函数，自动判断：
- LLM 最新回复有 `tool_calls` → 返回 `"tools"`
- 没有 `tool_calls` → 返回 `END`

## 消息类型说明

工具调用过程中会出现 4 种消息：

| 消息类型 | 谁产生 | 内容 |
|---------|-------|------|
| HumanMessage | 用户 | 原始问题 |
| AIMessage（含 tool_calls） | LLM | "我要调 get_dau" |
| ToolMessage | ToolNode | "get_dau 的结果是 52 万" |
| AIMessage（不含 tool_calls）| LLM | 最终回复给用户 |

## 运行顺序

```bash
# Step 1: 了解工具定义和元信息
conda run -n biagent python learn/05_tools/01_define_tool.py

# Step 2: 观察 LLM 如何选工具（tool_calls 格式）
conda run -n biagent python learn/05_tools/02_bind_and_call.py

# Step 3: 完整 ReAct 循环（工具真正被执行）
conda run -n biagent python learn/05_tools/03_tool_loop.py
```
