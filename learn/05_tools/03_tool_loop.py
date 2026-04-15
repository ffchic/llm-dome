"""
第五关 Step 3：完整 ReAct 循环

学习目标：
1. ToolNode 自动执行 LLM 选择的工具
2. tools_condition 内置路由：有 tool_calls → tools，没有 → END
3. 理解整个 ReAct 循环：LLM 思考 → 调工具 → 看结果 → 继续思考 → 回复

执行流程：
    START → llm_node → tools_condition ─┬─(有 tool_calls)→ tool_node → llm_node
                                         └─(无 tool_calls)→ END

运行方式：
    conda run -n biagent python learn/05_tools/03_tool_loop.py
"""
import os
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── 工具定义（★ 打印日志，方便观察工具何时被执行）────
@tool
def get_dau(date: str) -> str:
    """获取指定日期的日活跃用户数（DAU）。date 格式：YYYY-MM-DD"""
    print(f"  ★ [工具执行] get_dau(date={date!r})")
    return f"{date} 的 DAU 是 52 万"

@tool
def get_revenue(date: str) -> str:
    """获取指定日期的充值收入总额。date 格式：YYYY-MM-DD"""
    print(f"  ★ [工具执行] get_revenue(date={date!r})")
    return f"{date} 的充值收入是 156.8 万元"

tools = [get_dau, get_revenue]
llm_with_tools = llm.bind_tools(tools)


# ── State ─────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── 节点 ──────────────────────────────────────
def llm_node(state: State) -> dict:
    """LLM 思考：决定直接回答还是调工具"""
    # 统计 LLM 被调用的次数（通过含 tool_calls 属性的消息数量估算）
    call_count = sum(
        1 for m in state["messages"] if hasattr(m, "tool_calls")
    ) + 1
    print(f"\n[llm_node] 第 {call_count} 次调用，消息数：{len(state['messages'])}")

    response = llm_with_tools.invoke(state["messages"])
    has_tools = bool(getattr(response, "tool_calls", []))
    print(f"[llm_node] 决定调工具：{has_tools}")
    if has_tools:
        names = [tc["name"] for tc in response.tool_calls]
        print(f"[llm_node] 准备调用：{names}")
    return {"messages": [response]}

# ToolNode：自动找到工具 → 传入 LLM 生成的参数 → 返回 ToolMessage
tool_node = ToolNode(tools)


# ── 构建图 ────────────────────────────────────
builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")

# tools_condition 是内置路由函数：
# - 最新 AIMessage 有 tool_calls → 返回 "tools"
# - 没有 tool_calls → 返回 END
builder.add_conditional_edges("llm", tools_condition)

# 工具执行完后，把 ToolMessage 送回给 LLM 继续推理
builder.add_edge("tools", "llm")

graph = builder.compile()


# ── 执行 ──────────────────────────────────────
print("用户：2024-01-15 的日活和充值各是多少，帮我做个简单汇总。\n")

result = graph.invoke({
    "messages": [
        SystemMessage(content="你是一个 BI 数据分析助手，用中文简洁回答。"),
        HumanMessage(content="2024-01-15 的日活和充值各是多少，帮我做个简单汇总。"),
    ]
})

print(f"\n{'='*55}")
print("最终回复：")
print(result["messages"][-1].content)

print(f"\n{'='*55}")
print(f"完整消息历史（共 {len(result['messages'])} 条）：")
for msg in result["messages"]:
    name = type(msg).__name__
    content_preview = str(getattr(msg, "content", ""))[:60]
    tool_calls_info = ""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls_info = f" → 调工具: {[tc['name'] for tc in msg.tool_calls]}"
    print(f"  [{name:15s}] {content_preview}{tool_calls_info}")

# 预期消息历史：
# [SystemMessage    ] 系统提示
# [HumanMessage     ] 用户问题
# [AIMessage        ] → 调工具: ['get_dau', 'get_revenue']  （或分两次调）
# [ToolMessage      ] get_dau 结果
# [ToolMessage      ] get_revenue 结果
# [AIMessage        ] 最终汇总回复
