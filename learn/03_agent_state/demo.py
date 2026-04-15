"""
第三关：AgentState 与 add_messages reducer

学习目标：
1. 用 Annotated + add_messages 声明消息字段
2. 观察多轮对话中消息如何追加而不是覆盖
3. 理解节点只需返回"增量"，reducer 负责合并

运行方式：
    conda run -n biagent python learn/03_agent_state/demo.py
"""
import os
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── State 定义 ────────────────────────────────
# Annotated[list, add_messages] 的含义：
#   - 类型是 list
#   - 当节点返回 {"messages": [...]} 时，
#     LangGraph 用 add_messages 函数合并（追加），而不是覆盖
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── 节点 ──────────────────────────────────────
def chat_node(state: AgentState) -> dict:
    """调用 LLM，打印当前消息数，返回新消息（增量）"""
    msg_count = len(state["messages"])
    print(f"\n[chat_node] 调用前消息数：{msg_count}")
    for i, msg in enumerate(state["messages"]):
        label = type(msg).__name__
        print(f"  [{i}] {label:15s}: {str(msg.content)[:60]}")

    response = llm.invoke(state["messages"])

    # 关键：只返回新消息
    # add_messages reducer 会把 response 追加到 state["messages"] 末尾
    return {"messages": [response]}


# ── 图 ────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()


# ── 第一轮对话 ────────────────────────────────
print("=" * 50)
print("第一轮：自我介绍")
state = graph.invoke({
    "messages": [
        SystemMessage(content="你是一个 BI 数据分析助手，回答简短。"),
        HumanMessage(content="你好，你能帮我做什么？"),
    ]
})
print(f"\n第一轮 AI 回复：{state['messages'][-1].content}")
print(f"第一轮后消息总数：{len(state['messages'])}")
# 预期：3 条（System + Human + AI）


# ── 第二轮对话（在第一轮基础上继续）────────────
# 把第一轮的完整历史 + 新问题传给图
# 节点看到的 messages 里包含所有历史，LLM 能"记住"上文
print("\n" + "=" * 50)
print("第二轮：追问（沿用第一轮历史）")
new_messages = state["messages"] + [HumanMessage(content="能查充值数据吗？")]
state = graph.invoke({"messages": new_messages})
print(f"\n第二轮 AI 回复：{state['messages'][-1].content}")
print(f"第二轮后消息总数：{len(state['messages'])}")
# 预期：5 条（第一轮3条 + 新Human + 新AI）


# ── 手动验证 add_messages 的行为 ──────────────
print("\n" + "=" * 50)
print("手动验证 add_messages（追加而非覆盖）")
from langgraph.graph.message import add_messages as _add_messages

existing = [HumanMessage(content="旧消息1"), AIMessage(content="旧消息2")]
new = [HumanMessage(content="新消息3")]
merged = _add_messages(existing, new)

print(f"合并前：{len(existing)} 条")
print(f"合并后：{len(merged)} 条")   # 应该是 3，不是 1
print(f"内容：{[m.content for m in merged]}")
# 结论：add_messages 追加，而不是覆盖
