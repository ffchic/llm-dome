"""
第三关补充：两个节点的多轮对话

演示：
  START → chat_1（第一轮）→ chat_2（第二轮）→ END

chat_1 和 chat_2 共享同一个 messages 列表（add_messages 自动追加）
chat_2 调用 LLM 时能看到 chat_1 的回复历史，实现"对话延续"

运行方式：
    conda run -n biagent python learn/03_agent_state/demo_two_nodes.py
"""
import os
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chat_1(state: State) -> dict:
    """第一轮：用户问题 → LLM 第一次回复"""
    print(f"\n[chat_1] 收到消息数：{len(state['messages'])}")
    response = llm.invoke(state["messages"])
    print(f"[chat_1] LLM 回复：{response.content[:60]}...")
    return {"messages": [response]}
    # add_messages 自动把 response 追加到 messages 末尾


def chat_2(state: State) -> dict:
    """第二轮：在第一轮历史基础上追加新问题，再调一次 LLM"""
    # 此时 state["messages"] 已包含 chat_1 的回复
    print(f"\n[chat_2] 收到消息数：{len(state['messages'])}")  # 比 chat_1 多1条

    # 追加一个新问题
    follow_up = HumanMessage(content="那具体怎么查？给我一个思路。")
    all_messages = state["messages"] + [follow_up]

    response = llm.invoke(all_messages)
    print(f"[chat_2] LLM 回复：{response.content[:60]}...")

    # 返回新问题 + 新回复，add_messages 追加到历史
    return {"messages": [follow_up, response]}


# ── 构建图 ────────────────────────────────────
builder = StateGraph(State)
builder.add_node("chat_1", chat_1)
builder.add_node("chat_2", chat_2)
builder.add_edge(START, "chat_1")
builder.add_edge("chat_1", "chat_2")
builder.add_edge("chat_2", END)
graph = builder.compile()


# ── 执行 ──────────────────────────────────────
print("=" * 50)
initial_messages = [
    SystemMessage(content="你是一个 BI 数据分析助手，回答简短。"),
    HumanMessage(content="我想查最近7天的充值总额，可以吗？"),
]

result = graph.invoke({"messages": initial_messages})

print("\n" + "=" * 50)
print("=== 最终消息历史 ===")
for i, msg in enumerate(result["messages"]):
    label = type(msg).__name__
    print(f"[{i}] {label:15s}: {str(msg.content)[:80]}")

print(f"\n消息总数：{len(result['messages'])}")
# 预期：5 条（System + Human + AI + Human + AI）
