"""
第六关：Redis Checkpointer 多轮记忆

学习目标：
1. 用 AsyncRedisSaver 持久化对话状态到 Redis
2. 理解 thread_id 的作用（对话房间号）
3. 验证：两次独立运行之间，LLM 能记住之前说过的内容

运行方式（关键：运行两次，观察差异）：
    conda run -n biagent python learn/06_checkpointer/demo.py
    # 第一次：建立对话
    # 第二次：验证记忆
"""
import os
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── State & 图 ────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chat_node(state: State) -> dict:
    msg_count = len(state["messages"])
    print(f"[chat_node] 当前历史消息数（含所有轮次）：{msg_count}")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# ── 配置 ──────────────────────────────────────
# MemorySaver：内存存储，不依赖 Redis，进程结束后清空
# 生产环境换成 AsyncRedisSaver（需要 Redis Stack）
THREAD_ID = "langgraph_learn_demo_user_001"

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": THREAD_ID}}

# ── 第一轮：自我介绍 ──────────────────────────
print("=" * 50)
print("第一轮：告诉 AI 你的名字和职位")
print("=" * 50)
result = graph.invoke(
    {
        "messages": [
            SystemMessage(content="你是一个 BI 助手，回答要简短。"),
            HumanMessage(content="你好！我叫小明，是运营团队的数据分析师。"),
        ]
    },
    config=config,
)
print(f"AI 回复：{result['messages'][-1].content}")
print(f"本轮后历史总条数：{len(result['messages'])}")

# ── 第二轮：测试记忆 ──────────────────────────
print("\n" + "=" * 50)
print("第二轮：测试 AI 是否记得你的名字和职位")
print("=" * 50)
# 注意：只传新消息，不带历史
# checkpointer 通过 thread_id 自动恢复上一轮的历史
result = graph.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？我的职位是什么？")]},
    config=config,
)
print(f"AI 回复：{result['messages'][-1].content}")
print(f"本轮后历史总条数：{len(result['messages'])}")

# ── 第三轮：换一个 thread_id（不同对话，没有记忆）──
print("\n" + "=" * 50)
print("第三轮：换一个 thread_id，验证隔离性")
print("=" * 50)
other_config = {"configurable": {"thread_id": "other_user_999"}}
result = graph.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？")]},
    config=other_config,
)
print(f"AI 回复：{result['messages'][-1].content}")
# 预期：AI 不知道"小明"，因为是不同的 thread_id
