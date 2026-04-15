"""
第七关：流式输出（astream_events）

学习目标：
1. 用 astream_events 迭代图执行过程中产生的所有事件
2. 过滤 on_chat_model_stream 事件，实时打印每个 token
3. 理解流式 vs 非流式的体验差异

为什么重要：
    BIAgent 的 FastAPI 接口会用 SSE（Server-Sent Events）把流式数据
    推送给前端。本章是 SSE 的基础。
    核心就是：for each event in astream_events → 过滤 → 推送

运行方式：
    conda run -n biagent python learn/07_streaming/demo.py
"""
import os
import asyncio
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
    streaming=True,  # 必须开启，否则 on_chat_model_stream 事件不会触发
)

# ── 图（和前几章一样的最简结构）────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chat_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()


async def demo_streaming():
    """演示1：流式输出（一字一字打印）"""
    question = "用 150 字左右介绍老虎机游戏最关键的三个运营指标（DAU、充值、留存），以及为什么它们重要。"

    print(f"用户：{question}")
    print("\nAI（流式输出）：")
    print("-" * 50)

    collected_tokens = []

    # astream_events：异步迭代图执行中产生的所有事件
    # version="v2" 是当前推荐版本
    async for event in graph.astream_events(
        {
            "messages": [
                SystemMessage(content="你是一个游戏运营数据分析专家，用中文回答。"),
                HumanMessage(content=question),
            ]
        },
        version="v2",
    ):
        # 每个 event 是一个 dict，关键字段：
        #   event["event"]  → 事件类型（字符串）
        #   event["name"]   → 触发该事件的组件名
        #   event["data"]   → 事件数据

        if event["event"] == "on_chat_model_stream":
            # 每生成一个 token（通常是1-3个汉字）触发一次
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)  # flush 确保即时输出
                collected_tokens.append(chunk.content)

    print()  # 换行
    print("-" * 50)
    full_text = "".join(collected_tokens)
    print(f"（流式完成，共 {len(full_text)} 字）")
    return full_text


async def demo_all_events():
    """演示2：打印图执行中的所有事件类型（帮助理解事件体系）"""
    print("\n\n" + "=" * 50)
    print("图执行中的所有事件类型：")
    print("=" * 50)

    event_types = {}
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="用一个字回答：好")]},
        version="v2",
    ):
        et = event["event"]
        name = event.get("name", "")
        if et not in event_types:
            event_types[et] = name

    for et, name in sorted(event_types.items()):
        marker = " ← 我们用这个过滤 token" if et == "on_chat_model_stream" else ""
        print(f"  {et:35s} (触发者: {name}){marker}")


async def demo_sse_simulation():
    """演示3：模拟 SSE 推送格式（FastAPI 的 SSE 就长这样）"""
    print("\n\n" + "=" * 50)
    print("模拟 SSE 推送格式（data: <token>\\n\\n）：")
    print("=" * 50)

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="用两句话介绍什么是DAU")]},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                # SSE 格式：data: <内容>\n\n
                sse_line = f"data: {chunk.content}\n\n"
                print(sse_line, end="", flush=True)
                await asyncio.sleep(0.3)  # 放慢，方便观察每行格式


async def main():
    await demo_streaming()
    await demo_all_events()
    await demo_sse_simulation()

asyncio.run(main())
