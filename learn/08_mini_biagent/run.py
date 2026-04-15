"""
迷你 BIAgent 命令行入口

流式输出 AI 回复到终端。

运行方式（从项目根目录执行）：
    conda run -n biagent python -m learn.08_mini_biagent.run
    conda run -n biagent python -m learn.08_mini_biagent.run "充值相关的表有哪些？"
    conda run -n biagent python -m learn.08_mini_biagent.run "你好，你能做什么？"
"""
import sys
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from .graph import build_graph, REDIS_URL

THREAD_ID = "mini_biagent_learn_001"


async def main():
    # 从命令行参数读取用户输入，默认问一个引导性的问题
    user_input = sys.argv[1] if len(sys.argv) > 1 else "你好，你能帮我查什么数据？"

    print(f"用户：{user_input}")
    print(f"AI：", end="", flush=True)

    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        # 用 astream_events 实现流式输出
        # 这正是 BIAgent FastAPI SSE 接口的核心逻辑
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    print(chunk.content, end="", flush=True)

    print()  # 换行
    print(f"\n（Thread ID: {THREAD_ID}，对话历史已保存到 Redis）")


asyncio.run(main())
