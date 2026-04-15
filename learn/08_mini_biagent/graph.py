"""
迷你 BIAgent 的图组装

包含：
- 路由函数（按意图路由 / execute 后路由）
- 图构建函数 build_graph()
- Redis Checkpointer 配置
"""
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import intent_node, execute_node, chat_node, respond_node, tool_node

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "backend" / ".env")

# biagent_redis（db=1）
REDIS_URL = os.getenv("BIAGENT_REDIS_URL", "redis://localhost:6379/1")


# ── 路由函数 ──────────────────────────────────

def route_by_intent(state: AgentState) -> Literal["execute", "chat"]:
    """intent_node 之后：根据意图字段路由"""
    return "execute" if state["intent"] == "data_query" else "chat"


def route_after_execute(state: AgentState) -> Literal["tools", "respond"]:
    """execute_node 之后：
    - LLM 还想调工具 → tools
    - LLM 不再调工具 → respond（汇总输出）
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "respond"


# ── 图构建 ────────────────────────────────────

def build_graph(checkpointer=None):
    """构建并编译迷你 BIAgent 图

    Args:
        checkpointer: 可选，传入 AsyncRedisSaver 实例则开启多轮记忆
    """
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("intent", intent_node)
    builder.add_node("execute", execute_node)
    builder.add_node("tools", tool_node)
    builder.add_node("chat", chat_node)
    builder.add_node("respond", respond_node)

    # 连接节点
    builder.add_edge(START, "intent")

    builder.add_conditional_edges(
        "intent",
        route_by_intent,
        {"execute": "execute", "chat": "chat"},
    )

    builder.add_conditional_edges(
        "execute",
        route_after_execute,
        {"tools": "tools", "respond": "respond"},
    )

    # tools 执行完后把结果送回 execute 继续推理（ReAct 循环）
    builder.add_edge("tools", "execute")
    builder.add_edge("chat", END)
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)
