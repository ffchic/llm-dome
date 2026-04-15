"""
第四关：条件路由（conditional_edges）

学习目标：
1. 用 add_conditional_edges 让图根据状态走不同分支
2. 路由函数：接收 State，返回下一个节点的名称（字符串）
3. 理解"意图识别 → 路由"是 Agent 的核心模式

核心思想：
    普通边：A → B（固定）
    条件边：A → f(State) → B 或 C 或 D（动态）

运行方式：
    conda run -n biagent python learn/04_conditional_routing/demo.py
"""
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict
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

# ── State ─────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str  # 识别出的意图，路由函数会读取这个字段


# ── 节点 ──────────────────────────────────────

def intent_node(state: State) -> dict:
    """识别用户意图：data_query 或 general_chat"""
    user_msg = state["messages"][-1].content
    prompt = f"""判断以下用户输入的意图，只返回一个词（不要其他任何内容）：
- data_query：涉及数据查询（充值、用户、收入、DAU、GMV 等指标）
- general_chat：普通问候或非数据问题

用户输入：{user_msg}
意图："""
    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()
    # 容错：LLM 可能返回带解释的文字，提取关键词
    intent = "data_query" if "data_query" in intent else "general_chat"
    print(f"[intent_node] 识别意图：{intent}")
    return {"intent": intent}


def data_query_node(state: State) -> dict:
    """数据查询分支：模拟返回数据（后续章节会替换成真实工具）"""
    print("[data_query_node] 走数据查询分支 ✓")
    fake_result = "（模拟数据）今日 DAU：52 万，充值：156.8 万元，留存率：38%"
    return {
        "messages": [
            HumanMessage(content=fake_result, name="data_system")
        ]
    }


def general_chat_node(state: State) -> dict:
    """闲聊分支：直接 LLM 回复"""
    print("[general_chat_node] 走闲聊分支 ✓")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ── 路由函数 ──────────────────────────────────
# 路由函数规则：
#   输入：当前 State
#   输出：下一个节点的名称（字符串）
# Literal 类型注解是可选的，但让返回值更清晰
def route_by_intent(state: State) -> Literal["data_query", "general_chat"]:
    """读取 intent 字段，决定走哪条分支"""
    return "data_query" if state["intent"] == "data_query" else "general_chat"


# ── 构建图 ────────────────────────────────────
builder = StateGraph(State)
builder.add_node("intent", intent_node)
builder.add_node("data_query", data_query_node)
builder.add_node("general_chat", general_chat_node)

builder.add_edge(START, "intent")

# add_conditional_edges(
#   source_node,      ← 从哪个节点出发
#   routing_function, ← 路由函数（返回字符串）
#   path_map,         ← {返回值 → 目标节点名}
# )
builder.add_conditional_edges(
    "intent",
    route_by_intent,
    {
        "data_query": "data_query",
        "general_chat": "general_chat",
    },
)
builder.add_edge("data_query", END)
builder.add_edge("general_chat", END)

graph = builder.compile()


# ── 测试两种意图 ──────────────────────────────
test_cases = [
    "今天的充值金额和 DAU 各是多少？",
    "你好啊，今天心情怎么样？",
    "最近 7 天的新增用户趋势如何？",
]

for user_input in test_cases:
    print(f"\n{'='*55}")
    print(f"用户：{user_input}")
    result = graph.invoke({
        "messages": [HumanMessage(content=user_input)],
        "intent": "",
    })
    last = result["messages"][-1]
    print(f"回复：{last.content[:100]}")
