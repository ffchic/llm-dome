"""
第六关补充：MySQL 自定义持久化

演示：用 bi_sessions + bi_messages 手动管理对话历史
不依赖 LangGraph checkpointer，完全自己控制读写逻辑

核心思路：
  1. invoke 前：从 bi_messages 加载历史，重建消息列表
  2. invoke：图正常运行（用 MemorySaver 管单次内部状态）
  3. invoke 后：把新产生的消息写入 bi_messages，更新 bi_sessions

注意：SystemMessage（系统提示）不存库，每次由代码注入。
      bi_messages 只存 user / assistant / tool 三种角色。

运行方式：
    conda run -n biagent python learn/06_checkpointer/demo_mysql.py
    # 连续运行两次，第二次 AI 能记住第一次说的内容
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy import create_engine, text

# ── 加载配置 ──────────────────────────────────
# DeepSeek Key 来自 learn/.env，DB 连接来自 backend/.env
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "backend" / ".env")

# biagent_mysql_url 是 aiomysql（异步驱动），demo 用同步驱动替换
MYSQL_URL = os.getenv(
    "BIAGENT_MYSQL_URL",
    "mysql+pymysql://root@127.0.0.1:3306/biagent_db",
).replace("mysql+aiomysql", "mysql+pymysql")

engine = create_engine(MYSQL_URL)

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── LangGraph 图定义 ──────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: State) -> dict:
    print(f"[chat_node] 当前消息数（含历史）：{len(state['messages'])}")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()

SYSTEM_PROMPT = "你是一个 BI 数据分析助手，回答要简短。"


# ── MySQL 持久化工具函数 ───────────────────────

def ensure_session(session_id: str) -> None:
    """确保 bi_sessions 中存在该会话，不存在则创建。"""
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT id FROM bi_sessions WHERE session_id = :sid"),
            {"sid": session_id},
        ).fetchone()
        if not exists:
            now = datetime.now()
            conn.execute(
                text("""
                    INSERT INTO bi_sessions
                        (session_id, user_id, title, status, message_count, created_at, updated_at)
                    VALUES (:sid, 1, '演示对话', 'active', 0, :now, :now)
                """),
                {"sid": session_id, "now": now},
            )
            print(f"[DB] 新建会话：{session_id}")
        else:
            print(f"[DB] 加载已有会话：{session_id}")


def load_history(session_id: str) -> list:
    """从 bi_messages 加载历史消息，重建 LangChain 消息列表。"""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content FROM bi_messages
                WHERE session_id = :sid
                ORDER BY created_at ASC
            """),
            {"sid": session_id},
        ).fetchall()

    history = []
    for role, content in rows:
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    print(f"[DB] 加载历史消息：{len(history)} 条")
    return history


def save_messages(session_id: str, new_human: str, new_ai: str) -> None:
    """将本轮新产生的 user + assistant 消息写入 bi_messages，更新 bi_sessions。"""
    now = datetime.now()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO bi_messages (session_id, role, content, created_at)
                VALUES (:sid, 'user', :content, :ts)
            """),
            {"sid": session_id, "content": new_human, "ts": now},
        )
        conn.execute(
            text("""
                INSERT INTO bi_messages (session_id, role, content, created_at)
                VALUES (:sid, 'assistant', :content, :ts)
            """),
            {"sid": session_id, "content": new_ai, "ts": now},
        )
        conn.execute(
            text("""
                UPDATE bi_sessions
                SET message_count = message_count + 2,
                    last_message_at = :ts
                WHERE session_id = :sid
            """),
            {"sid": session_id, "ts": now},
        )
    print(f"[DB] 已保存本轮消息到 MySQL")


# ── 主流程 ────────────────────────────────────

# 固定 session_id，多次运行共用同一段历史
SESSION_ID = "demo-mysql-session-001"


def chat(user_input: str) -> str:
    """完整的一轮对话：加载历史 → invoke → 保存。"""
    ensure_session(SESSION_ID)

    # 1. 从 MySQL 加载历史
    history = load_history(SESSION_ID)

    # 2. 组装消息：SystemMessage（不存库）+ 历史 + 本轮用户输入
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=user_input)]

    # 3. 图执行（持久化完全由 MySQL 负责，不需要 checkpointer）
    result = graph.invoke({"messages": messages})

    ai_reply = result["messages"][-1].content

    # 4. 保存本轮新消息到 MySQL
    save_messages(SESSION_ID, user_input, ai_reply)

    return ai_reply


# ── 运行 ──────────────────────────────────────
# print("=" * 55)
# print(f"Session ID: {SESSION_ID}")
# print("=" * 55)

# user_input = "你好！我叫小明，是运营团队的数据分析师。"
# print(f"\n用户：{user_input}")
# reply = chat(user_input)
# print(f"AI  ：{reply}")

# print("\n" + "=" * 55)
# print("提示：再运行一次，把下面这行解注释，测试跨进程记忆")
# print("=" * 55)

# ── 第二轮（解注释后运行）────────────────────
user_input2 = "我叫什么名字？我的职位是什么？"
print(f"\n用户：{user_input2}")
reply2 = chat(user_input2)
print(f"AI  ：{reply2}")
# 预期：AI 回答"你叫小明，是运营团队的数据分析师"
# 因为第一次运行的消息已经存进了 MySQL，这次运行 load_history 能读到
