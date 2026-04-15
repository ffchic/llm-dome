"""
迷你 BIAgent 的节点定义

节点列表：
- intent_node：判断意图（data_query / general_chat）
- execute_node：调工具查数据（在 data_query 分支中循环执行）
- chat_node：普通闲聊回复（在 general_chat 分支中执行）
- respond_node：汇总工具结果，生成最终人性化回复
- tool_node：ToolNode 实例（自动执行 LLM 选中的工具）
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from .tools import query_metadata, execute_sql

# 加载 backend/.env（DB URL）和 learn/.env（DeepSeek Key）
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "backend" / ".env")
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    streaming=True,
)

_tools = [query_metadata, execute_sql]
_llm_with_tools = _llm.bind_tools(_tools)

# ToolNode：自动执行 LLM 选中的工具，返回 ToolMessage
tool_node = ToolNode(_tools)


def intent_node(state: AgentState) -> dict:
    """判断用户意图：data_query 或 general_chat"""
    user_msg = state["messages"][-1].content
    prompt = f"""判断以下用户输入的意图，只返回一个词（不加任何解释）：
- data_query：想查询游戏数据（充值、用户数、留存、DAU、GMV、收入、表结构等）
- general_chat：非数据问题（问候、闲聊、功能询问等）

用户输入：{user_msg}
意图："""
    response = _llm.invoke([HumanMessage(content=prompt)])
    intent = "data_query" if "data_query" in response.content.lower() else "general_chat"
    print(f"  [intent_node] 意图：{intent}")
    return {"intent": intent}


def execute_node(state: AgentState) -> dict:
    """数据查询节点：让 LLM 用工具查元数据并执行 SQL

    执行逻辑：
    1. 首次进入：LLM 调用 query_metadata 查相关表
    2. 看到元数据后：LLM 调用 execute_sql 执行 SQL
    3. 工具结果返回后：LLM 决定是否还需要继续调工具
    """
    system_msg = SystemMessage(content="""你是一个 BI 数据分析助手，帮助运营团队查询游戏数据。

工作步骤：
1. 先用 query_metadata 工具搜索相关表的元数据（了解有哪些表和字段）
2. 根据元数据，构造合适的 SELECT SQL，用 execute_sql 工具执行
3. 如果 SQL 结果为空，尝试调整查询条件

注意：只能执行 SELECT，不能修改数据。""")

    # 把系统提示加到消息历史前面
    messages = [system_msg] + list(state["messages"])
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


def chat_node(state: AgentState) -> dict:
    """普通闲聊节点：直接 LLM 回复"""
    system_msg = SystemMessage(content="你是一个友好的 BI 数据分析助手，用中文回答，简洁清晰。")
    messages = [system_msg] + list(state["messages"])
    response = _llm.invoke(messages)
    return {"messages": [response]}


def respond_node(state: AgentState) -> dict:
    """汇总节点：根据工具执行结果，生成最终人性化回复"""
    system_msg = SystemMessage(content="""你是一个 BI 数据分析助手。

根据对话历史中的数据查询结果，用通俗易懂的中文给出分析总结：
- 直接说明数据结果
- 如有多个指标，用列表或对比形式呈现
- 如果查询无结果，诚实说明，并建议用户检查数据或调整问题
- 回答要简洁，避免冗余""")

    messages = [system_msg] + list(state["messages"])
    response = _llm.invoke(messages)
    return {"messages": [response]}
