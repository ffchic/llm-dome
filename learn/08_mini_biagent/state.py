"""
AgentState：迷你 BIAgent 的共享状态

对应第三关学到的 TypedDict + Annotated + add_messages。
"""
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（add_messages 自动追加）
    intent: str   # 识别出的意图：data_query / general_chat
