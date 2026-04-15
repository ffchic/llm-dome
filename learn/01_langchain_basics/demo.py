"""
第一关：用 LangChain 调通 DeepSeek

学习目标：
1. 理解 BaseMessage 体系（HumanMessage / SystemMessage / AIMessage）
2. 用 ChatOpenAI 适配 DeepSeek API（OpenAI 兼容协议）
3. 理解 invoke() 的输入和输出

运行方式：
    conda run -n biagent python learn/01_langchain_basics/demo.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 加载 learn/.env（相对于本文件向上两层到项目根，再进 learn/）
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── 初始化 LLM ────────────────────────────────
# DeepSeek 兼容 OpenAI 协议，只需替换 base_url 和 model
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── 构造消息 ──────────────────────────────────
# LangChain 用消息列表代替原始字符串，每条消息有明确角色：
# - SystemMessage：系统提示，设定 AI 的角色和行为
# - HumanMessage：用户说的话
# - AIMessage：AI 的回复（invoke 后才有）
messages = [
    SystemMessage(content="你是一个帮助运营团队做数据分析的 BI 助手，回答要简洁。"),
    HumanMessage(content="你好，能帮我查今天的日活跃用户数吗？"),
]

print("=== 发送的消息 ===")
for msg in messages:
    print(f"[{type(msg).__name__:15s}] {msg.content}")

# ── 调用 LLM ──────────────────────────────────
print("\n=== 调用 LLM ===")
response: AIMessage = llm.invoke(messages)

print(f"\n回复内容：{response.content}")
print(f"消耗 tokens：{response.usage_metadata}")
print(f"response 类型：{type(response).__name__}")

# ── 多轮对话：手动追加消息历史 ────────────────
# 注意：这里要手动维护历史。第三关会用 add_messages 自动处理
print("\n\n=== 多轮对话演示 ===")
history = messages + [response]           # 把 AI 回复追加到历史
history.append(HumanMessage(content="那这个数据是从哪里来的？"))

response2 = llm.invoke(history)
print(f"第二轮回复：{response2.content}")
print(f"\n消息历史总条数：{len(history) + 1}")
# 可以看到：每轮对话都要把完整历史传给 LLM，LLM 才能"记得"上文
