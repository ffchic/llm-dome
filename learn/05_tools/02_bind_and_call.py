"""
第五关 Step 2：让 LLM 决定调哪个工具

学习目标：
1. bind_tools() 的作用：把工具 schema 注入 LLM 请求
2. 观察 AIMessage.tool_calls 的格式
3. 区分"LLM 说要调工具"和"工具真的被执行"（两个步骤）

运行方式：
    conda run -n biagent python learn/05_tools/02_bind_and_call.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── 工具定义 ──────────────────────────────────
@tool
def get_dau(date: str) -> str:
    """获取指定日期的日活跃用户数（DAU）。date 格式：YYYY-MM-DD"""
    return f"{date} 的 DAU 是 52 万"

@tool
def get_revenue(date: str) -> str:
    """获取指定日期的充值收入总额。date 格式：YYYY-MM-DD"""
    return f"{date} 的充值收入是 156.8 万元"

# bind_tools：把工具的 JSON Schema 注入到每次 LLM 请求
# 这样 LLM 知道"我有这些工具可以调用"
llm_with_tools = llm.bind_tools([get_dau, get_revenue])


# ── 测试1：问 DAU，LLM 应选 get_dau ──────────
print("=" * 55)
print("测试1：问 DAU（预期：LLM 选 get_dau）")
print("=" * 55)
response1 = llm_with_tools.invoke([HumanMessage(content="2024-01-15 的日活是多少？")])

print(f"content（回复文字）: {repr(response1.content)}")
# 通常是空字符串 ""，因为 LLM 决定调工具而不是直接回答
print(f"tool_calls: {response1.tool_calls}")
# 格式示例：
# [{'name': 'get_dau', 'args': {'date': '2024-01-15'}, 'id': 'call_xxx', 'type': 'tool_call'}]


# ── 测试2：问收入，LLM 应选 get_revenue ──────
print("\n" + "=" * 55)
print("测试2：问充值（预期：LLM 选 get_revenue）")
print("=" * 55)
response2 = llm_with_tools.invoke([HumanMessage(content="昨天（2024-01-14）的充值是多少？")])
print(f"content: {repr(response2.content)}")
print(f"tool_calls: {response2.tool_calls}")


# ── 测试3：普通问题，不调工具 ─────────────────
print("\n" + "=" * 55)
print("测试3：普通问题（预期：不调工具，直接回答）")
print("=" * 55)
response3 = llm_with_tools.invoke([HumanMessage(content="你是什么 AI？用一句话说。")])
print(f"content: {repr(response3.content[:100])}")
print(f"tool_calls: {response3.tool_calls}")   # 应为空列表 []


# ── 手动执行 LLM 选中的工具 ───────────────────
# 下一个脚本（03_tool_loop.py）会用 ToolNode 自动做这件事
# 这里先手动演示，方便理解
print("\n" + "=" * 55)
print("手动执行 LLM 选中的工具（理解 ToolNode 的本质）")
print("=" * 55)
tools_map = {"get_dau": get_dau, "get_revenue": get_revenue}

for tc in response1.tool_calls:
    print(f"工具名称：{tc['name']}")
    print(f"LLM 生成的参数：{tc['args']}")
    result = tools_map[tc["name"]].invoke(tc["args"])
    print(f"工具执行结果：{result}")
