"""
第二关：LangGraph 最小图

学习目标：
1. 理解 StateGraph 是什么（带状态的有向图）
2. 理解节点（node）= 普通 Python 函数
3. 理解边（edge）= 执行顺序
4. 理解 compile() 和 invoke() 的作用

核心思想：
    LangGraph 的图 = 状态机
    节点 = 转换函数（读取 State，返回更新）
    边 = 转换规则（谁执行完了轮到谁）

运行方式：
    conda run -n biagent python learn/02_langgraph_core/demo.py
"""
import os
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ── 1. 定义 State ─────────────────────────────
# State 是贯穿整张图的"共享黑板"
# 所有节点都能读取和修改 State 中的字段
class State(TypedDict):
    input: str    # 用户原始输入
    refined: str  # 节点处理后的中间结果
    output: str   # 最终 LLM 回复


# ── 2. 定义节点 ───────────────────────────────
# 节点是普通 Python 函数，规则：
#   输入：整个 State（dict）
#   输出：一个 dict，只包含要更新的字段（其他字段保持不变）

def preprocess_node(state: State) -> dict:
    """节点1：预处理用户输入，加上引导性前缀"""
    user_input = state["input"]
    print(f"[preprocess_node] 原始输入：{user_input}")
    refined = f"请用一句话简洁回答以下运营问题：{user_input}"
    return {"refined": refined}  # 只更新 refined 字段


def llm_node(state: State) -> dict:
    """节点2：调用 LLM 生成回答"""
    print(f"[llm_node] 处理后的提示：{state['refined']}")
    response = llm.invoke([HumanMessage(content=state["refined"])])
    return {"output": response.content}  # 只更新 output 字段

    
# ── 3. 构建图 ─────────────────────────────────
# StateGraph(State) 告诉 LangGraph 图的状态结构
builder = StateGraph(State)

# 注册节点：名称 → 函数
builder.add_node("preprocess", preprocess_node)
builder.add_node("llm", llm_node)

# 添加边：定义执行顺序
# START 和 END 是 LangGraph 内置的特殊节点
builder.add_edge(START, "preprocess")    # 图从 preprocess 开始
builder.add_edge("preprocess", "llm")   # preprocess 完后执行 llm
builder.add_edge("llm", END)             # llm 完后结束


# ── 4. 编译 ───────────────────────────────────
# compile() 验证图的合法性（无孤立节点、有 START/END 等），
# 并生成可执行对象
graph = builder.compile()


# ── 5. 执行（invoke）─────────────────────────
print("=== [invoke] 等全部跑完，返回最终 State ===\n")

result = graph.invoke({
    "input": "今天的新增用户数是多少？",
    "refined": "",
    "output": "",
})

print(f"\n=== 最终 State ===")
print(f"input:   {result['input']}")
print(f"refined: {result['refined']}")
print(f"output:  {result['output']}")

# ── 6. 执行（stream）─────────────────────────
print("\n\n=== [stream] 每个节点执行完立刻输出 ===\n")

for step in graph.stream({
    "input": "今天的新增用户数是多少？",
    "refined": "",
    "output": "",
}):
    # step 格式：{节点名: 该节点返回的 dict}
    node_name, node_output = next(iter(step.items()))
    print(f"[{node_name}] 输出：{node_output}")

# ── 小结 ──────────────────────────────────────
print("""
=== 关键概念回顾 ===
StateGraph  → 带状态的有向图容器
add_node()  → 注册节点（名称 + 函数）
add_edge()  → 添加确定性边（A 完成后一定走 B）
compile()   → 冻结图结构，生成可执行对象
invoke()    → 同步执行整张图，返回最终 State
""")
