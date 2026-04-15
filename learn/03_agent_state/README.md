# 第三关：AgentState 深度解析

## 为什么节点不能直接覆盖 messages？

假设图里有两个节点先后往 `messages` 写数据：

```python
# 节点 A 返回：{"messages": [msg_a]}
# 节点 B 返回：{"messages": [msg_b]}
# 默认合并（字典更新）：messages = [msg_b]  ← msg_a 被覆盖！
```

在对话 Agent 里，这是致命问题——每次 LLM 回复都会清掉历史记录，
LLM 会"失忆"。

## Reducer：告诉 LangGraph 如何合并字段

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    #                         ^^^^^^^^^^^
    #         Reducer 函数，告诉 LangGraph：
    #         合并 messages 时，把新消息追加到列表末尾，
    #         而不是替换整个列表
```

`add_messages` 的实际行为：
- 新消息的 id 不存在 → **追加**（新增对话）
- 新消息的 id 已存在 → **替换**（更新某条消息，用于流式场景）
- 同时支持 `BaseMessage` 对象和原始 `dict` 两种格式

## 节点只需返回"增量"

```python
def my_node(state: AgentState) -> dict:
    # state["messages"] 包含完整历史
    response = llm.invoke(state["messages"])
    
    # 只返回新消息，不需要手动追加到列表
    # add_messages reducer 会自动处理合并
    return {"messages": [response]}
```

## MessagesState：内置快捷方式

```python
from langgraph.graph import MessagesState

# 等价于：
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

BIAgent 的 AgentState 继承 MessagesState 再加额外字段（intent、sql_result 等）。

## 与第二关的对比

| 第二关 State | 第三关 AgentState |
|-------------|-----------------|
| 普通 TypedDict | Annotated + Reducer |
| 字段更新 = 覆盖 | messages 更新 = 追加 |
| 手动维护消息历史 | 自动追加 |
| 适合简单流水线 | 适合对话 Agent |

## 运行 demo.py

```bash
conda run -n biagent python learn/03_agent_state/demo.py
```

注意观察每步打印的消息数量，验证历史是追加而不是覆盖。
