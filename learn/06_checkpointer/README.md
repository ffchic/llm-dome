# 第六关：Redis Checkpointer（多轮记忆）

## 问题：没有 Checkpointer 的 Agent 是"金鱼记忆"

没有 checkpointer，每次 `graph.invoke()` 都是全新的、独立的状态，
Agent 完全不记得上一次说了什么。

## 解决方案：Checkpointer + thread_id

Checkpointer 是 LangGraph 的持久化层。
每次图执行后，它把完整状态保存到存储后端（这里是 Redis）。
下次执行时，通过 `thread_id` 恢复这段历史。

```python
config = {"configurable": {"thread_id": "user_001_session_1"}}
```

`thread_id` 相当于"对话房间号"：
- 同一 thread_id → 共享同一段对话历史
- 不同 thread_id → 完全独立，互不干扰

## 工作原理

```
graph.ainvoke(new_input, config={"configurable": {"thread_id": "xxx"}})
    ↓
1. 从 Redis 加载 thread_id 的最新 checkpoint（如果有）
2. 把历史状态 + 新输入合并
3. 执行图
4. 把新状态写回 Redis（存为新 checkpoint）
```

## AsyncRedisSaver 用法

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

# 必须用 async with，确保连接被正确关闭
async with AsyncRedisSaver.from_conn_string("redis://localhost:6379/1") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "my_thread"}}
    result = await graph.ainvoke(input, config=config)
```

## 本项目配置

使用 `biagent_redis`（db=1），与 game_redis（db=0）隔离。

## 安装

```bash
conda run -n biagent pip install langgraph-checkpoint-redis
```

## 运行（关键：运行两次！）

```bash
# 第一次运行：建立对话，告诉 AI 你的信息
conda run -n biagent python learn/06_checkpointer/demo.py

# 第二次运行：验证 AI 是否记得第一次的内容
conda run -n biagent python learn/06_checkpointer/demo.py
```

## 注意事项

- 图必须用 `builder.compile(checkpointer=checkpointer)` 编译（不能事后注入）
- 每次 `ainvoke` 都要传相同的 `config`（含 thread_id）
- `AsyncRedisSaver` 是异步的，整个调用链都需要 `async/await`
