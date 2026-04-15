# 第八关：迷你 BIAgent（整合所有概念）

## 这一章做什么

把前 7 关学到的所有技术整合成一个真实可用的迷你 BI Agent：

1. 接收用户自然语言问题
2. 判断意图（数据查询 or 闲聊）
3. 数据查询时：先查元数据（bi_table_metadata）→ 再生成并执行 SQL
4. 流式输出分析结果

## 架构图

```
用户提问
    ↓
[intent_node] ─── 判断意图
    ↓                  ↓
[execute_node]    [chat_node]
  （data_query）   （general_chat）
    ↓                  ↓
  调工具循环         直接回复
  query_metadata       ↓
  execute_sql         END
    ↓
[respond_node] ── 生成最终回复
    ↓
   END
```

## 文件职责

| 文件 | 职责 |
|------|------|
| state.py | AgentState：消息历史 + 意图字段 |
| tools.py | query_metadata（查元数据）+ execute_sql（只读SQL） |
| nodes.py | 4个节点的实现 + ToolNode |
| graph.py | 图组装 + Redis Checkpointer + 路由函数 |
| run.py | 命令行入口，流式打印结果 |

## 与真实 BIAgent 的关系

这一章是真实 BIAgent 的**简化原型**，核心逻辑完全一致：

| 真实 BIAgent | 本章迷你版 |
|-------------|----------|
| 5个节点（intent/plan/execute/analyze/respond）| 4个节点 |
| FastAPI SSE 接口 | 命令行直接打印 |
| bi_sessions 会话管理 | 固定 thread_id |
| 完整错误处理和日志 | 最小实现 |

读完这一章，再去看真实项目的 `backend/app/agent/` 目录，你会发现非常眼熟。

## 前置条件

1. **biagent_db** 的 `bi_table_metadata` 表中有数据
   （通过前端管理页面 `localhost:5174` 录入，或手动 INSERT）
2. **game_db** 已创建（可以是空库，SQL 会返回"无结果"）
3. **Redis** 在 6379 端口运行（`redis-server --daemonize yes`）
4. **backend/.env** 中有 DB 连接信息（`BIAGENT_DB_URL`、`GAME_DB_URL`）

## 运行

```bash
# 从项目根目录，以模块方式运行
cd /path/to/BIAgent
conda run -n biagent python -m learn.08_mini_biagent.run

# 或者带问题参数
conda run -n biagent python -m learn.08_mini_biagent.run "充值相关的表有哪些？"
conda run -n biagent python -m learn.08_mini_biagent.run "你好，你能做什么？"
```

## 注意事项

- 必须从项目根目录以 `-m` 方式运行（因为用了相对导入 `from .state import ...`）
- `tools.py` 使用同步 SQLAlchemy（pymysql），避免 async 工具的复杂性
- `backend/.env` 中的连接串是 `mysql+aiomysql://...`，tools.py 会自动替换为 `mysql+pymysql://...`
