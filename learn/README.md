# LangGraph 学习路线

本文件夹是 BIAgent 项目的配套学习材料，从零开始带你掌握 LangGraph。
所有文件均为**本地私有**，不提交 git。

## 学习顺序

| 章节 | 目录 | 核心概念 |
|------|------|---------|
| 第一关 | 01_langchain_basics/ | BaseMessage、ChatOpenAI、DeepSeek API |
| 第二关 | 02_langgraph_core/ | StateGraph、节点、边、图的执行 |
| 第三关 | 03_agent_state/ | TypedDict、Annotated、add_messages reducer |
| 第四关 | 04_conditional_routing/ | conditional_edges、路由函数 |
| 第五关 | 05_tools/ | @tool、bind_tools、ToolNode、ReAct 循环 |
| 第六关 | 06_checkpointer/ | Redis Checkpointer、thread_id、多轮记忆 |
| 第七关 | 07_streaming/ | astream_events、流式输出 |
| 第八关 | 08_mini_biagent/ | 整合：完整迷你 BI Agent |

## 环境准备

### 1. 复制并填写 .env 文件

```bash
cp learn/.env.example learn/.env
# 然后编辑 learn/.env，填入你的 DeepSeek API Key
```

### 2. 安装依赖（在 biagent conda 环境中）

```bash
conda run -n biagent pip install langchain-openai langgraph langgraph-checkpoint-redis python-dotenv
```

第 8 章还需要（项目中通常已安装）：

```bash
conda run -n biagent pip install sqlalchemy aiomysql pymysql
```

### 3. 运行任意 demo

```bash
# 方式一：激活环境后直接运行
conda activate biagent
python learn/01_langchain_basics/demo.py

# 方式二：不激活，直接用 conda run（推荐）
conda run -n biagent python learn/01_langchain_basics/demo.py
```

### 4. 第 6、8 章需要额外配置

- **Redis**：确保本地 Redis 在 6379 端口运行（`redis-server --daemonize yes`）
- **MySQL**：确保 biagent_db 和 game_db 已创建，且 biagent_db 中有 bi_table_metadata 数据
- **数据库连接**：第 6、8 章会读取 `backend/.env` 中的 DB 配置（已有则无需额外操作）

## 各章关系图

```
01 LangChain 基础
    ↓
02 LangGraph 图结构
    ↓
03 AgentState + add_messages
    ↓
04 条件路由
    ↓
05 工具（@tool + ToolNode）
    ↓
06 Redis Checkpointer（多轮记忆）
    ↓
07 流式输出（astream_events）
    ↓
08 迷你 BIAgent（整合所有概念）
```
