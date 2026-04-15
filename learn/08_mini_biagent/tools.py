"""
迷你 BIAgent 的两个核心工具：
1. query_metadata：在 bi_table_metadata 中搜索相关表信息
2. execute_sql：在 game_db 上执行只读 SQL

技术要点：
- 使用同步 SQLAlchemy（pymysql），避免 async 工具在 ToolNode 中的复杂性
- tools.py 从 backend/.env 读取 DB 连接信息
- execute_sql 有安全检查：只允许 SELECT 语句
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from sqlalchemy import create_engine, text

# 加载 backend/.env（包含 DB 连接信息）
_backend_env = Path(__file__).parent.parent.parent / "backend" / ".env"
load_dotenv(dotenv_path=_backend_env)

# 把 aiomysql 连接串转换为 pymysql（同步版本）
def _to_sync_url(url: str) -> str:
    return url.replace("mysql+aiomysql://", "mysql+pymysql://")

_BIAGENT_URL = _to_sync_url(
    os.getenv("BIAGENT_DB_URL", "mysql+pymysql://root:@localhost:3306/biagent_db")
)
_GAME_URL = _to_sync_url(
    os.getenv("GAME_DB_URL", "mysql+pymysql://root:@localhost:3306/game_db")
)


@tool
def query_metadata(keyword: str) -> str:
    """在 bi_table_metadata 表中搜索与关键词相关的表信息。
    返回表名、中文别名、业务描述，供 Agent 决定查哪张表、用哪些字段。

    Args:
        keyword: 搜索关键词，如"充值"、"用户"、"DAU"、"留存"
    """
    try:
        engine = create_engine(_BIAGENT_URL, echo=False, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT table_name, table_alias, description, columns_json
                    FROM bi_table_metadata
                    WHERE is_active = 1
                      AND (
                          table_name  LIKE :kw
                          OR table_alias  LIKE :kw
                          OR description  LIKE :kw
                          OR business_tags LIKE :kw
                      )
                    LIMIT 3
                """),
                {"kw": f"%{keyword}%"},
            ).fetchall()
        engine.dispose()

        if not rows:
            return f"未在元数据中找到与「{keyword}」相关的表，请尝试其他关键词。"

        parts = []
        for row in rows:
            cols_preview = str(row.columns_json)[:300] if row.columns_json else "（无）"
            parts.append(
                f"表名：{row.table_name}（{row.table_alias or '无别名'}）\n"
                f"描述：{row.description}\n"
                f"字段（部分）：{cols_preview}"
            )
        return "\n\n".join(parts)

    except Exception as e:
        return f"查询元数据时出错：{e}"


@tool
def execute_sql(sql: str) -> str:
    """在 game_db 上执行只读 SQL 查询（仅限 SELECT），返回前 20 行结果。

    Args:
        sql: 合法的 SELECT 语句。禁止使用 INSERT / UPDATE / DELETE / DROP 等写操作。
    """
    # 安全检查：只允许 SELECT
    sql_stripped = sql.strip()
    if not sql_stripped.upper().startswith("SELECT"):
        return "错误：execute_sql 只允许执行 SELECT 语句，请修改 SQL。"

    try:
        engine = create_engine(_GAME_URL, echo=False, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text(sql_stripped))
            rows = result.fetchmany(20)
            col_names = list(result.keys())
        engine.dispose()

        if not rows:
            return "SQL 执行成功，但查询结果为空（0 行）。"

        # 格式化为简单表格
        header = " | ".join(col_names)
        divider = "-" * min(len(header), 80)
        lines = [header, divider]
        for row in rows:
            lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))

        suffix = "\n（结果已截断，最多显示 20 行）" if len(rows) == 20 else ""
        return "\n".join(lines) + suffix

    except Exception as e:
        return f"SQL 执行失败：{e}\n请检查表名和字段名是否正确。"
