"""
第五关 Step 1：定义工具并查看元信息

学习目标：
1. @tool 装饰器的用法
2. 工具的三要素：name / description / args
3. 手动调用工具（不经过 LLM，直接 invoke）

运行方式：
    conda run -n biagent python learn/05_tools/01_define_tool.py
"""
from langchain_core.tools import tool


# ── 定义工具 ──────────────────────────────────
# @tool 把普通函数包装成 LangChain 工具对象
# 工具对象有 .name / .description / .args / .invoke() 等属性

@tool
def get_dau(date: str) -> str:
    """获取指定日期的日活跃用户数（DAU）。

    Args:
        date: 日期字符串，格式 YYYY-MM-DD，例如 2024-01-15

    Returns:
        该日期的 DAU 数字（模拟数据）
    """
    # 实际项目会查 Redis 或 MySQL，这里用假数据演示
    fake_data = {
        "2024-01-15": "52 万",
        "2024-01-14": "48 万",
        "2024-01-13": "51 万",
    }
    result = fake_data.get(date, "暂无数据")
    return f"{date} 的 DAU：{result}"


@tool
def get_revenue(date: str, game_id: int = 1) -> str:
    """获取指定日期和游戏的充值收入总额。

    Args:
        date: 日期字符串，格式 YYYY-MM-DD
        game_id: 游戏 ID，默认为 1（老虎机主游戏）

    Returns:
        充值金额字符串（模拟数据）
    """
    return f"游戏 {game_id} 在 {date} 的充值收入：156.8 万元（模拟）"


# ── 查看工具元信息 ────────────────────────────
# 这些元信息会被序列化后发送给 LLM，
# LLM 根据 description 判断什么时候该调哪个工具
print("=" * 55)
print("get_dau 工具元信息")
print("=" * 55)
print(f"name（工具名）:        {get_dau.name}")
print(f"description（描述）:   {get_dau.description}")
print(f"args（参数 schema）:   {get_dau.args}")

print()

print("=" * 55)
print("get_revenue 工具元信息")
print("=" * 55)
print(f"name:        {get_revenue.name}")
print(f"description: {get_revenue.description}")
print(f"args:        {get_revenue.args}")

# ── 手动调用工具（不经过 LLM）────────────────
# .invoke(dict) 直接执行工具函数
# 在图里 ToolNode 会自动调用，但了解手动调用有助于调试
print("\n" + "=" * 55)
print("手动调用工具")
print("=" * 55)

result1 = get_dau.invoke({"date": "2024-01-15"})
print(f"get_dau('2024-01-15')         → {result1}")

result2 = get_dau.invoke({"date": "2024-01-16"})   # 无数据的日期
print(f"get_dau('2024-01-16')         → {result2}")

result3 = get_revenue.invoke({"date": "2024-01-15", "game_id": 2})
print(f"get_revenue('2024-01-15', 2)  → {result3}")

result4 = get_revenue.invoke({"date": "2024-01-15"})  # 使用默认 game_id=1
print(f"get_revenue('2024-01-15')     → {result4}")
