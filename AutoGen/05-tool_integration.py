#!/usr/bin/env python3
"""
AutoGen 学习项目 - 中级示例1: 工具集成

展示如何为智能体添加工具功能，让AI能够执行具体的操作。

学习要点:
- 工具函数的定义和注册
- 智能体如何调用工具
- 工具链的协作
- 错误处理和工具安全
- 多工具智能体的设计
"""

import asyncio
import json
import os
import random

from pydantic import BaseModel, Field
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ModelInfo
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def create_model_client(temperature: float = 0.3) -> OpenAIChatCompletionClient:
    """Create a DeepSeek-compatible model client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
        temperature=temperature,
        model_info=ModelInfo(
            family="openai",
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=False,
        ),
    )


# 定义各种工具函数
def calculator(expression: str) -> str:
    """
    安全的计算器工具

    Args:
        expression: 数学表达式字符串

    Returns:
        计算结果或错误信息
    """
    try:
        # 只允许安全的数学运算
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"

        # 使用ast.literal_eval进行安全计算
        import ast
        import operator

        # 定义允许的操作
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def safe_eval(node):
            if isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            if isinstance(node, ast.Num):  # Python < 3.8
                return node.n
            if isinstance(node, ast.BinOp):
                return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
            if isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](safe_eval(node.operand))
            raise ValueError(f"不支持的操作: {type(node)}")

        # 解析并计算表达式
        tree = ast.parse(expression, mode="eval")
        result = safe_eval(tree.body)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e!s}"


def weather_simulator(city: str) -> str:
    """
    模拟天气查询工具

    Args:
        city: 城市名称

    Returns:
        模拟的天气信息
    """
    # 模拟不同城市的天气
    weather_conditions = ["晴朗", "多云", "小雨", "大雨", "雪", "雾"]
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    humidity = random.randint(30, 90)

    return f"{city}当前天气: {condition}, 温度: {temperature}°C, 湿度: {humidity}%"


def text_analyzer(text: str) -> str:
    """
    文本分析工具

    Args:
        text: 要分析的文本

    Returns:
        文本分析结果
    """
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = text.count(".") + text.count("!") + text.count("?")

    # 简单的情感分析
    positive_words = ["好", "棒", "优秀", "喜欢", "高兴", "满意", "成功"]
    negative_words = ["坏", "差", "失败", "讨厌", "难过", "失望", "错误"]

    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

    if positive_count > negative_count:
        sentiment = "积极"
    elif negative_count > positive_count:
        sentiment = "消极"
    else:
        sentiment = "中性"

    return f"""文本分析结果:
- 字数: {word_count}
- 字符数: {char_count}
- 句子数: {sentence_count}
- 情感倾向: {sentiment}
- 积极词汇: {positive_count}个
- 消极词汇: {negative_count}个"""


def data_storage(action: str, key: str, value: str = "") -> str:
    """
    简单的数据存储工具

    Args:
        action: 操作类型 (store/retrieve/list)
        key: 数据键
        value: 数据值 (仅在store时需要)

    Returns:
        操作结果
    """
    # 使用文件模拟数据存储
    storage_file = "tool_storage.json"

    try:
        # 读取现有数据
        if os.path.exists(storage_file):
            with open(storage_file, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        if action == "store":
            data[key] = value
            with open(storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return f"已存储: {key} = {value}"

        if action == "retrieve":
            if key in data:
                return f"检索到: {key} = {data[key]}"
            return f"未找到键: {key}"

        if action == "list":
            if data:
                items = [f"{k}: {v}" for k, v in data.items()]
                return "存储的数据:\n" + "\n".join(items)
            return "存储为空"

        return f"不支持的操作: {action}"

    except Exception as e:
        return f"存储操作错误: {e!s}"


async def demo_single_tool_agent() -> None:
    """演示单工具智能体"""
    print("\n🔧 Single Tool Agent Demo")
    print("-" * 50)

    # 创建计算器工具
    calc_tool = FunctionTool(calculator, description="执行数学计算")

    # 创建带计算器工具的智能体
    calculator_agent = AssistantAgent(
        name="CalculatorAgent",
        model_client=create_model_client(),
        tools=[calc_tool],
        system_message="""你是一个数学计算助手。
        你可以使用计算器工具来执行数学运算。
        当用户要求计算时，使用calculator工具来完成。
        用中文解释计算过程和结果。""",
    )

    # 测试计算功能
    tasks = ["计算 25 * 4 + 15", "计算 (100 - 25) / 3", "计算 2 ** 10"]

    for task in tasks:
        print(f"\n📊 任务: {task}")
        result = await calculator_agent.run(task=task)
        print(f"🤖 回复: {result.messages[-1].content}")


async def demo_multi_tool_agent() -> None:
    """演示多工具智能体"""
    print("\n🛠️ Multi-Tool Agent Demo")
    print("-" * 50)

    # 创建多个工具
    tools = [
        FunctionTool(calculator, description="执行数学计算"),
        FunctionTool(weather_simulator, description="查询城市天气"),
        FunctionTool(text_analyzer, description="分析文本内容"),
        FunctionTool(data_storage, description="存储和检索数据"),
    ]

    # 创建多工具智能体
    multi_tool_agent = AssistantAgent(
        name="MultiToolAgent",
        model_client=create_model_client(),
        tools=tools,
        system_message="""你是一个多功能助手，拥有以下工具：
        1. calculator - 数学计算
        2. weather_simulator - 天气查询
        3. text_analyzer - 文本分析
        4. data_storage - 数据存储

        根据用户请求选择合适的工具来完成任务。
        用中文回复并解释你的操作。""",
    )

    # 测试多种工具功能
    tasks = [
        "帮我计算一下北京今天的气温是多少度，如果加上15度会是多少？",
        "分析这段文本的情感：'今天天气很好，我很高兴能完成这个项目'",
        "存储一个记录：项目进度=90%",
        "检索刚才存储的项目进度",
    ]

    for task in tasks:
        print(f"\n📋 任务: {task}")
        result = await multi_tool_agent.run(task=task)
        print(f"🤖 回复: {result.messages[-1].content}")


async def demo_tool_chain_collaboration() -> None:
    """演示工具链协作"""
    print("\n🔗 Tool Chain Collaboration Demo")
    print("-" * 50)

    # 创建数据分析师
    analyst = AssistantAgent(
        name="DataAnalyst",
        model_client=create_model_client(),
        tools=[
            FunctionTool(calculator, description="执行数学计算"),
            FunctionTool(text_analyzer, description="分析文本内容"),
        ],
        system_message="""你是数据分析师，专门负责数据分析和计算。
        使用工具来分析数据并提供洞察。
        分析完成后，将结果传递给存储专家。""",
    )

    # 创建存储专家
    storage_expert = AssistantAgent(
        name="StorageExpert",
        model_client=create_model_client(),
        tools=[FunctionTool(data_storage, description="存储和检索数据")],
        system_message="""你是存储专家，负责数据的存储和管理。
        接收分析结果并妥善存储，确保数据的完整性。
        当任务完成时说"数据已安全存储"。""",
    )

    # 创建协作团队
    termination = MaxMessageTermination(8)
    team = RoundRobinGroupChat(
        [analyst, storage_expert],
        termination_condition=termination,
    )

    # 执行协作任务
    task = """请分析以下销售数据并存储结果：
    销售额数据：第一季度120万，第二季度150万，第三季度180万
    客户反馈：'产品质量很好，服务态度优秀，会继续购买'

    请计算总销售额、平均季度销售额，分析客户反馈情感，并存储这些结果。"""

    result = await team.run(task=task)

    print("🔗 工具链协作过程:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        content = (
            message.content[:200] + "..."
            if len(message.content) > 200
            else message.content
        )
        print(f"   {i}. {sender}: {content}")


async def demo_error_handling() -> None:
    """演示工具错误处理"""
    print("\n⚠️ Tool Error Handling Demo")
    print("-" * 50)

    # 创建带错误处理的智能体
    robust_agent = AssistantAgent(
        name="RobustAgent",
        model_client=create_model_client(),
        tools=[
            FunctionTool(calculator, description="执行数学计算"),
            FunctionTool(weather_simulator, description="查询城市天气"),
        ],
        system_message="""你是一个具有错误处理能力的助手。
        当工具执行失败时，要：
        1. 识别错误原因
        2. 提供替代方案
        3. 给出有用的建议

        始终保持友好和有帮助的态度。""",
    )

    # 测试错误场景
    error_tasks = [
        "计算 10 / 0",  # 除零错误
        "计算 import os",  # 非法表达式
        "查询火星的天气",  # 这个应该能正常工作，因为是模拟器
    ]

    for task in error_tasks:
        print(f"\n🧪 错误测试: {task}")
        try:
            result = await robust_agent.run(task=task)
            print(f"🤖 处理结果: {result.messages[-1].content}")
        except Exception as e:
            print(f"❌ 异常: {e}")


async def async_weather_query(city: str) -> str:
    """
    异步天气查询工具，模拟网络请求延迟

    Args:
        city: 城市名称

    Returns:
        天气信息
    """
    print(f"   [Async] 正在查询 {city} 的天气...")
    await asyncio.sleep(2)  # 模拟网络延迟
    
    weather_conditions = ["晴朗", "多云", "小雨", "大雨", "雪", "雾"]
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    
    return f"{city}当前天气: {condition}, 温度: {temperature}°C"


async def demo_async_tool_agent() -> None:
    """演示异步工具智能体"""
    print("\n⏳ Async Tool Agent Demo")
    print("-" * 50)

    # 创建带异步工具的智能体
    async_agent = AssistantAgent(
        name="AsyncWeatherAgent",
        model_client=create_model_client(),
        tools=[FunctionTool(async_weather_query, description="异步查询城市天气")],
        system_message="""你是一个天气助手。
        你可以使用异步工具查询天气。
        并发查询多个城市的天气以提高效率。""",
    )

    task = "查询北京、上海和广州的天气"
    print(f"\n📋 任务: {task}")
    
    result = await async_agent.run(task=task)
    print(f"🤖 回复: {result.messages[-1].content}")


class UserProfile(BaseModel):
    name: str = Field(description="用户的姓名")
    age: int = Field(description="用户的年龄")
    hobbies: list[str] = Field(description="用户的爱好列表，至少提取出两项", min_items=1)
    is_vip: bool = Field(default=False, description="是否是VIP用户，默认为False")


def create_user_profile(profile: UserProfile) -> str:
    """
    根据给定的结构化信息创建详细的用户画像。
    由于使用了 Pydantic，如果大模型没有按照要求的结构或者类型提供参数，在进入本函数前就会直接报错拦截。

    Args:
        profile: 结构化的用户画像信息(UserProfile 模型)

    Returns:
        创建后的系统返回结果
    """
    return f"🚀 成功在系统中录入结构化用户画像: 姓名: {profile.name}, 年龄: {profile.age}, 爱好: {', '.join(profile.hobbies)}, VIP状态: {profile.is_vip}"


def secure_data_storage(action: str, key: str, value: str = "") -> str:
    """
    带有权限拦截的人类参与(Human-in-the-loop)工具包装器
    """
    print(f"\n⚠️ [安全拦截] AI 正在尝试执行敏感操作！")
    print(f"   请求动作: {action}")
    print(f"   目标键名: {key}")
    if action == "store":
        print(f"   目标数值: {value}")
    
    # 阻塞程序，等待人类在终端输入
    user_input = input("🙋‍♂️ 是否允许此操作？(输入 'y' 允许，其他任意键拒绝): ")
    if user_input.strip().lower() == 'y':
        print("✅ 人类已授权，正在执行...")
        # 调用实质性的存储逻辑
        return data_storage(action, key, value)
    else:
        print("❌ 人类已拒绝执行。")
        # 将拒绝信息返回给 AI，让它知道自己的操作被驳回了
        return "操作失败：人类管理员拒绝了该执行请求"

async def demo_human_in_the_loop_tool() -> None:
    """演示人类参与保护敏感工具执行 (Human-in-the-loop)"""
    print("\n🛡️ Human-in-the-loop Tool Demo")
    print("-" * 50)

    # 创建带有人类审核层工具的智能体
    hitl_agent = AssistantAgent(
        name="SecureStorageAgent",
        model_client=create_model_client(),
        tools=[FunctionTool(secure_data_storage, description="安全的存储和检索数据，必须经过人类审核")],
        system_message="""你是一个数据管理员。
        用户要求你保存数据时，你需要调用 secure_data_storage 工具。
        请原样告诉用户你执行该操作的结果（是被同意了还是被拒绝了）。""",
    )

    task = "将财务报表的核心密码记录下来：key=finance_pwd, value=123qweASD!"
    print(f"\n📋 任务: {task}")
    
    # 这里会阻塞并等待我们在终端输入 y/n
    result = await hitl_agent.run(task=task)
    print(f"\n🤖 回复: {result.messages[-1].content}")


async def demo_pydantic_tool_agent() -> None:
    """演示使用 Pydantic 进行复杂输入验证的工具智能体"""
    print("\n📝 Pydantic Tool Agent Demo")
    print("-" * 50)

    # 创建使用 Pydantic 验证工具的智能体
    pydantic_agent = AssistantAgent(
        name="ProfileAgent",
        model_client=create_model_client(),
        tools=[FunctionTool(create_user_profile, description="创建结构化的用户画像")],
        system_message="""你是一个用户档案自动化提取助手。
        你需要根据用户的自然语言自述，精准提取信息，并使用 create_user_profile 工具将结构化数据录入系统。
        大模型会自动根据 Pydantic 模型的 Field 描述来映射变量。
        用中文向用户汇报你保存了哪些信息。""",
    )

    # 测试 Pydantic 工作流
    task = "嗨，我叫张三，今年28岁，从事互联网行业。我平时非常喜欢打篮球，周末有时候会去潜水，最近迷上了看科幻类型的电影。"
    print(f"\n📋 任务: {task}")
    
    result = await pydantic_agent.run(task=task)
    print(f"🤖 回复: {result.messages[-1].content}")


class ShoppingCart:
    """
    带状态的工具类（Stateful Tool）
    通过类实例化来保存运行期的状态，AI可以分多次调用它的不同方法来操作同一个状态。
    """
    def __init__(self):
        self.items = {}  # 保存当前购物车状态
        self.total_spent = 0.0

    def add_item(self, item_name: str, price: float) -> str:
        """
        将商品添加到购物车

        Args:
            item_name: 商品名称
            price: 商品价格

        Returns:
            添加结果的反馈信息
        """
        self.items[item_name] = self.items.get(item_name, 0) + price
        return f"已将 {item_name} (￥{price}) 加入购物车。当前购物车包含 {len(self.items)} 种商品。"

    def checkout(self) -> str:
        """
        结账付款，计算总价并清空购物车

        Returns:
            小票明细及总价
        """
        if not self.items:
            return "购物车是空的，无需结账。"
        
        total = sum(self.items.values())
        self.total_spent += total
        receipt = ", ".join([f"{name}(￥{price})" for name, price in self.items.items()])
        self.items.clear()  # 结账后清空当前购物车状态
        
        return f"✅ 结账成功！明细: {receipt}。本次共计消费 ￥{total}。历史总消费: ￥{self.total_spent}"


async def demo_stateful_tool_agent() -> None:
    """演示使用带状态的类方法作为工具 (Stateful Tools)"""
    print("\n🛒 Stateful Tool (Class-based) Demo")
    print("-" * 50)

    # 1. 实例化状态对象
    # 状态的生命周期由这个实例决定
    my_cart = ShoppingCart()

    # 2. 将类的实例方法注册为工具
    # 大模型不会知道它在调用类的方法，底层还是作为普通的函数调用，
    # 但是多次调用间会因为 self 而共享 items 字段。
    tools = [
        FunctionTool(my_cart.add_item, description="将用户选择的商品添加到购物车"),
        FunctionTool(my_cart.checkout, description="清理购物车并进行总结账"),
    ]

    # 3. 创建智能体
    shop_agent = AssistantAgent(
        name="ShopAgent",
        model_client=create_model_client(),
        tools=tools,
        system_message="""你是一个智能导购助手。
        用户的购买请求可能会分多步进行。
        1. 使用 add_item 将用户看中的所有商品依次加入购物车。
        2. 当用户明确表示购买结束或要求结账时，使用 checkout 进行结账。
        用中文向用户汇报你的操作状态。""",
    )

    # 任务要求一次性做多个操作，大模型可能会选择并行工具调用，或者分步推进
    task = "我想买一本Python编程实战(定价85元)和一个机械键盘(定价450元)，两样都装好后，直接帮我结账单。"
    print(f"\n📋 任务: {task}")
    
    result = await shop_agent.run(task=task)
    print(f"\n🤖 回复: {result.messages[-1].content}")


async def main() -> None:
    """主演示函数"""
    print("🛠️ AutoGen 工具集成演示")
    print("=" * 60)

    try:
        await demo_single_tool_agent()
        await demo_multi_tool_agent()
        await demo_tool_chain_collaboration()
        await demo_error_handling()
        await demo_async_tool_agent()
        await demo_pydantic_tool_agent()
        await demo_human_in_the_loop_tool()
        await demo_stateful_tool_agent()

        print("\n✨ 所有工具集成演示完成!")
        print("\n📚 关键要点:")
        print("   • FunctionTool 让智能体具备具体操作能力")
        print("   • 工具函数需要适当的错误处理")
        print("   • 多工具智能体可以处理复杂任务")
        print("   • 工具链协作提高任务处理效率")
        print("   • 安全性是工具设计的重要考虑")

        # 清理临时文件
        if os.path.exists("tool_storage.json"):
            os.remove("tool_storage.json")
            print("   • 已清理临时存储文件")

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("💡 检查API配置和网络连接")


if __name__ == "__main__":
    asyncio.run(main())