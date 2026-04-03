#!/usr/bin/env python3
"""
AutoGen 学习项目 - 示例2: 助手智能体深入探索

展示AssistantAgent的各种配置选项和功能特性。

学习要点:
- AssistantAgent的详细配置
- 系统消息的重要性
- 不同的模型参数设置
- 消息历史管理
- 错误处理和重试机制
"""

import asyncio
import os

# AssistantAgent 是 AutoGen 中最基础的对话智能体，负责与语言模型交互并生成回复
from autogen_agentchat.agents import AssistantAgent
# ModelInfo 用于描述所使用模型的能力元信息（如是否支持视觉、函数调用等）
from autogen_core.models import ModelInfo
# OpenAIChatCompletionClient 是兼容 OpenAI 接口格式的通用模型客户端，可对接 DeepSeek 等第三方服务
from autogen_ext.models.openai import OpenAIChatCompletionClient
# dotenv 用于从 .env 文件加载环境变量（如 API Key、模型名称等），避免硬编码敏感信息
from dotenv import load_dotenv

# 在程序启动时加载 .env 文件中的环境变量，后续通过 os.getenv() 读取
load_dotenv()


class AgentDemo:
    """
    AssistantAgent 功能演示类。

    封装了多个独立的演示方法，分别展示：
    - 基础助手的创建与使用
    - 创意写作助手（高温度参数）
    - 对话记忆与上下文保持
    - 错误处理与鲁棒性
    """

    def __init__(self):
        # 从环境变量中读取 API Key，若未设置则抛出异常，防止后续调用失败
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    def create_model_client(
        self,
        temperature: float = 0.7,   # 控制输出随机性：越高越有创意，越低越确定性
        max_tokens: int = 1000,      # 单次回复的最大 token 数量上限
    ) -> OpenAIChatCompletionClient:
        """
        创建并返回配置好的模型客户端（兼容 DeepSeek / OpenAI 接口）。

        Args:
            temperature: 采样温度，范围 0.0~1.0。
                         低值（如 0.2）使输出更确定，适合代码生成；
                         高值（如 0.9）使输出更多样，适合创意写作。
            max_tokens:  模型单次生成的最大 token 数，防止回复过长。

        Returns:
            配置完毕的 OpenAIChatCompletionClient 实例。
        """
        return OpenAIChatCompletionClient(
            # 模型名称从环境变量读取，默认使用 DeepSeek 的对话模型
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            api_key=self.api_key,
            # API 基础地址从环境变量读取，支持切换到不同的 OpenAI 兼容服务
            base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
            temperature=temperature,  # 输出创造性程度（0.0 最保守，1.0 最随机）
            max_tokens=max_tokens,    # 每次生成的最大 token 数
            top_p=0.9,                # Nucleus 采样阈值：只从累积概率前 90% 的 token 中采样，平衡多样性与质量
            model_info=ModelInfo(
                family="openai",          # 模型接口协议族，声明兼容 OpenAI 格式
                vision=False,             # 该模型不支持图像输入
                function_calling=True,    # 支持函数调用（Tool Call），可配合工具使用
                json_output=True,         # 支持结构化 JSON 输出模式
                structured_output=False,  # 不使用 OpenAI 的 Structured Output 严格模式
            ),
        )

    async def demo_basic_assistant(self) -> None:
        """
        演示：创建一个专注于编程辅导的基础助手并执行单次任务。

        要点：
        - 使用较低的 temperature（0.3）保证回答的准确性与一致性
        - 通过 system_message 赋予智能体特定角色和行为规范
        - 调用 agent.run(task=...) 发起单轮对话并获取结果
        """
        print("\n🔧 Basic Assistant Agent Demo")
        print("-" * 40)

        # temperature=0.3：低随机性，确保代码解释准确、逻辑清晰
        model_client = self.create_model_client(temperature=0.3)

        # 创建一个编程导师角色的助手，system_message 定义其职责与回复风格
        coding_assistant = AssistantAgent(
            name="CodingMentor",        # 智能体名称，用于标识和日志输出
            model_client=model_client,  # 绑定上面创建的模型客户端
            system_message="""你是一位专业的Python编程导师。
            你的职责是:
            1. 清晰地解释编程概念
            2. 提供带注释的代码示例
            3. 建议最佳实践
            4. 帮助调试问题

            总是用markdown格式化代码，并解释你的推理过程。""",
        )

        # 向智能体发送任务，run() 方法返回包含完整消息历史的 TaskResult 对象
        task = "解释Python中列表推导式和生成器表达式的区别，并提供示例。"
        result = await coding_assistant.run(task=task)

        # result.messages 是本次对话的消息列表，[-1] 取最后一条即模型的回复
        print(f"🤖 {coding_assistant.name} 说:")
        print(f"   {result.messages[-1].content[:200]}...")           # 只展示前 200 字符预览
        print(f"   [回复长度: {len(result.messages[-1].content)} 字符]")

    async def demo_creative_assistant(self) -> None:
        """
        演示：使用高 temperature 参数创建富有创意的写作助手。

        要点：
        - temperature=0.9 增加输出的随机性和多样性，更适合创意场景
        - system_message 中明确要求"富有想象力"来引导模型风格
        """
        print("\n🎨 Creative Assistant Demo")
        print("-" * 40)

        # temperature=0.9：高随机性，鼓励模型产出更有创意、多样化的文本
        model_client = self.create_model_client(temperature=0.9)

        creative_writer = AssistantAgent(
            name="CreativeWriter",
            model_client=model_client,
            system_message="""你是一位富有创意的写作助手。
            你擅长:
            - 创作引人入胜的故事
            - 创造生动的描述
            - 开发独特的角色
            - 以各种风格和类型写作

            在回复中要富有想象力和表现力。""",
        )

        task = "写一个关于AI发现自己能够做梦的科幻故事开头。"
        result = await creative_writer.run(task=task)

        # 展示创意输出的前 300 字符，感受高 temperature 带来的文风变化
        print(f"✨ {creative_writer.name} 创作:")
        print(f"   {result.messages[-1].content[:300]}...")

    async def demo_conversation_memory(self) -> None:
        """
        演示：AssistantAgent 的多轮对话上下文记忆能力。

        要点：
        - 同一个 AssistantAgent 实例在多次 run() 调用之间自动维护消息历史
        - 第二轮对话时模型可以引用第一轮中提到的信息（如用户姓名）
        - 通过 result.messages 的长度可以验证历史消息是否被正确累积
        """
        print("\n🧠 Conversation Memory Demo")
        print("-" * 40)

        model_client = self.create_model_client(temperature=0.5)

        # system_message 要求助手主动引用历史信息，便于验证记忆功能
        memory_assistant = AssistantAgent(
            name="MemoryKeeper",
            model_client=model_client,
            system_message="""你是一个拥有出色记忆力的助手。
            你能记住所有之前的对话，并可以引用它们。
            总是确认你从之前的互动中记住了什么。""",
        )

        # 第一轮：告知助手用户信息，建立上下文
        print("💬 第一次对话:")
        result1 = await memory_assistant.run(task="我的名字是Alice，我喜欢Python编程。")
        print(f"   助手: {result1.messages[-1].content}")

        # 第二轮：测试助手是否记住了第一轮中的信息
        # 由于同一实例维护历史，模型上下文中包含第一轮的完整对话
        print("\n💬 第二次对话 (测试记忆):")
        result2 = await memory_assistant.run(task="我的名字是什么？我喜欢什么？")
        print(f"   助手: {result2.messages[-1].content}")

        # 输出累积的消息总数，验证历史消息正确保留（应包含两轮的 user + assistant 消息）
        print(f"\n📊 对话中总消息数: {len(result2.messages)}")

    async def demo_error_handling(self) -> None:
        """
        演示：生产级代码中错误处理的重要性。

        要点：
        - 使用 try/except 捕获 API 调用可能抛出的各类异常（如超时、限流、token 超限）
        - 超长输入可能触发模型的 token 限制，需要做好边界处理
        - 捕获错误后应提供有意义的错误提示，而非让程序静默崩溃
        """
        print("\n⚠️  Error Handling Demo")
        print("-" * 40)

        # 使用默认参数（temperature=0.7）创建客户端
        model_client = self.create_model_client()

        assistant = AssistantAgent(
            name="RobustAssistant",
            model_client=model_client,
            system_message="你是一个优雅处理错误的助手。",
        )

        try:
            # 正常请求：简单的数学问题，预期能正常返回结果
            result = await assistant.run(task="2 + 2 等于多少？")
            print(f"✅ 正常请求: {result.messages[-1].content}")

            # 压力测试：构造一个极长的输入，可能触发模型的 token 上限错误
            # "非常 " 重复 1000 次约产生 2000 个字符，用于模拟异常输入场景
            long_task = "解释这个: " + "非常 " * 1000 + "关于AutoGen的长问题"
            result = await assistant.run(task=long_task)
            print(f"✅ 长请求处理: 回复长度 {len(result.messages[-1].content)}")

        except Exception as e:
            # 捕获所有异常类型，打印错误信息，避免程序崩溃影响其他流程
            print(f"❌ 捕获错误: {e}")
            print("💡 这展示了在生产代码中错误处理的重要性")


async def main() -> None:
    """
    主入口函数：依次执行所有 AssistantAgent 演示。

    执行顺序：
    1. 基础助手演示（低 temperature，代码辅导场景）
    2. 创意写作演示（高 temperature，开放性创作场景）
    3. 对话记忆演示（验证多轮上下文保持能力）
    4. 错误处理演示（模拟异常输入，展示鲁棒性）
    """
    print("🤖 AutoGen AssistantAgent 深入探索")
    print("=" * 50)

    try:
        # 初始化演示类，同时验证 API Key 是否已配置
        demo = AgentDemo()

        # 顺序执行各演示，每个演示相互独立、互不干扰
        await demo.demo_basic_assistant()        # 演示1：基础助手
        await demo.demo_creative_assistant()     # 演示2：创意写作
        await demo.demo_conversation_memory()    # 演示3：对话记忆
        await demo.demo_error_handling()         # 演示4：错误处理

        # 汇总本次学习的核心知识点
        print("\n✨ 所有演示成功完成!")
        print("\n📚 关键要点:")
        print("   • AssistantAgent 高度可配置")
        print("   • 系统消息定义智能体行为")
        print("   • Temperature 控制创造性与一致性")
        print("   • 智能体维护对话记忆")
        print("   • 适当的错误处理至关重要")

    except Exception as e:
        # 捕获初始化阶段的异常（如 API Key 缺失）
        print(f"❌ 演示失败: {e}")
        print("💡 确保你的API密钥设置正确且有效")


if __name__ == "__main__":
    # 程序入口：使用 asyncio.run() 启动异步事件循环并执行 main()
    # asyncio.run() 会自动管理事件循环的创建与销毁，是 Python 3.7+ 的推荐写法
    asyncio.run(main())