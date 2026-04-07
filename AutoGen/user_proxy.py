#!/usr/bin/env python3
"""
AutoGen 学习项目 - 示例3: 用户代理智能体

展示UserProxyAgent的功能，包括人机交互模式。

学习要点:
- UserProxyAgent的配置
- 人机交互模式
- 与AssistantAgent的协作
- 工作流控制
"""

import asyncio
import os

# AssistantAgent: 负责思考（对接大模型）
# UserProxyAgent: 负责代表人类行为或执行代码（如在终端输入、运行代码）
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
# MaxMessageTermination: 控制群聊对话何时终止（比如最多聊 N 句就停）
from autogen_agentchat.conditions import MaxMessageTermination
# RoundRobinGroupChat: 最基础的团队协作模式（按顺序一人一句轮流发言）
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def create_model_client() -> OpenAIChatCompletionClient:
    """创建配置好的大模型客户端"""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY not found in environment variables")

    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.3,
        model_info=ModelInfo(
            family="openai",
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=False,
        ),
    )


async def demo_basic_user_proxy() -> None:
    """
    演示：基础的 UserProxyAgent 与 AssistantAgent 协作

    工作流：
    1. 你将任务交给 Team（团队）。
    2. 第一轮：UserProxy 将任务发给 Assistant。
    3. 第二轮：Assistant 生成代码。
    4. 第三轮：触发对话终止（MaxMessageTermination(3) 限制了对话最多 3 轮）。
    """
    print("\n🤖 Basic UserProxy Demo")
    print("-" * 40)

    # 创建思考者（负责写代码）
    assistant = AssistantAgent(
        name="PythonHelper",
        model_client=create_model_client(),
        system_message="""你是一个Python编程助手。
        当用户需要代码时，提供完整的、可运行的Python代码。
        用中文解释你的代码逻辑。""",
    )

    # 创建用户代理。通常它会拦截代码并试图运行，或者等待人类在终端敲击回车。
    # 这里是一个基础配置，代表“用户”或执行环境存在于群聊中。
    user_proxy = UserProxyAgent(name="User", description="代表用户进行交互的代理")

    task = "写一个Python函数来计算斐波那契数列的前n项，并展示如何使用它。"

    # GroupChat 和 Termination 是新版 AutoGen 中的标准模式
    termination = MaxMessageTermination(3) # 限制整个对话进行 3 轮
    team = RoundRobinGroupChat(
        [assistant, user_proxy], # 按照 [assistant, user_proxy] 的顺序排排坐
        termination_condition=termination,
    )
    result = await Console(team.run_stream(task=task))

    # print("📝 对话结果:")
    for i, message in enumerate(result.messages[-3:], 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        content = (
            message.content[:200] + "..."
            if len(message.content) > 200
            else message.content
        )
        # print(f"   {i}. {sender}: {content}")


async def demo_code_execution() -> None:
    """
    演示：代码执行 (Code Execution)
    让 Agent 生成代码并在本地环境中执行。
    """
    print("\n🚀 Code Execution Demo")
    print("-" * 40)
    
    from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    from autogen_agentchat.agents import CodeExecutorAgent
    
    # 1. 创建本地代码执行器
    import warnings
    # 忽略本地执行的安全警告，因为我们已经确认风险并添加了人工审核
    warnings.filterwarnings("ignore", message=".*Using LocalCommandLineCodeExecutor.*")
    
    # work_dir 指定代码文件保存和执行的目录
    work_dir = os.path.join(os.path.dirname(__file__), "coding")
    os.makedirs(work_dir, exist_ok=True)
    local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
    
    # 2. 创建代码执行 Agent
    # 定义人工审核函数，用于在执行代码前进行确认
    def human_approval(*args, **kwargs):
        print("\n" + "="*40)
        print("⚠️ 助手生成了以下代码准备在本地执行：")
        
        # 提取并展示生成的代码块内容
        for arg in args:
            if isinstance(arg, list):
                for item in arg:
                    code_content = getattr(item, "code", str(item))
                    print(f"\n```\n{code_content}\n```")
            else:
                code_content = getattr(arg, "code", str(arg))
                print(f"\n```\n{code_content}\n```")
        
        print("="*40)
        response = input("是否允许执行? (y/n): ")
        is_approved = response.strip().lower() in ['y', 'yes']
        
        # 返回带有 approved 属性的对象 (适配 AutoGen 的 Approval 结构)
        class ApprovalResult:
            def __init__(self, approved: bool):
                self.approved = approved
        return ApprovalResult(approved=is_approved)

    executor_agent = CodeExecutorAgent(
        name="ExecutorAgent",
        code_executor=local_executor,
        description="负责执行Python代码并返回结果",
        approval_func=human_approval,  # 注入人工审核机制
    )
    
    # 3. 创建负责编写代码的 Assistant
    assistant = AssistantAgent(
        name="CoderAgent",
        model_client=create_model_client(),
        system_message="""你是一个有用的AI助手。
        当你需要解决问题时，你可以编写Python代码。
        请将代码放在 ```python 和 ``` 之间，它将被执行。
        警告：请不要在提供代码的同一轮回复中说出 'TERMINATE'！
        你必须等待代码被执行，看到执行成功并输出结果后，才能在下一轮单独回复 'TERMINATE' 来结束对话。""",
    )
    
    # 4. 组建团队，设置组合终止条件（只拦截由 CoderAgent 发出的 "TERMINATE" 消息，或者达到最大对话轮数）
    from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
    
    # 增加 sources=["CoderAgent"] 参数，防止用户初始任务中提到的 TERMINATE 直接触发终止！
    termination = TextMentionTermination("TERMINATE", sources=["CoderAgent"]) | MaxMessageTermination(20)
    
    team = RoundRobinGroupChat(
        [assistant, executor_agent], 
        termination_condition=termination
    )
    
    # 5. 指派需要执行代码的任务
    task = "写一个Python脚本，计算 1 到 100 之间所有偶数的平方和，打印出最终结果。"
    
    await Console(team.run_stream(task=task))


async def demo_custom_input_function() -> None:
    """
    演示：如何通过代码“模拟”真实用户的输入 (input_func)

    当我们需要自动化测试，或者不想一次次手动在终端打字时，
    可以通过 `input_func` 传入一个自定义函数，让机器代替人类回答大模型的问题。
    """
    print("\n🎯 Custom Input Function Demo")
    print("-" * 40)

    # 预设人类的剧本
    responses = [
        "我想学习数据分析",
        "我是初学者，主要想处理CSV文件",
        "好的，请给我一个简单的例子",
    ]
    response_index = 0

    # 每次 UserProxyAgent 轮到发言或请求人类干预时，它会调用这个函数
    def custom_input_func(prompt: str) -> str:
        nonlocal response_index
        if response_index < len(responses):
            response = responses[response_index]
            response_index += 1
            print(f"👤 模拟用户输入: {response}")
            return response
        print("谢谢你的帮助！")
        return "谢谢你的帮助！"

    assistant = AssistantAgent(
        name="InteractiveHelper",
        model_client=create_model_client(),
        system_message="""你是一个交互式助手。
        你会提出问题来更好地理解用户需求。
        当你需要更多信息时，明确询问。
        保持对话自然和有帮助。""",
    )

    # input_func 使得大模型可以通过提出问题来获得预先写好的剧本回调
    user_proxy = UserProxyAgent(
        name="SimulatedUser",
        description="模拟用户交互的代理",
        input_func=custom_input_func,
    )

    task = "我需要帮助，但不确定具体要什么。"

    termination = MaxMessageTermination(6) # 允许 6 轮交互，让一问一答持续进行
    team = RoundRobinGroupChat(
        [assistant, user_proxy],
        termination_condition=termination,
    )
    result = await Console(team.run_stream(task=task))

    # print("\n💬 交互式对话摘要:")
    # conversation_count = 0
    # for message in result.messages:
    #     if hasattr(message, "source"):
    #         conversation_count += 1
    #         if conversation_count <= 4:  # Show first few exchanges
    #             print(f"   {message.source}: {message.content[:100]}...")


async def demo_collaborative_workflow() -> None:
    """
    演示：协同工作流

    这里 UserProxyAgent 被当成了不写代码的“项目经理”角色使用。
    """
    print("\n🤝 Collaborative Workflow Demo")
    print("-" * 40)

    # 规划专家 (AssistantAgent)
    planner = AssistantAgent(
        name="TaskPlanner",
        model_client=create_model_client(),
        system_message="""你是一个任务规划专家。
        你的职责是：
        1. 分析用户需求
        2. 制定详细的执行计划
        3. 将复杂任务分解为简单步骤
        4. 提供清晰的指导""",
    )

    # 项目经理 (UserProxyAgent)：充当人类意志/验收机制的化身
    project_manager = UserProxyAgent(
        name="ProjectManager",
        description="项目经理，负责审核和指导任务执行",
    )

    task = "规划一个数据科学项目：分析电商网站的用户行为数据，找出提升转化率的机会。"

    termination = MaxMessageTermination(4)
    team = RoundRobinGroupChat(
        [planner, project_manager],
        termination_condition=termination,
    )
    result = await Console(team.run_stream(task=task))

    # print("📊 协作工作流结果:")
    # print(f"   总轮次: {len(result.messages)}")
    # for i, message in enumerate(result.messages[-2:], 1):
    #     sender = message.source if hasattr(message, "source") else "Unknown"
    #     print(f"   {i}. {sender}: {message.content[:150]}...")


async def demo_role_based_interaction() -> None:
    """
    演示：基于角色的教与学互动

    这和前一个例子类似，重点在于用 input_func 模拟对话流状态机制。
    """
    print("\n🎭 Role-Based Interaction Demo")
    print("-" * 40)

    # 教师角色
    teacher = AssistantAgent(
        name="PythonTeacher",
        model_client=create_model_client(),
        system_message="""你是一位耐心的Python编程老师。
        你的教学方式：
        1. 先解释概念
        2. 提供简单例子
        3. 询问学生是否理解
        4. 根据反馈调整教学节奏""",
    )

    # 学生的模拟反馈序列
    student_responses = [
        "我想学习Python的列表操作",
        "能举个具体例子吗？",
        "明白了，那字典怎么用？",
    ]
    response_idx = 0

    def student_input(prompt: str) -> str:
        nonlocal response_idx
        if response_idx < len(student_responses):
            response = student_responses[response_idx]
            response_idx += 1
            print(f"🎓 学生说: {response}")
            return response
        return "我理解了，谢谢老师！"

    # UserProxyAgent 包装了模拟学生的逻辑
    student = UserProxyAgent(
        name="Student",
        description="正在学习Python的学生",
        input_func=student_input,
    )

    # 开始上课
    task = "开始一节关于Python数据结构的课程。"

    termination = MaxMessageTermination(6)
    team = RoundRobinGroupChat([teacher, student], termination_condition=termination)
    result = await team.run(task=task)

    print("\n📚 教学互动总结:")
    print(f"   教学轮次: {len(result.messages)}")
    print("   最后的师生对话:")
    for message in result.messages[-2:]:
        sender = message.source if hasattr(message, "source") else "Unknown"
        print(f"   {sender}: {message.content[:120]}...")


async def demo_tool_based_termination() -> None:
    """
    演示：基于 Tool Calling 的工作流终止
    
    不再让大模型通过文本输出 'TERMINATE' 来结束对话（文本匹配不稳定，容易被大模型误触发）。
    而是要求大模型必须调用 `submit_result` 工具。
    工具执行后会返回系统的魔法字符串，团队只需监听到该特定字符串即可安全终止。
    """
    print("\n🛠️ Tool-Based Termination Demo ")
    print("-" * 40)
    
    from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

    # 1. 定义一个用于“交卷”的工具函数
    async def submit_result(final_answer: str) -> str:
        """
        当且仅当你完全解决了用户的问题并得出最终结果时，调用此工具提交答案以结束任务。
        
        Args:
            final_answer: 你对本次任务的最终解答结果内容
        """
        print(f"\n[系统拦截] 📥 成功接收到大模型通过工具提交的结果:\n{final_answer}")
        
        # 返回一个不对大模型公开的、极其特殊的终止识别词
        # 外部框架只监听这个特殊词来终结循环
        return "SYSTEM_AUTHORIZED_TERMINATION_SIGNAL"

    # 2. 创建负责干活的 AssistantAgent，并挂载工具
    assistant = AssistantAgent(
        name="SmartWorker",
        model_client=create_model_client(),
        tools=[submit_result],
        system_message="""你是一个计算小能手。
        【重要交卷规则】：
        你不允许在文本里自行输出 'TERMINATE' 或者任何终止词。
        当你算出最终结果后，必须且只能通过调用 `submit_result` 工具将结果提交即可。""",
    )

    # 3. 设置只认“系统授权信号”的系统级终止规则
    # TextMentionTermination 会检查所有消息内容，当工具被执行并返回了该字符串时，系统就会终止
    termination = TextMentionTermination("SYSTEM_AUTHORIZED_TERMINATION_SIGNAL") | MaxMessageTermination(10)

    # 4. 创建团队（这里只用这一个 Agent 演示自我调用工具交卷的流程）
    team = RoundRobinGroupChat(
        [assistant], 
        termination_condition=termination
    )

    task = "请计算一下 1 到 100 所有连续整数的和。完成计算后，请正式交卷。"

    await Console(team.run_stream(task=task))


async def main() -> None:
    """主入口函数：执行所有 UserProxyAgent 演示"""
    print("🤖 AutoGen UserProxyAgent 功能展示")
    print("=" * 50)

    # try:
    # await demo_basic_user_proxy()
    # await demo_custom_input_function()
    # await demo_collaborative_workflow()
    # await demo_role_based_interaction()
    # await demo_code_execution()
    await demo_tool_based_termination()

    print("\n✨ 所有UserProxy演示完成!")
    # print("\n📚 关键要点:")
    # print("   • UserProxyAgent 代表人类用户进行交互")
    # print("   • 可以配置自定义输入函数模拟用户行为")
    # print("   • 在团队中充当重要的协调角色")
    # print("   • 支持各种角色扮演和工作流模式")
    # print("   • 是人机协作/代码执行的重要桥梁")

    # except Exception as e:
    #     print(f"❌ 演示失败: {e}")
    #     print("💡 检查API配置和网络连接")


if __name__ == "__main__":
    asyncio.run(main())