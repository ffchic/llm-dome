#!/usr/bin/env python3
"""
AutoGen 学习项目 - 示例4: 简单对话系统

展示两个智能体之间的对话，以及如何控制对话流程。

学习要点:
- 双智能体对话
- RoundRobinGroupChat的使用
- 终止条件设置
- 不同的对话场景
- 对话控制和管理
"""

import asyncio  # 用于支持异步协程编程，AutoGen 的核心调度是异步的
import os       # 用于读取环境变量，如 API_KEY

from autogen_agentchat.agents import AssistantAgent  # AutoGen 核心角色类，代表一个拥有 LLM 大脑的智能体实例
# 各种对话终止条件检测器：
# - MaxMessageTermination: 当总对话数达到指定次数时强行终止
# - TextMentionTermination: 当任意智能体的回复里出现了指定的关键词时结束对话
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
# 定义团队如何交互的不同容器/环境：
# - RoundRobinGroupChat: 轮询对话，参与者按预定的数组顺序依次发言
# - SelectorGroupChat: 动态路由群聊，通过 LLM 每次决断接下来该由谁发言
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_core.models import ModelInfo  # 提供与底层模型交互时的基础元数据（声明模型的能力范围）
from autogen_ext.models.openai import OpenAIChatCompletionClient  # OpenAI 兼容格式的模型客户端
from dotenv import load_dotenv  # 用于从 .env 文件注入环境变量


load_dotenv()


def create_model_client(temperature: float = 0.7) -> OpenAIChatCompletionClient:
    """
    创建并返回配置好的模型客户端（兼容 DeepSeek 等 OpenAI API 格式）。
    
    参数:
        temperature (float): 控制模型生成的随机性。
                             - 较低的值 (例如 0.1-0.3) 使输出更确定、严谨 (适合代码生成、逻辑分析)。
                             - 较高的值 (例如 0.7-0.9) 使输出更具创造性、发散性 (适合头脑风暴、角色扮演)。
    
    返回:
        OpenAIChatCompletionClient: 封装好的可以调用大模型的客户端对象。
    """
    # 1. 从环境变量加载 API 密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # 2. 实例化客户端
    # 注意：这里配置了 model_info，这是 AutoGen 用来了解模型能力（如是否支持函数调用、视觉等）的元数据
    return OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
        temperature=temperature,
        model_info=ModelInfo(
            family="openai",            # 模型家族，用于确定内置的 prompt 模板体系
            vision=False,               # 是否支持图像输入
            function_calling=True,      # 是否支持工具/函数调用 (Tool calling)
            json_output=True,           # 是否严格支持 JSON 格式输出
            structured_output=False,    # 是否支持结构化输出 (Structured Outputs)
        ),
    )


async def demo_teacher_student_conversation() -> None:
    """
    演示：师生对话场景
    
    学习要点：
    1. 如何通过提示词区分角色的知识水平（老师低 temperature，学生高 temperature）
    2. 基于提到特定关键词（TextMentionTermination）的终止条件设定
    """
    print("\n👨‍🏫 Teacher-Student Conversation Demo")
    print("-" * 50)

    # 创建教师 Agent：需要准确和严谨，因此 temperature 设低为 0.3
    teacher = AssistantAgent(
        name="PythonTeacher",
        model_client=create_model_client(temperature=0.3),
        system_message="""你是一位耐心的Python编程老师。
        你的特点：
        - 用简单易懂的语言解释概念
        - 提供具体的代码示例
        - 鼓励学生提问
        - 循序渐进地教学

        当学生说"我明白了"时，结束这个话题。""",
    )

    # 创建学生 Agent：需要有好奇心和跳跃性思维，因此 temperature 设高为 0.8
    student = AssistantAgent(
        name="Student",
        model_client=create_model_client(temperature=0.8),
        system_message="""你是一个好学的Python初学者。
        你的特点：
        - 对编程概念好奇
        - 会提出具体的问题
        - 需要例子来理解概念
        - 学会后会说"我明白了"

        保持学习的热情，但不要问太多问题。""",
    )

    # 设置终止条件：当任何一条消息中包含"我明白了"这几个字时，对话即刻终止
    termination = TextMentionTermination("我明白了")
    team = RoundRobinGroupChat([teacher, student], termination_condition=termination)

    # 抛出初始话题，开始上课
    task = "老师，请教我Python中的列表是什么，怎么使用？"
    result = await team.run(task=task)

    print("📚 教学对话记录:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        content = (
            message.content[:150] + "..."
            if len(message.content) > 150
            else message.content
        )
        print(f"   {i}. {sender}: {content}")

    print("\n📊 对话统计:")
    print(f"   总消息数: {len(result.messages)}")
    print(f"   停止原因: {result.stop_reason}")


async def demo_debate_conversation() -> None:
    """
    演示：双智能体辩论场景
    
    学习要点：
    1. 为 Agent 设定完全相反的立场
    2. 如何让模型在逻辑上互相博弈，并在被对方说服达成共识时自然终止
    """
    print("\n🗣️ Debate Conversation Demo")
    print("-" * 50)

    # 创建 Python 支持者 Agent
    python_advocate = AssistantAgent(
        name="PythonAdvocate",
        model_client=create_model_client(temperature=0.6),
        system_message="""你是Python编程语言的支持者。
        你的观点：
        - Python简单易学
        - 生态系统丰富
        - 适合快速开发
        - 在AI/ML领域领先

        进行友好的辩论，提出有力的论据。当对方说"好吧，你说得有道理"时停止辩论。""",
    )

    # 创建 JavaScript 支持者 Agent
    js_advocate = AssistantAgent(
        name="JSAdvocate",
        model_client=create_model_client(temperature=0.6),
        system_message="""你是JavaScript编程语言的支持者。
        你的观点：
        - JavaScript无处不在
        - 前后端都能用
        - 性能在不断提升
        - 社区活跃度高

        进行友好的辩论，但要保持开放的心态。如果对方论据充分，可以说"好吧，你说得有道理"。""",
    )

    # 当一方说出"好吧，你说得有道理"时结束辩论
    termination = TextMentionTermination("好吧，你说得有道理")
    team = RoundRobinGroupChat(
        [python_advocate, js_advocate],
        termination_condition=termination,
    )

    # 抛出辩驳命题，开启辩论
    task = "让我们讨论一下：Python和JavaScript哪个更适合初学者学习编程？"
    result = await team.run(task=task)

    print("🎭 辩论记录:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        content = (
            message.content[:200] + "..."
            if len(message.content) > 200
            else message.content
        )
        print(f"   {i}. {sender}: {content}")

    print("\n📊 辩论统计:")
    print(f"   总轮次: {len(result.messages)}")
    print(
        f"   获胜者: "
        f"{result.messages[-2].source if len(result.messages) > 1 else 'Unknown'}",
    )


async def demo_creative_collaboration() -> None:
    """
    演示：创意写作协作场景
    
    学习要点：
    1. 极高的 Temperature 激发创作灵感（0.8-0.9）
    2. 通过系统提示词设计流水线接力写作模式
    """
    print("\n🎨 Creative Collaboration Demo")
    print("-" * 50)

    # 创建故事作家 Agent：发散性思维
    writer = AssistantAgent(
        name="StoryWriter",
        model_client=create_model_client(temperature=0.9),
        system_message="""你是一位创意作家。
        你的任务：
        - 开始一个有趣的故事
        - 创造生动的场景和角色
        - 留下悬念让编辑续写
        - 保持故事的连贯性

        每次写2-3句话，然后说"请编辑继续"。""",
    )

    # 创建故事编辑 Agent：推进情节接力
    editor = AssistantAgent(
        name="Editor",
        model_client=create_model_client(temperature=0.8),
        system_message="""你是一位故事编辑。
        你的任务：
        - 继续作家开始的故事
        - 发展情节和角色
        - 保持故事风格一致
        - 推进故事发展

        每次写2-3句话。如果故事达到高潮，说"故事完成"。""",
    )

    # 设置共创终止暗号
    termination = TextMentionTermination("故事完成")
    team = RoundRobinGroupChat([writer, editor], termination_condition=termination)

    # 启动接力小说任务
    task = "让我们一起创作一个关于时间旅行者的短篇科幻故事。"
    result = await team.run(task=task)

    print("📖 创作过程:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        print(f"   {i}. {sender}:")
        print(f"      {message.content}")
        print()

    print("📊 创作统计:")
    print(f"   总段落: {len(result.messages)}")
    print(
        f"   最终作者: "
        f"{result.messages[-2].source if len(result.messages) > 1 else 'Unknown'}",
    )


async def demo_problem_solving_team() -> None:
    """
    演示：问题解决协作组
    
    学习要点：
    分步骤专业诊断流水线：分析师先拆解问题 -> 解决方案专家后出策略
    """
    print("\n🧩 Problem Solving Team Demo")
    print("-" * 50)

    # 创建数据分析师：负责溯源问题
    analyst = AssistantAgent(
        name="DataAnalyst",
        model_client=create_model_client(temperature=0.3),
        system_message="""你是一位数据分析师。
        你的职责：
        - 分析问题和数据
        - 提出分析方法
        - 识别关键指标
        - 提供客观的见解

        分析完成后说"分析完成，请解决方案专家提出建议"。""",
    )

    # 创建解决架构师：基于前者的分析出详细应对策略
    solution_expert = AssistantAgent(
        name="SolutionExpert",
        model_client=create_model_client(temperature=0.5),
        system_message="""你是解决方案专家。
        你的职责：
        - 基于分析结果提出解决方案
        - 考虑实施的可行性
        - 提供具体的行动步骤
        - 评估风险和收益

        方案完成后说"解决方案已制定完成"。""",
    )

    # 定义专家团验收的终结词汇
    termination = TextMentionTermination("解决方案已制定完成")
    team = RoundRobinGroupChat(
        [analyst, solution_expert],
        termination_condition=termination,
    )

    # 向业务解决小组抛出痛点
    task = "我们的电商网站转化率下降了15%，需要分析原因并提出解决方案。"
    result = await team.run(task=task)

    print("🔍 问题解决过程:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        content = (
            message.content[:300] + "..."
            if len(message.content) > 300
            else message.content
        )
        print(f"   {i}. {sender}:")
        print(f"      {content}")
        print()

    print("📊 解决方案统计:")
    print(f"   分析轮次: {len(result.messages)}")


async def demo_max_message_termination() -> None:
    """
    演示：最大消息数量限制
    
    学习要点：
    如何防止无休止的水群（死循环），在指定对话轮数后强制拔电源掐断。
    """
    print("\n⏱️ Max Message Termination Demo")
    print("-" * 50)

    # 创建两个极其健谈的闲聊机器人
    agent1 = AssistantAgent(
        name="ChatterBox1",
        model_client=create_model_client(temperature=0.7),
        system_message="你是一个健谈的聊天机器人，喜欢讨论技术话题。每次回复要简短。",
    )

    agent2 = AssistantAgent(
        name="ChatterBox2",
        model_client=create_model_client(temperature=0.7),
        system_message="你是另一个健谈的聊天机器人，也喜欢技术讨论。每次回复要简短。",
    )

    # 强制终止条件：不管聊得多么火热，一旦团队总消息数达到 6 条立即强制拔电源
    termination = MaxMessageTermination(6)
    team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)

    # 抛出一个容易引发无限讨论的宽泛话题
    task = "聊聊人工智能的发展趋势吧！"
    result = await team.run(task=task)

    print("💬 限制消息数的对话:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        print(f"   {i}. {sender}: {message.content}")

    print("\n📊 对话统计:")
    print(f"   实际消息数: {len(result.messages)}")
    print(f"   停止原因: {result.stop_reason}")
    print("   ✅ 成功在达到消息限制时停止")


async def demo_selector_group_chat() -> None:
    """
    演示：复杂的动态群聊场景 (SelectorGroupChat)
    
    学习要点：
    1. 多个 Agent 参与的无固定顺序的群聊
    2. 由模型根据上下文动态选择下一个最适合发言的 Agent (动态路由)
    3. 模拟软件开发小组（产品经理、程序员、测试）的协作过程
    """
    print("\n👥 Selector Group Chat Demo (Dynamic Routing)")
    print("-" * 50)

    # 1. 创建产品经理Agent (Product Manager)
    # PM 负责定义需求，因此 temperature (0.5) 略微带点创造力但保证逻辑严密
    pm = AssistantAgent(
        name="ProductManager",
        model_client=create_model_client(temperature=0.5),
        system_message="""你是一位严谨的产品经理。
        你的职责：
        - 明确用户需求
        - 将需求转化为开发任务给 Developer (开发者)
        - 收到 Tester (测试人员) 的测试结果后进行最终验收，或者打回
        
        【停止指令】当测试通过，并且你认为所有需求已经满足，没有代码安全隐患或遗漏时，请务必直接输出 "任务完成" 这四个字来结束整个开发流程。千万不要在没测试的情况下结束。""",
    )

    # 2. 创建程序员Agent (Developer)
    # 程序员写代码需要高度确定性和正确性，所以 temperature 设定极低 (0.1或0.3)
    dev = AssistantAgent(
        name="Developer",
        model_client=create_model_client(temperature=0.1),
        system_message="""你是一位后端开发工程师。
        你的职责：
        - 根据 ProductManager 的需求编写高质量的 Python 代码
        - 修复 Tester (测试QA) 提出的各种 bug 与逻辑漏洞
        
        【交接指令】每次写完第一版代码，或者修复了 bug 之后，请明确在回复中请求 Tester ("Tester，代码已准备好，请进行测试审查") 进行测试。""",
    )

    # 3. 创建测试工程师Agent (QA Tester)
    # QA 负责挑错、想边界条件（比如空白、超长、非法字符），需要稍微有一点点严谨的散发性 (0.3)
    tester = AssistantAgent(
        name="Tester",
        model_client=create_model_client(temperature=0.3),
        system_message="""你是一位严格且具有怀疑精神的QA测试工程师。
        你的职责：
        - 审查 Developer 提供的代码，指出所有的逻辑漏洞 (例如正则表达式不够完善、没有处理边界条件等)。
        - 不需要你自己修改代码，只负责找出所有潜在的 Bug！
        
        【交接指令】如果发现问题，请严厉指出并明确让 Developer 重新返工修复；如果从代码逻辑上看绝对完美无瑕，请直接输出 "任务完成" 这四个字来结束整个开发流程 (这也会触发群聊强制终止)。""",
    )

    # 4. 设定整个系统的终止条件
    # 当收到 "任务完成" 这几个字的回复时，SelectorGroupChat 就会由于达到了 termination_condition 中止继续传球给别的 agent
    termination = TextMentionTermination("任务完成")

    # 5. 使用 SelectorGroupChat 进行动态路由 (也就是常说的 AI 在背后当群主/主持人)
    # 【核心知识点】，zu
    # - team 的核心容器模型 (model_client): 这里需要传入一个大模型实例。
    #   这个模型不在前台说话，而是每一轮都自动分析当前的聊天记录，并结合参与人（PM, DEV, Tester）的描述，
    #   决策出下一个发言的角色是谁。为了保证它能做出准确决定而不带无意义的创新发散，temperature 设定极低如 0.1。
    # - 工作流: PM发布工作 -> LLM群主分析当前上下文 -> 判定该交给 Developer -> 
    #   Developer 写代码请求测试 -> LLM判断下文应该交给 Tester -> QA发现问题 -> LLM群主交回给 Developer ->
    #   最终 QA通过并发出了 "任务完成" 导致条件满足。
    team = SelectorGroupChat(
        [pm, dev, tester],
        model_client=create_model_client(temperature=0.1),  
        termination_condition=termination,
    )

    # 6. 设定期望的工作输入 (Task)，触发真正的 AI 群聊执行
    # task 被塞给整个 team.run()，由 PM 接管或者是路由直接接管并启动群聊循环
    task = "需求：我们需要写一个Python函数，用来验证一个字符串是否是合法的中国大陆手机号码。请大家思考边界条件(比如空串、特殊字符)、完成代码编写并最终测试通过。"
    result = await team.run(task=task)  # 协程对象，等待整体群聊天结束并保存全记录

    print("💻 协作开发过程记录:")
    for i, message in enumerate(result.messages, 1):
        # Result 的 message 将保存所有经过群聊的回复
        sender = message.source if hasattr(message, "source") else "Unknown"
        # 截取过长的文本输出
        content = (
            message.content[:300] + "..."
            if len(message.content) > 300
            else message.content
        )
        print(f"   [{i}] {sender} 发言:\n      {content}\n")

    print("📊 开发协作统计:")
    print(f"   参与角色数: 3")
    print(f"   总交互消息数: {len(result.messages)}")
    print(f"   停止原因: {result.stop_reason}")


async def demo_research_team_selector() -> None:
    """
    演示：研究团队的智能协作 (基于 SelectorGroupChat)
    
    学习要点:
    展示 SelectorGroupChat 高级功能，不依赖简单的 RoundRobin 轮询，
    而是由模型做裁判决定发给“技术专家”还是“数据科学家”。
    """
    print("\n🔬 Research Team Demo (SelectorGroupChat)")
    print("-" * 50)

    # 1. 团队负责人：掌控流程和发言调用
    research_lead = AssistantAgent(
        name="ResearchLead",
        model_client=create_model_client(temperature=0.3),
        system_message="""你是研究团队负责人。
        职责：
        - 制定研究计划和方向
        - 分配任务给团队成员
        - 当需要具体的技术分析时，请 TechnicalExpert 发言。
        - 当需要数据分析时，请 DataScientist 发言。
        
        【停止指令】当研究完成时，务必输出"研究项目完成"。""",
    )

    # 2. 技术专家 (TechExpert)
    tech_expert = AssistantAgent(
        name="TechnicalExpert",
        model_client=create_model_client(temperature=0.4),
        system_message="""你是技术专家。专长是深度学习架构、算法和性能优化逻辑。
        只在涉及技术和算法体系构建时介入提供深度支持。""",
    )

    # 3. 数据科学家 (DataScientist)
    data_scientist = AssistantAgent(
        name="DataScientist",
        model_client=create_model_client(temperature=0.4),
        system_message="""你是数据科学家。专长是数据建模、统计分析和实验洞察。
        负责回答数据维度的合理性，以及针对数据的验证。""",
    )

    termination = TextMentionTermination("研究项目完成")
    research_team = SelectorGroupChat(
        participants=[research_lead, tech_expert, data_scientist],
        # 路由选择模型（裁判），temperature要低，保证按指令分发任务
        model_client=create_model_client(temperature=0.1),
        termination_condition=termination,
    )

    task = "我们需要研究如何提高推荐系统的准确性和用户满意度。请制定研究计划并分析关键技术及数据挑战。"
    result = await research_team.run(task=task)

    print("🔬 团队智能协作过程:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        content = message.content[:200] + "..." if len(message.content) > 200 else message.content
        print(f"   [{i}] {sender} 发言:\n      {content}\n")
    
    print("📊 运行统计:")
    print(f"   交互轮次: {len(result.messages)} | 停止原因: {result.stop_reason}")


async def demo_selector_with_custom_prompt() -> None:
    """
    演示：通过定制化路由提示词 (Custom Selector Prompt) 深入控制发言顺序
    
    学习要点:
    在 SelectorGroupChat 中，有时默认的路由提示词无法精准满足复杂的业务逻辑。
    我们可以通过自定义 `selector_prompt`，像写伪代码一样，明确告诉路由 LLM 按什么规则调度。
    """
    print("\n🎯 Custom Selector Prompt Demo (定制化路由)")
    print("-" * 50)

    # 定义各个职能角色的简单版本
    planner = AssistantAgent("Planner", 
                             model_client=create_model_client(0.1), 
                             description="负责拆解任务规划的计划员", 
                             system_message="你的职责是将任务拆分成子步骤，并且在完成时输出'计划完成'")
    executer = AssistantAgent("Executor", 
                              model_client=create_model_client(0.1), 
                              description="负责执行代码和操作的执行员", 
                              system_message="根据Planner的计划执行具体的任务，完成后回复'执行完毕'")
    reviewer = AssistantAgent("Reviewer", 
                             model_client=create_model_client(0.1), 
                             description="负责检查执行结果的审核员", 
                             system_message="检查Executor的产出，如果没问题则必须输出'全部任务验收通过'。发现问题则退回。")

    termination = TextMentionTermination("全部任务验收通过")
    
    # 核心：通过定制 selector_prompt 赋予群聊“状态机”式的严格跳转规则
    custom_prompt = """你是一个群聊的主持人。请根据以下规则选择下一个发言人：
    1. 刚开始必须由 Planner 发言。
    2. 如果 Planner 说'计划完成'，下一个必须是 Executor。
    3. 如果 Executor 说'执行完毕'，下一个必须是 Reviewer。
    4. 如果 Reviewer 提出修改建议，下一个必须是 Executor 重新执行。
    根据当前的对话历史，输出下一个应该发言的角色名称。
    """

    team = SelectorGroupChat(
        [planner, executer, reviewer],
        model_client=create_model_client(temperature=0.1),
        termination_condition=termination,
        selector_prompt=custom_prompt, # 注入自定义路由逻辑
        allow_repeated_speaker=False   # 禁止连续两次自己给自己传球
    )

    result = await team.run(task="请计划并执行一个简单的 'Hello World' 打印程序，并由审核员验收。")

    print("🎯 定制化路由协作过程:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        content = message.content[:200] + "..." if len(message.content) > 200 else message.content
        print(f"   [{i}] {sender} 发言:\n      {content}\n")


async def demo_custom_selector_function() -> None:
    """
    演示：闭包/函数式定制化路由 (Custom Speaker Selection Function)
    
    学习要点:
    在更严肃的工程场景中，我们不能完全信任大模型的路由（即使加了严苛的 Prompt）。
    新版 AutoGen 允许直接传入一个 Python 函数 (或闭包)，通过执行纯原生的 Python 逻辑
    （如基于正则、If-Else判断、甚至外挂查询）来强行指派下一个发言人。
    这就实现了代码级的“硬状态约束”（Hard State Graph）。
    """
    print("\n⚙️ Custom Selector Function Demo (硬代码闭包路由)")
    print("-" * 50)

    # 创建几个毫无智商，只知道复读的简单 Agent 用来做流转测试
    a = AssistantAgent("Agent_A", model_client=create_model_client(0.1), system_message="回复固定这句话: '我已经处理完A步骤，转交B'")
    b = AssistantAgent("Agent_B", model_client=create_model_client(0.1), system_message="回复固定这句话: '我已经处理完B步骤，再转交A'")
    
    # 我们用一个闭包外部变量来记录回合数，演示复杂的非文本状态约束
    class LoopState:
        turns = 0
        
    state = LoopState()

    # 重点：定义自定义选择器闭包函数 (Custom Speaker Selector)
    # 它接收当前对话的所有消息作为入参，返回字符串(下一个发言者的名字)
    def my_strict_router(messages) -> str:
        # 这个函数完全脱离大模型执行，靠纯代码跑逻辑
        if not messages:
            return "Agent_A"  # 启动时永远让 A 先说
            
        last_message = messages[-1].content
        last_speaker = messages[-1].source
        
        # 使用闭包外部变量做逻辑判定
        if state.turns >= 2:
            # 打破正常规则，如果玩了两个回合（A-B-A-B），强行让A说一句终止的话来满足下方 termination 条件
            if last_speaker == "Agent_B":
                 # 强行注入终止条件信号（实战中可以用来做系统熔断等）
                 return "Agent_A" 
                 
        if last_speaker == "Agent_A" and "转交B" in last_message:
            return "Agent_B"
            
        if last_speaker == "Agent_B" and "转交A" in last_message:
            state.turns += 1
            return "Agent_A"
            
        # 兜底选择
        return "Agent_A"

    # 基于回合数的自定义终止条件
    termination = MaxMessageTermination(5)

    # 构建 GroupChat，此时不再设置 model_client (或者设了也不会被用于路由)，
    # 而是传入 selector_func (在某些新版 API 中为 selector) 属性。
    # 注意：在最新的 autogen_agentchat 0.4 中，SelectorGroupChat 专门增加了 `selector_func`
    team = SelectorGroupChat(
        participants=[a, b],
        model_client=create_model_client(0.1), # 由于用了纯函数，这里的模型不怎么工作了
        termination_condition=termination,
        selector_func=my_strict_router # 将闭包函数挂载为图跳转的核心控制器
    )

    result = await team.run(task="开始运行硬约束路由流转测试！")

    print("⚙️ 硬代码闭包路由协作过程:")
    for i, message in enumerate(result.messages, 1):
        sender = message.source if hasattr(message, "source") else "Unknown"
        print(f"   [{i}] {sender} 发言:\n      {message.content}\n")


async def main() -> None:
    """
    主入口函数：依次执行所有双智能体对话模式的演示
    """
    print("🤖 AutoGen 简单对话系统演示")
    print("=" * 60)

    try:
        # await demo_teacher_student_conversation()
        # await demo_debate_conversation()
        # await demo_creative_collaboration()
        # await demo_problem_solving_team()
        # await demo_max_message_termination()

        # 运行动态路由群聊演示 (SelectorGroupChat)
        await demo_selector_group_chat()
        
        # 运行新增的研究团队动态委派演示
        await demo_research_team_selector()

        # 运行定制化路由与状态约束演示 （软约束）
        await demo_selector_with_custom_prompt()

        # 运行纯代码闭包路由演示 （硬约束）
        await demo_custom_selector_function()

        print("\n✨ 所有对话演示完成!")
        print("\n📚 关键要点:")
        print("   • RoundRobinGroupChat 管理按顺序的双智能体/多智能体轮询对话")
        print("   • SelectorGroupChat 管理通过模型自主判断“下一个是谁”的动态路由")
        print("   • 不同的终止条件控制对话流程")
        print("   • 系统消息定义智能体的角色和行为")
        print("   • Temperature 影响回复的创造性")
        print("   • 对话可以有各种应用场景")
        print("   • 适当的终止条件确保对话有意义地结束")

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("💡 检查API配置和网络连接")


if __name__ == "__main__":
    # Python 异步库启动事件循环进入 AutoGen
    # 这是异步框架的标准启动方式，确保里面包含到的网络请求和各种协程挂机可以顺畅执行
    asyncio.run(main())