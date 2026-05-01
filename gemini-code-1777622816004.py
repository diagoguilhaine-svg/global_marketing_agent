import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# ==========================================
# 配置参数：设置你的 LLM API Key
# (你可以使用 OpenAI、Google Gemini 或本地大模型，这里以 OpenAI 格式为例)
# ==========================================
os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"

# 初始化底层大语言模型驱动核心 (使用有良好推理能力的模型)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# ==========================================
# 第一步：定义多代理 (Agents) 的角色与能力
# ==========================================

# 1. 市场洞察代理：负责抓取卖点和受众分析
market_researcher = Agent(
    role='资深出海市场分析师',
    goal='分析 {product_info} 在 {target_market} 的受众痛点和核心卖点',
    backstory='你拥有10年跨境电商和SaaS出海经验，对全球各地的消费心理、文化禁忌和本地化趋势了如指掌。',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 2. 内容创作代理：负责生成符合渠道特性的营销文案
content_creator = Agent(
    role='跨平台营销文案专家',
    goal='基于市场分析，为 {channels} 创作引人入胜的营销内容',
    backstory='你精通各大社交媒体（Twitter, LinkedIn, TikTok, Instagram）的算法和用户偏好，擅长用爆款公式（如AIDA、PAS）写文案。',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 3. 多语言本地化代理：负责地道翻译与文化适配（不仅仅是翻译，而是创译）
localization_expert = Agent(
    role='多语言本地化总监',
    goal='将营销文案翻译并适配为地道的 {target_language}，确保符合当地俚语和表达习惯',
    backstory='你是母语级别的多语种专家，且对目标市场的流行文化、网络语境有极高的敏感度。你擅长“创译”（Transcreation），能保留营销钩子（Hook）的吸引力。',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 4. 品牌合规审核代理：防范风险与质量把控
compliance_reviewer = Agent(
    role='出海品牌合规与公关审查官',
    goal='审查最终的 {target_language} 内容，确保没有宗教/文化禁忌、政治敏感词汇，并符合品牌基调',
    backstory='你是个极其严谨的细节控，负责在内容发布前做最后一道把关，防止任何公关危机的发生，并优化最终的排版和 Hashtags。',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ==========================================
# 第二步：定义自动化任务流 (Tasks)
# ==========================================

research_task = Task(
    description='分析产品：{product_info}。针对目标市场：{target_market}。提取3个核心卖点和2个用户痛点，形成一份简短的市场洞察报告。',
    expected_output='一份包含产品核心卖点和用户痛点的市场洞察报告（Markdown 格式）。',
    agent=market_researcher
)

creation_task = Task(
    description='根据上一步的市场洞察报告，为以下渠道：{channels} 撰写营销文案草稿。针对每个渠道生成一条完整的帖子：必须包含吸引人的标题（Hook）、核心价值传递以及明确的行动呼吁（CTA）。',
    expected_output='按渠道分类的营销文案草稿集合（使用通用语言/英语撰写即可，方便下一步翻译）。',
    agent=content_creator
)

localization_task = Task(
    description='将上一步生成的营销文案，翻译并进行本地化改造，目标语言为：{target_language}。必须加上适合该语言生态和平台的 Emoji 以及热门本土标签（Hashtags）。',
    expected_output='完全本地化、地道且带有 Emoji 和 Hashtag 的目标语言（{target_language}）社交媒体文案。',
    agent=localization_expert
)

review_task = Task(
    description='仔细阅读本地化后的文案，检查是否符合 {target_market} 的文化习俗，有无排版错误、语法错误或合规风险。如果没有问题，输出最终的可发布版本。',
    expected_output='最终确认无误、格式整洁、可直接复制到各个渠道发布的最终终稿，确保排版适合阅读。',
    agent=compliance_reviewer
)

# ==========================================
# 第三步：组建多代理工作流 (Crew)
# ==========================================

marketing_crew = Crew(
    agents=[market_researcher, content_creator, localization_expert, compliance_reviewer],
    tasks=[research_task, creation_task, localization_task, review_task],
    process=Process.sequential # 采用顺序工作流：分析 -> 创作 -> 翻译 -> 审核
)

# ==========================================
# 第四步：执行系统与用例测试
# ==========================================

if __name__ == "__main__":
    print("🚀 正在启动出海跨语言多代理内容生产引擎...\n")
    
    # 模拟真实的业务输入参数
    inputs = {
        'product_info': '一款名为“SonicSync”的AI驱动降噪无线耳机，支持360度空间音频，单次续航长达50小时，专为运动和通勤设计，售价仅为Apple AirPods的一半。',
        'target_market': '日本年轻一代与都市通勤白领',
        'target_language': '日语',
        'channels': 'Twitter (X), Instagram'
    }
    
    # Kickoff 开始执行所有任务
    result = marketing_crew.kickoff(inputs=inputs)
    
    print("========================================")
    print("✨ 【系统产出】跨语言多渠道营销文案终稿：\n")
    print(result)