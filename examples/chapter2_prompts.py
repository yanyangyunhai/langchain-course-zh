"""
第2章示例代码：提示词工程
演示PromptTemplate的使用和邮件助手
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

def example_1_basic_template():
    """示例1：基础PromptTemplate"""
    print("=== 示例1：基础PromptTemplate ===")

    template = """
请回答以下问题：{question}
要求：回答简洁明了，不超过100字。
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"]
    )

    formatted = prompt.format(question="什么是LangChain？")
    print(f"格式化后的提示词：\n{formatted}\n")

def example_2_chat_template():
    """示例2：ChatPromptTemplate"""
    print("=== 示例2：ChatPromptTemplate ===")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}专家"),
        ("user", "{input}")
    ])

    messages = prompt.format_messages(
        role="Python编程",
        input="如何定义一个函数？"
    )

    print("消息列表：")
    for msg in messages:
        print(f"- {msg.type}: {msg.content}\n")

def example_3_partial_variables():
    """示例3：部分变量填充"""
    print("=== 示例3：部分变量填充 ===")

    template = """
作为{role}专家，请{action}。
主题：{topic}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["action", "topic"],
        partial_variables={"role": "AI技术"}  # 预先填充
    )

    formatted = prompt.format(
        action="编写教程",
        topic="LangChain基础"
    )

    print(f"格式化结果：\n{formatted}\n")

def example_4_few_shot_learning():
    """示例4：Few-shot学习"""
    print("=== 示例4：Few-shot学习 ===")

    template = """
将下面的句子转换成简洁版：

示例1：
原文：人工智能正在改变我们的生活方式
简洁版：AI改变生活

示例2：
原文：机器学习是人工智能的一个重要分支
简洁版：ML是AI分支

示例3：
原文：深度学习使用多层神经网络来模拟人脑
简洁版：深度学习用多层网络模拟人脑

原文：{original}
简洁版：
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["original"]
    )

    examples = [
        "自然语言处理让计算机能够理解人类语言",
        "计算机视觉让机器能够识别图像和视频"
    ]

    for example in examples:
        formatted = prompt.format(original=example)
        print(f"原文：{example}")
        print(f"提示词：\n{formatted}\n")

def example_5_email_assistant():
    """示例5：邮件助手"""
    print("=== 示例5：邮件助手 ===")

    class EmailAssistant:
        def __init__(self):
            self.llm = ChatOpenAI(model="gpt-3.5-turbo")
            self.system_prompt = """你是一个专业的邮件助手。"""

        def classify_email(self, email_content):
            """分类邮件"""
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""请将以下邮件分类为：工作/个人/促销/垃圾邮件/其他

邮件内容：
{email_content}

只返回类别名称。""")
            ])

            return self.llm.invoke(prompt.format_messages(
                email_content=email_content
            )).content.strip()

        def generate_reply(self, email_content):
            """生成回复"""
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""为以下邮件生成礼貌的自动回复：
{email_content}

回复要求：3-5句话，表达感谢。""")
            ])

            return self.llm.invoke(prompt.format_messages(
                email_content=email_content
            )).content.strip()

    assistant = EmailAssistant()

    email = """
    From: boss@company.com
    Subject: 项目进度汇报

    请在本周五前提交项目进展报告。
    """

    print(f"邮件内容：{email}")
    print(f"分类：{assistant.classify_email(email)}")
    print(f"回复：{assistant.generate_reply(email)}\n")

def example_6_chain_of_thought():
    """示例6：思维链"""
    print("=== 示例6：思维链 ===")

    template = """
请逐步思考并回答：

问题：{question}

思考过程：
1. 首先理解问题
2. 分析关键信息
3. 计算或推理
4. 得出结论

请显示详细思考过程。
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"]
    )

    questions = [
        "小明有5个苹果，吃了2个，又买了3个，现在有几个？",
        "一个正方形的边长是5厘米，周长是多少厘米？"
    ]

    for question in questions:
        formatted = prompt.format(question=question)
        print(f"问题：{question}")
        print(f"提示词：\n{formatted}\n")

if __name__ == "__main__":
    print("🚀 第2章示例代码：提示词工程\n")

    try:
        example_1_basic_template()
        example_2_chat_template()
        example_3_partial_variables()
        example_4_few_shot_learning()
        example_5_email_assistant()
        example_6_chain_of_thought()

        print("✅ 所有示例运行完成！")
    except Exception as e:
        print(f"❌ 运行出错：{e}")
        print("\n请确保：")
        print("1. 已安装所需依赖：pip install langchain langchain-openai openai")
        print("2. 已设置有效的OpenAI API密钥")
