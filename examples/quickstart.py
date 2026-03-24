"""
LangChain快速入门示例
运行此文件需要先安装：pip install langchain langchain-openai openai
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# 设置API密钥（请替换为你的实际密钥）
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

def example_1_basic_chat():
    """示例1：基础对话"""
    print("=== 示例1：基础对话 ===")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    response = llm.invoke([
        {"role": "user", "content": "你好，请用一句话介绍Python"}
    ])

    print(f"回答：{response.content}\n")

def example_2_prompt_template():
    """示例2：提示词模板"""
    print("=== 示例2：提示词模板 ===")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}专家"),
        ("user", "{question}")
    ])

    formatted_prompt = prompt.format_messages(
        role="Python编程",
        question="如何定义一个函数？"
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(formatted_prompt)

    print(f"回答：{response.content}\n")

def example_3_memory():
    """示例3：记忆功能"""
    print("=== 示例3：记忆功能 ===")

    memory = ConversationBufferMemory()

    # 保存对话
    memory.save_context(
        {"input": "我的名字是小明"},
        {"output": "你好小明！"}
    )

    # 检索记忆
    memory_variables = memory.load_memory_variables({})
    print(f"对话历史：{memory_variables['history']}\n")

def example_4_chain():
    """示例4：链式调用"""
    print("=== 示例4：链式调用 ===")

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手"),
        ("user", "{input}")
    ])
    memory = ConversationBufferMemory()

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    # 第一次对话
    response1 = chain.run("我叫小红")
    print(f"对话1：{response1}")

    # 第二次对话（记住上下文）
    response2 = chain.run("我的名字是什么？")
    print(f"对话2：{response2}\n")

def example_5_practical_bot():
    """示例5：实用的聊天机器人"""
    print("=== 示例5：实用的聊天机器人 ===")

    class SimpleBot:
        def __init__(self):
            self.llm = ChatOpenAI(model="gpt-3.5-turbo")
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个有帮助的AI助手"),
                ("user", "{input}")
            ])
            self.memory = ConversationBufferMemory()
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=self.memory
            )

        def chat(self, user_input):
            return self.chain.run(user_input)

    # 测试机器人
    bot = SimpleBot()

    questions = [
        "什么是LangChain？",
        "它有什么用途？",
        "我该如何学习它？"
    ]

    for q in questions:
        response = bot.chat(q)
        print(f"问：{q}")
        print(f"答：{response}\n")

if __name__ == "__main__":
    print("🚀 LangChain快速入门示例\n")

    # 运行所有示例
    try:
        example_1_basic_chat()
        example_2_prompt_template()
        example_3_memory()
        example_4_chain()
        example_5_practical_bot()

        print("✅ 所有示例运行完成！")
    except Exception as e:
        print(f"❌ 运行出错：{e}")
        print("\n请确保：")
        print("1. 已安装所需依赖：pip install langchain langchain-openai openai")
        print("2. 已设置有效的OpenAI API密钥")
