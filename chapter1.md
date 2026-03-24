# 第1章：LangChain基础

## 1.1 什么是LangChain？

### 诞生背景

随着大语言模型（LLM）的快速发展，开发者发现直接使用LLM API存在很多限制：
- 缺乏记忆能力
- 无法访问外部数据
- 难以进行复杂的推理
- 不方便集成其他工具

LangChain应运而生，它是一个专门为开发LLM应用设计的框架，解决了上述问题。

### 为什么选择LangChain？

1. **模块化设计**：将LLM应用拆解为独立组件，便于组合和复用
2. **丰富的集成**：支持多种LLM、向量数据库、工具等
3. **活跃的社区**：文档完善，更新迅速，问题容易解决
4. **生产级质量**：经过大量实战验证，稳定可靠

### 应用场景

- 智能问答系统
- 文档分析工具
- 自动化客服
- 数据分析助手
- 代码生成工具
- 知识库检索
- 聊天机器人

## 1.2 环境搭建

### Python环境配置

```bash
# 推荐Python版本：3.8 - 3.11
# 检查Python版本
python --version

# 创建虚拟环境（推荐）
python -m venv langchain_env
source langchain_env/bin/activate  # Linux/Mac
# langchain_env\Scripts\activate   # Windows
```

### 安装LangChain

```bash
# 安装核心库
pip install langchain

# 安装OpenAI集成
pip install langchain-openai

# 安装常用依赖
pip install openai tiktoken
```

### 配置API密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"

# 或在代码中设置
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 第一个LangChain程序

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# 创建语言模型实例
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# 创建消息
message = HumanMessage(content="你好，请介绍一下你自己")

# 发送请求
response = llm.invoke([message])

# 输出结果
print(response.content)
```

**输出示例：**
```
你好！我是基于大语言模型开发的AI助手，可以帮助你完成各种任务，
包括问答、写作、编程、分析等。有什么可以帮助你的吗？
```

## 1.3 核心概念

LangChain由以下核心组件组成：

### 1.3.1 Prompts（提示词）

提示词是与LLM交互的关键，决定了模型的理解和输出。

```python
from langchain.prompts import ChatPromptTemplate

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}助手"),
    ("user", "{input}")
])

# 填充变量
filled_prompt = prompt.format_messages(
    role="Python编程",
    input="如何用Python打印Hello World？"
)
```

### 1.3.2 Models（模型）

LangChain支持多种模型：
- LLM：基础文本生成模型
- Chat Model：对话模型
- Embedding Model：文本嵌入模型

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 对话模型
chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# 嵌入模型
embedding_model = OpenAIEmbeddings()
```

### 1.3.3 Indexes（索引）

索引用于管理和查询大量文本数据。

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 分割文档
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
texts = text_splitter.split_documents(documents)
```

### 1.3.4 Memory（记忆）

记忆组件用于存储和检索对话历史。

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory()

# 保存对话
memory.save_context(
    {"input": "我叫小明"},
    {"output": "你好小明！"}
)

# 查看记忆
print(memory.load_memory_variables({}))
```

### 1.3.5 Chains（链）

链将多个组件连接起来，形成完整的处理流程。

```python
from langchain.chains import LLMChain

# 创建链
chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    memory=memory
)

# 运行链
result = chain.run("我的名字是什么？")
print(result)
```

### 1.3.6 Agents（智能体）

智能体可以自主决策和执行任务。

```python
from langchain.agents import initialize_agent, Tool
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="用于数学计算"
    )
]

# 创建智能体
agent = initialize_agent(
    tools,
    chat_model,
    agent="zero-shot-react-description",
    verbose=True
)

# 运行智能体
result = agent.run("计算 123 * 456")
```

## 1.4 实战：简单的问答机器人

让我们创建一个简单的问答机器人，综合运用前面学到的知识。

### 完整代码

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

class QABot:
    def __init__(self):
        # 初始化模型
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )

        # 创建提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的AI助手，擅长回答各种问题"),
            ("user", "{input}")
        ])

        # 创建记忆
        self.memory = ConversationBufferMemory()

        # 创建链
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def chat(self, user_input):
        """与机器人对话"""
        response = self.chain.run(user_input)
        return response

# 使用示例
if __name__ == "__main__":
    bot = QABot()

    # 对话循环
    print("🤖 问答机器人已启动（输入'quit'退出）")
    while True:
        user_input = input("\n你: ")

        if user_input.lower() == 'quit':
            print("👋 再见！")
            break

        response = bot.chat(user_input)
        print(f"机器人: {response}")
```

### 运行示例

```
🤖 问答机器人已启动（输入'quit'退出）

你: 你好
机器人: 你好！很高兴见到你。有什么我可以帮助你的吗？

你: 今天天气怎么样？
机器人: 很抱歉，我无法获取实时天气信息。你可以查看天气APP或网站了解当地天气情况。

你: Python是什么？
机器人: Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。它具有简洁明了的语法、强大的标准库和丰富的第三方库，被广泛应用于Web开发、数据分析、人工智能、科学计算等多个领域。

你: quit
👋 再见！
```

### 优化建议

1. **添加更多功能**
   - 上下文记忆管理
   - 多轮对话支持
   - 情感识别

2. **提升回答质量**
   - 优化提示词
   - 添加知识库
   - 集成搜索工具

3. **增强用户体验**
   - 流式输出
   - 加载动画
   - 历史记录

## 本章小结

通过本章学习，你应该已经掌握了：
- LangChain的基本概念和应用场景
- 环境搭建和配置
- 核心组件的使用方法
- 简单问答机器人的开发

下一章，我们将深入学习提示词工程，掌握如何设计高效的提示词来提升AI应用的质量。

## 练习题

1. 使用LangChain创建一个翻译机器人，支持中英文互译。
2. 为问答机器人添加长对话记忆功能。
3. 尝试不同的温度参数，观察对输出结果的影响。

## 常见问题

**Q: LangChain支持哪些LLM？**
A: LangChain支持OpenAI、Anthropic、Cohere、HuggingFace等多种LLM。

**Q: 如何控制API成本？**
A: 可以通过设置max_tokens参数、使用缓存、选择合适的模型等方式控制成本。

**Q: LangChain与直接调用API有什么区别？**
A: LangChain提供了更高层次的抽象，让开发更简单，同时提供了记忆、工具集成等高级功能。

**Q: 需要付费使用LangChain吗？**
A: LangChain本身是开源免费的，但底层的LLM服务（如OpenAI）需要付费。

---

下一章预告：提示词工程 - 学习如何设计高效的提示词来提升AI应用的性能。
