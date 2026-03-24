# 第3章：记忆系统

## 3.1 为什么需要记忆？

在开发AI应用时，你会发现一个重要的问题：LLM本身是"无记忆"的。每次对话都是独立的，模型无法记住之前的对话内容。

### 问题的出现

**示例1：无记忆的问题**
```
用户：我叫小明
AI：你好小明！

用户：我的名字是什么？
AI：我不知道你的名字是什么。
```

**示例2：有记忆的好处**
```
用户：我叫小明
AI：你好小明！

用户：我的名字是什么？
AI：你的名字是小明。
```

### 记忆的重要性

1. **对话连续性**：用户期望AI能记住之前的对话
2. **上下文理解**：基于历史对话理解当前问题
3. **个性化体验**：记住用户偏好和习惯
4. **长期关系**：建立长期对话关系

## 3.2 LangChain Memory组件

LangChain提供了多种Memory组件，满足不同的使用场景：

### 3.2.1 ConversationBufferMemory

最简单的记忆类型，保存所有对话历史。

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory()

# 保存对话
memory.save_context(
    {"input": "我叫小明"},
    {"output": "你好小明！"}
)

# 再次保存
memory.save_context(
    {"input": "我的名字是什么？"},
    {"output": "你的名字是小明。"}
)

# 查看记忆
memory_variables = memory.load_memory_variables({})
print(memory_variables)
# 输出：{'history': 'Human: 我叫小明\nAI: 你好小明！\nHuman: 我的名字是什么？\nAI: 你的名字是小明。'}
```

**优点：**
- 简单易用
- 保留完整对话历史
- 准确回溯

**缺点：**
- 对话过长时消耗token
- 可能超出上下文限制

**适用场景：**
- 短对话（<10轮）
- 需要完整历史
- Token成本不敏感

### 3.2.2 ConversationBufferWindowMemory

只保留最近N轮对话，自动删除旧对话。

```python
from langchain.memory import ConversationBufferWindowMemory

# 创建记忆（保留最近3轮）
memory = ConversationBufferWindowMemory(k=3)

# 保存多轮对话
memory.save_context({"input": "问题1"}, {"output": "回答1"})
memory.save_context({"input": "问题2"}, {"output": "回答2"})
memory.save_context({"input": "问题3"}, {"output": "回答3"})
memory.save_context({"input": "问题4"}, {"output": "回答4"})

# 查看记忆（只保留最后3轮）
memory_variables = memory.load_memory_variables({})
print(memory_variables)
# 输出：{'history': 'Human: 问题2\nAI: 回答2\nHuman: 问题3\nAI: 回答3\nHuman: 问题4\nAI: 回答4'}
```

**优点：**
- 控制token使用
- 保持最近上下文
- 自动清理旧对话

**缺点：**
- 丢失早期信息
- 可能丢失重要上下文

**适用场景：**
- 长对话（>10轮）
- 只需要最近上下文
- Token成本敏感

### 3.2.3 ConversationSummaryMemory

自动总结对话历史，只保留摘要。

```python
from langchain.memory import ConversationSummaryMemory

# 创建记忆
memory = ConversationSummaryMemory(llm=ChatOpenAI(temperature=0))

# 保存对话
memory.save_context(
    {"input": "我叫小明，今年25岁，是一名程序员"},
    {"output": "你好小明！很高兴认识你。"}
)

memory.save_context(
    {"input": "我喜欢Python和JavaScript"},
    {"output": "很好的技术栈！"}
)

# 查看记忆（自动总结）
memory_variables = memory.load_memory_variables({})
print(memory_variables)
# 输出：{'history': '小明是一名25岁的程序员，擅长Python和JavaScript。'}
```

**优点：**
- 节省token
- 保留关键信息
- 自动摘要

**缺点：**
- 可能丢失细节
- 摘要质量依赖LLM
- 无法精确回溯

**适用场景：**
- 超长对话
- 需要关键信息
- Token成本严格限制

### 3.2.4 ConversationTokenBufferMemory

根据token数量限制，保留尽可能多的对话。

```python
from langchain.memory import ConversationTokenBufferMemory

# 创建记忆（最多1000 tokens）
memory = ConversationTokenBufferMemory(
    llm=ChatOpenAI(temperature=0),
    max_token_limit=1000
)

# 保存大量对话
for i in range(20):
    memory.save_context(
        {"input": f"问题{i}"},
        {"output": f"回答{i}"}
    )

# 查看记忆（自动裁剪到1000 tokens内）
memory_variables = memory.load_memory_variables({})
print(f"记忆长度：{len(memory_variables['history'])} tokens")
```

**优点：**
- 精确控制token使用
- 保留更多对话
- 自动裁剪

**缺点：**
- 可能不完整
- 需要LLM计算tokens

**适用场景：**
- Token预算固定
- 需要平衡历史和成本
- 长对话场景

### 3.2.5 VectorStoreMemory

使用向量数据库存储和检索记忆。

```python
from langchain.memory import VectorStoreMemory
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# 创建向量数据库
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings()
)

# 创建记忆
memory = VectorStoreMemory(
    vectorstore=vectorstore,
    memory_key="chat_history"
)

# 保存对话
memory.save_context(
    {"input": "我喜欢编程"},
    {"output": "编程很有趣！"}
)

# 检索相关记忆
result = memory.load_memory_variables({"input": "我擅长什么？"})
print(result)
```

**优点：**
- 支持大规模记忆
- 语义检索
- 持久化存储

**缺点：**
- 实现复杂
- 需要向量数据库
- 检索可能不精确

**适用场景：**
- 超长对话历史
- 需要语义检索
- 企业级应用

## 3.3 实战：具有记忆的客服机器人

让我们创建一个具有记忆功能的客服机器人。

### 完整代码

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

class CustomerServiceBot:
    def __init__(self, company_name="TechCorp"):
        self.llm = ChatOpenAI(temperature=0.7)

        # 创建记忆
        self.memory = ConversationBufferMemory(
            return_messages=True  # 返回消息对象
        )

        # 创建提示词模板
        template = """你是一家名为{company_name}公司的专业客服助手。

你的职责：
1. 友好耐心地回答用户问题
2. 记住用户的信息和偏好
3. 提供准确的产品信息
4. 处理用户投诉和建议

以下是对话历史：
{history}

当前用户问题：{input}

请根据对话历史回答用户问题。"""

        prompt = ChatPromptTemplate.from_template(template)

        # 创建对话链
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True
        )

    def chat(self, user_input):
        """与用户对话"""
        response = self.chain.predict(input=user_input)
        return response

    def get_memory(self):
        """获取对话历史"""
        return self.memory.load_memory_variables({})

    def clear_memory(self):
        """清空对话历史"""
        self.memory.clear()

# 使用示例
if __name__ == "__main__":
    bot = CustomerServiceBot()

    # 对话示例
    conversations = [
        "你好，我是新用户",
        "我想了解你们的产品",
        "我想购买Python培训课程",
        "我的名字叫小明",
        "请记住我的名字",
        "我的名字是什么？"
    ]

    for user_input in conversations:
        print(f"\n用户：{user_input}")
        response = bot.chat(user_input)
        print(f"客服：{response}")

    # 查看记忆
    print("\n=== 对话历史 ===")
    print(bot.get_memory())
```

### 运行示例

```
用户：你好，我是新用户
客服：你好！欢迎来到TechCorp。很高兴为您服务！请问有什么可以帮助您的？

用户：我想了解你们的产品
客服：当然！我们提供多种技术培训课程，包括Python、JavaScript、机器学习等。您对哪个方向感兴趣？

用户：我想购买Python培训课程
客服：很好的选择！我们的Python课程适合初学者和进阶者。课程包含实战项目，价格是¥299。您想了解更多详情吗？

用户：我的名字叫小明
客服：很高兴认识你，小明！我会记住你的名字。

用户：请记住我的名字
客服：好的，我已经记住了，小明！

用户：我的名字是什么？
客服：你的名字是小明。
```

## 3.4 记忆管理策略

### 策略1：分层记忆

**短期记忆（1-3天）：**
- 保存完整对话
- 使用ConversationBufferMemory
- 适合最近上下文

**中期记忆（1-2周）：**
- 保存关键信息摘要
- 使用ConversationSummaryMemory
- 适合回顾近期对话

**长期记忆（1个月+）：**
- 保存用户档案
- 使用VectorStoreMemory
- 适合用户画像

### 策略2：定期清理

```python
class SmartMemoryManager:
    def __init__(self):
        self.short_term = ConversationBufferMemory()
        self.long_term = ConversationSummaryMemory(llm=ChatOpenAI())

    def save_context(self, input_text, output_text):
        # 保存到短期记忆
        self.short_term.save_context(
            {"input": input_text},
            {"output": output_text}
        )

        # 如果对话超过10轮，转移到长期记忆
        history = self.short_term.load_memory_variables({})
        if history['history'].count('\n') > 20:
            self.long_term.save_context(
                {"input": input_text},
                {"output": output_text}
            )

    def get_memory(self, query=None):
        # 优先使用短期记忆
        short = self.short_term.load_memory_variables({})

        # 如果需要历史信息，查询长期记忆
        long = self.long_term.load_memory_variables({})

        return {
            "recent": short.get('history', ''),
            "summary": long.get('history', '')
        }
```

### 策略3：用户专属记忆

```python
# 为每个用户创建独立的记忆
user_memories = {}

def get_user_memory(user_id):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory()
    return user_memories[user_id]

# 使用
user_id = "user_123"
memory = get_user_memory(user_id)
memory.save_context(
    {"input": "我喜欢Python"},
    {"output": "好的，我记住了"}
)
```

## 本章小结

通过本章学习，你应该已经掌握了：
- 为什么需要记忆系统
- LangChain提供的多种Memory组件
- 不同Memory的适用场景
- 实际应用：记忆客服机器人
- 记忆管理策略

**关键要点：**
1. 记忆是AI应用的基础功能
2. 根据场景选择合适的Memory类型
3. 平衡记忆质量和Token成本
4. 实现分层记忆管理策略
5. 支持多用户独立记忆

## 练习题

1. 创建一个支持多用户的聊天机器人
2. 实现定期清理历史对话的功能
3. 对比不同Memory类型的性能差异

## 常见问题

**Q: Memory会占用多少token？**
A: 取决于Memory类型和使用方式。BufferMemory占用最多，SummaryMemory最少。

**Q: 如何选择Memory类型？**
A: 根据对话长度、token预算、是否需要完整历史来选择。

**Q: Memory可以持久化吗？**
A: 可以，使用save()和load()方法，或集成数据库。

**Q: 多个Memory可以一起使用吗？**
A: 可以，通过自定义Memory类实现组合功能。

---

下一章预告：链式调用 - 学习如何将多个组件连接起来，构建复杂的AI应用。
