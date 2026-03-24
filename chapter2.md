# 第2章：提示词工程

## 2.1 提示词设计原则

提示词工程是AI应用开发的核心技能。一个好的提示词可以让模型产生高质量的输出，而一个糟糕的提示词可能导致模型产生无意义或不准确的回答。

### 核心原则

#### 原则1：清晰明确的指令

**错误示例：**
```
写一篇文章
```

**正确示例：**
```
写一篇关于人工智能的文章，内容包括：
1. 人工智能的定义
2. 人工智能的发展历程
3. 人工智能的应用场景
4. 人工智能的未来展望

文章要求：
- 字数800-1000字
- 语言通俗易懂
- 包含具体案例
- 结构清晰，有标题和段落
```

**为什么？** 明确的指令帮助模型准确理解你的需求。

#### 原则2：提供上下文信息

**错误示例：**
```
这个错误怎么解决？
```

**正确示例：**
```
我在使用Python开发Web应用时遇到了以下错误：

错误信息：ConnectionError: HTTPConnectionPool(host='api.example.com', port=80): Max retries exceeded

代码片段：
```python
import requests
response = requests.get('https://api.example.com/data')
print(response.json())
```

我尝试过的方法：
1. 检查网络连接
2. 尝试增加超时时间
3. 检查API地址是否正确

问题依然存在，请帮我分析可能的原因和解决方案。
```

**为什么？** 上下文信息让模型能够基于实际情况给出准确的建议。

#### 原则3：使用示例驱动学习（Few-shot Learning）

**错误示例：**
```
将下面的句子转换成简洁版：
人工智能正在改变我们的生活方式
```

**正确示例：**
```
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

原文：自然语言处理让计算机能够理解人类语言
简洁版：
```

**为什么？** 示例帮助模型理解转换的规律和风格。

#### 原则4：迭代优化

提示词不是一次就能完美的，需要不断测试和优化：

**第1版：**
```
写一个Python函数计算斐波那契数列
```

**第2版：**
```
写一个Python函数计算斐波那契数列，要求：
1. 使用递归方式实现
2. 包含注释
3. 有输入验证
4. 返回第n个斐波那契数
```

**第3版：**
```
写一个Python函数计算斐波那契数列，要求：
1. 使用递归方式实现
2. 包含详细注释解释算法逻辑
3. 有输入验证，确保n为正整数
4. 返回第n个斐波那契数
5. 包含时间复杂度分析
6. 提供测试用例

函数签名：
def fibonacci(n: int) -> int:
    ...
```

**为什么？** 迭代优化可以让提示词越来越精确，输出质量越来越高。

### 提示词结构建议

一个优秀的提示词应该包含以下几个部分：

```
1. 角色定义（可选）
   "你是一个经验丰富的Python开发者"

2. 任务描述
   "编写一个Python函数..."

3. 详细要求
   "1. 使用递归实现
    2. 包含注释
    3. ..."

4. 输出格式
   "输出格式：函数代码 + 测试用例"

5. 限制条件
   "限制：不使用第三方库"
```

## 2.2 PromptTemplate详解

LangChain的PromptTemplate是管理提示词的核心组件。

### 基础PromptTemplate

```python
from langchain.prompts import PromptTemplate

# 创建简单模板
template = """
请回答以下问题：{question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)

# 使用模板
formatted = prompt.format(question="什么是Python？")
print(formatted)
```

### ChatPromptTemplate

对话应用通常使用ChatPromptTemplate：

```python
from langchain.prompts import ChatPromptTemplate

# 创建对话模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}专家"),
    ("user", "{input}")
])

# 填充变量
messages = prompt.format_messages(
    role="Python编程",
    input="如何定义一个函数？"
)

print(messages)
```

### 部分变量填充

```python
from langchain.prompts import PromptTemplate

template = """
作为{role}专家，请{action}。
主题：{topic}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["action", "topic"],
    partial_variables={"role": "AI技术"}  # 预先填充
)

# 只需填充剩余变量
formatted = prompt.format(
    action="编写教程",
    topic="LangChain基础"
)

print(formatted)
```

### 复杂模板构建

```python
from langchain.prompts import ChatPromptTemplate

# 构建多轮对话模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}顾问"),
    ("human", "我的需求是：{user_input}"),
    ("ai", "我理解你的需求是{user_input}，请问你的预算是多少？"),
    ("human", "我的预算是{budget}"),
    ("ai", "了解了，我会根据你的预算提供{role}方面的建议")
])

messages = prompt.format_messages(
    role="网站开发",
    user_input="我想建一个电商网站",
    budget="5万元"
)
```

## 2.3 实战：智能邮件助手

让我们创建一个智能邮件助手，展示提示词工程的实际应用。

### 完整代码

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

class EmailAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")

        # 系统提示词
        self.system_prompt = """你是一个专业的邮件助手，擅长处理各种邮件任务：
1. 邮件分类：将邮件分类为工作、个人、促销、垃圾邮件等
2. 自动回复：根据邮件内容生成适当的回复
3. 邮件摘要：提取邮件的关键信息生成摘要
4. 邮件优化：改进邮件的语气和结构

回复要求：
- 保持专业礼貌的语气
- 信息准确完整
- 结构清晰有条理
- 适当的长度和详略"""

    def classify_email(self, email_content):
        """分类邮件"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""请将以下邮件分类为以下类别之一：
- 工作：与职业相关的邮件
- 个人：与私人生活相关的邮件
- 促销：广告、优惠等商业邮件
- 垃圾邮件：无关或恶意邮件
- 其他：不属于以上类别的邮件

邮件内容：
{email_content}

只返回类别名称，不要解释。""")
        ])

        response = self.llm.invoke(prompt.format_messages(
            email_content=email_content
        ))

        return response.content.strip()

    def generate_reply(self, email_content, reply_type="polite"):
        """生成自动回复"""
        if reply_type == "polite":
            tone = "礼貌、专业、感谢"
        elif reply_type == "direct":
            tone = "直接、简洁、高效"
        else:
            tone = "友好、亲切、热情"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""请为以下邮件生成自动回复：
{email_content}

回复要求：
- 语气：{tone}
- 包含对原邮件的适当回应
- 如果有具体问题需要回答，请回答
- 结尾表达感谢或期待回复
- 长度控制在3-5句话""")
        ])

        response = self.llm.invoke(prompt.format_messages(
            email_content=email_content
        ))

        return response.content.strip()

    def summarize_email(self, email_content):
        """生成邮件摘要"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""请为以下邮件生成摘要（不超过100字）：
{email_content}

摘要要求：
- 提取关键信息
- 突出重要事项
- 简洁明了""")
        ])

        response = self.llm.invoke(prompt.format_messages(
            email_content=email_content
        ))

        return response.content.strip()

    def improve_email(self, email_content, improvement_type="tone"):
        """优化邮件"""
        if improvement_type == "tone":
            instruction = "改善邮件的语气，使其更加专业礼貌"
        elif improvement_type == "structure":
            instruction = "优化邮件的结构，使其更加清晰有条理"
        elif improvement_type == "clarity":
            instruction = "提高邮件的表达清晰度，去除冗余信息"
        else:
            instruction = "全面提升邮件质量"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""请{instruction}：
{email_content}

优化要求：
- 保持原意不变
- 提升专业度
- 增强表达效果""")
        ])

        response = self.llm.invoke(prompt.format_messages(
            email_content=email_content
        ))

        return response.content.strip()

# 使用示例
if __name__ == "__main__":
    assistant = EmailAssistant()

    # 示例邮件
    email = """
    From: boss@company.com
    Subject: 关于项目进度的讨论
    To: team@company.com

    各位同事，

    下周我们需要完成项目的第一阶段。请大家在本周五之前提交各自的任务进展报告。

    报告内容应包括：
    1. 已完成的工作
    2. 遇到的问题
    3. 下周计划
    4. 需要的支持

    谢谢大家的配合！

    王经理
    """

    print("=== 邮件分类 ===")
    category = assistant.classify_email(email)
    print(f"分类结果：{category}\n")

    print("=== 邮件摘要 ===")
    summary = assistant.summarize_email(email)
    print(f"摘要：{summary}\n")

    print("=== 自动回复 ===")
    reply = assistant.generate_reply(email, "polite")
    print(f"回复：{reply}\n")
```

### 运行示例

```
=== 邮件分类 ===
分类结果：工作

=== 邮件摘要 ===
王经理要求团队周五前提交第一阶段项目进展报告，包括已完成工作、问题、下周计划和所需支持。

=== 自动回复 ===
收到，我会按时提交项目进展报告。感谢您的安排。
```

## 2.4 提示词调试技巧

### 技巧1：添加推理步骤

**原始提示词：**
```
计算：123 × 456 = ?
```

**优化后：**
```
请按照以下步骤计算：123 × 456

步骤1：将456分解为400 + 50 + 6
步骤2：分别计算123 × 400、123 × 50、123 × 6
步骤3：将结果相加
步骤4：给出最终答案

请显示详细计算过程。
```

### 技巧2：使用思维链（Chain of Thought）

**原始提示词：**
```
判断：小明有5个苹果，吃了2个，又买了3个，现在有几个？
```

**优化后：**
```
请逐步思考并回答：

问题：小明有5个苹果，吃了2个，又买了3个，现在有几个？

思考过程：
1. 小明一开始有5个苹果
2. 吃了2个后，还剩5 - 2 = 3个
3. 又买了3个，现在有3 + 3 = 6个

答案：6个
```

### 技巧3：提供角色和风格

**原始提示词：**
```
写一封道歉信
```

**优化后：**
```
你是一位专业的商务沟通顾问。请写一封真诚、专业的道歉信，语气诚恳，表达歉意，并提出补救措施。

背景信息：
- 公司产品出现质量问题
- 影响了客户的使用体验
- 需要向客户道歉

要求：
- 开头直接表达歉意
- 中间说明情况和原因
- 结尾提出解决方案和承诺
- 全程保持专业和真诚
```

## 本章小结

通过本章学习，你应该已经掌握了：
- 提示词设计的核心原则
- PromptTemplate的使用方法
- 实际应用场景的实现
- 调试和优化技巧

**关键要点：**
1. 清晰明确的指令是高质量输出的基础
2. 上下文和示例能显著提升效果
3. 迭代优化是必要的步骤
4. 实战是掌握提示词工程的最好方式

## 练习题

1. 优化以下提示词：
   - "写一篇博客"
   - "分析这个代码"
   - "推荐一些电影"

2. 为你的项目设计3个不同用途的PromptTemplate

3. 实现一个智能聊天机器人，能识别用户意图并给出相应回答

## 常见问题

**Q: 提示词多长合适？**
A: 没有固定长度，关键是要包含足够的信息。一般来说，100-500字是合理的范围。

**Q: 如何判断提示词是否有效？**
A: 通过输出质量来判断。如果输出符合预期，说明提示词有效。如果不符合，需要调整。

**Q: 一次尝试多少个提示词版本比较合适？**
A: 建议2-3个版本，对比输出效果，选择最好的。

**Q: 提示词会被模型记住吗？**
A: 不会，每次调用都是独立的。但你可以使用对话记忆功能来保持上下文。

---

下一章预告：记忆系统 - 学习如何让AI记住对话历史，实现多轮对话。
