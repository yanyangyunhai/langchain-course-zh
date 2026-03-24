# 第4章：链式调用

## 4.1 Chain基础概念

在LangChain中，Chain（链）是核心组件之一。它允许你将多个组件连接起来，形成一个完整的处理流程。

### 什么是Chain？

简单来说，Chain就是将多个操作步骤串联起来，前一步的输出作为后一步的输入。

**类比：**
- 传统编程：函数A的输出 → 函数B的输入 → 函数C的输入
- LangChain：Prompt → LLM → Output Chain → Next Component

### 为什么需要Chain？

1. **模块化**：将复杂任务拆解为简单步骤
2. **可复用**：定义好的Chain可以重复使用
3. **可组合**：多个Chain可以组合成更复杂的流程
4. **可维护**：修改某个环节不影响其他部分

### Chain的基本结构

```
输入
  ↓
[组件1] → 输出1
  ↓
[组件2] → 输出2
  ↓
[组件3] → 输出3
  ↓
最终输出
```

## 4.2 基础Chain类型

### 4.2.1 LLMChain

最简单的Chain类型，只包含一个LLM和一个Prompt。

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 创建LLM
llm = ChatOpenAI(temperature=0.7)

# 创建Prompt
prompt = PromptTemplate(
    input_variables=["product"],
    template="请为{product}写一段简短的产品介绍。"
)

# 创建Chain
chain = LLMChain(llm=llm, prompt=prompt)

# 运行Chain
result = chain.run("iPhone 15")
print(result)
```

**输出示例：**
```
iPhone 15是苹果公司最新发布的旗舰智能手机，搭载强大的A17仿生芯片，
配备先进的摄像系统和超视网膜XDR显示屏，提供卓越的性能和用户体验。
```

### 4.2.2 SimpleChain

最基础的Chain，只有一个输入和一个输出。

```python
from langchain.chains import LLMChain, SimpleSequentialChain

# Chain 1：生成标题
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="为'{topic}'这个主题生成一个吸引人的标题。"
)

chain1 = LLMChain(llm=llm, prompt=prompt1)

# Chain 2：根据标题生成内容
prompt2 = PromptTemplate(
    input_variables=["title"],
    template="根据标题'{title}'写一篇简短的文章。"
)

chain2 = LLMChain(llm=llm, prompt=prompt2)

# 串联两个Chain
overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

# 运行
result = overall_chain.run("人工智能")
```

### 4.2.3 SequentialChain

按顺序执行多个Chain，每个Chain的输出传递给下一个Chain。

```python
from langchain.chains import SequentialChain

# Chain 1：生成大纲
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="为'{topic}'生成文章大纲，包含3个主要部分。"
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="outline")

# Chain 2：扩展大纲
prompt2 = PromptTemplate(
    input_variables=["outline"],
    template="扩展这个大纲：{outline}，为每个部分添加2-3个要点。"
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="detailed_outline")

# Chain 3：生成完整文章
prompt3 = PromptTemplate(
    input_variables=["detailed_outline"],
    template="根据详细大纲写一篇完整的文章：{detailed_outline}"
)
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="article")

# 串联所有Chain
overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["topic"],
    output_variables=["article"],
    verbose=True
)

# 运行
result = overall_chain("机器学习")
print(result["article"])
```

## 4.3 高级Chain用法

### 4.3.1 RouterChain

根据输入内容选择不同的处理路径。

```python
from langchain.chains.router import MultiPromptChain
from langchain.prompts import PromptTemplate

# 定义多个Prompt
code_prompt = PromptTemplate(
    template="你是一个编程助手。回答这个问题：{input}"
)

general_prompt = PromptTemplate(
    template="你是一个通用助手。回答这个问题：{input}"
)

# 定义Router
router_template = """根据用户的问题，选择最合适的处理方式：
- 如果问题涉及编程，选择 'code'
- 否则，选择 'general'

用户问题：{input}

只返回 'code' 或 'general'。"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"]
)

# 创建RouterChain
chain = MultiPromptChain(
    prompt_infos=[
        {"name": "code", "prompt": code_prompt},
        {"name": "general", "prompt": general_prompt}
    ],
    llm=llm,
    verbose=True
)

# 运行
result1 = chain.run("如何用Python定义函数？")
result2 = chain.run("今天天气怎么样？")
```

### 4.3.2 TransformChain

在Chain中插入数据处理步骤。

```python
from langchain.chains import TransformChain

# 定义转换函数
def uppercase(inputs):
    return {"text": inputs["text"].upper()}

# 创建TransformChain
transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["text"],
    transform=uppercase
)

# 创建LLM Chain
prompt = PromptTemplate(
    input_variables=["text"],
    template="处理这段文本：{text}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# 组合Chain
from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[transform_chain, llm_chain],
    input_variables=["text"],
    output_variables=["text"],
    verbose=True
)

# 运行
result = overall_chain("hello world")
```

### 4.3.3 RefineChain

逐步完善答案，多次迭代生成更好的结果。

```python
from langchain.chains import RefineChain

# 初始Prompt
initial_prompt = PromptTemplate(
    input_variables=["question"],
    template="回答这个问题：{question}"
)

# 优化Prompt
refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer"],
    template="""这是初始答案：{existing_answer}

请根据以下问题优化答案：{question}

优化要求：
1. 添加更多细节
2. 提供更好的解释
3. 保持准确性"""
)

# 创建RefineChain
refine_chain = RefineChain(
    initial_prompt=initial_prompt,
    refine_prompt=refine_prompt,
    llm=llm,
    verbose=True
)

# 运行
result = refine_chain.run("什么是机器学习？")
```

## 4.4 实战：文档分析工具

让我们创建一个完整的文档分析工具，展示Chain的强大功能。

### 完整代码

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import LLMChain

class DocumentAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)

    def analyze(self, document):
        """分析文档"""
        # Chain 1：摘要生成
        prompt1 = PromptTemplate(
            input_variables=["document"],
            template="为以下文档生成简短摘要（不超过100字）：\n\n{document}"
        )
        chain1 = LLMChain(
            llm=self.llm,
            prompt=prompt1,
            output_key="summary"
        )

        # Chain 2：关键词提取
        prompt2 = PromptTemplate(
            input_variables=["document"],
            template="从文档中提取5个关键词，用逗号分隔：\n\n{document}"
        )
        chain2 = LLMChain(
            llm=self.llm,
            prompt=prompt2,
            output_key="keywords"
        )

        # Chain 3：生成分析报告
        prompt3 = PromptTemplate(
            input_variables=["summary", "keywords", "document"],
            template="""基于以下信息生成文档分析报告：

摘要：{summary}

关键词：{keywords}

原文档：{document}

报告格式：
1. 文档主题
2. 关键内容
3. 建议行动"""
        )
        chain3 = LLMChain(
            llm=self.llm,
            prompt=prompt3,
            output_key="report"
        )

        # 串联所有Chain
        overall_chain = SequentialChain(
            chains=[chain1, chain2, chain3],
            input_variables=["document"],
            output_variables=["report"],
            verbose=True
        )

        # 运行
        result = overall_chain.run(document)
        return result["report"]

# 使用示例
if __name__ == "__main__":
    analyzer = DocumentAnalyzer()

    document = """
    人工智能（AI）是计算机科学的一个重要分支，
    它致力于创建能够执行通常需要人类智能的任务的系统。
    机器学习是AI的核心技术之一，通过数据训练模型，
    使计算机能够从经验中学习并改进性能。
    深度学习是机器学习的一个子领域，使用神经网络
    来模拟人脑的工作方式。
    """

    report = analyzer.analyze(document)
    print(report)
```

### 运行示例

```
> Entering new SequentialChain chain...

> Finished chain.

文档分析报告：

1. 文档主题
本文档主要介绍人工智能、机器学习和深度学习的基本概念及其关系。

2. 关键内容
文档阐述了AI作为计算机科学分支的定位，解释了机器学习通过数据训练实现计算机学习能力的机制，并说明了深度学习利用神经网络模拟人脑的技术原理。

3. 建议行动
建议读者进一步学习具体的应用案例，了解这些技术在实际项目中的使用方法，并尝试实践相关算法和模型。
```

## 4.5 Chain最佳实践

### 1. 命名清晰
```python
# 好的命名
summarization_chain = LLMChain(...)
title_generation_chain = LLMChain(...)

# 避免模糊命名
chain1 = LLMChain(...)
chain2 = LLMChain(...)
```

### 2. 使用verbose调试
```python
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True  # 在开发时启用
)
```

### 3. 定义输入输出变量
```python
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_key="generated_text"  # 明确输出变量名
)
```

### 4. 处理错误
```python
try:
    result = chain.run(inputs)
except Exception as e:
    print(f"Chain执行失败: {e}")
    result = None
```

### 5. 性能优化
```python
# 使用批量处理
results = chain.batch([input1, input2, input3])

# 使用缓存
from langchain.cache import InMemoryCache
llm.cache = InMemoryCache()
```

## 本章小结

通过本章学习，你应该已经掌握了：
- Chain的基本概念和作用
- 不同类型的Chain及其用途
- 高级Chain的构建方法
- 实际应用：文档分析工具
- Chain最佳实践

**关键要点：**
1. Chain是模块化和可复用的核心
2. SequentialChain适合顺序处理
3. RouterChain实现条件分支
4. TransformChain插入数据处理
5. 清晰的命名和调试很重要

## 练习题

1. 创建一个Chain，实现从主题到文章的完整流程
2. 使用RouterChain实现多语言支持
3. 优化文档分析工具，添加更多分析维度

## 常见问题

**Q: Chain和Agent有什么区别？**
A: Chain是预定义的处理流程，Agent是自主决策的智能体。Chain更可控，Agent更灵活。

**Q: Chain可以嵌套吗？**
A: 可以，一个Chain的输出可以作为另一个Chain的输入。

**Q: 如何调试Chain？**
A: 启用verbose=True，查看每一步的输入输出。

**Q: Chain的性能如何优化？**
A: 使用缓存、批量处理、选择合适的模型。

---

下一章预告：RAG（检索增强生成）- 学习如何让AI访问和利用外部知识。
