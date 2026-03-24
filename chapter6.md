# 第6章：Agent开发

## 6.1 什么是Agent？

Agent（智能体）是LangChain中最强大的组件之一。它不是简单地按照预定义的步骤执行任务，而是能够自主决策、规划步骤、调用工具的AI智能体。

### Agent vs Chain

**Chain的特点：**
- 预定义的处理流程
- 按顺序执行
- 固定的输入输出
- 可控性强

**Agent的特点：**
- 自主决策能力
- 动态规划步骤
- 灵活调用工具
- 具有推理能力

### 核心能力

1. **决策能力** - 根据任务选择合适的行动
2. **规划能力** - 将复杂任务分解为步骤
3. **工具调用** - 使用外部工具完成任务
4. **推理能力** - 思考和优化策略
5. **学习能力** - 从错误中学习

### Agent的工作流程

```
用户输入
    ↓
[思考] - Agent分析任务
    ↓
[规划] - 制定行动计划
    ↓
[行动] - 调用工具执行
    ↓
[观察] - 查看执行结果
    ↓
[判断] - 是否需要继续？
    ↓ 是 → [行动] → [观察] → [判断]
    ↓ 否
    ↓
返回最终结果
```

## 6.2 Tools（工具）

Tools是Agent的"手脚"，让Agent能够执行具体的操作。

### 6.2.1 内置Tools

LangChain提供了丰富的内置工具：

```python
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from langchain.utilities.python import PythonREPL

# 搜索工具
search = SerpAPIWrapper()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="用于搜索互联网信息"
)

# Python REPL工具
python_repl = PythonREPL()
python_tool = Tool(
    name="Python_REPL",
    func=python_repl.run,
    description="用于执行Python代码"
)
```

### 6.2.2 自定义Tools

```python
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel

# 定义输入Schema
class CalculatorInput(BaseModel):
    expression: str = Field(description="要计算的数学表达式")

# 定义Tool
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "用于执行数学计算"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str):
        """执行计算"""
        try:
            result = eval(expression)
            return f"计算结果：{expression} = {result}"
        except Exception as e:
            return f"计算错误：{e}"

# 使用工具
calculator = CalculatorTool()
result = calculator._run("2 + 3 * 4")
print(result)  # 输出：计算结果：2 + 3 * 4 = 14
```

### 6.2.3 常用工具类型

#### 1. 搜索工具
```python
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
result = search.run("什么是LangChain？")
```

#### 2. 计算工具
```python
from langchain.chains import LLMMathChain
from langchain_openai import OpenAI

llm_math = LLMMathChain.from_llm(OpenAI(temperature=0))
result = llm_math.run("2 + 2 = ?")
```

#### 3. 文件操作工具
```python
from langchain.tools import BaseTool
import os

class FileReaderTool(BaseTool):
    name = "file_reader"
    description = "读取文件内容"
    
    def _run(self, file_path: str):
        with open(file_path, 'r') as f:
            return f.read()
```

## 6.3 Agent类型

### 6.3.1 Zero-shot Agent

最简单的Agent类型，一次性完成任务。

```python
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI

# 创建工具
tools = [
    Tool(
        name="calculator",
        func=lambda x: str(eval(x)),
        description="用于数学计算"
    )
]

# 创建LLM
llm = ChatOpenAI(temperature=0)

# 初始化Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 运行Agent
result = agent.run("计算 123 * 456")
print(result)
```

### 6.3.2 ReAct Agent

使用"推理-行动"循环，逐步完成任务。

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# 定义ReAct Prompt
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input", "agent_scratchpad", "tools"],
)

# 创建Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 创建执行器
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# 运行
result = agent_executor.invoke({"input": "计算 123 * 456"})
```

### 6.3.3 Conversational Agent

支持多轮对话的Agent。

```python
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 多轮对话
result1 = agent.run("我的名字是小明")
result2 = agent.run("我的名字是什么？")
```

## 6.4 实战：全能助手Agent

让我们创建一个功能强大的全能助手。

### 完整代码

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import PythonREPL
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain

class UniversalAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 创建工具集
        self.tools = self.create_tools()
        
        # 初始化Agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def create_tools(self):
        """创建工具集"""
        tools = []
        
        # 1. 搜索工具
        search = DuckDuckGoSearchRun()
        tools.append(Tool(
            name="Search",
            func=search.run,
            description="搜索互联网信息，用于查找最新资料、新闻、技术文档"
        ))
        
        # 2. 数学计算工具
        math_chain = LLMMathChain.from_llm(self.llm)
        tools.append(Tool(
            name="Calculator",
            func=math_chain.run,
            description="执行数学计算，支持加减乘除、括号等运算"
        ))
        
        # 3. Python代码执行工具
        python_repl = PythonREPL()
        tools.append(Tool(
            name="Python_REPL",
            func=python_repl.run,
            description="执行Python代码，用于数据分析、文本处理等"
        ))
        
        return tools

    def chat(self, user_input):
        """对话"""
        try:
            result = self.agent.run(user_input)
            return result
        except Exception as e:
            return f"抱歉，处理请求时出错：{e}"

# 使用示例
if __name__ == "__main__":
    assistant = UniversalAssistant()

    # 对话示例
    conversations = [
        "搜索什么是LangChain？",
        "计算 123 * 456",
        "用Python生成一个随机数",
        "我记得刚才聊了什么？"
    ]

    for msg in conversations:
        print(f"\n用户：{msg}")
        response = assistant.chat(msg)
        print(f"助手：{response}")
```

### 运行示例

```
用户：搜索什么是LangChain？
助手：
> Entering new AgentExecutor chain...
Action: Search
Action Input: 什么是LangChain
Observation: LangChain是一个开源框架，用于开发由语言模型驱动的应用程序...
Thought: 我现在知道了LangChain是什么
Final Answer: LangChain是一个开源框架，用于开发由语言模型驱动的应用程序，提供了模块化组件来构建复杂的AI应用。

用户：计算 123 * 456
助手：
> Entering new AgentExecutor chain...
Action: Calculator
Action Input: 123 * 456
Observation: 56088
Thought: 计算完成
Final Answer: 123 * 456 = 56088

用户：用Python生成一个随机数
助手：
> Entering new AgentExecutor chain...
Action: Python_REPL
Action Input: import random; random.randint(1, 100)
Observation: 42
Thought: Python执行完成
Final Answer: 生成的随机数是 42

用户：我记得刚才聊了什么？
助手：
> Entering new AgentExecutor chain...
Thought: 我需要查看对话历史
Final Answer: 我们刚才讨论了：1) 什么是LangChain，2) 计算了123 * 456 = 56088，3) 用Python生成了随机数42。
```

## 6.5 Agent最佳实践

### 1. 工具定义清晰
```python
# 好的描述
Tool(
    name="calculator",
    description="执行数学计算，支持加减乘除运算"
)

# 避免模糊描述
Tool(
    name="tool1",  # ❌ 太模糊
    description="做某事"  # ❌ 不明确
)
```

### 2. 错误处理
```python
try:
    result = tool.run(input_data)
except Exception as e:
    return f"工具执行失败：{e}"
```

### 3. 工具选择优化
```python
# 限制工具数量，提高决策速度
essential_tools = [
    search_tool,
    calculator_tool
]

# 可选工具根据需要添加
optional_tools = [
    python_tool,
    file_tool
]
```

### 4. 记忆管理
```python
# 对于长期对话，使用总结记忆
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=1000
)
```

## 6.6 高级Agent技巧

### 6.6.1 动态工具选择

```python
from langchain.agents import Tool, get_all_tool_names

def dynamic_agent(input_text):
    # 根据输入动态选择工具
    if "计算" in input_text or "数学" in input_text:
        return [calculator_tool]
    elif "搜索" in input_text:
        return [search_tool]
    else:
        return all_tools

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
```

### 6.6.2 工具组合

```python
class CombinedTool(BaseTool):
    """组合多个工具"""
    name = "combined_search"
    description = "组合使用多个搜索工具"
    
    def _run(self, query: str):
        # 使用多个搜索引擎
        results = []
        for search_engine in [duckduckgo, google, bing]:
            try:
                result = search_engine.run(query)
                results.append(result)
            except:
                continue
        return "\n".join(results)
```

### 6.6.3 自定义推理策略

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# 自定义推理Prompt
custom_template = """你是一个专业的研究助理。

{tools}

工作流程：
1. 理解用户的问题
2. 分析需要什么信息
3. 选择合适的工具
4. 执行搜索和计算
5. 综合所有信息
6. 提供详细答案

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=custom_template,
    input_variables=["input", "agent_scratchpad", "tools"],
)
```

## 本章小结

通过本章学习，你应该已经掌握了：
- Agent的核心概念和工作原理
- Tools的定义和使用
- 不同类型的Agent及其特点
- 实际应用：全能助手Agent
- Agent最佳实践和高级技巧

**关键要点：**
1. Agent具有自主决策能力
2. Tools是Agent的能力扩展
3. 不同Agent类型适用于不同场景
4. 合理的工具选择和组合至关重要
5. 错误处理和记忆管理不能忽视

## 练习题

1. 创建一个能够回答编程问题的Agent
2. 实现一个具有文件操作能力的Agent
3. 优化Agent的工具选择策略

## 常见问题

**Q: Agent和Chain可以结合使用吗？**
A: 可以，Chain可以作为工具集成到Agent中。

**Q: Agent的工具数量有限制吗？**
A: 理论上没有，但工具越多决策越慢，建议控制在5-10个。

**Q: 如何调试Agent？**
A: 启用verbose=True，观察每一步的思考和行动。

**Q: Agent的推理过程可以自定义吗？**
A: 可以，通过修改Prompt模板自定义推理策略。

---

下一章预告：高级主题 - 学习LangSmith调试、LangServe部署等高级功能。
