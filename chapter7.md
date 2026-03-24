# 第7章：高级主题

## 7.1 LangSmith - 开发调试平台

LangSmith是LangChain官方提供的开发、测试和监控平台。

### 7.1.1 为什么需要LangSmith？

在开发复杂的LLM应用时，你会遇到：
1. **调试困难** - 不知道哪里出错了
2. **性能瓶颈** - 不知道哪个环节慢
3. **版本管理** - 难以追踪不同版本的表现
4. **用户反馈** - 难以重现用户遇到的问题

LangSmith解决了这些问题。

### 7.1.2 集成LangSmith

```python
from langchain_openai import ChatOpenAI
from langchain.smith import LangSmith

# 配置环境变量
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# 创建LLM
llm = ChatOpenAI(temperature=0.7)

# 使用时自动追踪
response = llm.invoke("你好")
```

### 7.1.3 查看运行记录

```python
from langchain.smith import Client

client = Client()

# 获取最近的运行记录
runs = client.list_runs(
    project_name="my-project",
    limit=10
)

for run in runs:
    print(f"Run ID: {run.id}")
    print(f"Status: {run.status}")
    print(f"Latency: {run.end_time - run.start_time}ms")
```

## 7.2 LangServe - 部署服务

LangServe允许你将LangChain应用部署为REST API服务。

### 7.2.1 基础部署

```python
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.schema import StrOutputParser
from fastapi import FastAPI
from pydantic import BaseModel

# 创建Chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = llm | StrOutputParser()

# 定义API
app = FastAPI()

class Input(BaseModel):
    text: str

class Output(BaseModel):
    result: str

@app.post("/chain")
async def run_chain(input: Input) -> Output:
    result = chain.invoke(input.text)
    return Output(result=result)

# 运行服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 7.2.2 使用LangServe CLI

```bash
# 创建服务文件
echo 'from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from fastapi import FastAPI
from pydantic import BaseModel

llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = llm | StrOutputParser()

app = FastAPI()

class Input(BaseModel):
    text: str

class Output(BaseModel):
    result: str

@app.post("/chain")
async def run_chain(input: Input) -> Output:
    return Output(result=chain.invoke(input.text))' > app.py

# 启动服务
langchain serve app.py
```

### 7.2.3 客户端调用

```python
import requests

# 调用API
response = requests.post(
    "http://localhost:8000/chain/invoke",
    json={"text": "你好"}
)

print(response.json())
```

## 7.3 性能优化

### 7.3.1 缓存机制

```python
from langchain.cache import InMemoryCache

# 启用缓存
llm = ChatOpenAI()
llm.cache = InMemoryCache()

# 缓存结果
result1 = llm.invoke("你好")
result2 = llm.invoke("你好")  # 从缓存获取
```

### 7.3.2 批处理

```python
# 批量处理提升效率
inputs = ["问题1", "问题2", "问题3"]
results = llm.batch(inputs)
```

### 7.3.3 异步调用

```python
import asyncio
from langchain_openai import AsyncChatOpenAI

llm = AsyncChatOpenAI()

async def async_query():
    tasks = [llm.ainvoke(f"问题{i}") for i in range(10)]
    results = await asyncio.gather(*tasks)
    return results

# 运行
results = asyncio.run(async_query())
```

### 7.3.4 Token优化

```python
# 控制Token使用
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    max_tokens=500,  # 限制输出长度
    temperature=0.7
)

# 使用流式输出
for chunk in llm.stream("介绍一下自己"):
    print(chunk.content, end="", flush=True)
```

## 7.4 生产环境配置

### 7.4.1 错误处理

```python
from langchain.schema import BaseOutputParser
from langchain.schema.exceptions import OutputParserException

class OutputParser(BaseOutputParser):
    def parse(self, text: str):
        try:
            # 尝试解析
            return self._parse(text)
        except Exception as e:
            raise OutputParserException(
                f"Failed to parse: {text}",
                llm_output=text,
            )

# 使用
try:
    result = chain.invoke(input_data)
except OutputParserException as e:
    # 处理解析错误
    result = f"解析错误：{e}"
```

### 7.4.2 日志记录

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 在Chain中添加日志
class LoggingChain:
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, inputs):
        logging.info(f"Chain输入: {inputs}")
        result = self.llm.invoke(inputs)
        logging.info(f"Chain输出: {result}")
        return result
```

### 7.4.3 监控指标

```python
import time
from datetime import datetime

class MonitoredChain:
    def __init__(self, chain):
        self.chain = chain
        self.call_count = 0
        self.total_time = 0
    
    def __call__(self, inputs):
        self.call_count += 1
        start_time = time.time()
        
        try:
            result = self.chain(inputs)
            duration = time.time() - start_time
            self.total_time += duration
            
            logging.info(f"调用#{self.call_count} - 耗时: {duration:.2f}s")
            return result
        except Exception as e:
            logging.error(f"调用#{self.call_count} - 错误: {e}")
            raise

# 使用
monitored_chain = MonitoredChain(chain)
result = monitored_chain.invoke(input_data)
print(f"平均耗时: {monitored_chain.total_time / monitored_chain.call_count:.2f}s")
```

### 7.4.4 负载均衡

```python
from langchain_openai import ChatOpenAI
import random

# 配置多个API Key
api_keys = [
    "sk-xxx1",
    "sk-xxx2",
    "sk-xxx3"
]

# 负载均衡
class LoadBalancedLLM:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0
        self.models = [
            ChatOpenAI(api_key=key) for key in api_keys
        ]
    
    def invoke(self, prompt):
        # 轮询选择
        model = self.models[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.models)
        return model.invoke(prompt)

# 使用
llm = LoadBalancedLLM(api_keys)
result = llm.invoke("测试负载均衡")
```

## 7.5 安全考虑

### 7.5.1 输入验证

```python
from pydantic import BaseModel, validator

class SafeInput(BaseModel):
    text: str
    
    @validator('text')
    def validate_text(cls, v):
        # 限制长度
        if len(v) > 1000:
            raise ValueError("输入过长")
        # 过滤敏感词
        sensitive_words = ["密码", "secret"]
        for word in sensitive_words:
            if word in v.lower():
                raise ValueError(f"包含敏感词: {word}")
        return v

# 使用
safe_input = SafeInput(text="正常文本")
result = chain.invoke(safe_input.text)
```

### 7.5.2 输出过滤

```python
def filter_output(text):
    """过滤敏感信息"""
    # 移除潜在的敏感模式
    import re
    patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
        r'\b\d{3}-\d{2}-\d{4}\b',  # 社保号
        r'\b\d{11}\b',  # 身份证号
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '[已隐藏]', text)
    
    return text

# 使用
result = chain.invoke(input_text)
safe_result = filter_output(result)
```

### 7.5.3 速率限制

```python
from functools import wraps
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(list)
    
    def is_allowed(self, key):
        now = time.time()
        # 清理过期记录
        self.calls[key] = [t for t in self.calls[key] if now - t < self.time_window]
        
        # 检查是否超限
        if len(self.calls[key]) >= self.max_calls:
            return False
        
        self.calls[key].append(now)
        return True

# 使用
rate_limiter = RateLimiter(max_calls=10, time_window=60)

def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        user_id = kwargs.get('user_id', 'default')
        if not rate_limiter.is_allowed(user_id):
            raise Exception("请求过于频繁")
        return func(*args, **kwargs)
    
    return wrapper

@rate_limit
def chain_invoke(prompt, user_id='default'):
    return chain.invoke(prompt)
```

## 本章小结

通过本章学习，你应该已经掌握了：
- LangSmith的开发和调试功能
- LangServe的部署和服务化
- 性能优化技巧
- 生产环境配置
- 安全考虑和最佳实践

**关键要点：**
1. LangSmith大大简化了开发调试
2. LangServe让应用易于部署和扩展
3. 缓存、批处理、异步调用能显著提升性能
4. 生产环境需要完善的错误处理和日志
5. 安全是不可忽视的重要方面

## 练习题

1. 将你的Chain部署为LangServe服务
2. 实现一个带有缓存的Chain
3. 为Chain添加输入验证和输出过滤

## 常见问题

**Q: LangSmith免费吗？**
A: LangSmith有免费额度，超出后按使用量计费。

**Q: LangServe支持哪些部署方式？**
A: Docker、Kubernetes、云平台等。

**Q: 如何监控生产环境？**
A: 使用LangSmith监控、日志系统、APM工具等。

**Q: 性能瓶颈在哪里？**
A: 通常是LLM API调用和向量检索，需要针对性优化。

---

下一章预告：项目实战 - 综合运用所学知识完成完整的项目。
