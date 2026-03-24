# 第8章：项目实战

## 8.1 项目概述

在本章中，我们将综合运用前面学到的所有知识，完成一个完整的LangChain项目。

### 项目目标

创建一个**智能文档问答系统**，具备以下功能：
- ✅ 文档上传和解析
- ✅ 向量检索
- ✅ 智能问答
- ✅ 对话历史
- ✅ 多文档支持
- ✅ API部署

### 技术栈

- **框架：** LangChain
- **LLM：** OpenAI GPT-3.5-turbo
- **向量数据库：** ChromaDB
- **API框架：** FastAPI
- **部署：** LangServe

## 8.2 项目1：企业知识库

### 8.2.1 需求分析

**用户需求：**
- 员工可以上传企业文档
- 通过自然语言提问
- 系统从文档中找到答案
- 支持多轮对话

**技术方案：**
- 使用RAG技术
- 向量数据库存储文档
- Conversational Agent处理对话

### 8.2.2 完整代码

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import shutil
import os

app = FastAPI(title="企业知识库系统")

# 配置
llm = ChatOpenAI(temperature=0.3)
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 全局变量
vectorstore = None
qa_chain = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class DocumentUpload(BaseModel):
    title: str

# API端点
@app.post("/upload")
async def upload_document(file: UploadFile, title: str):
    """上传文档"""
    global vectorstore, qa_chain
    
    # 保存文件
    file_path = f"documents/{file.filename}"
    os.makedirs("documents", exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 加载文档
    if file.filename.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    
    # 更新向量数据库
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings
        )
    else:
        vectorstore.add_documents(texts)
    
    # 重建QA链
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return {
        "status": "success",
        "title": title,
        "chunks": len(texts)
    }

@app.post("/query")
async def query(request: QueryRequest):
    """查询"""
    global qa_chain
    
    if qa_chain is None:
        return {"error": "请先上传文档"}
    
    result = qa_chain({
        "question": request.question
    })
    
    return {
        "answer": result["answer"],
        "sources": [doc.page_content[:100] + "..." 
                  for doc in result.get("source_documents", [])]
    }

@app.get("/documents")
async def list_documents():
    """列出所有文档"""
    documents = []
    docs_dir = "documents"
    
    if os.path.exists(docs_dir):
        for file_name in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, file_name)
            documents.append({
                "name": file_name,
                "size": os.path.getsize(file_path),
                "uploaded_at": os.path.getctime(file_path)
            })
    
    return {"documents": documents}

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 8.2.3 客户端使用

```python
import requests

# 上传文档
with open("employee_handbook.pdf", "rb") as f:
    files = {"file": f}
    data = {"title": "员工手册"}
    response = requests.post("http://localhost:8000/upload", files=files, data=data)
    print(response.json())

# 查询
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "公司的年假政策是什么？"}
)
print(response.json()["answer"])

# 列出文档
response = requests.get("http://localhost:8000/documents")
print(response.json())
```

## 8.3 项目2：自动化客服系统

### 8.3.1 需求分析

**用户需求：**
- 24/7自动回答客户问题
- 识别客户意图
- 处理常见问题
- 复杂问题转人工

**技术方案：**
- Agent + Knowledge Base
- 意图识别
- 多轮对话

### 8.3.2 完整代码

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

class CustomerServiceSystem:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.embeddings = OpenAIEmbeddings()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.knowledge_base = None
        
    def initialize_knowledge_base(self, qa_file):
        """初始化知识库"""
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        loader = TextLoader(qa_file)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = splitter.split_documents(documents)
        
        self.knowledge_base = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
    
    def create_tools(self):
        """创建工具集"""
        tools = []
        
        # 知识库检索工具
        if self.knowledge_base:
            retriever = self.knowledge_base.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            class KnowledgeBaseTool(BaseTool):
                name = "knowledge_base"
                description = "查询常见问题知识库"
                
                def _run(self, query):
                    result = qa_chain({"query": query})
                    return result["result"]
            
            tools.append(KnowledgeBaseTool())
        
        return tools
    
    def create_agent(self):
        """创建客服Agent"""
        tools = self.create_tools()
        
        system_prompt = """你是一个专业的客服助手。

你的职责：
1. 礼貌友好地回答客户问题
2. 首先查询知识库
3. 如果知识库没有答案，提供通用建议
4. 对于复杂问题，建议联系人工客服

常见问题类型：
- 产品功能
- 价格政策
- 售后服务
- 退款流程
- 配送政策"""
        
        agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            agent_kwargs={
                "system_message": system_prompt
            },
            verbose=True
        )
        
        return agent
    
    def chat(self, user_message):
        """对话"""
        if not hasattr(self, 'agent'):
            self.agent = self.create_agent()
        
        try:
            result = self.agent.run(user_message)
            return result
        except Exception as e:
            return f"抱歉，处理您的问题时出现错误：{e}。您可以联系人工客服。"

# 使用示例
if __name__ == "__main__":
    system = CustomerServiceSystem()
    
    # 初始化知识库
    system.initialize_knowledge_base("customer_qa.txt")
    
    # 对话示例
    conversations = [
        "你们的退货政策是什么？",
        "这个产品多少钱？",
        "我想退款",
        "谢谢"
    ]
    
    for msg in conversations:
        print(f"客户：{msg}")
        response = system.chat(msg)
        print(f"客服：{response}\n")
```

## 8.4 项目3：数据分析助手

### 8.4.1 需求分析

**用户需求：**
- 上传数据文件
- 用自然语言分析数据
- 生成报告和图表
- 支持SQL查询

**技术方案：**
- Python代码执行
- Pandas数据处理
- 自然语言转SQL
- 可视化生成

### 8.4.2 完整代码

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import PythonREPL
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

class DataAnalysisAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def create_tools(self):
        """创建分析工具"""
        tools = []
        
        # Python代码执行工具
        python_tool = PythonREPLTool()
        tools.append(python_tool)
        
        return tools
    
    def create_agent(self):
        """创建分析Agent"""
        tools = self.create_tools()
        
        system_prompt = """你是一个数据分析助手。

你的能力：
1. 使用Python和Pandas分析数据
2. 生成统计报告
3. 创建可视化图表
4. 执行SQL查询

注意事项：
- 对于简单的查询，直接使用Python代码
- 对于复杂的分析，逐步完成
- 如果代码执行失败，尝试简化
- 最后提供清晰的分析结论"""
        
        agent = create_python_agent(
            self.llm,
            tools,
            verbose=True
        )
        
        return agent
    
    def analyze(self, user_request, data=None):
        """分析数据"""
        if data:
            # 预加数据到上下文
            system_prompt = f"""
你有以下数据可用：
{data}

请基于这个数据进行分析。
"""
            # 重新创建Agent
            from langchain.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}")
            ])
            
            from langchain.chains import ConversationChain
            chain = ConversationChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory,
                verbose=True
            )
            
            return chain.run(user_request)
        else:
            if not hasattr(self, 'agent'):
                self.agent = self.create_agent()
            
            return self.agent.run(user_request)

# 使用示例
if __name__ == "__main__":
    assistant = DataAnalysisAssistant()
    
    # 示例数据
    sample_data = """
产品,销售额,利润
A产品,1000,200
B产品,800,150
C产品,1200,300
"""
    
    # 分析请求
    requests = [
        "计算总销售额和总利润",
        "哪个产品利润率最高？",
        "生成销售报告"
    ]
    
    for req in requests:
        print(f"用户：{req}")
        response = assistant.analyze(req, sample_data)
        print(f"助手：{response}\n")
```

## 8.5 项目部署

### 8.5.1 Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8.5.2 Docker Compose

```yaml
version: '3.8'

services:
  langchain-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./documents:/app/documents
    restart: always
```

### 8.5.3 Kubernetes部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langchain-app
  template:
    metadata:
      labels:
        app: langchain-app
    spec:
      containers:
      - name: app
        image: langchain-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: langchain-service
spec:
  selector:
    matchLabels:
      app: langchain-app
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

## 8.6 项目总结

### 8.6.1 技术亮点

1. **RAG技术** - 准确检索相关知识
2. **对话记忆** - 支持多轮对话
3. **Agent自主决策** - 灵活处理复杂任务
4. **工具集成** - 扩展AI能力
5. **API部署** - 易于集成和扩展

### 8.6.2 扩展方向

1. **添加更多数据源** - 数据库、API、实时数据
2. **实现用户系统** - 认证、权限管理
3. **优化检索算法** - 混合检索、重排序
4. **添加监控告警** - LangSmith、日志分析
5. **性能优化** - 缓存、批处理、异步调用

### 8.6.3 实际应用场景

- **企业知识库** - 内部文档智能问答
- **客户服务** - 24/7自动客服
- **数据分析** - 自然语言数据查询
- **教育辅导** - 个性化学习助手
- **法律咨询** - 案例检索和建议

## 本章小结

通过本章学习，你应该已经掌握了：
- 如何综合运用LangChain组件
- 如何设计和实现完整项目
- 如何部署和维护生产系统
- 如何优化和扩展项目

**关键要点：**
1. 实践是最好的学习方式
2. 从小项目开始，逐步复杂
3. 注重代码质量和可维护性
4. 做好错误处理和日志记录
5. 持续优化和迭代改进

## 课程总结

恭喜你完成了《LangChain从入门到精通》课程的学习！

### 你现在能够：

✅ **基础能力**
- 理解LangChain的核心概念
- 使用PromptTemplate和Memory
- 构建和管理Chain

✅ **进阶能力**
- 实现RAG系统
- 开发自主Agent
- 优化性能和安全

✅ **高级能力**
- 使用LangSmith调试
- 部署LangServe服务
- 构建生产级应用

✅ **实战能力**
- 完成完整项目
- 部署到生产环境
- 持续优化改进

### 下一步学习建议

1. **深入学习**
   - 探索更多高级功能
   - 学习其他LLM框架
   - 研究Agent架构

2. **实践项目**
   - 构建自己的应用
   - 参与开源社区
   - 分享你的经验

3. **持续关注**
   - LangChain更新
   - LLM技术发展
   - 社区最佳实践

## 练习题

1. 构建一个你自己的LangChain应用
2. 将应用部署到生产环境
3. 优化应用的性能和用户体验

## 常见问题

**Q: 如何选择合适的向量数据库？**
A: 根据数据规模、性能需求、部署方式选择。

**Q: Agent一定会比Chain好吗？**
A: 不一定，简单任务用Chain更高效。

**Q: 如何处理实时数据？**
A: 使用流式处理和实时向量更新。

**Q: 如何保证数据隐私？**
A: 使用本地部署、加密存储、访问控制。

---

**感谢学习本课程！**
**祝你成为优秀的LangChain开发者！**
