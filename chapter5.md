# 第5章：RAG（检索增强生成）

## 5.1 什么是RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合了检索和生成的技术，让AI能够访问和利用外部知识库。

### 为什么需要RAG？

LLM虽然强大，但有以下局限：
1. **知识截止**：训练数据有时间限制
2. **幻觉问题**：可能编造不存在的信息
3. **私有数据**：无法访问企业内部文档
4. **准确性**：对专业领域知识不足

RAG通过检索相关知识，然后生成答案，解决了这些问题。

### RAG的优势

1. **准确性提升**：基于真实文档生成答案
2. **可追溯性**：可以显示信息来源
3. **灵活性**：知识库可以随时更新
4. **专业性**：针对特定领域优化
5. **成本效益**：相比微调更经济

## 5.2 RAG架构

### 基本流程

```
用户提问
    ↓
[向量检索] → 从知识库检索相关文档
    ↓
[文档拼接] → 将检索到的文档组合成上下文
    ↓
[提示词构建] → 构建包含上下文的提示词
    ↓
[LLM生成] → 基于上下文生成答案
    ↓
[返回答案] → 返回最终答案和来源
```

### 核心组件

1. **文档加载器**：加载各种格式的文档
2. **文档分割器**：将长文档分割成小块
3. **向量嵌入**：将文本转换为向量
4. **向量数据库**：存储和检索向量
5. **检索器**：根据查询检索相关文档
6. **生成器**：基于检索结果生成答案

## 5.3 向量数据库

### 5.3.1 什么是向量数据库？

向量数据库专门用于存储和检索高维向量，常用于：
- 语义搜索
- 推荐系统
- 图像检索
- 文档检索

### 5.3.2 常用向量数据库

#### ChromaDB
```python
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# 创建向量数据库
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings()
)

# 添加文档
vectorstore.add_texts(["文档1", "文档2"])

# 检索
results = vectorstore.similarity_search("搜索查询")
```

#### FAISS
```python
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# 创建向量数据库
vectorstore = FAISS.from_texts(
    ["文档1", "文档2"],
    OpenAIEmbeddings()
)

# 检索
results = vectorstore.similarity_search("搜索查询")
```

#### Pinecone（云端）
```python
from langchain_community.vectorstores import Pinecone
import pinecone

# 初始化Pinecone
pinecone.init(api_key="your-api-key", environment="us-east-1")

# 创建索引
index_name = "my-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

# 创建向量数据库
vectorstore = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=OpenAIEmbeddings()
)
```

## 5.4 文档处理

### 5.4.1 文档加载

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

# 加载文本文件
loader = TextLoader("document.txt")
documents = loader.load()

# 加载PDF文件
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 加载CSV文件
loader = CSVLoader("data.csv")
documents = loader.load()

# 加载Markdown文件
loader = UnstructuredMarkdownLoader("README.md")
documents = loader.load()
```

### 5.4.2 文档分割

```python
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

# 字符分割器
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n\n"
)

# 递归分割器（推荐）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，"]
)

# 分割文档
texts = splitter.split_documents(documents)
```

### 5.4.3 向量嵌入

```python
from langchain.embeddings.openai import OpenAIEmbeddings

# 创建嵌入模型
embeddings = OpenAIEmbeddings()

# 嵌入单个文本
text_embedding = embeddings.embed_query("这是一段文本")

# 嵌入多个文本
text_embeddings = embeddings.embed_documents(["文本1", "文本2"])
```

## 5.5 构建RAG系统

### 完整代码示例

```python
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGSystem:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self, file_path):
        """加载文档"""
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)
        documents = loader.load()
        return documents

    def split_documents(self, documents):
        """分割文档"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        return texts

    def create_vectorstore(self, texts):
        """创建向量数据库"""
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings
        )

    def create_qa_chain(self):
        """创建问答链"""
        # 自定义提示词
        prompt_template = """使用以下上下文信息回答问题。
如果你不知道答案，就说你不知道，不要编造答案。

上下文：
{context}

问题：{question}

答案："""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # 创建检索器
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 检索最相关的3个文档
        )

        # 创建问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def query(self, question):
        """查询"""
        if self.qa_chain is None:
            raise ValueError("请先创建QA链")

        result = self.qa_chain({"query": question})

        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

    def add_document(self, file_path):
        """添加新文档"""
        documents = self.load_documents(file_path)
        texts = self.split_documents(documents)

        if self.vectorstore is None:
            self.create_vectorstore(texts)
        else:
            self.vectorstore.add_documents(texts)

        # 重新创建QA链
        self.create_qa_chain()

# 使用示例
if __name__ == "__main__":
    rag = RAGSystem()

    # 加载文档
    documents = rag.load_documents("knowledge.txt")
    texts = rag.split_documents(documents)
    rag.create_vectorstore(texts)
    rag.create_qa_chain()

    # 查询
    result = rag.query("什么是机器学习？")
    print("答案：", result["answer"])
    print("来源：", [doc.page_content[:100] + "..." for doc in result["sources"]])
```

## 5.6 实战：智能知识库问答

### 创建完整的知识库问答系统

```python
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class IntelligentKnowledgeBase:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def initialize(self, documents):
        """初始化知识库"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # 创建向量数据库
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings
        )

        # 创建对话式检索链
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True
        )

    def chat(self, question):
        """对话"""
        if self.qa_chain is None:
            raise ValueError("请先初始化知识库")

        result = self.qa_chain({"question": question})

        return {
            "answer": result["answer"],
            "sources": result.get("source_documents", [])
        }

    def add_knowledge(self, new_documents):
        """添加新知识"""
        if self.vectorstore is None:
            self.initialize(new_documents)
        else:
            self.vectorstore.add_documents(new_documents)

# 使用示例
if __name__ == "__main__":
    # 创建知识库
    kb = IntelligentKnowledgeBase()

    # 示例文档
    documents = [
        "Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。",
        "Python具有简洁明了的语法，易于学习和使用。",
        "Python广泛应用于Web开发、数据分析、人工智能等领域。"
    ]

    kb.initialize(documents)

    # 对话
    questions = [
        "什么是Python？",
        "Python有什么特点？",
        "Python可以做什么？"
    ]

    for q in questions:
        result = kb.chat(q)
        print(f"问题：{q}")
        print(f"答案：{result['answer']}")
        print()
```

## 5.7 RAG优化技巧

### 1. 提高检索质量

```python
# 使用不同的检索策略
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7  # 只返回相似度高于0.7的文档
    }
)
```

### 2. 优化文档分割

```python
# 根据文档类型选择合适的分割策略
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 根据需求调整
    chunk_overlap=200,  # 保持一定的重叠
    separators=["\n\n", "\n", "。", "！", "？", "；", "，"]
)
```

### 3. 混合检索

```python
# 结合关键词检索和语义检索
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance
    search_kwargs={"k": 5, "fetch_k": 10}
)
```

### 4. 缓存检索结果

```python
from langchain.cache import InMemoryCache

# 缓存LLM响应
llm.cache = InMemoryCache()

# 缓存检索结果
from langchain.storage import LocalFileStore
store = LocalFileStore("./cache/")
embeddings_cache = OpenAIEmbeddings(cache=store)
```

## 本章小结

通过本章学习，你应该已经掌握了：
- RAG的概念和优势
- 向量数据库的使用
- 文档加载、分割、嵌入
- 构建完整的RAG系统
- RAG优化技巧

**关键要点：**
1. RAG解决了LLM的知识限制
2. 向量数据库是RAG的核心
3. 文档分割质量影响检索效果
4. 可以根据场景选择不同的检索策略
5. 持续优化是RAG系统的关键

## 练习题

1. 构建一个RAG系统，支持PDF和Word文档
2. 实现混合检索（关键词+语义）
3. 添加文档来源引用功能

## 常见问题

**Q: RAG和微调哪个更好？**
A: 各有优劣。RAG更灵活、成本更低；微调性能更好但成本高。

**Q: 向量数据库选择标准？**
A: 考虑数据规模、查询性能、部署方式、成本等因素。

**Q: 如何处理多语言RAG？**
A: 使用多语言嵌入模型，或分别为不同语言建立索引。

**Q: RAG系统的性能瓶颈在哪里？**
A: 通常是嵌入和检索，可以通过缓存和优化向量数据库提升。

---

下一章预告：Agent开发 - 学习如何创建自主决策的AI智能体。
