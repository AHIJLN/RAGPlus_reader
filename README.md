# RAGPlus Reader 完整使用教程

> **版权声明**: 本项目遵循 MIT 开源协议

## 目录
1. [项目简介](#项目简介)
2. [快速开始（初学者）](#快速开始初学者)
3. [项目架构详解（专业者）](#项目架构详解专业者)
4. [高级功能与扩展（高级用户）](#高级功能与扩展高级用户)

---

## 项目简介

RAGPlus Reader 是一个智能文档阅读系统，结合了 RAG（检索增强生成）技术和概念提取功能，让您可以通过自然语言对话的方式深入理解文档内容。

### 核心特性
- 🚀 **双模式运行**：基础模式（快速问答）和增强模式（深度理解）
- 📚 **多格式支持**：PDF、TXT 文档
- 🧠 **本地嵌入**：使用本地模型，保护数据隐私
- 💡 **智能分块**：SAT（语义自适应分块）技术
- 🔍 **概念提取**：自动提取文档关键概念（增强模式）
- 🌐 **API 灵活**：支持 Deepseek 和 OpenRouter API

---

## 快速开始（初学者）

### 什么是 RAG？
RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的AI技术。简单来说，它能：
1. 将您的文档切分成小块并建立索引
2. 当您提问时，找到最相关的文档片段
3. 基于这些片段生成准确的回答

### 1. 安装项目

打开终端（Windows 用户使用 PowerShell 或 CMD），执行以下命令：

```bash
# 克隆项目
git clone https://github.com/AHIJLN/RAGPlus_reader.git
cd RAGPlus_reader

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
## Windows:
venv\Scripts\activate

## macOS/Linux:
source venv/bin/activate

# 运行智能安装脚本（推荐）
python install_requirements.py
```

安装脚本会自动：
- ✅ 检查 Python 版本
- ✅ 安装所有依赖
- ✅ 创建项目目录结构
- ✅ 配置兼容层
- ✅ 生成配置文件模板

### 2. 配置 API

编辑项目根目录下的 `.env` 文件，选择并配置一个 API 服务：

#### 推荐：使用 Deepseek API（性价比最高）

1. 访问 [https://platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
2. 注册并获取 API 密钥
3. 在 `.env` 文件中配置：

```env
OPENAI_API_KEY="your-deepseek-api-key-here"
OPENAI_API_BASE="https://api.deepseek.com/v1"
CHAT_MODEL="deepseek-chat"
```

### 3. 准备文档

将您要阅读的 PDF 或 TXT 文件放入 `data/documents/` 目录：

```bash
RAGPlus_reader/
├── data/
│   └── documents/
│       └── your_document.pdf  # 在这里放置您的文档
```

### 4. 开始使用

#### 基础模式（推荐初次使用）
```bash
python main.py data/documents/your_document.pdf
```

#### 增强模式（获得更深入的理解）
```bash
python main_enhanced.py data/documents/your_document.pdf
```

### 5. 交互式对话

程序启动后，您可以：
- 直接输入问题进行查询
- 输入 `info` 查看文档信息
- 输入 `help` 查看帮助
- 输入 `quit` 退出程序

**示例对话：**
```
> 这篇文档的主要内容是什么？
> 作者提出了哪些关键观点？
> 第三章讲了什么？
```

---

## 项目架构详解（专业者）

### 项目结构
```
RAGPlus_reader/
├── src/                          # 源代码目录
│   ├── contextgem_compat/       # ContextGem 兼容层
│   │   ├── __init__.py
│   │   └── core.py              # 概念提取核心实现
│   ├── document_loader.py       # 文档加载器
│   ├── sat_text_splitter.py     # SAT 智能分块器
│   ├── embedding_manager.py     # 本地嵌入管理
│   ├── vectorstore_local.py     # 向量存储管理
│   ├── rag_chain.py            # 基础 RAG 链
│   └── enhanced_rag_chain.py   # 增强 RAG 链
├── data/documents/              # 文档存放目录
├── output/                      # 输出目录
│   ├── cache/                   # 向量存储缓存
│   └── logs/                    # 日志文件
├── models/                      # 本地嵌入模型
│   └── all-MiniLM-L6-v2/       # 默认嵌入模型
├── main.py                      # 基础模式入口
├── main_enhanced.py             # 增强模式入口
└── .env                         # 配置文件
```

### 核心模块说明

#### 1. 文档处理流程
```python
文档加载 (document_loader.py)
    ↓
SAT智能分块 (sat_text_splitter.py)
    ↓
本地嵌入 (embedding_manager.py)
    ↓
向量存储 (vectorstore_local.py)
    ↓
RAG问答 (rag_chain.py / enhanced_rag_chain.py)
```

#### 2. SAT（Semantic Adaptive Tiling）分块策略
- **语义识别**：识别标题、段落、列表、代码块等结构
- **重要性评分**：基于类型、位置、长度计算重要性
- **智能组合**：根据语义边界和大小限制组合块

#### 3. 两种运行模式对比

| 特性 | 基础模式 | 增强模式 |
|------|---------|----------|
| 启动速度 | 快 | 稍慢 |
| 内存占用 | 低 | 中等 |
| 回答质量 | 良好 | 优秀 |
| 概念提取 | ❌ | ✅ |
| 文档洞察 | ❌ | ✅ |
| 适用场景 | 快速查询 | 深度理解 |

### 配置优化建议

#### 1. 分块参数调整
编辑 `.env` 文件：
```env
CHUNK_SIZE=1000  # 目标块大小（字符数）
# 建议值：
# - 技术文档：800-1200
# - 文学作品：1500-2000
# - 学术论文：1000-1500
```

#### 2. 缓存管理
```bash
# 清除特定文档的缓存
rm -rf output/cache/your_document_vectorstore

# 禁用缓存运行
python main.py data/documents/your_document.pdf --no-cache
```

#### 3. 日志级别设置
在代码中修改：
```python
logging.basicConfig(level=logging.DEBUG)  # 详细日志
```

### 性能优化技巧

1. **使用缓存**：第二次加载同一文档会使用缓存，速度提升 10x
2. **GPU 加速**：如果有 NVIDIA GPU，会自动启用 CUDA 加速
3. **批量处理**：可以修改代码支持目录批量加载

---

## 高级功能与扩展（高级用户）

### 自定义概念提取

在 `enhanced_rag_chain.py` 中添加新的概念定义：

```python
@staticmethod
def get_custom_concept():
    return JsonObjectConcept(
        name="技术术语",
        description="文档中的专业术语及其定义",
        structure={
            "terms": list,  # 术语列表
            "definitions": dict,  # 术语:定义 映射
            "categories": list  # 术语分类
        }
    )
```

### 扩展文档格式支持

在 `document_loader.py` 中添加新的加载器：

```python
def _load_docx(self, file_path: str) -> List[Document]:
    """加载 Word 文档"""
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return [Document(page_content=text, metadata={"source": file_path})]

# 注册新格式
self.supported_extensions['.docx'] = self._load_docx
```

### 自定义嵌入模型

替换默认的 `all-MiniLM-L6-v2` 模型：

1. 下载新模型到 `models/` 目录
2. 修改初始化参数：
```python
self.vector_manager = LocalVectorStore(
    embedding_model="your-custom-model"
)
```

### 集成到其他应用

```python
from src.enhanced_rag_chain import EnhancedRAGChain
from src.vectorstore_local import LocalVectorStore

class MyApp:
    def __init__(self):
        self.vector_store = LocalVectorStore()
        self.rag_chain = EnhancedRAGChain(
            vectorstore=None,
            model_name="deepseek-chat"
        )
    
    def process_document(self, file_path):
        # 自定义处理逻辑
        pass
```

### 高级 API 配置

#### 使用代理
```env
# HTTP 代理
HTTP_PROXY="http://your-proxy:port"
HTTPS_PROXY="http://your-proxy:port"
```

#### 自定义 API 端点
```python
# 在代码中直接配置
llm = ChatOpenAI(
    openai_api_base="https://your-custom-endpoint.com/v1",
    openai_api_key="your-key",
    model_name="custom-model"
)
```

### 批量文档处理脚本

创建 `batch_process.py`：

```python
import os
from pathlib import Path
from main_enhanced import EnhancedDocumentReader

def batch_process_documents(directory: str):
    reader = EnhancedDocumentReader()
    
    for file_path in Path(directory).glob("*.pdf"):
        print(f"\n处理文档: {file_path}")
        reader.load_document(str(file_path))
        
        # 自动提取关键信息
        insights = reader.rag_chain.get_document_insights()
        
        # 保存结果
        output_file = f"output/{file_path.stem}_insights.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(insights, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    batch_process_documents("data/documents")
```

### 性能监控与调试

添加性能监控代码：

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.2f}秒")
        return result
    return wrapper

# 应用到关键函数
@timing_decorator
def load_document(self, file_path: str):
    # 原有代码
```

---

## 常见问题解答

### Q: 第一次运行很慢？
A: 首次运行需要下载嵌入模型（约 90MB），之后会使用本地缓存。

### Q: 如何处理大型 PDF？
A: 建议使用增强模式并启用缓存，处理过的文档会保存向量索引。

### Q: API 调用失败？
A: 检查网络连接和 API 密钥，确保 `.env` 配置正确。

### Q: 内存不足？
A: 可以减小 `CHUNK_SIZE` 或使用基础模式。

---

## 最佳实践推荐

1. **初学者**：使用 Deepseek API + 基础模式开始
2. **日常使用**：增强模式 + 缓存，获得最佳体验
3. **专业分析**：自定义概念提取 + 批量处理
4. **隐私优先**：完全本地化部署（使用本地 LLM）

---

## 项目贡献

本项目遵循 MIT 协议，欢迎贡献代码、报告问题或提出建议。

---

*Happy Reading with RAGPlus! 🚀*
