# src/rag_chain.py
"""OpenRouter适配的RAG问答链模块 - 修复版"""

from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
import logging
import os

logger = logging.getLogger(__name__)


class OpenRouterRAGChain:
    """适配OpenRouter的RAG问答链"""
    
    def __init__(self, 
                 vectorstore,
                 model_name: str = "openai/gpt-4o-mini",
                 temperature: float = 0.0):
        """
        初始化RAG链
        
        Args:
            vectorstore: 向量存储对象（可以为None，稍后设置）
            model_name: OpenRouter模型名称
            temperature: 生成温度
        """
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.temperature = temperature
        
        # 初始化适配OpenRouter的LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
            max_retries=3,
            request_timeout=30
        )
        
        # 创建自定义提示模板
        self.prompt_template = self._create_prompt_template()
        
        # 初始化问答链 - 只在vectorstore存在时创建
        self.qa_chain = None
        if self.vectorstore is not None:
            self.qa_chain = self._create_qa_chain()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """创建优化的提示模板"""
        template = """你是一个专业的文档分析助手。请基于提供的上下文回答问题。

上下文信息：
{context}

问题：{question}

请遵循以下原则回答：
1. 只使用上下文中提供的信息
2. 如果上下文中没有相关信息，明确说明"根据提供的文档内容，无法找到相关信息"
3. 回答要准确、具体、有条理
4. 如果可能，引用具体的原文内容
5. 保持客观，不要添加个人推测

回答："""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_qa_chain(self) -> Optional[RetrievalQA]:
        """创建问答链"""
        if self.vectorstore is None:
            logger.warning("无法创建QA链：vectorstore为None")
            return None
            
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={
                "prompt": self.prompt_template,
                "verbose": False
            },
            return_source_documents=True
        )
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        回答问题
        
        Args:
            question: 用户问题
            
        Returns:
            Dict: 包含答案和源文档的字典
        """
        logger.info(f"收到问题: {question}")
        
        # 检查qa_chain是否已初始化
        if self.qa_chain is None:
            logger.error("QA链未初始化，请先加载文档")
            return {
                "question": question,
                "answer": "系统错误：请先加载文档",
                "sources": [],
                "model": self.model_name,
                "status": "error"
            }
        
        try:
            # 注意：OpenRouter可能不支持token计数回调
            result = self.qa_chain.invoke({"query": question})
            
            # 处理返回结果
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # 格式化源文档信息
            sources = []
            for i, doc in enumerate(source_docs):
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "relevance_rank": i + 1
                })
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "model": self.model_name,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"问答处理失败: {str(e)}")
            return {
                "question": question,
                "answer": f"处理失败: {str(e)}",
                "sources": [],
                "model": self.model_name,
                "status": "error"
            }
