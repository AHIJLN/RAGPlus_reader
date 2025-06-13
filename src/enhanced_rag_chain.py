# src/enhanced_rag_chain.py
"""增强的RAG链 - 集成ContextGem概念提取"""

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, TYPE_CHECKING

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 确保父类被正确导入
from rag_chain import OpenRouterRAGChain

# 用于类型检查，避免循环导入问题
if TYPE_CHECKING:
    from contextgem_compat import Document, DocumentLLM, JsonObjectConcept

# 全局变量，用于缓存导入结果，避免重复尝试
CONTEXTGEM_AVAILABLE = None
contextgem_classes = {}

logger = logging.getLogger(__name__)

def _lazy_import_contextgem():
    """延迟导入ContextGem模块，以处理复杂的启动依赖关系。"""
    global CONTEXTGEM_AVAILABLE, contextgem_classes
    if CONTEXTGEM_AVAILABLE is not None:
        return CONTEXTGEM_AVAILABLE

    try:
        from contextgem_compat import Document, DocumentLLM, JsonObjectConcept
        contextgem_classes['Document'] = Document
        contextgem_classes['DocumentLLM'] = DocumentLLM
        contextgem_classes['JsonObjectConcept'] = JsonObjectConcept
        CONTEXTGEM_AVAILABLE = True
        logger.info("ContextGem兼容层成功加载。")
    except ImportError as e:
        CONTEXTGEM_AVAILABLE = False
        logger.warning(f"ContextGem不可用，将使用基础RAG功能。错误: {e}")
    
    return CONTEXTGEM_AVAILABLE


class ConceptDefinitions:
    """预定义的文档概念。"""
    @staticmethod
    def get_document_overview():
        if not _lazy_import_contextgem(): return None
        return contextgem_classes['JsonObjectConcept'](
            name="文档概览",
            description="文档的整体信息，包括标题、作者、摘要、主要主题和文档类型。",
            structure={"title": str, "author": str, "summary": str, "main_topics": list, "document_type": str}
        )
    
    @staticmethod
    def get_key_points():
        if not _lazy_import_contextgem(): return None
        return contextgem_classes['JsonObjectConcept'](
            name="关键要点",
            description="文档中的主要观点、结论和建议。",
            structure={"main_arguments": list, "conclusions": list, "recommendations": list}
        )

class EnhancedRAGChain(OpenRouterRAGChain):
    """增强的RAG链 - 集成ContextGem概念提取"""
    
    def __init__(self, vectorstore, model_name: str = "openai/gpt-4o-mini", **kwargs):
        """初始化增强的RAG链"""
        
        # 初始化子类属性（必须在调用父类构造函数之前）
        self.enable_contextgem = _lazy_import_contextgem()
        self.auto_extract_concepts = kwargs.pop('auto_extract_concepts', True)
        self.extracted_concepts = {}
        self.document_llm = None
        
        # 现在可以安全地调用父类的构造函数
        super().__init__(vectorstore, model_name, **kwargs)
        
        # 父类初始化完成后，再执行子类依赖于父类组件的逻辑
        if self.enable_contextgem:
            self._init_contextgem()
        else:
            logger.info("增强模式(ContextGem)未启用或不可用。")
            
    def _init_contextgem(self):
        try:
            DocumentLLM = contextgem_classes['DocumentLLM']
            self.document_llm = DocumentLLM(
                model=self.model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logger.info("ContextGem功能已成功初始化。")
        except Exception as e:
            logger.error(f"ContextGem初始化失败: {e}", exc_info=True)
            self.enable_contextgem = False

    def set_document_concepts(self, full_text: str):
        if not self.enable_contextgem or not self.auto_extract_concepts or not full_text:
            return
        
        logger.info("开始自动提取文档核心概念...")
        try:
            concepts_to_extract = [
                c for c in [ConceptDefinitions.get_document_overview(), ConceptDefinitions.get_key_points()] if c
            ]
            if not concepts_to_extract:
                logger.warning("没有可用的概念定义，跳过提取。")
                return

            self.extracted_concepts = self.extract_concepts_from_text(full_text, concepts_to_extract)
            logger.info(f"概念提取完成，共提取了 {len(self.extracted_concepts)} 个概念。")
            # 概念提取后，需要重建问答链以使用新的prompt
            self.qa_chain = self._create_qa_chain()
        except Exception as e:
            logger.error(f"自动概念提取过程中发生错误: {e}", exc_info=True)

    def extract_concepts_from_text(self, text: str, concepts: List['JsonObjectConcept']) -> Dict[str, Any]:
        if not self.enable_contextgem or not self.document_llm:
            return {}
        
        Document = contextgem_classes['Document']
        doc = Document(raw_text=text)
        doc.add_concepts(concepts)
        
        extracted = self.document_llm.extract_concepts_from_document(doc)
        
        result = {}
        for concept in extracted:
            if hasattr(concept, 'extracted_items') and concept.extracted_items:
                result[concept.name] = concept.extracted_items[0].data
        return result

    def _create_prompt_template(self) -> PromptTemplate:
        if self.enable_contextgem and self.extracted_concepts:
            template = """你是一个专业的文档分析助手。请基于提供的上下文和已提取的文档核心概念来回答问题。

[文档核心概念]
{concepts}

[相关上下文片段]
{context}

[问题]
{question}

请遵循以下原则回答：
1. 优先利用[文档核心概念]中的结构化信息进行回答。
2. 使用[相关上下文片段]来补充细节和证据。
3. 如果信息不足，请明确说明。

回答："""
            return PromptTemplate(template=template, input_variables=["context", "question", "concepts"])
        return super()._create_prompt_template()

    def ask(self, question: str) -> Dict[str, Any]:
        logger.info(f"收到问题: {question}")
        
        try:
            if not self.enable_contextgem or not self.extracted_concepts:
                # 如果增强模式未启用或无概念，直接使用父类的RAG方法
                return super().ask(question)

            # --- 增强的问答流程 ---
            logger.info(f"增强模式处理问题...")
            
            # 准备传递给链的输入
            input_data = {"query": question}

            # 如果使用增强模板，需要动态添加concepts变量
            # 注意：Langchain的RetrievalQA链的invoke方法不直接接受额外的模板变量。
            # 我们需要重建chain或者使用更底层的LLMChain。
            # 这里采用一个更简单的方法：直接格式化prompt并调用LLM。
            
            docs = self.vectorstore.similarity_search(question, k=4)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            concepts_text = json.dumps(self.extracted_concepts, indent=2, ensure_ascii=False)
            
            final_prompt = self.prompt_template.format(
                context=context,
                question=question,
                concepts=concepts_text
            )
            
            response = self.llm.invoke(final_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "question": question, "answer": answer,
                "sources": [{"content": doc.page_content[:200] + "...", "metadata": doc.metadata} for doc in docs],
                "concepts_used": list(self.extracted_concepts.keys()),
                "status": "success",
                "model": self.model_name
            }

        except Exception as e:
            logger.error(f"问答处理失败: {e}", exc_info=True)
            return {"question": question, "answer": f"处理失败: {str(e)}", "sources": [], "status": "error"}


    def get_document_insights(self) -> Dict[str, Any]:
        if not self.enable_contextgem or not self.extracted_concepts:
            return {"error": "洞察不可用，因为概念未提取或功能未启用。"}

        insights = {
            "concepts_extracted": len(self.extracted_concepts),
            "concept_names": list(self.extracted_concepts.keys()),
            "details": {}
        }
        
        if "文档概览" in self.extracted_concepts:
            overview = self.extracted_concepts["文档概览"]
            insights["details"]["文档类型"] = overview.get("document_type", "未知")
            insights["details"]["主要主题"] = overview.get("main_topics", [])
        
        if "关键要点" in self.extracted_concepts:
            key_points = self.extracted_concepts["关键要点"]
            insights["details"]["结论数量"] = len(key_points.get("conclusions", []))
            insights["details"]["包含建议"] = bool(key_points.get("recommendations"))
        
        return insights
