"""ContextGem核心类的兼容实现 - 优化版"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import re
import json
import logging

logger = logging.getLogger(__name__)

# --- Data Classes ---

@dataclass
class ExtractedItem:
    """封装从文本中提取的单个结构化数据项。"""
    data: Dict[str, Any]
    references: Optional[List[Dict[str, Any]]] = None

@dataclass
class BaseConcept:
    """所有概念定义的基类。"""
    name: str
    description: str
    extracted_items: List[ExtractedItem] = field(default_factory=list)

@dataclass
class JsonObjectExample:
    """为JsonObjectConcept提供示例，以提高提取准确性。"""
    data: Dict[str, Any]
    description: Optional[str] = None

@dataclass
class JsonObjectConcept(BaseConcept):
    """定义一个要从文本中提取的JSON对象的结构。"""
    structure: Dict[str, type] = field(default_factory=dict)
    add_references: bool = False
    reference_depth: str = "paragraphs"
    examples: List[JsonObjectExample] = field(default_factory=list)
    
    def __post_init__(self):
        """确保在初始化后，即使子类未提供，也能正确设置父类字段。"""
        super().__init__(self.name, self.description)
        if self.structure is None:
            self.structure = {}

@dataclass
class Document:
    """表示一个待处理的文本文档。"""
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    concepts: List[BaseConcept] = field(default_factory=list)
    _paragraphs: Optional[List[str]] = None

    def add_concepts(self, concepts: List[BaseConcept]):
        """将概念定义与文档关联起来。"""
        self.concepts.extend(concepts)

    def get_concept_by_name(self, name: str) -> Optional[BaseConcept]:
        """通过名称查找已关联的概念。"""
        return next((c for c in self.concepts if c.name == name), None)

    @property
    def paragraphs(self) -> List[str]:
        """将原始文本分割成段落列表（懒加载）。"""
        if self._paragraphs is None:
            self._paragraphs = [p.strip() for p in self.raw_text.split('\n\n') if p.strip()]
        return self._paragraphs

# --- Core Logic Class ---

class DocumentLLM:
    """使用LLM从文档中提取概念的处理器。"""

    def __init__(self, model: str, api_key: str, **kwargs):
        """
        初始化DocumentLLM。
        
        Args:
            model: 要使用的LLM模型名称 (例如 'openai/gpt-4o-mini')。
            api_key: API密钥。
        """
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化OpenAI客户端，支持OpenRouter。"""
        try:
            from openai import OpenAI
            api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
            self.client = OpenAI(api_key=self.api_key, base_url=api_base)
            logger.info(f"DocumentLLM client initialized for model '{self.model}' using base URL '{api_base}'.")
        except ImportError:
            raise ImportError("请安装openai库: pip install openai>=1.0.0")
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}", exc_info=True)

    def extract_concepts_from_document(self, document: Document) -> List[BaseConcept]:
        """从单个文档中提取所有已定义的概念。"""
        if not self.client:
            logger.warning("LLM客户端未初始化，无法提取概念，返回空结果。")
            return []
            
        for concept in document.concepts:
            if isinstance(concept, JsonObjectConcept):
                extracted_data = self._extract_json_concept(document, concept)
                concept.extracted_items = extracted_data
        
        return document.concepts

    def _extract_json_concept(self, document: Document, concept: JsonObjectConcept) -> List[ExtractedItem]:
        """使用LLM提取单个JSON概念。"""
        prompt = self._build_extraction_prompt(document, concept)
        
        try:
            # 移除模型名称中的 'openai/' 前缀，以适配OpenRouter的API要求
            model_name_for_api = self.model.replace("openai/", "")
            
            response = self.client.chat.completions.create(
                model=model_name_for_api,
                messages=[
                    {"role": "system", "content": "你是一个专业的信息提取助手。请严格按照用户要求的JSON结构提取信息，不要添加任何额外的解释或说明。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000,
                response_format={"type": "json_object"} # 请求JSON输出
            )
            
            content = response.choices[0].message.content
            if not content:
                logger.warning(f"LLM为概念'{concept.name}'返回了空内容。")
                return []

            # 解析JSON响应
            result_data = json.loads(content)
            
            # 如果需要，添加引用信息
            if concept.add_references:
                self._add_references(result_data, document, concept)
            
            return [ExtractedItem(data=result_data)]
            
        except json.JSONDecodeError:
            logger.error(f"提取概念'{concept.name}'时JSON解析失败。模型返回: {content}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"提取概念'{concept.name}'时发生未知错误: {e}", exc_info=True)
            return []

    def _build_extraction_prompt(self, document: Document, concept: JsonObjectConcept) -> str:
        """构建用于信息提取的详细提示。"""
        type_mapping = {str: "string", int: "integer", float: "number", list: "array", dict: "object", bool: "boolean"}
        
        structure_parts = []
        for field_name, field_type in concept.structure.items():
            type_name = type_mapping.get(field_type, "any")
            structure_parts.append(f'  "{field_name}": "{type_name}"')
        
        structure_json = "{\n" + ",\n".join(structure_parts) + "\n}"

        prompt = f"""
从以下提供的文档内容中，提取关于“{concept.name}”的信息。

概念描述: {concept.description}

请严格按照以下JSON格式返回提取的信息。不要包含任何Markdown格式符(如```json)或额外的解释。

JSON结构:
{structure_json}

文档内容:
---
{document.raw_text[:4000]}...
---
"""
        return prompt.strip()

    def _add_references(self, data: Dict[str, Any], document: Document, concept: JsonObjectConcept):
        """（简化版）为提取的数据添加来源段落的引用。"""
        if concept.reference_depth != "paragraphs":
            return

        references = {}
        for key, value in data.items():
            if isinstance(value, str) and value:
                for i, para in enumerate(document.paragraphs):
                    if value in para:
                        references[key] = {
                            "paragraph_index": i,
                            "paragraph_text_preview": para[:100] + "..."
                        }
                        break
        if references:
            data["_references"] = references

    async def extract_concepts_from_document_async(self, document: Document) -> List[BaseConcept]:
        """异步提取（为保持接口兼容性，实际仍为同步执行）。"""
        return self.extract_concepts_from_document(document)
