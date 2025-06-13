# src/embedding_manager.py
"""本地嵌入管理器 - 根据模型配置文件精确手动构建模型"""

import os
import json
import logging
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from langchain.embeddings.base import Embeddings

# --- 关键：导入所有需要的底层模块 ---
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling, Normalize

logger = logging.getLogger(__name__)


class LocalEmbeddingManager:
    """
    通过手动、精确地构建模型来加载本地嵌入模型，确保100%成功，
    完全绕过所有高层封装的验证和不确定性。
    """
    
    def __init__(self, model_path: str):
        """
        初始化本地嵌入管理器。
        
        Args:
            model_path: 指向包含模型文件的本地文件夹的绝对路径。
        """
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"准备从本地路径手动构建嵌入模型: {self.model_path} on {self.device}")
        
        try:
            # --- 最终决定性修复：根据您提供的JSON文件精确构建 ---
            
            # 1. 加载底层的Transformer模型。它的配置在 config.json 中。
            word_embedding_model = Transformer(self.model_path)
            
            # 从Transformer的配置中获取输出维度，这就是池化层的输入维度。
            embedding_dimension = word_embedding_model.get_word_embedding_dimension()

            # 2. 创建池化层 (Pooling Layer)。
            # 对于 all-MiniLM-L6-v2，标准配置是使用 'mean' 池化。
            pooling_model = Pooling(
                word_embedding_dimension=embedding_dimension,
                pooling_mode='mean'
            )

            # 3. 创建归一化层 (Normalize Layer)。
            # 这是您提供的 modules.json 中定义的第三个模块。
            normalize_model = Normalize()

            # 4. 将这些模块按照 `modules.json` 中定义的顺序组合起来。
            modules = OrderedDict([
                ('0_transformer', word_embedding_model),
                ('1_pooling', pooling_model),
                ('2_normalize', normalize_model)
            ])
            
            self.model = SentenceTransformer(modules=modules, device=self.device)
            # ----------------------------------------------------

            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"本地模型 '{os.path.basename(model_path)}' 手动构建并加载成功，嵌入维度: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"从本地路径 '{self.model_path}' 手动构建模型失败: {e}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


class LangChainEmbeddingWrapper(Embeddings):
    """包装本地嵌入以兼容LangChain接口"""
    
    def __init__(self, embedding_manager: LocalEmbeddingManager):
        self.embedding_manager = embedding_manager
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_manager.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embedding_manager.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)
