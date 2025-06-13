# src/vectorstore_local.py
"""本地向量存储模块 - 使用FAISS和本地嵌入"""

import os
import logging
from pathlib import Path
import json
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from embedding_manager import LocalEmbeddingManager, LangChainEmbeddingWrapper

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """本地向量存储管理器"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        初始化向量存储。
        
        Args:
            embedding_model: 模型的名称，将用于在 'models/' 目录下查找对应的文件夹。
        """
        # --- 关键修复：在这里构建本地路径，并传递给嵌入管理器 ---
        # 1. 获取项目根目录 (src/.. -> project_root)
        project_root = Path(__file__).parent.parent.resolve()
        # 2. 构建模型的本地绝对路径
        local_model_path = project_root / 'models' / embedding_model
        
        # 3. 将这个绝对路径作为模型标识符传递
        self.embedding_manager = LocalEmbeddingManager(model_path=str(local_model_path))
        # -----------------------------------------------------------
        
        self.embeddings = LangChainEmbeddingWrapper(self.embedding_manager)
        self.vectorstore: Optional[FAISS] = None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """创建向量存储"""
        logger.info(f"正在创建本地向量存储，共{len(documents)}个文档块...")
        try:
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            logger.info("向量存储创建完成")
            return self.vectorstore
        except Exception as e:
            logger.error(f"创建向量存储失败: {e}", exc_info=True)
            raise
    
    def save_vectorstore(self, path: str):
        """保存向量存储到磁盘"""
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore.save_local(str(save_path))
        
        metadata = {
            'embedding_model_path': self.embedding_manager.model_path,
            'num_documents': len(self.vectorstore.docstore._dict)
        }
        (save_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        logger.info(f"向量存储已保存到: {save_path}")
    
    def load_vectorstore(self, path: str) -> FAISS:
        """从磁盘加载向量存储"""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"向量存储目录不存在: {load_path}")
        
        # 加载FAISS索引
        self.vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"从 '{load_path}' 加载向量存储成功。")
        return self.vectorstore

    # ... 其他方法保持不变 ...
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vectorstore is None: raise ValueError("向量存储未初始化")
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        if self.vectorstore is None: raise ValueError("向量存储未初始化")
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc, 1/(1+score)) for doc, score in results]

    def add_documents(self, documents: List[Document]):
        if self.vectorstore is None: raise ValueError("向量存储未初始化")
        self.vectorstore.add_documents(documents)
