# src/document_loader.py
"""文档加载模块 - 支持多种格式的文档读取"""

import os
from typing import List, Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalDocumentLoader:
    """通用文档加载器，支持PDF、TXT等格式"""
    
    def __init__(self):
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
        }
    
    def load(self, file_path: str) -> List[Document]:
        """
        加载文档文件
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            List[Document]: LangChain Document对象列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        logger.info(f"正在加载文档: {file_path}")
        return self.supported_extensions[ext](file_path)
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """加载PDF文档"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"成功加载PDF文档，共{len(docs)}页")
        return docs
    
    def _load_text(self, file_path: str) -> List[Document]:
        """加载文本文档"""
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        logger.info(f"成功加载文本文档")
        return docs
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """加载目录下的所有支持的文档"""
        all_docs = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if os.path.isfile(file_path) and ext in self.supported_extensions:
                try:
                    docs = self.load(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"加载文件失败 {filename}: {str(e)}")
        
        logger.info(f"共加载{len(all_docs)}个文档片段")
        return all_docs