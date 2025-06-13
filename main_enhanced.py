# main_enhanced.py
"""增强版文档阅读系统 - 集成ContextGem概念提取"""

import os
import sys
import logging
import json
import argparse
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# 添加src目录到Python路径
# 这是启动脚本必须做的事情，以确保所有'src'下的模块能被找到
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_loader import UniversalDocumentLoader
from sat_text_splitter import SATTextSplitter
from vectorstore_local import LocalVectorStore
from enhanced_rag_chain import EnhancedRAGChain

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class EnhancedDocumentReader:
    """增强版文档阅读系统"""
    
    def __init__(self, use_enhancements: bool = True):
        logger.info("初始化文档阅读系统...")
        self._validate_config()
        
        # 从环境变量获取模型名称，并提供一个合理的默认值
        self.chat_model_name = os.getenv("CHAT_MODEL", "deepseek-chat")
        logger.info(f"使用聊天模型: {self.chat_model_name}")

        self.loader = UniversalDocumentLoader()
        self.splitter = SATTextSplitter(target_chunk_size=int(os.getenv("CHUNK_SIZE", 1000)))
        self.vector_manager = LocalVectorStore(embedding_model="all-MiniLM-L6-v2")
        
        # 将获取到的模型名称传递给 EnhancedRAGChain
        self.rag_chain = EnhancedRAGChain(
            vectorstore=None, # 初始为空，加载文档后创建
            model_name=self.chat_model_name, # <-- **关键修改**
            auto_extract_concepts=use_enhancements
        )
        self.use_enhancements = self.rag_chain.enable_contextgem
        
        self.current_doc_path: Optional[str] = None
        self.full_document_text: Optional[str] = None
        
        logger.info(f"系统初始化完成 (增强模式: {'已启用' if self.use_enhancements else '已禁用'})")
    
    def _validate_config(self):
        required = ["OPENAI_API_KEY", "OPENAI_API_BASE"]
        if any(not os.getenv(var) for var in required):
            raise ValueError(f"缺少环境变量: {', '.join(v for v in required if not os.getenv(v))}")
    
    def load_document(self, file_path: str, use_cache: bool = True):
        logger.info(f"正在加载文档: {file_path}")
        self.current_doc_path = file_path
        cache_path = f"output/cache/{Path(file_path).stem}_vectorstore"
        
        vectorstore = None
        if use_cache and os.path.exists(cache_path):
            logger.info("发现缓存，正在加载...")
            try:
                vectorstore = self.vector_manager.load_vectorstore(cache_path)
                logger.info("从缓存加载向量存储成功")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}，将重新处理文档")

        if vectorstore is None:
            documents = self.loader.load(file_path)
            self.full_document_text = "\n\n".join([doc.page_content for doc in documents])
            logger.info("SAT分块...")
            chunks = self.splitter.split_documents(documents)
            logger.info("创建向量存储...")
            vectorstore = self.vector_manager.create_vectorstore(chunks)
            if use_cache:
                self.vector_manager.save_vectorstore(cache_path)
        
        # 确保完整文本被加载，用于概念提取
        if self.full_document_text is None:
            self._load_full_text(file_path)

        # 更新RAG链
        self.rag_chain.vectorstore = vectorstore
        self.rag_chain.qa_chain = self.rag_chain._create_qa_chain() # 重新绑定retriever
        if self.use_enhancements:
            self.rag_chain.set_document_concepts(self.full_document_text)

    def _load_full_text(self, file_path: str):
        docs = self.loader.load(file_path)
        self.full_document_text = "\n\n".join([d.page_content for d in docs])
    
    def interactive_mode(self):
        print("\n=== 文档阅读系统 - 交互模式 ===")
        print(f"模式: {'增强' if self.use_enhancements else '基础'}")
        print("命令: quit, help, info" + (", insights, concepts" if self.use_enhancements else ""))
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input: continue
                cmd = user_input.lower()

                if cmd == 'quit': break
                elif cmd == 'help': self._show_help()
                elif cmd == 'info': self._display_document_info()
                elif cmd == 'insights' and self.use_enhancements: self._display_insights()
                elif cmd == 'concepts' and self.use_enhancements: self._display_concepts()
                else: self._display_answer(self.rag_chain.ask(user_input))
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                logger.error(f"交互模式出错: {e}", exc_info=True)
                print(f"\n错误: {e}")
        print("\n感谢使用，再见！")

    def _show_help(self):
        print("\n帮助:\n- 直接提问\n- info: 文档信息\n- quit: 退出")
        if self.use_enhancements: print("- insights: 文档洞察\n- concepts: 提取的概念")

    def _display_insights(self):
        insights = self.rag_chain.get_document_insights()
        print("\n=== 文档洞察 ===")
        print(json.dumps(insights, indent=2, ensure_ascii=False))

    def _display_concepts(self):
        concepts = self.rag_chain.extracted_concepts
        if concepts:
            print("\n=== 提取的概念 ===")
            print(json.dumps(concepts, indent=2, ensure_ascii=False))
        else:
            print("未提取或无概念。")

    def _display_document_info(self):
        if not self.vector_manager.vectorstore: print("未加载文档。"); return
        print("\n=== 文档信息 ===")
        print(f"  路径: {self.current_doc_path}")
        print(f"  模式: {'增强' if self.use_enhancements else '基础'}")
        print(f"  聊天模型: {self.rag_chain.model_name}")
        print(f"  嵌入模型: {self.vector_manager.embedding_manager.model_name}")
        print(f"  向量块数: {len(self.vector_manager.vectorstore.docstore._dict)}")

    def _display_answer(self, result: Dict[str, Any]):
        print("\n" + "="*60)
        if result.get('status') == 'error':
            print(f"【错误】\n{result.get('answer', '未知错误')}")
        else:
            print(f"【回答】\n{result.get('answer', '无回答')}")
            if result.get('concepts_used'):
                print(f"\n【使用概念】: {', '.join(result['concepts_used'])}")
            if result.get('sources'):
                print("\n【参考来源】:")
                for i, src in enumerate(result['sources']):
                    print(f"  {i+1}. {src['content'].strip().replace(chr(10), ' ')}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="文档阅读系统")
    parser.add_argument("file_path", nargs="?", help="文档路径")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument("--basic", action="store_true", help="强制使用基础模式")
    args = parser.parse_args()

    if not args.file_path:
        parser.print_help()
        print("\n示例: python main_enhanced.py data/documents/your_file.pdf")
        return

    print("=== 增强版文档阅读系统启动... ===")
    os.makedirs("output/cache", exist_ok=True)
    
    try:
        reader = EnhancedDocumentReader(use_enhancements=not args.basic)
        reader.load_document(args.file_path, use_cache=not args.no_cache)
        print("\n文档加载成功！")
        reader.interactive_mode()
    except Exception as e:
        logger.error(f"程序运行失败: {e}", exc_info=True)
        print(f"\n程序出现严重错误: {e}")

if __name__ == "__main__":
    main()
