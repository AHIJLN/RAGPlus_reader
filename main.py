# main.py
"""OpenRouter适配版文档阅读系统 - 基础模式"""

import os
import sys
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from document_loader import UniversalDocumentLoader
from sat_text_splitter import SATTextSplitter
from vectorstore_local import LocalVectorStore
from rag_chain import OpenRouterRAGChain

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


class OpenRouterDocumentReader:
    """OpenRouter适配的文档阅读系统"""
    
    def __init__(self):
        """初始化文档阅读系统"""
        logger.info("初始化OpenRouter文档阅读系统...")
        
        # 验证环境配置
        self._validate_config()
        
        self.loader = UniversalDocumentLoader()
        
        # 使用SAT智能分块器
        self.splitter = SATTextSplitter(
            target_chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            min_chunk_size=200,
            max_chunk_size=2000,
            model_name=os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
        )
        
        # 使用本地向量存储（避免OpenRouter的embedding限制）
        self.vector_manager = LocalVectorStore(
            embedding_model="all-MiniLM-L6-v2"  # 使用本地模型
        )
        
        self.rag_chain: Optional[OpenRouterRAGChain] = None
        self.current_doc_path: Optional[str] = None
        
        logger.info("系统初始化完成")
    
    def _validate_config(self):
        """验证配置"""
        required_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE"]
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"缺少必要的环境变量: {', '.join(missing)}")
        
        logger.info(f"API Base: {os.getenv('OPENAI_API_BASE')}")
        logger.info(f"Chat Model: {os.getenv('CHAT_MODEL', 'openai/gpt-4o-mini')}")
    
    def load_document(self, file_path: str, use_cache: bool = True):
        """加载并处理文档"""
        logger.info(f"正在加载文档: {file_path}")
        self.current_doc_path = file_path
        
        # 生成缓存路径
        file_hash = Path(file_path).stem
        cache_path = f"output/cache/{file_hash}_vectorstore"
        
        # 尝试从缓存加载
        if use_cache and os.path.exists(cache_path):
            logger.info("发现缓存的向量存储，正在加载...")
            try:
                vectorstore = self.vector_manager.load_vectorstore(cache_path)
                
                # 初始化RAG链
                self.rag_chain = OpenRouterRAGChain(
                    vectorstore=vectorstore,
                    model_name=os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
                )
                
                logger.info("从缓存加载成功")
                return {
                    "file_path": file_path,
                    "from_cache": True,
                    "status": "success"
                }
            except Exception as e:
                logger.warning(f"加载缓存失败: {str(e)}，将重新处理文档")
        
        # 1. 加载文档
        documents = self.loader.load(file_path)
        logger.info(f"文档加载完成，共{len(documents)}页")
        
        # 2. 使用SAT智能分块
        logger.info("开始SAT智能分块...")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"分块完成，共{len(chunks)}个语义块")
        
        # 3. 创建本地向量存储
        logger.info("创建向量存储...")
        vectorstore = self.vector_manager.create_vectorstore(chunks)
        
        # 保存到缓存
        if use_cache:
            self.vector_manager.save_vectorstore(cache_path)
            logger.info(f"向量存储已缓存")
        
        # 4. 初始化RAG链
        self.rag_chain = OpenRouterRAGChain(
            vectorstore=vectorstore,
            model_name=os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
        )
        
        return {
            "file_path": file_path,
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "from_cache": False,
            "status": "success"
        }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """回答关于文档的问题"""
        if self.rag_chain is None:
            raise ValueError("请先加载文档")
        
        logger.info(f"处理问题: {question}")
        return self.rag_chain.ask(question)
    
    def interactive_mode(self):
        """交互式问答模式"""
        print("\n=== OpenRouter文档阅读系统 - 交互模式 ===")
        print("输入 'quit' 退出")
        print("输入 'help' 查看帮助")
        print("输入 'info' 查看文档信息\n")
        
        while True:
            try:
                user_input = input("\n请输入您的问题: ").strip()
                
                if user_input.lower() == 'quit':
                    print("感谢使用，再见！")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n帮助信息：")
                    print("- 直接输入问题进行查询")
                    print("- 输入 'info' 查看当前文档信息")
                    print("- 输入 'quit' 退出程序")
                
                elif user_input.lower() == 'info':
                    self._display_document_info()
                
                elif user_input:
                    result = self.ask_question(user_input)
                    self._display_answer(result)
                
            except KeyboardInterrupt:
                print("\n\n程序被中断")
                break
            except Exception as e:
                logger.error(f"错误: {str(e)}")
                print(f"\n发生错误: {str(e)}")
    
    def _display_document_info(self):
        """显示文档信息"""
        if self.vector_manager.vectorstore is None:
            print("尚未加载文档")
            return
        
        print("\n=== 文档信息 ===")
        print(f"文件路径: {self.current_doc_path}")
        print(f"向量存储文档数: {len(self.vector_manager.vectorstore.docstore._dict)}")
        print(f"嵌入模型: {self.vector_manager.embedding_manager.model_name}")
        print(f"聊天模型: {os.getenv('CHAT_MODEL', 'openai/gpt-4o-mini')}")
        print("===============\n")
    
    def _display_answer(self, result: Dict[str, Any]):
        """格式化显示答案"""
        print("\n" + "="*60)
        
        if result['status'] == 'error':
            print("【错误】")
            print(result['answer'])
        else:
            print("【回答】")
            print(result['answer'])
            
            if result.get('sources'):
                print("\n【参考来源】")
                for i, source in enumerate(result['sources']):
                    print(f"\n{i+1}. {source['content']}")
                    if 'main_heading' in source['metadata']:
                        print(f"   来自章节: {source['metadata']['main_heading']}")
        
        print("="*60)


def check_dependencies():
    """检查必要的依赖"""
    missing = []
    
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import faiss
    except ImportError:
        try:
            import faiss_cpu
        except ImportError:
            missing.append("faiss-cpu")
    
    if missing:
        print(f"错误: 缺少必要的依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)} 或运行 python install_requirements.py")
        return False
    
    return True


def main():
    """主函数"""
    print("=== OpenRouter文档阅读系统 (基础模式) ===\n")
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 创建必要的目录
    os.makedirs("output/cache", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)
    os.makedirs("data/documents", exist_ok=True)
    
    try:
        # 创建文档阅读器
        reader = OpenRouterDocumentReader()
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        print(f"\n初始化失败: {str(e)}")
        return
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return
        
        # 加载文档
        try:
            # 支持--no-cache参数
            use_cache = "--no-cache" not in sys.argv
            
            info = reader.load_document(file_path, use_cache=use_cache)
            
            print(f"\n文档加载成功！")
            if info.get('from_cache'):
                print("（从缓存加载）")
            else:
                print(f"- 总页数: {info.get('total_pages', 'N/A')}")
                print(f"- 语义块: {info.get('total_chunks', 'N/A')}")
            
            # 进入交互模式
            reader.interactive_mode()
            
        except Exception as e:
            logger.error(f"加载文档失败: {str(e)}")
            print(f"\n错误: {str(e)}")
    
    else:
        print("使用方法: python main.py <文档路径> [--no-cache]")
        print("示例: python main.py data/documents/test.pdf")
        print("\n请将您的PDF或TXT文件放在 data/documents/ 目录下")


if __name__ == "__main__":
    main()
