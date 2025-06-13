#!/usr/bin/env python3
"""智能安装脚本 - 自动解决所有兼容性问题并配置项目"""

import os
import sys
import subprocess
import shutil
import traceback  # 添加缺失的导入
from pathlib import Path

class SmartInstaller:
    def __init__(self):
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.project_root = Path(__file__).parent.resolve()
        
    def run(self):
        """执行完整的智能安装和配置流程"""
        print(f"🚀 智能文档阅读系统安装程序")
        print(f"📌 Python版本: {self.python_version}")
        print("="*50)
        
        self.setup_project_structure()
        self.install_dependencies()
        self.setup_compatibility_layer()
        self.create_env_template()
        self.verify_installation()
        
        print("\n✅ 安装与配置完成！")
        print("\n下一步：")
        print("1. 编辑 .env 文件，填入您的OpenRouter API密钥。")
        print("2. 将您的文档放入 'data/documents' 目录。")
        print("3. 运行增强模式: python main_enhanced.py data/documents/your_file.pdf")
        print("4. 或运行基础模式: python main.py data/documents/your_file.pdf")

    def setup_project_structure(self):
        """创建项目所需的所有目录"""
        print("\n📁 正在创建项目目录结构...")
        dirs = [
            "data/documents",
            "output/cache",
            "output/logs",
            "src"
        ]
        for d in dirs:
            (self.project_root / d).mkdir(parents=True, exist_ok=True)
        print("✓ 目录结构准备就绪。")

    def install_dependencies(self):
        """安装所有必要的Python依赖"""
        print("\n📦 正在安装依赖库...")
        requirements_content = """# 核心框架
langchain==0.1.20
langchain-community==0.0.38
langchain-openai==0.1.6

# API客户端与环境
openai>=1.12.0
python-dotenv==1.0.1

# 文档处理
pypdf==4.2.0

# 本地嵌入 (高性能)
sentence-transformers==2.7.0
torch>=2.0.0
# huggingface-hub 会作为sentence-transformers的依赖自动安装

# 向量存储
faiss-cpu==1.8.0

# 工具库
numpy==1.26.4
tqdm==4.66.4

# 注意：不直接安装contextgem，使用内置的兼容层
"""
        req_file = self.project_root / 'requirements_smart.txt'
        req_file.write_text(requirements_content, encoding='utf-8')
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ])
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(req_file)
            ])
            print("✓ 依赖安装完成。")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            print("请检查您的Python环境和网络连接，然后重试。")
            sys.exit(1)
        finally:
            req_file.unlink() # 清理临时文件

    def setup_compatibility_layer(self):
        """设置ContextGem兼容层，从源文件复制"""
        print("\n🔧 正在配置ContextGem兼容层...")
        
        compat_dir = self.project_root / 'src/contextgem_compat'
        compat_dir.mkdir(exist_ok=True)
        
        # 1. 创建 __init__.py
        init_content = '''"""ContextGem兼容层 - 解决Python 3.12兼容性问题"""

from .core import (
    Document, DocumentLLM, JsonObjectConcept, JsonObjectExample,
    ExtractedItem, BaseConcept
)

__all__ = [
    "Document", "DocumentLLM", "JsonObjectConcept", "JsonObjectExample",
    "ExtractedItem", "BaseConcept"
]
'''
        (compat_dir / '__init__.py').write_text(init_content, encoding='utf-8')
        
        # 2. 从项目根目录复制 core.py 作为兼容层的核心
        source_core_py = self.project_root / 'core.py'
        target_core_py = compat_dir / 'core.py'
        
        if not source_core_py.exists():
            print(f"❌ 错误: 兼容层源文件 'core.py' 不在项目根目录中。")
            sys.exit(1)
            
        shutil.copy(str(source_core_py), str(target_core_py))
        print("✓ 兼容层配置完成。")

    def create_env_template(self):
        """创建.env文件模板（如果不存在）"""
        env_file = self.project_root / '.env'
        if not env_file.exists():
            print("\n📝 正在创建.env文件模板...")
            env_content = """# OpenRouter API配置 (必须)
# 前往 https://openrouter.ai 获取你的API密钥
OPENAI_API_KEY="your-openrouter-api-key-here"
OPENAI_API_BASE="https://openrouter.ai/api/v1"

# 模型配置 (可选)
# 你可以在OpenRouter上选择任何你喜欢的模型
CHAT_MODEL="openai/gpt-4o-mini"

# RAG配置 (可选)
CHUNK_SIZE=1000
"""
            env_file.write_text(env_content, encoding='utf-8')
            print("✓ .env文件已创建。请务必填入您的API密钥。")

    def verify_installation(self):
        """验证安装是否成功"""
        print("\n🔍 正在验证安装...")
        sys.path.insert(0, str(self.project_root / 'src'))
        try:
            print("  - 正在测试基础模块...")
            from document_loader import UniversalDocumentLoader
            from sat_text_splitter import SATTextSplitter
            from vectorstore_local import LocalVectorStore
            print("  ✓ 基础模块导入成功。")
            
            print("  - 正在测试嵌入模块...")
            from embedding_manager import LocalEmbeddingManager
            # 修复：使用正确的参数创建实例
            model_path = self.project_root / 'models' / 'all-MiniLM-L6-v2'
            manager = LocalEmbeddingManager(model_path=str(model_path))
            print("  ✓ 嵌入模块初始化成功。")

            print("  - 正在测试ContextGem兼容层...")
            from contextgem_compat import Document, JsonObjectConcept
            doc = Document(raw_text="测试")
            concept = JsonObjectConcept(name="测试", description="描述", structure={"key": str})
            print("  ✓ 兼容层导入和实例化成功。")
            
            print("  - 正在测试增强模式RAG链...")
            from enhanced_rag_chain import EnhancedRAGChain
            print("  ✓ 增强模式模块导入成功。")
            
            print("\n🎉 所有核心模块验证通过！")
            return True
        except Exception as e:
            print(f"\n❌ 验证失败: {e}")
            traceback.print_exc()
            print("请检查上面的错误信息。如果问题与依赖库有关，请尝试重新运行此安装脚本。")
            return False

if __name__ == "__main__":
    installer = SmartInstaller()
    installer.run()
