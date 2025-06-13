# src/huggingface_hub_patch.py
"""
猴子补丁，用于解决huggingface_hub和sentence-transformers在某些环境下的兼容性问题。
该补丁通过在程序早期设置环境变量来避免隐式token检查引发的潜在问题。
"""
import os
import logging

logger = logging.getLogger(__name__)

def apply_patch():
    """应用兼容性补丁"""
    try:
        # 禁用huggingface_hub的隐式token检查，这是某些版本冲突的常见原因
        if "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in os.environ:
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            logger.debug("HuggingFace Hub patch applied: HF_HUB_DISABLE_IMPLICIT_TOKEN set to 1.")
        
        # 禁用遥测数据，有时也能避免不必要的网络请求和潜在问题
        if "HF_HUB_DISABLE_TELEMETRY" not in os.environ:
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            logger.debug("HuggingFace Hub patch applied: HF_HUB_DISABLE_TELEMETRY set to 1.")
            
    except Exception as e:
        logger.warning(f"Failed to apply HuggingFace Hub patch: {e}", exc_info=True)

# 在模块导入时立即应用补丁
apply_patch()
