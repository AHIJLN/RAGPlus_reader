# src/sat_text_splitter.py
"""SAT (Semantic Adaptive Tiling) 智能文本分块模块 - 简化版"""

from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
import re
import logging
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SemanticSegment:
    """语义段落"""
    content: str
    start_idx: int
    end_idx: int
    segment_type: str
    importance_score: float
    metadata: Dict[str, Any]


class SATTextSplitter:
    """基于语义自适应分块的文本分割器（简化版，不依赖LLM）"""
    
    def __init__(self, 
                 target_chunk_size: int = 1000,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 2000,
                 model_name: str = None):  # 保留接口兼容性
        """
        初始化SAT分割器
        
        Args:
            target_chunk_size: 目标块大小
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
            model_name: 保留参数以保持兼容性
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # 结构标识符
        self.structure_patterns = {
            'heading1': r'^#{1}\s+.+$',  # # 一级标题
            'heading2': r'^#{2}\s+.+$',  # ## 二级标题
            'heading3': r'^#{3,6}\s+.+$',  # ### 三级及以下标题
            'numbered_heading': r'^\d+\.?\s+[A-Z].+$',  # 1. 标题
            'chinese_heading': r'^第[一二三四五六七八九十百千]+[章节条款]',  # 中文章节
            'list_item': r'^[-•*]\s+|^\d+\.\s+|^[a-z]\)\s+',  # 列表项
            'code_block': r'^```',  # 代码块
            'table_row': r'\|.+\|',  # 表格行
            'blank_line': r'^\s*$',  # 空行
        }
        
        # 重要性权重
        self.type_weights = {
            'heading1': 1.0,
            'heading2': 0.9,
            'heading3': 0.8,
            'numbered_heading': 0.85,
            'chinese_heading': 0.9,
            'list_item': 0.6,
            'code_block': 0.7,
            'table_row': 0.7,
            'paragraph': 0.5,
            'blank_line': 0.1
        }
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        使用SAT方法智能分割文档
        
        Args:
            documents: 原始文档列表
            
        Returns:
            List[Document]: 语义完整的文档块
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            logger.info(f"处理文档 {doc_idx + 1}/{len(documents)}")
            
            # 1. 识别文档结构
            segments = self._identify_semantic_segments(doc.page_content)
            
            # 2. 计算段落重要性
            segments = self._calculate_importance_scores(segments)
            
            # 3. 智能组合段落成块
            chunks = self._create_semantic_chunks(segments, doc.metadata)
            
            all_chunks.extend(chunks)
        
        logger.info(f"SAT分割完成: {len(documents)}个文档 -> {len(all_chunks)}个语义块")
        return all_chunks
    
    def _identify_semantic_segments(self, text: str) -> List[SemanticSegment]:
        """识别文本中的语义段落"""
        segments = []
        lines = text.split('\n')
        current_segment = []
        current_type = 'paragraph'
        start_idx = 0
        
        for i, line in enumerate(lines):
            line_type = self._classify_line(line)
            
            # 检测段落边界
            if self._is_segment_boundary(current_type, line_type, current_segment, line):
                # 保存当前段落
                if current_segment and any(l.strip() for l in current_segment):
                    content = '\n'.join(current_segment)
                    segments.append(SemanticSegment(
                        content=content,
                        start_idx=start_idx,
                        end_idx=i,
                        segment_type=current_type,
                        importance_score=0.5,
                        metadata={}
                    ))
                
                # 开始新段落
                current_segment = [line] if line.strip() or line_type == 'code_block' else []
                current_type = line_type if line.strip() else 'paragraph'
                start_idx = i
            else:
                current_segment.append(line)
        
        # 处理最后一个段落
        if current_segment and any(l.strip() for l in current_segment):
            content = '\n'.join(current_segment)
            segments.append(SemanticSegment(
                content=content,
                start_idx=start_idx,
                end_idx=len(lines),
                segment_type=current_type,
                importance_score=0.5,
                metadata={}
            ))
        
        return segments
    
    def _classify_line(self, line: str) -> str:
        """分类文本行的类型"""
        # 检查各种模式
        for pattern_type, pattern in self.structure_patterns.items():
            if re.match(pattern, line):
                return pattern_type
        
        # 默认为段落
        return 'paragraph' if line.strip() else 'blank_line'
    
    def _is_segment_boundary(self, current_type: str, new_type: str, 
                           current_segment: List[str], new_line: str) -> bool:
        """判断是否为段落边界"""
        # 空行通常表示段落结束
        if new_type == 'blank_line' and current_segment:
            return True
        
        # 标题总是开始新段落
        if new_type in ['heading1', 'heading2', 'heading3', 'numbered_heading', 'chinese_heading']:
            return True
        
        # 从普通段落到特殊结构的转换
        if current_type == 'paragraph' and new_type in ['list_item', 'code_block', 'table_row']:
            return True
        
        # 从特殊结构回到普通段落
        if current_type in ['list_item', 'table_row'] and new_type == 'paragraph' and new_line.strip():
            return True
        
        # 代码块的开始和结束
        if new_type == 'code_block':
            return True
        
        return False
    
    def _calculate_importance_scores(self, segments: List[SemanticSegment]) -> List[SemanticSegment]:
        """计算每个段落的重要性分数"""
        if not segments:
            return segments
        
        for i, segment in enumerate(segments):
            # 基础权重
            base_score = self.type_weights.get(segment.segment_type, 0.5)
            
            # 长度因素（较长的段落可能更重要）
            length_factor = min(len(segment.content) / 500, 1.0)
            
            # 位置因素（开头和结尾的段落更重要）
            position_factor = 1.0
            if i < 3:  # 前三个段落
                position_factor = 1.2
            elif i >= len(segments) - 3:  # 最后三个段落
                position_factor = 1.1
            
            # 内容密度（非空字符比例）
            non_empty_ratio = len(segment.content.strip()) / max(len(segment.content), 1)
            
            # 综合计算
            segment.importance_score = base_score * (0.5 + 0.3 * length_factor + 0.2 * non_empty_ratio) * position_factor
            
            # 存储额外信息
            segment.metadata['length'] = len(segment.content)
            segment.metadata['position'] = i
        
        return segments
    
    def _create_semantic_chunks(self, 
                               segments: List[SemanticSegment], 
                               doc_metadata: Dict) -> List[Document]:
        """基于语义关系创建文档块"""
        chunks = []
        current_chunk_segments = []
        current_size = 0
        
        for i, segment in enumerate(segments):
            segment_size = len(segment.content)
            
            # 决定是否开始新块
            should_split = self._should_split_here(
                current_chunk_segments, 
                segment, 
                current_size,
                segments[i+1] if i+1 < len(segments) else None
            )
            
            if should_split and current_chunk_segments:
                # 创建当前块
                chunk = self._create_chunk_from_segments(current_chunk_segments, doc_metadata)
                chunks.append(chunk)
                
                # 开始新块
                current_chunk_segments = []
                current_size = 0
            
            # 添加到当前块
            current_chunk_segments.append(segment)
            current_size += segment_size
        
        # 处理最后的段落
        if current_chunk_segments:
            chunk = self._create_chunk_from_segments(current_chunk_segments, doc_metadata)
            chunks.append(chunk)
        
        return chunks
    
    def _should_split_here(self,
                          current_segments: List[SemanticSegment],
                          new_segment: SemanticSegment,
                          current_size: int,
                          next_segment: Optional[SemanticSegment]) -> bool:
        """决定是否在此处分割"""
        new_size = current_size + len(new_segment.content)
        
        # 1. 硬性大小限制
        if new_size > self.max_chunk_size:
            return True
        
        # 2. 如果当前块太小，继续添加
        if current_size < self.min_chunk_size:
            return False
        
        # 3. 重要标题边界
        if (new_segment.segment_type in ['heading1', 'heading2'] and 
            new_segment.importance_score > 0.8 and
            current_size > self.min_chunk_size * 0.5):
            return True
        
        # 4. 内容类型变化
        if current_segments:
            last_type = current_segments[-1].segment_type
            # 从文本到代码/表格的转换
            if last_type in ['paragraph', 'list_item'] and new_segment.segment_type in ['code_block', 'table_row']:
                return current_size > self.min_chunk_size * 0.7
        
        # 5. 接近目标大小，且下一个是标题
        if (new_size > self.target_chunk_size * 0.9 and 
            next_segment and 
            next_segment.segment_type in ['heading1', 'heading2', 'heading3']):
            return True
        
        # 6. 超过目标大小
        if new_size > self.target_chunk_size:
            return True
        
        return False
    
    def _create_chunk_from_segments(self, 
                                   segments: List[SemanticSegment], 
                                   doc_metadata: Dict) -> Document:
        """从段落列表创建文档块"""
        # 组合内容
        content_parts = []
        for i, seg in enumerate(segments):
            # 在段落之间添加适当的分隔
            if i > 0 and segments[i-1].segment_type != 'blank_line':
                # 如果前一个段落不是空行，添加换行
                if seg.segment_type in ['heading1', 'heading2', 'heading3', 'paragraph']:
                    content_parts.append('')  # 添加空行
            content_parts.append(seg.content)
        
        content = '\n'.join(content_parts)
        
        # 构建元数据
        metadata = doc_metadata.copy()
        metadata.update({
            'chunk_type': 'semantic',
            'segment_types': list(set(seg.segment_type for seg in segments)),
            'importance_score': float(np.mean([seg.importance_score for seg in segments])),
            'start_segment': segments[0].start_idx,
            'end_segment': segments[-1].end_idx,
            'num_segments': len(segments),
            'chunk_size': len(content)
        })
        
        # 如果包含标题，记录主要标题
        headings = [seg for seg in segments if 'heading' in seg.segment_type]
        if headings:
            metadata['main_heading'] = headings[0].content.strip()
        
        return Document(
            page_content=content.strip(),
            metadata=metadata
        )