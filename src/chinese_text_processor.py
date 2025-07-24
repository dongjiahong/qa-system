"""
Chinese Text Processing Optimization Module
中文文本处理优化模块
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from loguru import logger

from .models import FileProcessingError
from .config import get_config


@dataclass
class ChineseTextStats:
    """中文文本统计信息"""
    total_chars: int
    chinese_chars: int
    english_chars: int
    punctuation_chars: int
    chinese_ratio: float
    sentences: int
    paragraphs: int


class ChineseTextProcessor:
    """
    中文文本处理优化器
    
    专门针对中文文本的分词、清理和优化处理
    """
    
    def __init__(self):
        """初始化中文文本处理器"""
        self.config = get_config()
        
        # 中文字符范围
        self.chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
        
        # 中文标点符号
        self.chinese_punctuation = '，。！？；：""''（）【】《》、'
        
        # 英文标点符号
        self.english_punctuation = ',.!?;:"\'()[]{}/<>-'
        
        # 句子分割模式
        self.sentence_pattern = re.compile(r'[。！？\.\!\?]+')
        
        # 段落分割模式
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        # 无意义字符模式
        self.noise_pattern = re.compile(r'[^\w\s\u4e00-\u9fff' + 
                                      re.escape(self.chinese_punctuation) + 
                                      re.escape(self.english_punctuation) + ']')
        
        logger.info("ChineseTextProcessor initialized")
    
    def analyze_text(self, text: str) -> ChineseTextStats:
        """
        分析文本的中文特征
        
        Args:
            text: 输入文本
            
        Returns:
            ChineseTextStats: 文本统计信息
        """
        if not text:
            return ChineseTextStats(0, 0, 0, 0, 0.0, 0, 0)
        
        # 统计字符类型
        chinese_chars = len(self.chinese_char_pattern.findall(text))
        total_chars = len(text.strip())
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        punctuation_chars = len(re.findall(r'[' + re.escape(self.chinese_punctuation + self.english_punctuation) + ']', text))
        
        # 计算中文比例
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0.0
        
        # 统计句子和段落
        sentences = len(self.sentence_pattern.split(text)) - 1
        paragraphs = len(self.paragraph_pattern.split(text))
        
        return ChineseTextStats(
            total_chars=total_chars,
            chinese_chars=chinese_chars,
            english_chars=english_chars,
            punctuation_chars=punctuation_chars,
            chinese_ratio=chinese_ratio,
            sentences=max(1, sentences),
            paragraphs=max(1, paragraphs)
        )
    
    def optimize_chinese_text(self, text: str) -> str:
        """
        优化中文文本处理
        
        Args:
            text: 原始文本
            
        Returns:
            str: 优化后的文本
        """
        if not text:
            return ""
        
        # 1. 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 2. 处理中文标点符号
        text = self._normalize_chinese_punctuation(text)
        
        # 3. 移除噪声字符
        text = self.noise_pattern.sub(' ', text)
        
        # 4. 优化中英文混排
        text = self._optimize_mixed_text(text)
        
        # 5. 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _normalize_chinese_punctuation(self, text: str) -> str:
        """标准化中文标点符号"""
        # 统一引号
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        # 统一省略号
        text = re.sub(r'…+', '...', text)
        
        # 统一破折号
        text = re.sub(r'——+', '——', text)
        
        return text
    
    def _optimize_mixed_text(self, text: str) -> str:
        """优化中英文混排文本"""
        # 在中文和英文之间添加适当的空格
        text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)
        
        # 在中文和数字之间添加空格
        text = re.sub(r'([\u4e00-\u9fff])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([\u4e00-\u9fff])', r'\1 \2', text)
        
        return text
    
    def smart_chunk_chinese_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> List[str]:
        """
        智能中文文本分块
        
        Args:
            text: 原始文本
            chunk_size: 块大小
            overlap: 重叠大小
            
        Returns:
            List[str]: 文本块列表
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 如果不是最后一块，尝试在合适的位置分割
            if end < len(text):
                # 优先在句号处分割
                sentence_end = self._find_best_split_point(text, start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个开始位置，考虑重叠
            start = max(start + 1, end - overlap)
            
            # 避免无限循环
            if start >= len(text):
                break
        
        return chunks
    
    def _find_best_split_point(self, text: str, start: int, end: int) -> int:
        """找到最佳分割点"""
        # 在指定范围内寻找最佳分割点
        search_range = text[start:end]
        
        # 优先级：句号 > 问号感叹号 > 分号冒号 > 逗号 > 空格
        split_patterns = [
            (r'[。！？]', 1),  # 句子结束符
            (r'[；：]', 1),    # 分号冒号
            (r'[，、]', 1),    # 逗号顿号
            (r'\s', 0),       # 空格
        ]
        
        for pattern, offset in split_patterns:
            matches = list(re.finditer(pattern, search_range))
            if matches:
                # 选择最后一个匹配
                last_match = matches[-1]
                return start + last_match.end() + offset
        
        # 如果没有找到合适的分割点，返回原始结束位置
        return end
    
    def batch_process_texts(
        self, 
        texts: List[str], 
        max_workers: int = 4
    ) -> List[str]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            max_workers: 最大工作线程数
            
        Returns:
            List[str]: 处理后的文本列表
        """
        if not texts:
            return []
        
        logger.info(f"Batch processing {len(texts)} texts with {max_workers} workers")
        
        processed_texts = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.optimize_chinese_text, text): i 
                for i, text in enumerate(texts)
            }
            
            # 收集结果，保持原始顺序
            results = [None] * len(texts)
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Failed to process text {index}: {str(e)}")
                    results[index] = texts[index]  # 使用原始文本作为备用
        
        # 过滤掉None值
        processed_texts = [text for text in results if text is not None]
        
        logger.info(f"Successfully processed {len(processed_texts)} texts")
        return processed_texts
    
    async def async_process_texts(self, texts: List[str]) -> List[str]:
        """
        异步处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            List[str]: 处理后的文本列表
        """
        if not texts:
            return []
        
        logger.info(f"Async processing {len(texts)} texts")
        
        async def process_single_text(text: str) -> str:
            """异步处理单个文本"""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.optimize_chinese_text, text)
        
        # 创建异步任务
        tasks = [process_single_text(text) for text in texts]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        processed_texts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process text {i}: {str(result)}")
                processed_texts.append(texts[i])  # 使用原始文本作为备用
            else:
                processed_texts.append(result)
        
        logger.info(f"Successfully async processed {len(processed_texts)} texts")
        return processed_texts
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        提取中文关键词
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            
        Returns:
            List[str]: 关键词列表
        """
        if not text:
            return []
        
        # 简单的关键词提取（基于词频）
        # 移除标点符号
        clean_text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 分割成词（简单按空格和长度分割）
        words = []
        for word in clean_text.split():
            if len(word) >= 2:  # 至少2个字符
                # 对于中文，按2-4字符长度分割
                if self.chinese_char_pattern.search(word):
                    for i in range(len(word) - 1):
                        for length in [2, 3, 4]:
                            if i + length <= len(word):
                                candidate = word[i:i + length]
                                if len(self.chinese_char_pattern.findall(candidate)) >= length // 2:
                                    words.append(candidate)
                else:
                    words.append(word)
        
        # 统计词频
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序并返回前N个
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords] if freq > 1]
        
        return keywords
    
    def measure_processing_time(self, text: str) -> Tuple[str, float]:
        """
        测量文本处理时间
        
        Args:
            text: 输入文本
            
        Returns:
            Tuple[str, float]: (处理后的文本, 处理时间)
        """
        start_time = time.time()
        processed_text = self.optimize_chinese_text(text)
        processing_time = time.time() - start_time
        
        return processed_text, processing_time
    
    def get_processing_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        获取批量处理统计信息
        
        Args:
            texts: 文本列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not texts:
            return {
                "total_texts": 0,
                "total_chars": 0,
                "avg_chinese_ratio": 0.0,
                "processing_time": 0.0,
                "chars_per_second": 0.0
            }
        
        start_time = time.time()
        
        total_chars = 0
        total_chinese_chars = 0
        
        for text in texts:
            stats = self.analyze_text(text)
            total_chars += stats.total_chars
            total_chinese_chars += stats.chinese_chars
        
        processing_time = time.time() - start_time
        avg_chinese_ratio = total_chinese_chars / total_chars if total_chars > 0 else 0.0
        chars_per_second = total_chars / processing_time if processing_time > 0 else 0.0
        
        return {
            "total_texts": len(texts),
            "total_chars": total_chars,
            "avg_chinese_ratio": avg_chinese_ratio,
            "processing_time": processing_time,
            "chars_per_second": chars_per_second
        }


# 全局中文文本处理器实例
_chinese_text_processor: Optional[ChineseTextProcessor] = None


def get_chinese_text_processor() -> ChineseTextProcessor:
    """
    获取全局中文文本处理器实例
    
    Returns:
        ChineseTextProcessor: 中文文本处理器实例
    """
    global _chinese_text_processor
    
    if _chinese_text_processor is None:
        _chinese_text_processor = ChineseTextProcessor()
    
    return _chinese_text_processor


def reset_chinese_text_processor() -> None:
    """重置全局中文文本处理器实例（主要用于测试）"""
    global _chinese_text_processor
    _chinese_text_processor = None