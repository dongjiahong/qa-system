"""
Metadata Extractors for Knowledge QA System
元数据抽取器模块，用于从文档中提取结构化信息以提升问答效果
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from loguru import logger
from llama_index.core import Document

from .models import KnowledgeSystemError
from .llm_client import OllamaClient, get_ollama_client
from .chinese_text_processor import get_chinese_text_processor


class ExtractorType(Enum):
    """元数据抽取器类型"""
    QUESTIONS_ANSWERED = "questions_answered"
    KEY_CONCEPTS = "key_concepts"
    TOPICS = "topics"
    ENTITIES = "entities"
    RELATIONSHIPS = "relationships"
    DIFFICULTY_LEVEL = "difficulty_level"


@dataclass
class ExtractedMetadata:
    """抽取的元数据"""
    extractor_type: ExtractorType
    content: Dict[str, Any]
    confidence: float
    source_text: str


class BaseMetadataExtractor(ABC):
    """元数据抽取器基类"""
    
    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or get_ollama_client()
        self.chinese_processor = get_chinese_text_processor()
    
    @abstractmethod
    def extract(self, document: Document) -> ExtractedMetadata:
        """抽取元数据"""
        pass
    
    @abstractmethod
    def get_extractor_type(self) -> ExtractorType:
        """获取抽取器类型"""
        pass


class QuestionsAnsweredExtractor(BaseMetadataExtractor):
    """问答对抽取器 - 从文档中识别已回答的问题"""
    
    def __init__(self, llm_client: Optional[OllamaClient] = None):
        super().__init__(llm_client)
        
        # 问题模式匹配
        self.question_patterns = [
            r'问题?\s*[:：]\s*(.+?)[？?]',
            r'Q\s*[:：]\s*(.+?)[？?]',
            r'疑问\s*[:：]\s*(.+?)[？?]',
            r'(.+?)[？?](?=\s*答案?[:：])',
            r'(.+?)[？?](?=\s*A\s*[:：])',
            r'(.+?)[？?](?=\s*解答[:：])',
        ]
        
        # 答案模式匹配
        self.answer_patterns = [
            r'答案?\s*[:：]\s*(.+?)(?=\n\n|\n问题?[:：]|\n疑问[:：]|\nQ\s*[:：]|$)',
            r'A\s*[:：]\s*(.+?)(?=\n\n|\n问题?[:：]|\n疑问[:：]|\nQ\s*[:：]|$)',
            r'解答\s*[:：]\s*(.+?)(?=\n\n|\n问题?[:：]|\n疑问[:：]|\nQ\s*[:：]|$)',
        ]
    
    def extract(self, document: Document) -> ExtractedMetadata:
        """从文档中抽取问答对"""
        try:
            text = document.text
            qa_pairs = []
            
            # 1. 使用正则表达式匹配明显的问答格式
            regex_pairs = self._extract_with_regex(text)
            qa_pairs.extend(regex_pairs)
            
            # 2. 使用LLM识别隐含的问答关系
            if len(qa_pairs) < 3:  # 如果正则匹配结果较少，使用LLM补充
                llm_pairs = self._extract_with_llm(text)
                qa_pairs.extend(llm_pairs)
            
            # 3. 去重和质量过滤
            qa_pairs = self._filter_and_deduplicate(qa_pairs)
            
            # 4. 计算置信度
            confidence = self._calculate_confidence(qa_pairs, text)
            
            logger.info(f"Extracted {len(qa_pairs)} Q&A pairs with confidence {confidence:.2f}")
            
            return ExtractedMetadata(
                extractor_type=ExtractorType.QUESTIONS_ANSWERED,
                content={
                    "qa_pairs": qa_pairs,
                    "total_pairs": len(qa_pairs),
                    "extraction_methods": ["regex", "llm"] if len(qa_pairs) > len(regex_pairs) else ["regex"]
                },
                confidence=confidence,
                source_text=text[:500] + "..." if len(text) > 500 else text
            )
            
        except Exception as e:
            logger.error(f"QuestionsAnsweredExtractor failed: {str(e)}")
            return ExtractedMetadata(
                extractor_type=ExtractorType.QUESTIONS_ANSWERED,
                content={"qa_pairs": [], "error": str(e)},
                confidence=0.0,
                source_text=document.text[:200] + "..."
            )
    
    def _extract_with_regex(self, text: str) -> List[Dict[str, str]]:
        """使用正则表达式抽取问答对"""
        qa_pairs = []
        
        # 尝试不同的问答格式模式
        for q_pattern in self.question_patterns:
            for a_pattern in self.answer_patterns:
                matches = self._find_qa_matches(text, q_pattern, a_pattern)
                qa_pairs.extend(matches)
        
        return qa_pairs
    
    def _find_qa_matches(self, text: str, q_pattern: str, a_pattern: str) -> List[Dict[str, str]]:
        """查找问答匹配"""
        qa_pairs = []
        
        # 查找问题
        questions = re.finditer(q_pattern, text, re.MULTILINE | re.DOTALL)
        
        for q_match in questions:
            question = q_match.group(1).strip()
            
            # 在问题后查找答案
            search_start = q_match.end()
            remaining_text = text[search_start:search_start + 1000]  # 限制搜索范围
            
            answer_match = re.search(a_pattern, remaining_text, re.MULTILINE | re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
                
                if self._is_valid_qa_pair(question, answer):
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "extraction_method": "regex",
                        "confidence": 0.8
                    })
        
        return qa_pairs
    
    def _extract_with_llm(self, text: str) -> List[Dict[str, str]]:
        """使用LLM抽取隐含的问答关系"""
        try:
            # 限制文本长度
            if len(text) > 2000:
                text = text[:2000] + "..."
            
            prompt = f"""分析以下文本，识别其中包含的问题和对应答案。即使问题和答案不是明确的问答格式，也要识别出文本中回答了哪些问题。

文本内容：
{text}

请按照以下JSON格式返回结果：
{{
    "qa_pairs": [
        {{
            "question": "识别出的问题",
            "answer": "对应的答案",
            "confidence": 0.7
        }}
    ]
}}

要求：
1. 问题应该是具体的、可回答的
2. 答案应该基于文本内容
3. 每个问答对都要有置信度评分(0-1)
4. 最多返回5个问答对
5. 确保返回有效的JSON格式"""

            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500
            )
            
            # 解析LLM响应
            response_text = response.response.strip()
            qa_data = self._parse_llm_response(response_text)
            
            if qa_data and "qa_pairs" in qa_data:
                pairs = qa_data["qa_pairs"]
                for pair in pairs:
                    pair["extraction_method"] = "llm"
                return pairs
            
            return []
            
        except Exception as e:
            logger.warning(f"LLM extraction failed: {str(e)}")
            return []
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """解析LLM响应"""
        try:
            # 查找JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                return json.loads(json_str)
            
            return None
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return None
    
    def _is_valid_qa_pair(self, question: str, answer: str) -> bool:
        """验证问答对的有效性"""
        # 基本长度检查
        if len(question.strip()) < 5 or len(answer.strip()) < 10:
            return False
        
        # 问题必须包含疑问词或问号
        question_indicators = ['什么', '如何', '为什么', '怎样', '哪些', '谁', '何时', '何地', '？', '?']
        if not any(indicator in question for indicator in question_indicators):
            return False
        
        # 答案不应该是另一个问题
        if '？' in answer or '?' in answer:
            return False
        
        # 避免重复内容
        if question.lower() == answer.lower():
            return False
        
        return True
    
    def _filter_and_deduplicate(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """过滤和去重问答对"""
        if not qa_pairs:
            return []
        
        # 去重
        seen_questions = set()
        unique_pairs = []
        
        for pair in qa_pairs:
            question_key = pair["question"].lower().strip()
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                unique_pairs.append(pair)
        
        # 按置信度排序
        unique_pairs.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
        
        # 限制数量
        return unique_pairs[:10]
    
    def _calculate_confidence(self, qa_pairs: List[Dict[str, str]], text: str) -> float:
        """计算整体置信度"""
        if not qa_pairs:
            return 0.0
        
        # 基于问答对数量和质量计算置信度
        pair_confidences = [pair.get("confidence", 0.5) for pair in qa_pairs]
        avg_confidence = sum(pair_confidences) / len(pair_confidences)
        
        # 考虑问答对数量的影响
        quantity_factor = min(len(qa_pairs) / 5.0, 1.0)  # 5个问答对为满分
        
        # 考虑文本长度的影响
        text_length_factor = min(len(text) / 1000.0, 1.0)  # 1000字符为满分
        
        final_confidence = avg_confidence * 0.6 + quantity_factor * 0.3 + text_length_factor * 0.1
        
        return min(final_confidence, 1.0)
    
    def get_extractor_type(self) -> ExtractorType:
        return ExtractorType.QUESTIONS_ANSWERED


class KeyConceptsExtractor(BaseMetadataExtractor):
    """关键概念抽取器"""
    
    def extract(self, document: Document) -> ExtractedMetadata:
        """抽取关键概念"""
        try:
            text = document.text
            
            # 使用LLM抽取关键概念
            concepts = self._extract_concepts_with_llm(text)
            
            # 计算置信度
            confidence = self._calculate_concept_confidence(concepts, text)
            
            return ExtractedMetadata(
                extractor_type=ExtractorType.KEY_CONCEPTS,
                content={
                    "concepts": concepts,
                    "total_concepts": len(concepts)
                },
                confidence=confidence,
                source_text=text[:300] + "..."
            )
            
        except Exception as e:
            logger.error(f"KeyConceptsExtractor failed: {str(e)}")
            return ExtractedMetadata(
                extractor_type=ExtractorType.KEY_CONCEPTS,
                content={"concepts": [], "error": str(e)},
                confidence=0.0,
                source_text=document.text[:200] + "..."
            )
    
    def _extract_concepts_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """使用LLM抽取关键概念"""
        try:
            if len(text) > 1500:
                text = text[:1500] + "..."
            
            prompt = f"""分析以下文本，提取其中的关键概念。关键概念应该是重要的术语、定义、原理或核心思想。

文本内容：
{text}

请按照以下JSON格式返回结果：
{{
    "concepts": [
        {{
            "name": "概念名称",
            "definition": "概念定义或描述",
            "importance": 0.8,
            "category": "概念类别"
        }}
    ]
}}

要求：
1. 提取5-10个最重要的概念
2. 每个概念都要有清晰的定义
3. importance评分范围0-1
4. category可以是：定义、原理、方法、工具、理论等
5. 确保返回有效的JSON格式"""

            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1000
            )
            
            response_text = response.response.strip()
            concept_data = self._parse_llm_response(response_text)
            
            if concept_data and "concepts" in concept_data:
                return concept_data["concepts"]
            
            return []
            
        except Exception as e:
            logger.warning(f"Concept extraction failed: {str(e)}")
            return []
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """解析LLM响应"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                return json.loads(json_str)
            
            return None
            
        except json.JSONDecodeError:
            return None
    
    def _calculate_concept_confidence(self, concepts: List[Dict], text: str) -> float:
        """计算概念抽取置信度"""
        if not concepts:
            return 0.0
        
        # 基于概念数量和重要性计算
        importance_scores = [c.get("importance", 0.5) for c in concepts]
        avg_importance = sum(importance_scores) / len(importance_scores)
        
        quantity_factor = min(len(concepts) / 8.0, 1.0)
        
        return avg_importance * 0.7 + quantity_factor * 0.3
    
    def get_extractor_type(self) -> ExtractorType:
        return ExtractorType.KEY_CONCEPTS


class MetadataExtractorManager:
    """元数据抽取器管理器"""
    
    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm_client = llm_client or get_ollama_client()
        
        # 注册抽取器
        self.extractors = {
            ExtractorType.QUESTIONS_ANSWERED: QuestionsAnsweredExtractor(self.llm_client),
            ExtractorType.KEY_CONCEPTS: KeyConceptsExtractor(self.llm_client),
        }
        
        logger.info(f"MetadataExtractorManager initialized with {len(self.extractors)} extractors")
    
    def extract_all_metadata(self, document: Document) -> Dict[ExtractorType, ExtractedMetadata]:
        """抽取所有类型的元数据"""
        results = {}
        
        for extractor_type, extractor in self.extractors.items():
            try:
                logger.debug(f"Running {extractor_type.value} extractor")
                metadata = extractor.extract(document)
                results[extractor_type] = metadata
            except Exception as e:
                logger.error(f"Extractor {extractor_type.value} failed: {str(e)}")
                results[extractor_type] = ExtractedMetadata(
                    extractor_type=extractor_type,
                    content={"error": str(e)},
                    confidence=0.0,
                    source_text=""
                )
        
        return results
    
    def extract_specific_metadata(
        self, 
        document: Document, 
        extractor_types: List[ExtractorType]
    ) -> Dict[ExtractorType, ExtractedMetadata]:
        """抽取特定类型的元数据"""
        results = {}
        
        for extractor_type in extractor_types:
            if extractor_type in self.extractors:
                try:
                    extractor = self.extractors[extractor_type]
                    metadata = extractor.extract(document)
                    results[extractor_type] = metadata
                except Exception as e:
                    logger.error(f"Extractor {extractor_type.value} failed: {str(e)}")
                    results[extractor_type] = ExtractedMetadata(
                        extractor_type=extractor_type,
                        content={"error": str(e)},
                        confidence=0.0,
                        source_text=""
                    )
            else:
                logger.warning(f"Extractor type {extractor_type.value} not registered")
        
        return results
    
    def get_available_extractors(self) -> List[ExtractorType]:
        """获取可用的抽取器类型"""
        return list(self.extractors.keys())
    
    def register_extractor(self, extractor: BaseMetadataExtractor) -> None:
        """注册新的抽取器"""
        extractor_type = extractor.get_extractor_type()
        self.extractors[extractor_type] = extractor
        logger.info(f"Registered extractor: {extractor_type.value}")
    
    def get_extraction_summary(
        self, 
        results: Dict[ExtractorType, ExtractedMetadata]
    ) -> Dict[str, Any]:
        """获取抽取结果摘要"""
        summary = {
            "total_extractors": len(results),
            "successful_extractions": 0,
            "failed_extractions": 0,
            "average_confidence": 0.0,
            "extractor_results": {}
        }
        
        confidences = []
        
        for extractor_type, metadata in results.items():
            is_successful = "error" not in metadata.content
            
            if is_successful:
                summary["successful_extractions"] += 1
                confidences.append(metadata.confidence)
            else:
                summary["failed_extractions"] += 1
            
            summary["extractor_results"][extractor_type.value] = {
                "success": is_successful,
                "confidence": metadata.confidence,
                "content_keys": list(metadata.content.keys())
            }
        
        if confidences:
            summary["average_confidence"] = sum(confidences) / len(confidences)
        
        return summary


# 全局管理器实例
_metadata_manager: Optional[MetadataExtractorManager] = None


def get_metadata_extractor_manager() -> MetadataExtractorManager:
    """获取全局元数据抽取器管理器实例"""
    global _metadata_manager
    
    if _metadata_manager is None:
        _metadata_manager = MetadataExtractorManager()
    
    return _metadata_manager


def reset_metadata_extractor_manager() -> None:
    """重置全局管理器实例（主要用于测试）"""
    global _metadata_manager
    _metadata_manager = None