"""
Question Generator module for Knowledge QA System
问题生成器模块
"""

import random
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from loguru import logger

from .models import (
    Question, 
    QuestionDifficulty, 
    KnowledgeSystemError, 
    ModelServiceError,
    VectorStoreError,
    KnowledgeBaseNotFoundError
)
from .vector_store import VectorStore, SearchResult
from .llm_client import OllamaClient, get_ollama_client, clean_model_response
from .config import get_config


class ContentSelectionStrategy(Enum):
    """内容选择策略"""
    RANDOM = "random"  # 随机选择
    DIVERSE = "diverse"  # 多样化选择
    RECENT = "recent"  # 最近添加的内容
    COMPREHENSIVE = "comprehensive"  # 综合性内容


@dataclass
class QuestionGenerationContext:
    """问题生成上下文"""
    content: str
    source_metadata: Dict[str, Any]
    difficulty: QuestionDifficulty
    strategy: ContentSelectionStrategy


class QuestionQualityValidator:
    """问题质量验证器"""
    
    def __init__(self):
        self.config = get_config()
        
        # 问题质量检查规则
        self.min_length = 5  # 调整为更合理的最小长度
        self.max_length = 500
        self.required_patterns = [
            r'[？?]',  # 必须包含问号
        ]
        self.forbidden_patterns = [
            r'^(请|请问)',  # 不应以"请"开头
            r'(以下|下面).*选择',  # 避免选择题格式
            r'[ABCD][\.\)、]',  # 避免选项格式
        ]
        self.quality_keywords = [
            '什么', '如何', '为什么', '怎样', '哪些', '谁', '何时', '何地',
            '解释', '描述', '分析', '比较', '评价', '讨论'
        ]
    
    def validate_question(self, question: str) -> Tuple[bool, List[str]]:
        """
        验证问题质量
        
        Args:
            question: 问题文本
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 问题列表)
        """
        issues = []
        
        # 基本长度检查
        if len(question.strip()) < self.min_length:
            issues.append(f"问题过短，至少需要{self.min_length}个字符")
        
        if len(question.strip()) > self.max_length:
            issues.append(f"问题过长，不能超过{self.max_length}个字符")
        
        # 必需模式检查
        for pattern in self.required_patterns:
            if not re.search(pattern, question):
                issues.append("问题必须包含问号")
        
        # 禁止模式检查
        for pattern in self.forbidden_patterns:
            if re.search(pattern, question):
                issues.append("问题格式不符合要求")
        
        # 质量关键词检查
        has_quality_keyword = any(keyword in question for keyword in self.quality_keywords)
        if not has_quality_keyword:
            issues.append("问题应包含疑问词或引导词")
        
        # 检查是否为空或只有标点
        if not question.strip() or question.strip() in ['？', '?']:
            issues.append("问题内容为空")
        
        return len(issues) == 0, issues


class QuestionGenerator:
    """
    问题生成器
    
    基于知识库内容生成学习问题，支持多种内容选择策略和难度级别
    """
    
    def __init__(
        self, 
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[OllamaClient] = None
    ):
        """
        初始化问题生成器
        
        Args:
            vector_store: 向量存储实例
            llm_client: LLM客户端实例
        """
        self.config = get_config()
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client or get_ollama_client()
        self.validator = QuestionQualityValidator()
        
        # 问题历史记录，用于去重
        self.question_history: Dict[str, set] = {}  # kb_name -> set of content_hash
        
        logger.info("QuestionGenerator initialized")
    
    def generate_question(
        self, 
        kb_name: str,
        difficulty: QuestionDifficulty = QuestionDifficulty.EASY,
        strategy: ContentSelectionStrategy = ContentSelectionStrategy.RANDOM,
        max_retries: int = 3,
        allow_duplicates: bool = False
    ) -> Question:
        """
        生成问题
        
        Args:
            kb_name: 知识库名称
            difficulty: 问题难度
            strategy: 内容选择策略
            max_retries: 最大重试次数
            allow_duplicates: 是否允许重复问题
            
        Returns:
            Question: 生成的问题对象
            
        Raises:
            KnowledgeBaseNotFoundError: 知识库不存在
            ModelServiceError: 模型服务错误
            KnowledgeSystemError: 其他系统错误
        """
        logger.info(f"Generating question for kb '{kb_name}' with difficulty '{difficulty.value}' and strategy '{strategy.value}'")
        
        # 检查知识库是否存在
        if not self.vector_store.collection_exists(kb_name):
            raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' does not exist")
        
        # 获取知识库统计信息
        stats = self.vector_store.get_collection_stats(kb_name)
        if stats["document_count"] == 0:
            raise KnowledgeSystemError(f"Knowledge base '{kb_name}' is empty")
        
        # 初始化该知识库的问题历史记录
        if kb_name not in self.question_history:
            self.question_history[kb_name] = set()
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Question generation attempt {attempt + 1}/{max_retries}")
                
                # 1. 选择内容
                context = self._select_content_for_question(kb_name, strategy)
                
                # 2. 生成问题
                question_data = self._generate_question_content(context, difficulty)
                question_content = question_data["question"]
                background_info = question_data["background"]
                
                # 3. 验证问题质量
                is_valid, issues = self.validator.validate_question(question_content)
                if not is_valid:
                    logger.warning(f"Generated question failed validation: {issues}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise KnowledgeSystemError(f"Failed to generate valid question after {max_retries} attempts: {issues}")
                
                # 4. 创建问题对象
                question = Question(
                    content=question_content,
                    kb_name=kb_name,
                    source_context=context.content,
                    difficulty=difficulty,
                    background_info=background_info
                )
                
                # 5. 检查是否重复（如果不允许重复）
                if not allow_duplicates and question.content_hash in self.question_history[kb_name]:
                    logger.warning(f"Generated question is duplicate, retrying...")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.warning(f"All attempts generated duplicate questions, returning anyway")
                
                # 6. 记录问题到历史
                self.question_history[kb_name].add(question.content_hash)
                
                logger.info(f"Successfully generated question: {question.id}")
                return question
                
            except (ModelServiceError, VectorStoreError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    break
            except Exception as e:
                last_error = KnowledgeSystemError(f"Unexpected error during question generation: {str(e)}")
                logger.error(f"Unexpected error: {str(e)}")
                break
        
        # 所有重试都失败了
        if last_error:
            raise last_error
        else:
            raise KnowledgeSystemError("Failed to generate question after all retries")
    
    def _select_content_for_question(
        self, 
        kb_name: str, 
        strategy: ContentSelectionStrategy
    ) -> QuestionGenerationContext:
        """
        根据策略选择用于生成问题的内容
        
        Args:
            kb_name: 知识库名称
            strategy: 选择策略
            
        Returns:
            QuestionGenerationContext: 问题生成上下文
        """
        logger.debug(f"Selecting content using strategy: {strategy.value}")
        
        try:
            if strategy == ContentSelectionStrategy.RANDOM:
                return self._select_random_content(kb_name)
            elif strategy == ContentSelectionStrategy.DIVERSE:
                return self._select_diverse_content(kb_name)
            elif strategy == ContentSelectionStrategy.RECENT:
                return self._select_recent_content(kb_name)
            elif strategy == ContentSelectionStrategy.COMPREHENSIVE:
                return self._select_comprehensive_content(kb_name)
            else:
                logger.warning(f"Unknown strategy {strategy}, falling back to random")
                return self._select_random_content(kb_name)
                
        except Exception as e:
            logger.error(f"Content selection failed: {str(e)}")
            raise VectorStoreError(f"Failed to select content: {str(e)}")
    
    def _select_random_content(self, kb_name: str) -> QuestionGenerationContext:
        """随机选择内容"""
        # 使用随机查询词进行搜索
        random_queries = [
            "知识", "内容", "信息", "概念", "方法", "原理", "特点", "作用", "意义", "应用"
        ]
        
        query = random.choice(random_queries)
        results = self.vector_store.similarity_search(kb_name, query, k=5)
        
        if not results:
            raise VectorStoreError(f"No content found in knowledge base '{kb_name}'")
        
        # 随机选择一个结果
        selected_result = random.choice(results)
        
        return QuestionGenerationContext(
            content=selected_result.document.content,
            source_metadata=selected_result.document.metadata,
            difficulty=QuestionDifficulty.MEDIUM,
            strategy=ContentSelectionStrategy.RANDOM
        )
    
    def _select_diverse_content(self, kb_name: str) -> QuestionGenerationContext:
        """多样化内容选择"""
        # 使用多个不同的查询词
        diverse_queries = [
            "定义", "特征", "分类", "应用", "优缺点", "发展", "影响", "关系"
        ]
        
        all_results = []
        for query in diverse_queries[:3]:  # 限制查询数量
            results = self.vector_store.similarity_search(kb_name, query, k=2)
            all_results.extend(results)
        
        if not all_results:
            # 回退到随机选择
            return self._select_random_content(kb_name)
        
        # 选择相似度最高的结果
        best_result = max(all_results, key=lambda x: x.score)
        
        return QuestionGenerationContext(
            content=best_result.document.content,
            source_metadata=best_result.document.metadata,
            difficulty=QuestionDifficulty.MEDIUM,
            strategy=ContentSelectionStrategy.DIVERSE
        )
    
    def _select_recent_content(self, kb_name: str) -> QuestionGenerationContext:
        """选择最近的内容"""
        # 由于ChromaDB没有直接的时间排序，使用元数据过滤
        results = self.vector_store.similarity_search(kb_name, "最新", k=10)
        
        if not results:
            return self._select_random_content(kb_name)
        
        # 选择第一个结果（假设是最相关的）
        selected_result = results[0]
        
        return QuestionGenerationContext(
            content=selected_result.document.content,
            source_metadata=selected_result.document.metadata,
            difficulty=QuestionDifficulty.MEDIUM,
            strategy=ContentSelectionStrategy.RECENT
        )
    
    def _select_comprehensive_content(self, kb_name: str) -> QuestionGenerationContext:
        """选择综合性内容"""
        # 搜索包含综合性关键词的内容
        comprehensive_queries = [
            "总结", "概述", "综合", "整体", "全面", "系统"
        ]
        
        all_results = []
        for query in comprehensive_queries:
            results = self.vector_store.similarity_search(kb_name, query, k=3)
            all_results.extend(results)
        
        if not all_results:
            return self._select_random_content(kb_name)
        
        # 选择内容最长的结果（通常更综合）
        longest_result = max(all_results, key=lambda x: len(x.document.content))
        
        return QuestionGenerationContext(
            content=longest_result.document.content,
            source_metadata=longest_result.document.metadata,
            difficulty=QuestionDifficulty.MEDIUM,
            strategy=ContentSelectionStrategy.COMPREHENSIVE
        )
    
    def _generate_question_content(
        self, 
        context: QuestionGenerationContext, 
        difficulty: QuestionDifficulty
    ) -> str:
        """
        基于上下文生成问题内容
        
        Args:
            context: 问题生成上下文
            difficulty: 问题难度
            
        Returns:
            str: 生成的问题内容
        """
        # 截断过长的内容
        content = context.content
        if len(content) > self.config.max_context_length:
            content = content[:self.config.max_context_length] + "..."
        
        # 根据难度构建不同的提示词
        prompt = self._create_question_prompt(content, difficulty)
        
        try:
            # 调用LLM生成问题
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.config.question_generation_temperature,
                max_tokens=1000  # 限制问题长度
            )
            
            response_text = response.response.strip()
            
            # 尝试解析JSON格式的响应
            try:
                import json
                response_data = json.loads(self._clean_json_response(response_text))
                question = response_data.get("question", "").strip()
                background = response_data.get("background", "").strip()
            except (json.JSONDecodeError, KeyError):
                # 如果不是JSON格式，按原来的方式处理
                question = self._clean_generated_question(response_text)
                background = ""
            
            if not question:
                raise ModelServiceError("Generated question is empty after cleaning")
            
            return {"question": question, "background": background}
            
        except Exception as e:
            logger.error(f"Failed to generate question content: {str(e)}")
            raise ModelServiceError(f"Question generation failed: {str(e)}")
    
    def _create_question_prompt(self, content: str, difficulty: QuestionDifficulty) -> str:
        """
        创建问题生成提示词
        
        Args:
            content: 知识内容
            difficulty: 问题难度
            
        Returns:
            str: 提示词
        """
        difficulty_instructions = {
            QuestionDifficulty.EASY: {
                "description": "简单",
                "requirements": [
                    "测试基本概念和定义的理解",
                    "问题应该直接明了",
                    "答案应该能在原文中直接找到"
                ]
            },
            QuestionDifficulty.MEDIUM: {
                "description": "中等",
                "requirements": [
                    "测试对概念的理解和应用",
                    "需要一定的分析和推理",
                    "可能需要结合多个知识点"
                ]
            },
            QuestionDifficulty.HARD: {
                "description": "困难",
                "requirements": [
                    "测试深层理解和批判性思维",
                    "需要综合分析和评价",
                    "可能涉及比较、对比或创新应用"
                ]
            }
        }
        
        diff_info = difficulty_instructions[difficulty]
        requirements_text = "\n".join([f"- {req}" for req in diff_info["requirements"]])
        
        prompt = f"""基于以下知识内容，生成一个{diff_info["description"]}难度的学习问题，并提供相关的背景信息。

知识内容：
{content}

问题要求：
{requirements_text}

生成规则：
1. 问题必须用中文表达
2. 问题应该清晰、具体、有针对性
3. 问题必须以问号结尾
4. 避免是非题和选择题格式
5. 问题应该能够测试对知识的真正理解
6. 问题长度控制在10-100字之间

请按照以下JSON格式返回结果：
{{
    "question": "生成的问题内容",
    "background": "问题相关的背景信息，帮助理解问题的上下文和重要性"
}}

请确保返回有效的JSON格式："""

        return prompt
    
    def _clean_generated_question(self, question: str) -> str:
        """
        清理生成的问题
        
        Args:
            question: 原始问题
            
        Returns:
            str: 清理后的问题
        """
        # 首先使用通用的模型响应清理函数
        question = clean_model_response(question)
        
        # 移除多余的空白字符
        question = re.sub(r'\s+', ' ', question.strip())
        
        # 移除可能的前缀
        prefixes_to_remove = [
            r'^问题[:：]\s*',
            r'^问[:：]\s*',
            r'^题目[:：]\s*',
            r'^\d+[\.\)、]\s*',
        ]
        
        for prefix in prefixes_to_remove:
            question = re.sub(prefix, '', question)
        
        # 确保以问号结尾
        if not question.endswith(('？', '?')):
            if question.endswith(('。', '.', '！', '!')):
                question = question[:-1] + '？'
            else:
                question += '？'
        
        # 移除多余的标点符号
        question = re.sub(r'[？?]{2,}', '？', question)
        
        return question.strip()
    
    def _clean_json_response(self, response: str) -> str:
        """
        清理JSON响应
        
        Args:
            response: 原始响应
            
        Returns:
            str: 清理后的JSON字符串
        """
        # 首先使用通用的模型响应清理函数
        response = clean_model_response(response)
        
        # 移除可能的前缀和后缀
        response = response.strip()
        
        # 查找JSON开始和结束位置
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx + 1]
        
        # 修复常见的JSON格式问题
        response = re.sub(r',\s*}', '}', response)  # 移除尾随逗号
        response = re.sub(r',\s*]', ']', response)  # 移除数组尾随逗号
        
        return response
    
    def generate_multiple_questions(
        self,
        kb_name: str,
        count: int = 5,
        difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM,
        strategy: ContentSelectionStrategy = ContentSelectionStrategy.DIVERSE
    ) -> List[Question]:
        """
        批量生成多个问题
        
        Args:
            kb_name: 知识库名称
            count: 生成问题数量
            difficulty: 问题难度
            strategy: 内容选择策略
            
        Returns:
            List[Question]: 问题列表
        """
        logger.info(f"Generating {count} questions for kb '{kb_name}'")
        
        questions = []
        failed_count = 0
        max_failures = count  # 允许的最大失败次数
        
        for i in range(count):
            try:
                question = self.generate_question(kb_name, difficulty, strategy)
                questions.append(question)
                logger.debug(f"Generated question {i + 1}/{count}")
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to generate question {i + 1}: {str(e)}")
                
                if failed_count >= max_failures:
                    logger.error(f"Too many failures ({failed_count}), stopping generation")
                    break
        
        logger.info(f"Successfully generated {len(questions)} questions, {failed_count} failures")
        return questions
    
    def get_question_suggestions(self, kb_name: str, topic: str) -> List[str]:
        """
        根据主题获取问题建议
        
        Args:
            kb_name: 知识库名称
            topic: 主题关键词
            
        Returns:
            List[str]: 问题建议列表
        """
        try:
            # 搜索相关内容
            results = self.vector_store.similarity_search(kb_name, topic, k=3)
            
            if not results:
                return []
            
            suggestions = []
            
            # 为每个结果生成问题建议
            for result in results:
                try:
                    context = QuestionGenerationContext(
                        content=result.document.content,
                        source_metadata=result.document.metadata,
                        difficulty=QuestionDifficulty.MEDIUM,
                        strategy=ContentSelectionStrategy.DIVERSE
                    )
                    
                    question_data = self._generate_question_content(context, QuestionDifficulty.MEDIUM)
                    suggestions.append(question_data["question"])
                    
                except Exception as e:
                    logger.warning(f"Failed to generate suggestion: {str(e)}")
                    continue
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get question suggestions: {str(e)}")
            return []
    
    def validate_question_quality(self, question: str) -> Dict[str, Any]:
        """
        验证问题质量并返回详细信息
        
        Args:
            question: 问题文本
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        is_valid, issues = self.validator.validate_question(question)
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "length": len(question),
            "has_question_mark": '？' in question or '?' in question,
            "quality_score": self._calculate_quality_score(question)
        }
    
    def _calculate_quality_score(self, question: str) -> float:
        """
        计算问题质量分数
        
        Args:
            question: 问题文本
            
        Returns:
            float: 质量分数 (0-10)
        """
        score = 10.0
        
        # 长度评分
        if len(question) < 5:
            score -= 3.0
        elif len(question) > 200:
            score -= 2.0
        
        # 问号检查
        if not ('？' in question or '?' in question):
            score -= 4.0
        
        # 质量关键词检查
        quality_keywords = self.validator.quality_keywords
        keyword_count = sum(1 for keyword in quality_keywords if keyword in question)
        if keyword_count == 0:
            score -= 2.0
        elif keyword_count > 2:
            score += 1.0
        
        # 禁止模式检查
        for pattern in self.validator.forbidden_patterns:
            if re.search(pattern, question):
                score -= 2.5
        
        return max(0.0, min(10.0, score))
    
    def clear_question_history(self, kb_name: str) -> None:
        """
        清除指定知识库的问题历史记录
        
        Args:
            kb_name: 知识库名称
        """
        if kb_name in self.question_history:
            self.question_history[kb_name].clear()
            logger.info(f"Cleared question history for kb '{kb_name}'")
    
    def get_question_history_count(self, kb_name: str) -> int:
        """
        获取指定知识库的历史问题数量
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            int: 历史问题数量
        """
        return len(self.question_history.get(kb_name, set()))
    
    def is_question_duplicate(self, question_content: str, kb_name: str) -> bool:
        """
        检查问题是否重复
        
        Args:
            question_content: 问题内容
            kb_name: 知识库名称
            
        Returns:
            bool: 是否重复
        """
        import hashlib
        content_for_hash = f"{kb_name}:{question_content.strip().lower()}"
        content_hash = hashlib.md5(content_for_hash.encode('utf-8')).hexdigest()
        
        return content_hash in self.question_history.get(kb_name, set())
    
    def generate_question_with_skip_support(
        self,
        kb_name: str,
        difficulty: QuestionDifficulty = QuestionDifficulty.EASY,
        strategy: ContentSelectionStrategy = ContentSelectionStrategy.RANDOM,
        max_attempts: int = 5
    ) -> Question:
        """
        生成问题，支持跳过重复问题
        
        Args:
            kb_name: 知识库名称
            difficulty: 问题难度
            strategy: 内容选择策略
            max_attempts: 最大尝试次数
            
        Returns:
            Question: 生成的问题对象
            
        Raises:
            KnowledgeSystemError: 无法生成新问题
        """
        for attempt in range(max_attempts):
            try:
                question = self.generate_question(
                    kb_name=kb_name,
                    difficulty=difficulty,
                    strategy=strategy,
                    allow_duplicates=False
                )
                return question
            except Exception as e:
                if attempt == max_attempts - 1:
                    # 最后一次尝试，允许重复
                    logger.warning(f"Failed to generate unique question after {max_attempts} attempts, allowing duplicates")
                    return self.generate_question(
                        kb_name=kb_name,
                        difficulty=difficulty,
                        strategy=strategy,
                        allow_duplicates=True
                    )
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                continue
        
        raise KnowledgeSystemError(f"Failed to generate question after {max_attempts} attempts")


# 全局问题生成器实例
_question_generator: Optional[QuestionGenerator] = None


def get_question_generator() -> QuestionGenerator:
    """
    获取全局问题生成器实例
    
    Returns:
        QuestionGenerator: 问题生成器实例
    """
    global _question_generator
    
    if _question_generator is None:
        _question_generator = QuestionGenerator()
    
    return _question_generator


def reset_question_generator() -> None:
    """重置全局问题生成器实例（主要用于测试）"""
    global _question_generator
    _question_generator = None