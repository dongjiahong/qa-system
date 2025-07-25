"""
Answer Evaluator module for Knowledge QA System
答案评估器模块
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from .models import (
    EvaluationResult,
    EvaluationStatus,
    KnowledgeSystemError,
    ModelServiceError,
    VectorStoreError,
    KnowledgeBaseNotFoundError
)
from .vector_store import VectorStore, SearchResult
from .llm_client import OllamaClient, get_ollama_client, clean_model_response
from .config import get_config


class EvaluationCriteria(Enum):
    """评估标准"""
    ACCURACY = "accuracy"  # 事实准确性
    COMPLETENESS = "completeness"  # 完整性
    RELEVANCE = "relevance"  # 相关性
    CLARITY = "clarity"  # 清晰度


@dataclass
class EvaluationContext:
    """评估上下文"""
    question: str
    user_answer: str
    reference_context: str
    relevant_documents: List[SearchResult]


class AnswerValidator:
    """答案验证器"""
    
    def __init__(self):
        self.config = get_config()
        
        # 答案验证规则
        self.min_length = 2
        self.max_length = 2000
        
        # 无效答案模式
        self.invalid_patterns = [
            r'^(不知道|不清楚|没有|无|没)$',
            r'^[？?]+$',
            r'^[。.]+$',
            r'^[，,]+$',
        ]
        
        # 质量指标关键词
        self.quality_indicators = {
            'positive': ['因为', '由于', '根据', '通过', '可以', '能够', '具体', '详细'],
            'negative': ['可能', '大概', '也许', '估计', '应该']
        }
    
    def validate_answer(self, answer: str) -> Tuple[bool, List[str]]:
        """
        验证答案有效性
        
        Args:
            answer: 用户答案
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 问题列表)
        """
        issues = []
        
        # 基本长度检查
        if len(answer.strip()) < self.min_length:
            issues.append(f"答案过短，至少需要{self.min_length}个字符")
        
        if len(answer.strip()) > self.max_length:
            issues.append(f"答案过长，不能超过{self.max_length}个字符")
        
        # 无效模式检查
        for pattern in self.invalid_patterns:
            if re.search(pattern, answer.strip(), re.IGNORECASE):
                issues.append("答案内容无效或过于简单")
                break
        
        # 检查是否为空
        if not answer.strip():
            issues.append("答案不能为空")
        
        return len(issues) == 0, issues


class AnswerEvaluator:
    """
    答案评估器
    
    评估用户答案的正确性并提供反馈，支持多维度评估
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[OllamaClient] = None
    ):
        """
        初始化答案评估器
        
        Args:
            vector_store: 向量存储实例
            llm_client: LLM客户端实例
        """
        self.config = get_config()
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client or get_ollama_client()
        self.validator = AnswerValidator()
        
        logger.info("AnswerEvaluator initialized")
    
    def evaluate_answer(
        self,
        question: str,
        user_answer: str,
        kb_name: str,
        max_retries: int = 3
    ) -> EvaluationResult:
        """
        评估用户答案
        
        Args:
            question: 问题
            user_answer: 用户答案
            kb_name: 知识库名称
            max_retries: 最大重试次数
            
        Returns:
            EvaluationResult: 评估结果
            
        Raises:
            KnowledgeBaseNotFoundError: 知识库不存在
            ModelServiceError: 模型服务错误
            KnowledgeSystemError: 其他系统错误
        """
        logger.info(f"Evaluating answer for question in kb '{kb_name}'")
        
        # 检查知识库是否存在
        if not self.vector_store.collection_exists(kb_name):
            raise KnowledgeBaseNotFoundError(f"Knowledge base '{kb_name}' does not exist")
        
        # 验证输入
        if not question.strip():
            raise KnowledgeSystemError("Question cannot be empty")
        
        # 验证答案
        is_valid, issues = self.validator.validate_answer(user_answer)
        if not is_valid:
            logger.warning(f"Invalid answer provided: {issues}")
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback=f"答案无效: {'; '.join(issues)}",
                reference_answer="请提供有效的答案",
                missing_points=issues,
                strengths=[],
                status=EvaluationStatus.ERROR
            )
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Answer evaluation attempt {attempt + 1}/{max_retries}")
                
                # 1. 检索相关上下文
                context = self._retrieve_evaluation_context(question, user_answer, kb_name)
                
                # 2. 执行评估
                evaluation_result = self._perform_evaluation(context)
                
                # 3. 验证评估结果
                self._validate_evaluation_result(evaluation_result)
                
                logger.info(f"Successfully evaluated answer with score: {evaluation_result.score}")
                return evaluation_result
                
            except (ModelServiceError, VectorStoreError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    break
            except Exception as e:
                last_error = KnowledgeSystemError(f"Unexpected error during answer evaluation: {str(e)}")
                logger.error(f"Unexpected error: {str(e)}")
                break
        
        # 所有重试都失败了，返回错误状态的评估结果
        if last_error:
            logger.error(f"Answer evaluation failed after all retries: {str(last_error)}")
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback=f"评估失败: {str(last_error)}",
                reference_answer="无法获取参考答案",
                missing_points=["评估系统错误"],
                strengths=[],
                status=EvaluationStatus.ERROR
            )
        else:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="评估失败: 未知错误",
                reference_answer="无法获取参考答案",
                missing_points=["系统错误"],
                strengths=[],
                status=EvaluationStatus.ERROR
            )
    
    def _retrieve_evaluation_context(
        self,
        question: str,
        user_answer: str,
        kb_name: str
    ) -> EvaluationContext:
        """
        检索评估上下文
        
        Args:
            question: 问题
            user_answer: 用户答案
            kb_name: 知识库名称
            
        Returns:
            EvaluationContext: 评估上下文
        """
        logger.debug("Retrieving evaluation context")
        
        try:
            # 使用问题和答案作为查询来检索相关文档
            query_text = f"{question} {user_answer}"
            
            # 检索相关文档
            relevant_documents = self.vector_store.similarity_search(
                kb_name=kb_name,
                query=query_text,
                k=5  # 获取前5个最相关的文档
            )
            
            if not relevant_documents:
                logger.warning("No relevant documents found for evaluation")
                # 尝试仅使用问题进行搜索
                relevant_documents = self.vector_store.similarity_search(
                    kb_name=kb_name,
                    query=question,
                    k=3
                )
            
            # 构建参考上下文
            reference_context = self._build_reference_context(relevant_documents)
            
            return EvaluationContext(
                question=question,
                user_answer=user_answer,
                reference_context=reference_context,
                relevant_documents=relevant_documents
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve evaluation context: {str(e)}")
            raise VectorStoreError(f"Context retrieval failed: {str(e)}")
    
    def _build_reference_context(self, documents: List[SearchResult]) -> str:
        """
        构建参考上下文
        
        Args:
            documents: 相关文档列表
            
        Returns:
            str: 参考上下文
        """
        if not documents:
            return "没有找到相关的参考内容"
        
        context_parts = []
        max_context_length = self.config.max_context_length
        current_length = 0
        
        for i, doc_result in enumerate(documents):
            content = doc_result.document.content.strip()
            
            # 检查是否会超过最大长度
            if current_length + len(content) > max_context_length:
                # 截断内容
                remaining_length = max_context_length - current_length
                if remaining_length > 100:  # 至少保留100字符
                    content = content[:remaining_length] + "..."
                else:
                    break
            
            context_parts.append(f"参考内容{i+1}:\n{content}")
            current_length += len(content)
            
            if current_length >= max_context_length:
                break
        
        return "\n\n".join(context_parts)
    
    def _perform_evaluation(self, context: EvaluationContext) -> EvaluationResult:
        """
        执行答案评估
        
        Args:
            context: 评估上下文
            
        Returns:
            EvaluationResult: 评估结果
        """
        logger.debug("Performing answer evaluation")
        
        # 构建评估提示词
        prompt = self._create_evaluation_prompt(context)
        
        try:
            # 调用LLM进行评估
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.config.evaluation_temperature,
                max_tokens=1000
            )
            
            # 解析评估结果
            evaluation_result = self._parse_evaluation_response(response.response)
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Failed to perform evaluation: {str(e)}")
            raise ModelServiceError(f"Evaluation failed: {str(e)}")
    
    def _create_evaluation_prompt(self, context: EvaluationContext) -> str:
        """
        创建评估提示词
        
        Args:
            context: 评估上下文
            
        Returns:
            str: 评估提示词
        """
        prompt = f"""请评估以下用户答案的正确性和质量。

问题：
{context.question}

用户答案：
{context.user_answer}

参考知识：
{context.reference_context}

请按照以下JSON格式返回评估结果：
{{
    "is_correct": true/false,
    "score": 0-10的分数,
    "feedback": "详细的反馈说明，包括答案的优缺点",
    "reference_answer": "基于参考知识的标准答案",
    "missing_points": ["缺失的要点1", "缺失的要点2"],
    "strengths": ["答案的优点1", "答案的优点2"]
}}

评估标准：
1. 事实准确性（4分）：答案是否符合参考知识中的事实
2. 完整性（3分）：答案是否涵盖了问题的关键要点
3. 相关性（2分）：答案是否直接回答了问题
4. 清晰度（1分）：答案表达是否清晰易懂

评估要求：
- 如果答案基本正确且完整，is_correct为true，分数7分以上
- 如果答案部分正确但有重要遗漏，is_correct为false，分数4-7分
- 如果答案错误或严重不完整，is_correct为false，分数4分以下
- feedback应该具体指出答案的问题和改进建议
- reference_answer应该基于参考知识给出完整准确的答案
- missing_points列出答案中缺失的重要要点
- strengths列出答案中的优点和正确之处

请确保返回有效的JSON格式："""

        return prompt
    
    def _parse_evaluation_response(self, response: str) -> EvaluationResult:
        """
        解析评估响应
        
        Args:
            response: LLM响应
            
        Returns:
            EvaluationResult: 评估结果
        """
        try:
            # 清理响应文本
            cleaned_response = self._clean_json_response(response)
            
            # 解析JSON
            evaluation_data = json.loads(cleaned_response)
            
            # 验证必需字段
            required_fields = ["is_correct", "score", "feedback", "reference_answer"]
            for field in required_fields:
                if field not in evaluation_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # 确保可选字段存在
            evaluation_data.setdefault("missing_points", [])
            evaluation_data.setdefault("strengths", [])
            
            # 验证数据类型和范围
            if not isinstance(evaluation_data["is_correct"], bool):
                evaluation_data["is_correct"] = str(evaluation_data["is_correct"]).lower() == "true"
            
            score = float(evaluation_data["score"])
            if not (0 <= score <= 10):
                logger.warning(f"Score {score} out of range, clamping to 0-10")
                score = max(0, min(10, score))
            
            # 创建评估结果
            return EvaluationResult(
                is_correct=evaluation_data["is_correct"],
                score=score,
                feedback=str(evaluation_data["feedback"]),
                reference_answer=str(evaluation_data["reference_answer"]),
                missing_points=evaluation_data["missing_points"],
                strengths=evaluation_data["strengths"],
                status=EvaluationStatus.SUCCESS
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response}")
            # 尝试从响应中提取基本信息
            return self._create_fallback_evaluation(response)
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {str(e)}")
            return self._create_fallback_evaluation(response)
    
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
    
    def _create_fallback_evaluation(self, response: str) -> EvaluationResult:
        """
        创建备用评估结果
        
        Args:
            response: 原始响应
            
        Returns:
            EvaluationResult: 备用评估结果
        """
        logger.warning("Creating fallback evaluation result")
        
        # 尝试从响应中提取一些基本信息
        is_correct = "正确" in response or "对" in response
        
        # 简单的分数估算
        if "优秀" in response or "很好" in response:
            score = 8.5
        elif "良好" in response or "不错" in response:
            score = 7.0
        elif "一般" in response or "部分正确" in response:
            score = 5.5
        else:
            score = 3.0
        
        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            feedback=f"评估结果解析失败，原始反馈：{response[:200]}...",
            reference_answer="无法解析参考答案",
            missing_points=["评估结果解析失败"],
            strengths=[] if not is_correct else ["答案包含正确信息"],
            status=EvaluationStatus.PARTIAL
        )
    
    def _validate_evaluation_result(self, result: EvaluationResult) -> None:
        """
        验证评估结果
        
        Args:
            result: 评估结果
            
        Raises:
            KnowledgeSystemError: 评估结果无效
        """
        try:
            # 使用模型的验证方法
            result.validate()
        except Exception as e:
            logger.error(f"Evaluation result validation failed: {str(e)}")
            raise KnowledgeSystemError(f"Invalid evaluation result: {str(e)}")
    
    def evaluate_multiple_answers(
        self,
        questions_and_answers: List[Tuple[str, str]],
        kb_name: str
    ) -> List[EvaluationResult]:
        """
        批量评估多个答案
        
        Args:
            questions_and_answers: 问题和答案对列表
            kb_name: 知识库名称
            
        Returns:
            List[EvaluationResult]: 评估结果列表
        """
        logger.info(f"Evaluating {len(questions_and_answers)} answers for kb '{kb_name}'")
        
        results = []
        
        for i, (question, answer) in enumerate(questions_and_answers):
            try:
                result = self.evaluate_answer(question, answer, kb_name)
                results.append(result)
                logger.debug(f"Evaluated answer {i + 1}/{len(questions_and_answers)}")
            except Exception as e:
                logger.error(f"Failed to evaluate answer {i + 1}: {str(e)}")
                # 添加错误结果
                error_result = EvaluationResult(
                    is_correct=False,
                    score=0.0,
                    feedback=f"评估失败: {str(e)}",
                    reference_answer="无法获取参考答案",
                    missing_points=["评估系统错误"],
                    strengths=[],
                    status=EvaluationStatus.ERROR
                )
                results.append(error_result)
        
        logger.info(f"Completed batch evaluation: {len(results)} results")
        return results
    
    def get_evaluation_statistics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        获取评估统计信息
        
        Args:
            results: 评估结果列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not results:
            return {
                "total_count": 0,
                "correct_count": 0,
                "accuracy_rate": 0.0,
                "average_score": 0.0,
                "score_distribution": {},
                "status_distribution": {}
            }
        
        correct_count = sum(1 for r in results if r.is_correct)
        total_score = sum(r.score for r in results)
        
        # 分数分布
        score_ranges = {
            "9-10": 0,
            "8-8.9": 0,
            "7-7.9": 0,
            "6-6.9": 0,
            "5-5.9": 0,
            "0-4.9": 0
        }
        
        for result in results:
            if result.score >= 9:
                score_ranges["9-10"] += 1
            elif result.score >= 8:
                score_ranges["8-8.9"] += 1
            elif result.score >= 7:
                score_ranges["7-7.9"] += 1
            elif result.score >= 6:
                score_ranges["6-6.9"] += 1
            elif result.score >= 5:
                score_ranges["5-5.9"] += 1
            else:
                score_ranges["0-4.9"] += 1
        
        # 状态分布
        status_distribution = {}
        for result in results:
            status = result.status.value
            status_distribution[status] = status_distribution.get(status, 0) + 1
        
        return {
            "total_count": len(results),
            "correct_count": correct_count,
            "accuracy_rate": correct_count / len(results) * 100,
            "average_score": total_score / len(results),
            "score_distribution": score_ranges,
            "status_distribution": status_distribution
        }


# 全局答案评估器实例
_answer_evaluator: Optional[AnswerEvaluator] = None


def get_answer_evaluator() -> AnswerEvaluator:
    """
    获取全局答案评估器实例
    
    Returns:
        AnswerEvaluator: 答案评估器实例
    """
    global _answer_evaluator
    
    if _answer_evaluator is None:
        _answer_evaluator = AnswerEvaluator()
    
    return _answer_evaluator


def reset_answer_evaluator() -> None:
    """重置全局答案评估器实例（主要用于测试）"""
    global _answer_evaluator
    _answer_evaluator = None