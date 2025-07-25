"""
Unit tests for Answer Evaluator module
答案评估器模块单元测试
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.answer_evaluator import (
    AnswerEvaluator,
    AnswerValidator,
    EvaluationContext,
    EvaluationCriteria,
    get_answer_evaluator,
    reset_answer_evaluator
)
from src.models import (
    EvaluationResult,
    EvaluationStatus,
    KnowledgeSystemError,
    ModelServiceError,
    VectorStoreError,
    KnowledgeBaseNotFoundError
)
from src.vector_store import VectorStore, SearchResult, DocumentChunk
from src.llm_client import OllamaClient, GenerationResponse


class TestAnswerValidator:
    """答案验证器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.validator = AnswerValidator()
    
    def test_validate_valid_answer(self):
        """测试有效答案验证"""
        answer = "这是一个详细的答案，包含了相关的知识点和解释。"
        is_valid, issues = self.validator.validate_answer(answer)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_empty_answer(self):
        """测试空答案验证"""
        answer = ""
        is_valid, issues = self.validator.validate_answer(answer)
        
        assert is_valid is False
        assert "答案不能为空" in issues
    
    def test_validate_too_short_answer(self):
        """测试过短答案验证"""
        answer = "是"
        is_valid, issues = self.validator.validate_answer(answer)
        
        assert is_valid is False
        assert any("过短" in issue for issue in issues)
    
    def test_validate_too_long_answer(self):
        """测试过长答案验证"""
        answer = "很长的答案" * 500  # 超过2000字符
        is_valid, issues = self.validator.validate_answer(answer)
        
        assert is_valid is False
        assert any("过长" in issue for issue in issues)
    
    def test_validate_invalid_patterns(self):
        """测试无效模式答案"""
        invalid_answers = ["不知道", "不清楚", "没有", "???", "..."]
        
        for answer in invalid_answers:
            is_valid, issues = self.validator.validate_answer(answer)
            assert is_valid is False
            assert any("无效" in issue for issue in issues)
    
    def test_validate_whitespace_answer(self):
        """测试只有空白字符的答案"""
        answer = "   \n\t   "
        is_valid, issues = self.validator.validate_answer(answer)
        
        assert is_valid is False
        assert "答案不能为空" in issues


class TestAnswerEvaluator:
    """答案评估器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.mock_vector_store = Mock(spec=VectorStore)
        self.mock_llm_client = Mock(spec=OllamaClient)
        
        self.evaluator = AnswerEvaluator(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client
        )
    
    def test_init(self):
        """测试初始化"""
        assert self.evaluator.vector_store == self.mock_vector_store
        assert self.evaluator.llm_client == self.mock_llm_client
        assert isinstance(self.evaluator.validator, AnswerValidator)
    
    def test_evaluate_answer_success(self):
        """测试成功评估答案"""
        # 设置模拟
        self.mock_vector_store.collection_exists.return_value = True
        
        # 模拟搜索结果
        mock_doc = DocumentChunk(
            id="doc1",
            content="这是相关的知识内容，包含了问题的答案。",
            metadata={"source": "test.txt"}
        )
        mock_search_result = SearchResult(
            document=mock_doc,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        # 模拟LLM响应
        evaluation_json = {
            "is_correct": True,
            "score": 85.0,
            "feedback": "答案正确且完整",
            "reference_answer": "标准答案",
            "missing_points": [],
            "strengths": ["答案准确", "表达清晰"]
        }
        mock_response = GenerationResponse(
            response=json.dumps(evaluation_json, ensure_ascii=False),
            model="qwen3:1.7b",
            created_at="2024-01-01",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        # 执行测试
        result = self.evaluator.evaluate_answer(
            question="什么是机器学习？",
            user_answer="机器学习是一种人工智能技术，通过算法让计算机从数据中学习。",
            kb_name="test_kb"
        )
        
        # 验证结果
        assert isinstance(result, EvaluationResult)
        assert result.is_correct is True
        assert result.score == 85.0
        assert result.feedback == "答案正确且完整"
        assert result.reference_answer == "标准答案"
        assert result.status == EvaluationStatus.SUCCESS
        
        # 验证调用
        self.mock_vector_store.collection_exists.assert_called_once_with("test_kb")
        self.mock_vector_store.similarity_search.assert_called()
        self.mock_llm_client.generate.assert_called_once()
    
    def test_evaluate_answer_knowledge_base_not_found(self):
        """测试知识库不存在"""
        self.mock_vector_store.collection_exists.return_value = False
        
        with pytest.raises(KnowledgeBaseNotFoundError):
            self.evaluator.evaluate_answer(
                question="测试问题",
                user_answer="测试答案",
                kb_name="nonexistent_kb"
            )
    
    def test_evaluate_answer_empty_question(self):
        """测试空问题"""
        self.mock_vector_store.collection_exists.return_value = True
        
        with pytest.raises(KnowledgeSystemError, match="Question cannot be empty"):
            self.evaluator.evaluate_answer(
                question="",
                user_answer="测试答案",
                kb_name="test_kb"
            )
    
    def test_evaluate_answer_invalid_user_answer(self):
        """测试无效用户答案"""
        self.mock_vector_store.collection_exists.return_value = True
        
        result = self.evaluator.evaluate_answer(
            question="测试问题",
            user_answer="不知道",
            kb_name="test_kb"
        )
        
        assert result.is_correct is False
        assert result.score == 0.0
        assert "答案无效" in result.feedback
        assert result.status == EvaluationStatus.ERROR
    
    def test_evaluate_answer_with_retry(self):
        """测试重试机制"""
        self.mock_vector_store.collection_exists.return_value = True
        
        # 第一次调用失败，第二次成功
        self.mock_vector_store.similarity_search.side_effect = [
            VectorStoreError("临时错误"),
            [SearchResult(
                document=DocumentChunk(id="doc1", content="测试内容", metadata={}),
                score=0.8,
                distance=0.2
            )]
        ]
        
        evaluation_json = {
            "is_correct": True,
            "score": 80.0,
            "feedback": "答案正确",
            "reference_answer": "参考答案",
            "missing_points": [],
            "strengths": ["正确"]
        }
        mock_response = GenerationResponse(
            response=json.dumps(evaluation_json, ensure_ascii=False),
            model="qwen3:1.7b",
            created_at="2024-01-01",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        result = self.evaluator.evaluate_answer(
            question="测试问题",
            user_answer="测试答案",
            kb_name="test_kb"
        )
        
        assert result.is_correct is True
        assert result.score == 80.0
        assert self.mock_vector_store.similarity_search.call_count == 2
    
    def test_evaluate_answer_all_retries_failed(self):
        """测试所有重试都失败"""
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.similarity_search.side_effect = VectorStoreError("持续错误")
        
        result = self.evaluator.evaluate_answer(
            question="测试问题",
            user_answer="测试答案",
            kb_name="test_kb",
            max_retries=2
        )
        
        assert result.is_correct is False
        assert result.score == 0.0
        assert "评估失败" in result.feedback
        assert result.status == EvaluationStatus.ERROR
        assert self.mock_vector_store.similarity_search.call_count == 2
    
    def test_retrieve_evaluation_context(self):
        """测试检索评估上下文"""
        # 模拟搜索结果
        mock_docs = [
            SearchResult(
                document=DocumentChunk(id="doc1", content="内容1", metadata={}),
                score=0.9,
                distance=0.1
            ),
            SearchResult(
                document=DocumentChunk(id="doc2", content="内容2", metadata={}),
                score=0.8,
                distance=0.2
            )
        ]
        self.mock_vector_store.similarity_search.return_value = mock_docs
        
        context = self.evaluator._retrieve_evaluation_context(
            question="测试问题",
            user_answer="测试答案",
            kb_name="test_kb"
        )
        
        assert isinstance(context, EvaluationContext)
        assert context.question == "测试问题"
        assert context.user_answer == "测试答案"
        assert "参考内容1" in context.reference_context
        assert "参考内容2" in context.reference_context
        assert len(context.relevant_documents) == 2
    
    def test_retrieve_evaluation_context_no_results(self):
        """测试没有搜索结果的情况"""
        # 第一次搜索无结果，第二次有结果
        self.mock_vector_store.similarity_search.side_effect = [
            [],  # 第一次搜索无结果
            [SearchResult(
                document=DocumentChunk(id="doc1", content="内容", metadata={}),
                score=0.7,
                distance=0.3
            )]  # 第二次搜索有结果
        ]
        
        context = self.evaluator._retrieve_evaluation_context(
            question="测试问题",
            user_answer="测试答案",
            kb_name="test_kb"
        )
        
        assert isinstance(context, EvaluationContext)
        assert len(context.relevant_documents) == 1
        assert self.mock_vector_store.similarity_search.call_count == 2
    
    def test_build_reference_context(self):
        """测试构建参考上下文"""
        mock_docs = [
            SearchResult(
                document=DocumentChunk(id="doc1", content="第一个文档内容", metadata={}),
                score=0.9,
                distance=0.1
            ),
            SearchResult(
                document=DocumentChunk(id="doc2", content="第二个文档内容", metadata={}),
                score=0.8,
                distance=0.2
            )
        ]
        
        context = self.evaluator._build_reference_context(mock_docs)
        
        assert "参考内容1" in context
        assert "参考内容2" in context
        assert "第一个文档内容" in context
        assert "第二个文档内容" in context
    
    def test_build_reference_context_empty(self):
        """测试空文档列表"""
        context = self.evaluator._build_reference_context([])
        assert context == "没有找到相关的参考内容"
    
    def test_create_evaluation_prompt(self):
        """测试创建评估提示词"""
        context = EvaluationContext(
            question="什么是机器学习？",
            user_answer="机器学习是AI的一个分支",
            reference_context="机器学习是人工智能的重要分支...",
            relevant_documents=[]
        )
        
        prompt = self.evaluator._create_evaluation_prompt(context)
        
        assert "什么是机器学习？" in prompt
        assert "机器学习是AI的一个分支" in prompt
        assert "机器学习是人工智能的重要分支" in prompt
        assert "JSON格式" in prompt
        assert "事实准确性" in prompt
        assert "完整性" in prompt
        assert "相关性" in prompt
        assert "清晰度" in prompt
    
    def test_parse_evaluation_response_valid_json(self):
        """测试解析有效JSON响应"""
        response = json.dumps({
            "is_correct": True,
            "score": 85.0,
            "feedback": "答案正确",
            "reference_answer": "参考答案",
            "missing_points": ["缺失点1"],
            "strengths": ["优点1", "优点2"]
        }, ensure_ascii=False)
        
        result = self.evaluator._parse_evaluation_response(response)
        
        assert isinstance(result, EvaluationResult)
        assert result.is_correct is True
        assert result.score == 85.0
        assert result.feedback == "答案正确"
        assert result.reference_answer == "参考答案"
        assert result.missing_points == ["缺失点1"]
        assert result.strengths == ["优点1", "优点2"]
        assert result.status == EvaluationStatus.SUCCESS
    
    def test_parse_evaluation_response_missing_fields(self):
        """测试缺少必需字段的响应"""
        response = json.dumps({
            "is_correct": True,
            "score": 85.0,
            "feedback": "答案正确"
            # 缺少 reference_answer
        }, ensure_ascii=False)
        
        result = self.evaluator._parse_evaluation_response(response)
        
        # 应该创建备用评估结果
        assert isinstance(result, EvaluationResult)
        assert result.status == EvaluationStatus.PARTIAL
    
    def test_parse_evaluation_response_invalid_json(self):
        """测试无效JSON响应"""
        response = "这不是有效的JSON格式"
        
        result = self.evaluator._parse_evaluation_response(response)
        
        assert isinstance(result, EvaluationResult)
        assert result.status == EvaluationStatus.PARTIAL
        assert "评估结果解析失败" in result.feedback
    
    def test_parse_evaluation_response_score_out_of_range(self):
        """测试分数超出范围"""
        response = json.dumps({
            "is_correct": True,
            "score": 150.0,  # 超出范围
            "feedback": "答案正确",
            "reference_answer": "参考答案"
        }, ensure_ascii=False)
        
        result = self.evaluator._parse_evaluation_response(response)
        
        assert result.score == 10.0  # 应该被限制在10
    
    def test_clean_json_response(self):
        """测试清理JSON响应"""
        # 测试带前缀和后缀的响应
        response = '这是前缀 {"is_correct": true, "score": 85} 这是后缀'
        cleaned = self.evaluator._clean_json_response(response)
        assert cleaned == '{"is_correct": true, "score": 85}'
        
        # 测试带尾随逗号的响应
        response = '{"is_correct": true, "score": 85,}'
        cleaned = self.evaluator._clean_json_response(response)
        assert cleaned == '{"is_correct": true, "score": 85}'
    
    def test_create_fallback_evaluation(self):
        """测试创建备用评估结果"""
        response = "答案正确，很好的回答"
        result = self.evaluator._create_fallback_evaluation(response)
        
        assert isinstance(result, EvaluationResult)
        assert result.is_correct is True  # 包含"正确"
        assert result.status == EvaluationStatus.PARTIAL
        assert "评估结果解析失败" in result.feedback
    
    def test_validate_evaluation_result(self):
        """测试验证评估结果"""
        # 有效结果
        valid_result = EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="答案正确",
            reference_answer="参考答案"
        )
        
        # 不应该抛出异常
        self.evaluator._validate_evaluation_result(valid_result)
        
        # 无效结果 - 由于模型验证在初始化时进行，我们需要测试已创建的无效结果
        # 创建一个有效的结果，然后修改其属性来测试验证
        invalid_result = EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="答案正确",
            reference_answer="参考答案"
        )
        # 修改为无效值
        invalid_result.score = 150.0  # 超出范围
        invalid_result.feedback = ""  # 空反馈
        invalid_result.reference_answer = ""  # 空参考答案
        
        with pytest.raises(KnowledgeSystemError):
            self.evaluator._validate_evaluation_result(invalid_result)
    
    def test_evaluate_multiple_answers(self):
        """测试批量评估多个答案"""
        self.mock_vector_store.collection_exists.return_value = True
        
        # 模拟搜索结果
        mock_search_result = SearchResult(
            document=DocumentChunk(id="doc1", content="测试内容", metadata={}),
            score=0.8,
            distance=0.2
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        # 模拟LLM响应
        evaluation_json = {
            "is_correct": True,
            "score": 80.0,
            "feedback": "答案正确",
            "reference_answer": "参考答案",
            "missing_points": [],
            "strengths": ["正确"]
        }
        mock_response = GenerationResponse(
            response=json.dumps(evaluation_json, ensure_ascii=False),
            model="qwen3:1.7b",
            created_at="2024-01-01",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        questions_and_answers = [
            ("问题1", "答案1"),
            ("问题2", "答案2"),
            ("问题3", "答案3")
        ]
        
        results = self.evaluator.evaluate_multiple_answers(
            questions_and_answers,
            "test_kb"
        )
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, EvaluationResult)
            assert result.is_correct is True
            assert result.score == 80.0
    
    def test_evaluate_multiple_answers_with_failures(self):
        """测试批量评估中的部分失败"""
        self.mock_vector_store.collection_exists.return_value = True
        
        # 设置第二次调用失败，但由于重试机制，需要让所有重试都失败
        def side_effect_func(*args, **kwargs):
            if side_effect_func.call_count == 1:
                # 第一次调用成功
                side_effect_func.call_count += 1
                return [SearchResult(
                    document=DocumentChunk(id="doc1", content="内容", metadata={}),
                    score=0.8,
                    distance=0.2
                )]
            elif side_effect_func.call_count <= 4:  # 第二个问题的3次重试都失败
                side_effect_func.call_count += 1
                raise VectorStoreError("搜索失败")
            else:
                # 第三次调用成功
                side_effect_func.call_count += 1
                return [SearchResult(
                    document=DocumentChunk(id="doc2", content="内容", metadata={}),
                    score=0.8,
                    distance=0.2
                )]
        
        side_effect_func.call_count = 0
        self.mock_vector_store.similarity_search.side_effect = side_effect_func
        
        evaluation_json = {
            "is_correct": True,
            "score": 80.0,
            "feedback": "答案正确",
            "reference_answer": "参考答案",
            "missing_points": [],
            "strengths": ["正确"]
        }
        mock_response = GenerationResponse(
            response=json.dumps(evaluation_json, ensure_ascii=False),
            model="qwen3:1.7b",
            created_at="2024-01-01",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        questions_and_answers = [
            ("问题1", "答案1"),
            ("问题2", "答案2"),
            ("问题3", "答案3")
        ]
        
        results = self.evaluator.evaluate_multiple_answers(
            questions_and_answers,
            "test_kb"
        )
        
        assert len(results) == 3
        assert results[0].status == EvaluationStatus.SUCCESS
        assert results[1].status == EvaluationStatus.ERROR
        assert results[2].status == EvaluationStatus.SUCCESS
    
    def test_get_evaluation_statistics(self):
        """测试获取评估统计信息"""
        results = [
            EvaluationResult(
                is_correct=True,
                score=9.5,
                feedback="优秀",
                reference_answer="答案1",
                status=EvaluationStatus.SUCCESS
            ),
            EvaluationResult(
                is_correct=True,
                score=8.5,
                feedback="良好",
                reference_answer="答案2",
                status=EvaluationStatus.SUCCESS
            ),
            EvaluationResult(
                is_correct=False,
                score=4.5,
                feedback="需要改进",
                reference_answer="答案3",
                status=EvaluationStatus.SUCCESS
            ),
            EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="错误",
                reference_answer="答案4",
                status=EvaluationStatus.ERROR
            )
        ]
        
        stats = self.evaluator.get_evaluation_statistics(results)
        
        assert stats["total_count"] == 4
        assert stats["correct_count"] == 2
        assert stats["accuracy_rate"] == 50.0
        assert stats["average_score"] == 5.625
        assert stats["score_distribution"]["9-10"] == 1
        assert stats["score_distribution"]["8-8.9"] == 1
        assert stats["score_distribution"]["0-4.9"] == 2
        assert stats["status_distribution"]["success"] == 3
        assert stats["status_distribution"]["error"] == 1
    
    def test_get_evaluation_statistics_empty(self):
        """测试空结果列表的统计"""
        stats = self.evaluator.get_evaluation_statistics([])
        
        assert stats["total_count"] == 0
        assert stats["correct_count"] == 0
        assert stats["accuracy_rate"] == 0.0
        assert stats["average_score"] == 0.0


class TestGlobalFunctions:
    """全局函数测试"""
    
    def test_get_answer_evaluator(self):
        """测试获取全局答案评估器实例"""
        reset_answer_evaluator()  # 重置
        
        evaluator1 = get_answer_evaluator()
        evaluator2 = get_answer_evaluator()
        
        assert isinstance(evaluator1, AnswerEvaluator)
        assert evaluator1 is evaluator2  # 应该是同一个实例
    
    def test_reset_answer_evaluator(self):
        """测试重置全局答案评估器实例"""
        evaluator1 = get_answer_evaluator()
        reset_answer_evaluator()
        evaluator2 = get_answer_evaluator()
        
        assert evaluator1 is not evaluator2  # 应该是不同的实例


class TestEvaluationScenarios:
    """评估场景测试"""
    
    def setup_method(self):
        """测试设置"""
        self.mock_vector_store = Mock(spec=VectorStore)
        self.mock_llm_client = Mock(spec=OllamaClient)
        
        self.evaluator = AnswerEvaluator(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client
        )
        
        # 设置基本模拟
        self.mock_vector_store.collection_exists.return_value = True
        mock_search_result = SearchResult(
            document=DocumentChunk(id="doc1", content="相关知识内容", metadata={}),
            score=0.8,
            distance=0.2
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
    
    def test_correct_complete_answer(self):
        """测试正确完整的答案"""
        evaluation_json = {
            "is_correct": True,
            "score": 90.0,
            "feedback": "答案正确且完整，涵盖了所有关键要点",
            "reference_answer": "完整的参考答案",
            "missing_points": [],
            "strengths": ["准确", "完整", "清晰"]
        }
        
        mock_response = GenerationResponse(
            response=json.dumps(evaluation_json, ensure_ascii=False),
            model="qwen3:1.7b",
            created_at="2024-01-01",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        result = self.evaluator.evaluate_answer(
            question="什么是机器学习？",
            user_answer="机器学习是人工智能的一个重要分支，通过算法让计算机从数据中自动学习模式和规律。",
            kb_name="test_kb"
        )
        
        assert result.is_correct is True
        assert result.score == 90.0
        assert len(result.strengths) == 3
        assert len(result.missing_points) == 0
    
    def test_partially_correct_answer(self):
        """测试部分正确的答案"""
        evaluation_json = {
            "is_correct": False,
            "score": 60.0,
            "feedback": "答案部分正确，但缺少一些重要细节",
            "reference_answer": "更完整的参考答案",
            "missing_points": ["缺少算法类型说明", "未提及应用场景"],
            "strengths": ["基本概念正确"]
        }
        
        mock_response = GenerationResponse(
            response=json.dumps(evaluation_json, ensure_ascii=False),
            model="qwen3:1.7b",
            created_at="2024-01-01",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        result = self.evaluator.evaluate_answer(
            question="什么是机器学习？",
            user_answer="机器学习是AI技术。",
            kb_name="test_kb"
        )
        
        assert result.is_correct is False
        assert result.score == 60.0
        assert len(result.missing_points) == 2
        assert len(result.strengths) == 1
    
    def test_incorrect_answer(self):
        """测试错误答案"""
        evaluation_json = {
            "is_correct": False,
            "score": 20.0,
            "feedback": "答案存在明显错误，与参考知识不符",
            "reference_answer": "正确的参考答案",
            "missing_points": ["基本概念错误", "理解有偏差"],
            "strengths": []
        }
        
        mock_response = GenerationResponse(
            response=json.dumps(evaluation_json, ensure_ascii=False),
            model="qwen3:1.7b",
            created_at="2024-01-01",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        result = self.evaluator.evaluate_answer(
            question="什么是机器学习？",
            user_answer="机器学习就是机器人学习走路。",
            kb_name="test_kb"
        )
        
        assert result.is_correct is False
        assert result.score == 20.0
        assert len(result.missing_points) == 2
        assert len(result.strengths) == 0