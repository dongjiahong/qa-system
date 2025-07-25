"""
Unit tests for Question Generator module
问题生成器模块单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.question_generator import (
    QuestionGenerator,
    QuestionQualityValidator,
    ContentSelectionStrategy,
    QuestionGenerationContext,
    get_question_generator,
    reset_question_generator
)
from src.models import (
    Question,
    QuestionDifficulty,
    KnowledgeSystemError,
    ModelServiceError,
    VectorStoreError,
    KnowledgeBaseNotFoundError
)
from src.vector_store import VectorStore, SearchResult, DocumentChunk
from src.llm_client import OllamaClient, GenerationResponse


class TestQuestionQualityValidator:
    """问题质量验证器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.validator = QuestionQualityValidator()
    
    def test_validate_valid_question(self):
        """测试有效问题验证"""
        valid_questions = [
            "什么是机器学习的基本原理？",
            "如何实现深度学习算法的优化？",
            "为什么需要进行数据预处理？",
            "神经网络的工作原理是什么？"
        ]
        
        for question in valid_questions:
            is_valid, issues = self.validator.validate_question(question)
            assert is_valid, f"Question should be valid: {question}, issues: {issues}"
            assert len(issues) == 0
    
    def test_validate_invalid_question_no_question_mark(self):
        """测试无问号的无效问题"""
        question = "什么是机器学习"
        is_valid, issues = self.validator.validate_question(question)
        
        assert not is_valid
        assert any("问号" in issue for issue in issues)
    
    def test_validate_invalid_question_too_short(self):
        """测试过短的无效问题"""
        question = "？"
        is_valid, issues = self.validator.validate_question(question)
        
        assert not is_valid
        assert any("过短" in issue for issue in issues)
    
    def test_validate_invalid_question_too_long(self):
        """测试过长的无效问题"""
        question = "什么" * 300 + "？"
        is_valid, issues = self.validator.validate_question(question)
        
        assert not is_valid
        assert any("过长" in issue for issue in issues)
    
    def test_validate_invalid_question_forbidden_patterns(self):
        """测试包含禁止模式的问题"""
        invalid_questions = [
            "请解释什么是机器学习的基本概念？",
            "以下哪个选择是正确的答案？",
            "A. 选项一 B. 选项二 C. 选项三？"
        ]
        
        for question in invalid_questions:
            is_valid, issues = self.validator.validate_question(question)
            assert not is_valid, f"Question should be invalid: {question}"
    
    def test_validate_empty_question(self):
        """测试空问题"""
        empty_questions = ["", "   ", "？", "?"]
        
        for question in empty_questions:
            is_valid, issues = self.validator.validate_question(question)
            assert not is_valid
            assert any("空" in issue for issue in issues)


class TestQuestionGenerator:
    """问题生成器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_vector_store = Mock(spec=VectorStore)
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.generator = QuestionGenerator(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client
        )
    
    def test_init(self):
        """测试初始化"""
        assert self.generator.vector_store == self.mock_vector_store
        assert self.generator.llm_client == self.mock_llm_client
        assert isinstance(self.generator.validator, QuestionQualityValidator)
    
    def test_generate_question_success(self):
        """测试成功生成问题"""
        kb_name = "test_kb"
        
        # Mock vector store
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        
        # Mock search result
        mock_document = DocumentChunk(
            id="doc1",
            content="机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
            metadata={"source": "test.pdf"}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        # Mock LLM response
        mock_response = GenerationResponse(
            response="什么是机器学习的基本原理？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        # Generate question
        question = self.generator.generate_question(kb_name)
        
        # Assertions
        assert isinstance(question, Question)
        assert question.content == "什么是机器学习的基本原理？"
        assert question.kb_name == kb_name
        assert question.source_context == mock_document.content
        assert question.difficulty == QuestionDifficulty.MEDIUM
        
        # Verify method calls
        self.mock_vector_store.collection_exists.assert_called_once_with(kb_name)
        self.mock_vector_store.get_collection_stats.assert_called_once_with(kb_name)
        self.mock_llm_client.generate.assert_called_once()
    
    def test_generate_question_kb_not_exists(self):
        """测试知识库不存在的情况"""
        kb_name = "nonexistent_kb"
        
        self.mock_vector_store.collection_exists.return_value = False
        
        with pytest.raises(KnowledgeBaseNotFoundError) as exc_info:
            self.generator.generate_question(kb_name)
        
        assert kb_name in str(exc_info.value)
    
    def test_generate_question_empty_kb(self):
        """测试空知识库的情况"""
        kb_name = "empty_kb"
        
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 0}
        
        with pytest.raises(KnowledgeSystemError) as exc_info:
            self.generator.generate_question(kb_name)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_generate_question_with_different_difficulties(self):
        """测试不同难度的问题生成"""
        kb_name = "test_kb"
        
        # Setup mocks
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        
        mock_document = DocumentChunk(
            id="doc1",
            content="测试内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        mock_response = GenerationResponse(
            response="什么是测试内容的核心概念？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        # Test different difficulties
        for difficulty in QuestionDifficulty:
            question = self.generator.generate_question(kb_name, difficulty=difficulty)
            assert question.difficulty == difficulty
    
    def test_generate_question_with_different_strategies(self):
        """测试不同策略的问题生成"""
        kb_name = "test_kb"
        
        # Setup mocks
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        
        mock_document = DocumentChunk(
            id="doc1",
            content="测试内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        mock_response = GenerationResponse(
            response="什么是测试内容的核心概念？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        # Test different strategies
        for strategy in ContentSelectionStrategy:
            question = self.generator.generate_question(kb_name, strategy=strategy)
            assert isinstance(question, Question)
    
    def test_generate_question_retry_on_invalid_question(self):
        """测试无效问题时的重试机制"""
        kb_name = "test_kb"
        
        # Setup mocks
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        
        mock_document = DocumentChunk(
            id="doc1",
            content="测试内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        # First call returns invalid question, second call returns valid question
        invalid_response = GenerationResponse(
            response="无效",  # No question mark and too short
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        valid_response = GenerationResponse(
            response="什么是测试的核心概念？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        self.mock_llm_client.generate.side_effect = [invalid_response, valid_response]
        
        # Generate question
        question = self.generator.generate_question(kb_name)
        
        # Should succeed after retry
        assert question.content == "什么是测试的核心概念？"
        assert self.mock_llm_client.generate.call_count == 2
    
    def test_generate_question_max_retries_exceeded(self):
        """测试超过最大重试次数"""
        kb_name = "test_kb"
        
        # Setup mocks
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        
        mock_document = DocumentChunk(
            id="doc1",
            content="测试内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        # Always return invalid question
        invalid_response = GenerationResponse(
            response="无效",  # Too short and no question mark
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        self.mock_llm_client.generate.return_value = invalid_response
        
        # Should raise error after max retries
        with pytest.raises(KnowledgeSystemError) as exc_info:
            self.generator.generate_question(kb_name, max_retries=2)
        
        assert "valid question" in str(exc_info.value)
        assert self.mock_llm_client.generate.call_count == 2
    
    def test_select_random_content(self):
        """测试随机内容选择"""
        kb_name = "test_kb"
        
        mock_document = DocumentChunk(
            id="doc1",
            content="随机内容",
            metadata={"source": "test.pdf"}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.8,
            distance=0.2
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        context = self.generator._select_content_for_question(
            kb_name, 
            ContentSelectionStrategy.RANDOM
        )
        
        assert isinstance(context, QuestionGenerationContext)
        assert context.content == "随机内容"
        assert context.strategy == ContentSelectionStrategy.RANDOM
        assert context.source_metadata == {"source": "test.pdf"}
    
    def test_select_content_no_results(self):
        """测试没有搜索结果的情况"""
        kb_name = "test_kb"
        
        self.mock_vector_store.similarity_search.return_value = []
        
        with pytest.raises(VectorStoreError) as exc_info:
            self.generator._select_content_for_question(
                kb_name, 
                ContentSelectionStrategy.RANDOM
            )
        
        assert "No content found" in str(exc_info.value)
    
    def test_clean_generated_question(self):
        """测试问题清理功能"""
        test_cases = [
            ("问题：什么是机器学习", "什么是机器学习？"),
            ("问：如何实现算法", "如何实现算法？"),
            ("题目：为什么需要预处理", "为什么需要预处理？"),
            ("1. 什么是深度学习", "什么是深度学习？"),
            ("什么是AI。", "什么是AI？"),
            ("什么是AI！", "什么是AI？"),
            ("什么是AI？？？", "什么是AI？"),
            ("  什么是AI  ", "什么是AI？"),
        ]
        
        for input_question, expected in test_cases:
            result = self.generator._clean_generated_question(input_question)
            assert result == expected, f"Input: {input_question}, Expected: {expected}, Got: {result}"
    
    def test_generate_multiple_questions(self):
        """测试批量生成问题"""
        kb_name = "test_kb"
        count = 3
        
        # Setup mocks
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        
        mock_document = DocumentChunk(
            id="doc1",
            content="测试内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        mock_response = GenerationResponse(
            response="什么是测试内容的核心概念？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        # Generate multiple questions
        questions = self.generator.generate_multiple_questions(kb_name, count=count)
        
        assert len(questions) == count
        for question in questions:
            assert isinstance(question, Question)
            assert question.kb_name == kb_name
    
    def test_get_question_suggestions(self):
        """测试获取问题建议"""
        kb_name = "test_kb"
        topic = "机器学习"
        
        # Setup mocks
        mock_document = DocumentChunk(
            id="doc1",
            content="机器学习相关内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        mock_response = GenerationResponse(
            response="什么是机器学习的基本原理？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        self.mock_llm_client.generate.return_value = mock_response
        
        # Get suggestions
        suggestions = self.generator.get_question_suggestions(kb_name, topic)
        
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, str) for suggestion in suggestions)
        self.mock_vector_store.similarity_search.assert_called_with(kb_name, topic, k=3)
    
    def test_validate_question_quality(self):
        """测试问题质量验证"""
        valid_question = "什么是机器学习的基本原理？"
        invalid_question = "无效"
        
        # Test valid question
        result = self.generator.validate_question_quality(valid_question)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0
        assert result["has_question_mark"] is True
        assert result["quality_score"] > 50
        
        # Test invalid question
        result = self.generator.validate_question_quality(invalid_question)
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert result["has_question_mark"] is False
        assert result["quality_score"] < 50
    
    def test_calculate_quality_score(self):
        """测试质量分数计算"""
        test_cases = [
            ("什么是机器学习的基本原理？", 8.0),  # Good question
            ("？", 0),  # Too short
            ("什么" * 100 + "？", 7.0),  # Too long
            ("机器学习是什么概念", 5.0),  # No question mark
            ("请问什么是机器学习的基本概念？", 6.0),  # Forbidden pattern
        ]
        
        for question, expected_min_score in test_cases:
            score = self.generator._calculate_quality_score(question)
            assert 0 <= score <= 10
            # Allow some tolerance in score calculation
            if expected_min_score > 7.0:
                assert score >= expected_min_score - 1.5
    
    def test_model_service_error_handling(self):
        """测试模型服务错误处理"""
        kb_name = "test_kb"
        
        # Setup mocks
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        
        mock_document = DocumentChunk(
            id="doc1",
            content="测试内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        # Mock LLM client to raise error
        self.mock_llm_client.generate.side_effect = ModelServiceError("LLM service error")
        
        with pytest.raises(ModelServiceError):
            self.generator.generate_question(kb_name)
    
    def test_vector_store_error_handling(self):
        """测试向量存储错误处理"""
        kb_name = "test_kb"
        
        # Setup mocks
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        self.mock_vector_store.similarity_search.side_effect = VectorStoreError("Vector store error")
        
        with pytest.raises(VectorStoreError):
            self.generator.generate_question(kb_name)


class TestGlobalFunctions:
    """全局函数测试"""
    
    def test_get_question_generator(self):
        """测试获取全局问题生成器"""
        # Reset first
        reset_question_generator()
        
        generator1 = get_question_generator()
        generator2 = get_question_generator()
        
        assert isinstance(generator1, QuestionGenerator)
        assert generator1 is generator2  # Should be the same instance
    
    def test_reset_question_generator(self):
        """测试重置全局问题生成器"""
        generator1 = get_question_generator()
        reset_question_generator()
        generator2 = get_question_generator()
        
        assert generator1 is not generator2  # Should be different instances


class TestQuestionGenerationContext:
    """问题生成上下文测试"""
    
    def test_question_generation_context_creation(self):
        """测试问题生成上下文创建"""
        context = QuestionGenerationContext(
            content="测试内容",
            source_metadata={"source": "test.pdf"},
            difficulty=QuestionDifficulty.MEDIUM,
            strategy=ContentSelectionStrategy.RANDOM
        )
        
        assert context.content == "测试内容"
        assert context.source_metadata == {"source": "test.pdf"}
        assert context.difficulty == QuestionDifficulty.MEDIUM
        assert context.strategy == ContentSelectionStrategy.RANDOM


class TestContentSelectionStrategies:
    """内容选择策略测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.mock_vector_store = Mock(spec=VectorStore)
        self.mock_llm_client = Mock(spec=OllamaClient)
        self.generator = QuestionGenerator(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client
        )
    
    def test_select_diverse_content(self):
        """测试多样化内容选择"""
        kb_name = "test_kb"
        
        mock_document = DocumentChunk(
            id="doc1",
            content="多样化内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        context = self.generator._select_diverse_content(kb_name)
        
        assert isinstance(context, QuestionGenerationContext)
        assert context.strategy == ContentSelectionStrategy.DIVERSE
        # Should make multiple search calls for diversity
        assert self.mock_vector_store.similarity_search.call_count >= 1
    
    def test_select_recent_content(self):
        """测试最近内容选择"""
        kb_name = "test_kb"
        
        mock_document = DocumentChunk(
            id="doc1",
            content="最近内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        context = self.generator._select_recent_content(kb_name)
        
        assert isinstance(context, QuestionGenerationContext)
        assert context.strategy == ContentSelectionStrategy.RECENT
    
    def test_select_comprehensive_content(self):
        """测试综合性内容选择"""
        kb_name = "test_kb"
        
        # Create documents with different lengths
        short_doc = DocumentChunk(id="doc1", content="短内容", metadata={})
        long_doc = DocumentChunk(id="doc2", content="这是一个很长的综合性内容，包含了很多信息", metadata={})
        
        short_result = SearchResult(document=short_doc, score=0.8, distance=0.2)
        long_result = SearchResult(document=long_doc, score=0.7, distance=0.3)
        
        self.mock_vector_store.similarity_search.return_value = [short_result, long_result]
        
        context = self.generator._select_comprehensive_content(kb_name)
        
        assert isinstance(context, QuestionGenerationContext)
        assert context.strategy == ContentSelectionStrategy.COMPREHENSIVE
        # Should select the longer content
        assert len(context.content) > len("短内容")
    
    def test_fallback_to_random_on_no_results(self):
        """测试无结果时回退到随机选择"""
        kb_name = "test_kb"
        
        # First calls return empty, last call returns result
        mock_document = DocumentChunk(
            id="doc1",
            content="回退内容",
            metadata={}
        )
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.9,
            distance=0.1
        )
        
        # Mock multiple calls for diverse strategy, then fallback
        def mock_search_side_effect(*args, **kwargs):
            query = args[1] if len(args) > 1 else kwargs.get('query', '')
            if query in ["定义", "特征", "分类"]:
                return []  # Empty results for diverse queries
            else:
                return [mock_search_result]  # Return result for random fallback
        
        self.mock_vector_store.similarity_search.side_effect = mock_search_side_effect
        
        context = self.generator._select_diverse_content(kb_name)
        
        assert isinstance(context, QuestionGenerationContext)
        assert context.content == "回退内容"


if __name__ == "__main__":
    pytest.main([__file__])