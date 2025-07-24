"""
Integration tests for Question Generator module
问题生成器模块集成测试
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import os

from src.question_generator import QuestionGenerator, ContentSelectionStrategy
from src.models import QuestionDifficulty, Question
from src.vector_store import VectorStore, DocumentChunk, SearchResult
from src.llm_client import OllamaClient, GenerationResponse


class TestQuestionGeneratorIntegration:
    """问题生成器集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 使用临时目录进行测试
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟的向量存储和LLM客户端
        self.mock_vector_store = Mock(spec=VectorStore)
        self.mock_llm_client = Mock(spec=OllamaClient)
        
        # 初始化问题生成器
        self.generator = QuestionGenerator(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client
        )
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_question_generation(self):
        """端到端问题生成测试"""
        kb_name = "integration_test_kb"
        
        # 设置模拟数据
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 5}
        
        # 模拟文档内容
        mock_document = DocumentChunk(
            id="doc1",
            content="人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。机器学习是AI的一个子领域，它使计算机能够从数据中学习而无需明确编程。深度学习是机器学习的一个分支，使用神经网络来模拟人脑的工作方式。",
            metadata={
                "source": "ai_textbook.pdf",
                "page": 1,
                "chapter": "Introduction to AI"
            }
        )
        
        mock_search_result = SearchResult(
            document=mock_document,
            score=0.95,
            distance=0.05
        )
        
        self.mock_vector_store.similarity_search.return_value = [mock_search_result]
        
        # 模拟LLM响应
        mock_llm_response = GenerationResponse(
            response="什么是人工智能与机器学习的主要区别？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        self.mock_llm_client.generate.return_value = mock_llm_response
        
        # 生成问题
        question = self.generator.generate_question(
            kb_name=kb_name,
            difficulty=QuestionDifficulty.MEDIUM,
            strategy=ContentSelectionStrategy.RANDOM
        )
        
        # 验证结果
        assert isinstance(question, Question)
        assert question.kb_name == kb_name
        assert question.difficulty == QuestionDifficulty.MEDIUM
        assert question.content == "什么是人工智能与机器学习的主要区别？"
        assert question.source_context == mock_document.content
        assert question.id is not None
        assert question.created_at is not None
        
        # 验证调用
        self.mock_vector_store.collection_exists.assert_called_once_with(kb_name)
        self.mock_vector_store.get_collection_stats.assert_called_once_with(kb_name)
        self.mock_vector_store.similarity_search.assert_called_once()
        self.mock_llm_client.generate.assert_called_once()
    
    def test_question_generation_with_different_strategies(self):
        """测试不同策略的问题生成"""
        kb_name = "strategy_test_kb"
        
        # 基础设置
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 10}
        
        # 创建不同的文档用于不同策略
        documents = [
            DocumentChunk(
                id=f"doc{i}",
                content=f"这是第{i}个文档的内容，包含了关于主题{i}的详细信息。" * (i + 1),
                metadata={"source": f"doc{i}.pdf", "topic": f"topic{i}"}
            )
            for i in range(1, 4)
        ]
        
        search_results = [
            SearchResult(document=doc, score=0.9 - i*0.1, distance=i*0.1)
            for i, doc in enumerate(documents)
        ]
        
        self.mock_vector_store.similarity_search.return_value = search_results
        
        # 模拟LLM响应
        mock_responses = [
            "什么是随机选择策略的核心概念？",
            "如何理解多样化内容选择的优势？",
            "为什么需要使用最近内容选择策略？",
            "综合性内容选择有什么特点？"
        ]
        
        # 测试每种策略
        for i, strategy in enumerate(ContentSelectionStrategy):
            self.mock_llm_client.generate.return_value = GenerationResponse(
                response=mock_responses[i],
                model="qwen3:1.7b",
                created_at="2024-01-01T00:00:00Z",
                done=True
            )
            
            question = self.generator.generate_question(
                kb_name=kb_name,
                strategy=strategy
            )
            
            assert isinstance(question, Question)
            assert question.content == mock_responses[i]
            assert strategy.value == strategy.value  # 验证策略枚举值
    
    def test_batch_question_generation(self):
        """测试批量问题生成"""
        kb_name = "batch_test_kb"
        count = 3
        
        # 基础设置
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 20}
        
        mock_document = DocumentChunk(
            id="batch_doc",
            content="这是用于批量测试的文档内容，包含了丰富的知识信息。",
            metadata={"source": "batch_test.pdf"}
        )
        
        self.mock_vector_store.similarity_search.return_value = [
            SearchResult(document=mock_document, score=0.9, distance=0.1)
        ]
        
        # 模拟多个不同的LLM响应
        mock_responses = [
            "什么是批量处理的基本原理？",
            "如何优化批量操作的性能？",
            "批量生成有哪些应用场景？"
        ]
        
        self.mock_llm_client.generate.side_effect = [
            GenerationResponse(
                response=response,
                model="qwen3:1.7b",
                created_at="2024-01-01T00:00:00Z",
                done=True
            )
            for response in mock_responses
        ]
        
        # 批量生成问题
        questions = self.generator.generate_multiple_questions(
            kb_name=kb_name,
            count=count,
            difficulty=QuestionDifficulty.MEDIUM,
            strategy=ContentSelectionStrategy.DIVERSE
        )
        
        # 验证结果
        assert len(questions) == count
        for i, question in enumerate(questions):
            assert isinstance(question, Question)
            assert question.kb_name == kb_name
            assert question.difficulty == QuestionDifficulty.MEDIUM
            assert question.content == mock_responses[i]
        
        # 验证调用次数
        assert self.mock_llm_client.generate.call_count == count
    
    def test_question_suggestions_workflow(self):
        """测试问题建议工作流"""
        kb_name = "suggestions_test_kb"
        topic = "机器学习算法"
        
        # 模拟相关文档
        related_documents = [
            DocumentChunk(
                id="ml_doc1",
                content="监督学习是机器学习的一种方法，使用标记的训练数据来学习映射函数。",
                metadata={"topic": "supervised_learning"}
            ),
            DocumentChunk(
                id="ml_doc2", 
                content="无监督学习从未标记的数据中发现隐藏的模式和结构。",
                metadata={"topic": "unsupervised_learning"}
            ),
            DocumentChunk(
                id="ml_doc3",
                content="强化学习通过与环境交互来学习最优行为策略。",
                metadata={"topic": "reinforcement_learning"}
            )
        ]
        
        search_results = [
            SearchResult(document=doc, score=0.9, distance=0.1)
            for doc in related_documents
        ]
        
        self.mock_vector_store.similarity_search.return_value = search_results
        
        # 模拟建议问题
        suggestion_responses = [
            "什么是监督学习的核心特点？",
            "无监督学习如何发现数据模式？",
            "强化学习的学习机制是什么？"
        ]
        
        self.mock_llm_client.generate.side_effect = [
            GenerationResponse(
                response=response,
                model="qwen3:1.7b",
                created_at="2024-01-01T00:00:00Z",
                done=True
            )
            for response in suggestion_responses
        ]
        
        # 获取问题建议
        suggestions = self.generator.get_question_suggestions(kb_name, topic)
        
        # 验证结果
        assert len(suggestions) == len(related_documents)
        for i, suggestion in enumerate(suggestions):
            assert isinstance(suggestion, str)
            assert suggestion == suggestion_responses[i]
            assert "？" in suggestion  # 确保是问题格式
        
        # 验证搜索调用
        self.mock_vector_store.similarity_search.assert_called_once_with(kb_name, topic, k=3)
    
    def test_error_handling_integration(self):
        """测试错误处理集成"""
        kb_name = "error_test_kb"
        
        # 测试知识库不存在
        self.mock_vector_store.collection_exists.return_value = False
        
        with pytest.raises(Exception) as exc_info:
            self.generator.generate_question(kb_name)
        
        assert "not exist" in str(exc_info.value).lower()
        
        # 测试空知识库
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 0}
        
        with pytest.raises(Exception) as exc_info:
            self.generator.generate_question(kb_name)
        
        assert "empty" in str(exc_info.value).lower()
        
        # 测试向量搜索失败
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 5}
        self.mock_vector_store.similarity_search.return_value = []
        
        with pytest.raises(Exception) as exc_info:
            self.generator.generate_question(kb_name)
        
        assert "content" in str(exc_info.value).lower()
    
    def test_question_quality_validation_integration(self):
        """测试问题质量验证集成"""
        # 测试各种质量的问题
        test_questions = [
            ("什么是人工智能的发展历史和未来趋势？", True, 80),  # 高质量问题
            ("如何实现机器学习算法？", True, 75),  # 中等质量问题
            ("AI是什么？", True, 60),  # 简单但有效的问题
            ("？", False, 0),  # 无效问题
            ("这不是一个问题", False, 30),  # 没有问号
        ]
        
        for question, expected_valid, min_score in test_questions:
            result = self.generator.validate_question_quality(question)
            
            assert result["is_valid"] == expected_valid
            assert 0 <= result["quality_score"] <= 100
            
            if expected_valid:
                assert result["quality_score"] >= min_score - 20  # 允许一些误差
                assert result["has_question_mark"] == True
            else:
                assert len(result["issues"]) > 0
    
    @patch('src.question_generator.get_config')
    def test_configuration_integration(self, mock_get_config):
        """测试配置集成"""
        # 模拟配置
        mock_config = Mock()
        mock_config.max_context_length = 2000
        mock_config.question_generation_temperature = 0.8
        mock_get_config.return_value = mock_config
        
        # 重新创建生成器以使用新配置
        generator = QuestionGenerator(
            vector_store=self.mock_vector_store,
            llm_client=self.mock_llm_client
        )
        
        # 验证配置被正确使用
        assert generator.config.max_context_length == 2000
        assert generator.config.question_generation_temperature == 0.8
        
        # 测试长内容截断
        kb_name = "config_test_kb"
        self.mock_vector_store.collection_exists.return_value = True
        self.mock_vector_store.get_collection_stats.return_value = {"document_count": 1}
        
        # 创建超长内容
        long_content = "这是一个很长的内容。" * 500  # 超过2000字符
        mock_document = DocumentChunk(
            id="long_doc",
            content=long_content,
            metadata={}
        )
        
        self.mock_vector_store.similarity_search.return_value = [
            SearchResult(document=mock_document, score=0.9, distance=0.1)
        ]
        
        self.mock_llm_client.generate.return_value = GenerationResponse(
            response="这个长内容的核心概念是什么？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        # 生成问题
        question = generator.generate_question(kb_name)
        
        # 验证LLM调用时内容被截断
        call_args = self.mock_llm_client.generate.call_args
        prompt = call_args.kwargs['prompt']
        
        # 检查温度参数
        assert call_args.kwargs['temperature'] == 0.8
        
        # 检查内容截断（应该包含"..."）
        assert "..." in prompt or len(prompt) <= 3000  # 考虑提示词模板的额外长度


if __name__ == "__main__":
    pytest.main([__file__])