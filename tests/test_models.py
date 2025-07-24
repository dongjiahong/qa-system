"""
Unit tests for core data models and exception classes
核心数据模型和异常类的单元测试
"""

import json
import pytest
from datetime import datetime
from unittest.mock import patch

from src.models import (
    # Exception classes
    KnowledgeSystemError,
    FileProcessingError,
    ModelServiceError,
    DatabaseError,
    ValidationError,
    KnowledgeBaseNotFoundError,
    VectorStoreError,
    
    # Enums
    QuestionDifficulty,
    EvaluationStatus,
    
    # Data models
    KnowledgeBase,
    Question,
    EvaluationResult,
    QARecord,
)


# ============================================================================
# Exception Classes Tests 异常类测试
# ============================================================================

class TestExceptionClasses:
    """测试异常类"""
    
    def test_knowledge_system_error_basic(self):
        """测试基础异常类"""
        error = KnowledgeSystemError("测试错误")
        assert str(error) == "测试错误"
        assert error.message == "测试错误"
        assert error.details == {}
        assert isinstance(error.timestamp, datetime)
    
    def test_knowledge_system_error_with_details(self):
        """测试带详细信息的异常"""
        details = {"code": 500, "context": "测试上下文"}
        error = KnowledgeSystemError("测试错误", details)
        assert error.details == details
    
    def test_file_processing_error(self):
        """测试文件处理异常"""
        error = FileProcessingError("文件处理失败")
        assert isinstance(error, KnowledgeSystemError)
        assert str(error) == "文件处理失败"
    
    def test_model_service_error(self):
        """测试模型服务异常"""
        error = ModelServiceError("模型服务不可用")
        assert isinstance(error, KnowledgeSystemError)
        assert str(error) == "模型服务不可用"
    
    def test_database_error(self):
        """测试数据库异常"""
        error = DatabaseError("数据库连接失败")
        assert isinstance(error, KnowledgeSystemError)
        assert str(error) == "数据库连接失败"
    
    def test_validation_error(self):
        """测试验证异常"""
        error = ValidationError("数据验证失败")
        assert isinstance(error, KnowledgeSystemError)
        assert str(error) == "数据验证失败"
    
    def test_knowledge_base_not_found_error(self):
        """测试知识库不存在异常"""
        error = KnowledgeBaseNotFoundError("知识库不存在")
        assert isinstance(error, KnowledgeSystemError)
        assert str(error) == "知识库不存在"
    
    def test_vector_store_error(self):
        """测试向量存储异常"""
        error = VectorStoreError("向量存储操作失败")
        assert isinstance(error, KnowledgeSystemError)
        assert str(error) == "向量存储操作失败"


# ============================================================================
# Enums Tests 枚举测试
# ============================================================================

class TestEnums:
    """测试枚举类"""
    
    def test_question_difficulty_enum(self):
        """测试问题难度枚举"""
        assert QuestionDifficulty.EASY.value == "easy"
        assert QuestionDifficulty.MEDIUM.value == "medium"
        assert QuestionDifficulty.HARD.value == "hard"
        
        # 测试枚举比较
        assert QuestionDifficulty.EASY != QuestionDifficulty.MEDIUM
        assert QuestionDifficulty.MEDIUM == QuestionDifficulty.MEDIUM
    
    def test_evaluation_status_enum(self):
        """测试评估状态枚举"""
        assert EvaluationStatus.SUCCESS.value == "success"
        assert EvaluationStatus.FAILED.value == "failed"
        assert EvaluationStatus.PARTIAL.value == "partial"
        assert EvaluationStatus.ERROR.value == "error"


# ============================================================================
# KnowledgeBase Tests 知识库模型测试
# ============================================================================

class TestKnowledgeBase:
    """测试知识库数据模型"""
    
    def test_knowledge_base_creation_valid(self):
        """测试有效的知识库创建"""
        now = datetime.now()
        kb = KnowledgeBase(
            name="测试知识库",
            created_at=now,
            file_count=5,
            document_count=10,
            description="这是一个测试知识库"
        )
        
        assert kb.name == "测试知识库"
        assert kb.created_at == now
        assert kb.file_count == 5
        assert kb.document_count == 10
        assert kb.description == "这是一个测试知识库"
    
    def test_knowledge_base_creation_minimal(self):
        """测试最小参数创建知识库"""
        now = datetime.now()
        kb = KnowledgeBase(name="测试知识库", created_at=now)
        
        assert kb.name == "测试知识库"
        assert kb.created_at == now
        assert kb.file_count == 0
        assert kb.document_count == 0
        assert kb.description is None
    
    def test_knowledge_base_validation_empty_name(self):
        """测试空名称验证"""
        with pytest.raises(ValidationError, match="知识库名称不能为空"):
            KnowledgeBase(name="", created_at=datetime.now())
        
        with pytest.raises(ValidationError, match="知识库名称不能为空"):
            KnowledgeBase(name="   ", created_at=datetime.now())
    
    def test_knowledge_base_validation_long_name(self):
        """测试名称长度验证"""
        long_name = "a" * 101
        with pytest.raises(ValidationError, match="知识库名称不能超过100个字符"):
            KnowledgeBase(name=long_name, created_at=datetime.now())
    
    def test_knowledge_base_validation_negative_counts(self):
        """测试负数计数验证"""
        with pytest.raises(ValidationError, match="文件数量不能为负数"):
            KnowledgeBase(name="测试", created_at=datetime.now(), file_count=-1)
        
        with pytest.raises(ValidationError, match="文档数量不能为负数"):
            KnowledgeBase(name="测试", created_at=datetime.now(), document_count=-1)
    
    def test_knowledge_base_validation_long_description(self):
        """测试描述长度验证"""
        long_desc = "a" * 501
        with pytest.raises(ValidationError, match="描述不能超过500个字符"):
            KnowledgeBase(name="测试", created_at=datetime.now(), description=long_desc)
    
    def test_knowledge_base_to_dict(self):
        """测试序列化为字典"""
        now = datetime.now()
        kb = KnowledgeBase(
            name="测试知识库",
            created_at=now,
            file_count=5,
            document_count=10,
            description="测试描述"
        )
        
        expected = {
            "name": "测试知识库",
            "created_at": now.isoformat(),
            "file_count": 5,
            "document_count": 10,
            "description": "测试描述",
        }
        
        assert kb.to_dict() == expected
    
    def test_knowledge_base_from_dict(self):
        """测试从字典反序列化"""
        now = datetime.now()
        data = {
            "name": "测试知识库",
            "created_at": now.isoformat(),
            "file_count": 5,
            "document_count": 10,
            "description": "测试描述",
        }
        
        kb = KnowledgeBase.from_dict(data)
        assert kb.name == "测试知识库"
        assert kb.created_at == now
        assert kb.file_count == 5
        assert kb.document_count == 10
        assert kb.description == "测试描述"
    
    def test_knowledge_base_json_serialization(self):
        """测试JSON序列化和反序列化"""
        now = datetime.now()
        kb = KnowledgeBase(
            name="测试知识库",
            created_at=now,
            file_count=5,
            document_count=10,
            description="测试描述"
        )
        
        json_str = kb.to_json()
        kb_restored = KnowledgeBase.from_json(json_str)
        
        assert kb_restored.name == kb.name
        assert kb_restored.created_at == kb.created_at
        assert kb_restored.file_count == kb.file_count
        assert kb_restored.document_count == kb.document_count
        assert kb_restored.description == kb.description


# ============================================================================
# Question Tests 问题模型测试
# ============================================================================

class TestQuestion:
    """测试问题数据模型"""
    
    def test_question_creation_valid(self):
        """测试有效的问题创建"""
        question = Question(
            content="什么是机器学习？",
            kb_name="AI知识库",
            source_context="机器学习是人工智能的一个分支...",
            difficulty=QuestionDifficulty.MEDIUM
        )
        
        assert question.content == "什么是机器学习？"
        assert question.kb_name == "AI知识库"
        assert question.source_context == "机器学习是人工智能的一个分支..."
        assert question.difficulty == QuestionDifficulty.MEDIUM
        assert question.id is not None
        assert isinstance(question.created_at, datetime)
    
    def test_question_creation_default_values(self):
        """测试默认值创建问题"""
        question = Question(
            content="什么是机器学习？",
            kb_name="AI知识库",
            source_context="机器学习是人工智能的一个分支..."
        )
        
        assert question.difficulty == QuestionDifficulty.MEDIUM
        assert question.id is not None
        assert isinstance(question.created_at, datetime)
    
    def test_question_validation_empty_content(self):
        """测试空内容验证"""
        with pytest.raises(ValidationError, match="问题内容不能为空"):
            Question(
                content="",
                kb_name="AI知识库",
                source_context="上下文"
            )
    
    def test_question_validation_long_content(self):
        """测试内容长度验证"""
        long_content = "a" * 1001
        with pytest.raises(ValidationError, match="问题内容不能超过1000个字符"):
            Question(
                content=long_content,
                kb_name="AI知识库",
                source_context="上下文"
            )
    
    def test_question_validation_empty_kb_name(self):
        """测试空知识库名称验证"""
        with pytest.raises(ValidationError, match="知识库名称不能为空"):
            Question(
                content="问题",
                kb_name="",
                source_context="上下文"
            )
    
    def test_question_validation_empty_context(self):
        """测试空上下文验证"""
        with pytest.raises(ValidationError, match="问题来源上下文不能为空"):
            Question(
                content="问题",
                kb_name="知识库",
                source_context=""
            )
    
    def test_question_validation_long_context(self):
        """测试上下文长度验证"""
        long_context = "a" * 5001
        with pytest.raises(ValidationError, match="问题来源上下文不能超过5000个字符"):
            Question(
                content="问题",
                kb_name="知识库",
                source_context=long_context
            )
    
    def test_question_validation_invalid_difficulty(self):
        """测试无效难度验证"""
        question = Question(
            content="问题",
            kb_name="知识库",
            source_context="上下文"
        )
        # 直接设置无效的难度值来触发验证错误
        question.difficulty = "invalid"
        with pytest.raises(ValidationError, match="问题难度必须是有效的枚举值"):
            question.validate()
    
    def test_question_json_serialization(self):
        """测试JSON序列化和反序列化"""
        question = Question(
            content="什么是机器学习？",
            kb_name="AI知识库",
            source_context="机器学习是人工智能的一个分支...",
            difficulty=QuestionDifficulty.HARD
        )
        
        json_str = question.to_json()
        question_restored = Question.from_json(json_str)
        
        assert question_restored.content == question.content
        assert question_restored.kb_name == question.kb_name
        assert question_restored.source_context == question.source_context
        assert question_restored.difficulty == question.difficulty
        assert question_restored.id == question.id
        assert question_restored.created_at == question.created_at


# ============================================================================
# EvaluationResult Tests 评估结果模型测试
# ============================================================================

class TestEvaluationResult:
    """测试评估结果数据模型"""
    
    def test_evaluation_result_creation_valid(self):
        """测试有效的评估结果创建"""
        result = EvaluationResult(
            is_correct=True,
            score=85.5,
            feedback="答案基本正确，但缺少一些细节。",
            reference_answer="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
            missing_points=["缺少算法类型说明"],
            strengths=["概念理解正确", "表达清晰"],
            status=EvaluationStatus.SUCCESS
        )
        
        assert result.is_correct is True
        assert result.score == 85.5
        assert result.feedback == "答案基本正确，但缺少一些细节。"
        assert result.reference_answer == "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。"
        assert result.missing_points == ["缺少算法类型说明"]
        assert result.strengths == ["概念理解正确", "表达清晰"]
        assert result.status == EvaluationStatus.SUCCESS
    
    def test_evaluation_result_creation_minimal(self):
        """测试最小参数创建评估结果"""
        result = EvaluationResult(
            is_correct=False,
            score=60.0,
            feedback="答案不够准确。",
            reference_answer="正确答案是..."
        )
        
        assert result.missing_points == []
        assert result.strengths == []
        assert result.status == EvaluationStatus.SUCCESS
    
    def test_evaluation_result_validation_score_range(self):
        """测试分数范围验证"""
        with pytest.raises(ValidationError, match="分数必须在0-100之间"):
            EvaluationResult(
                is_correct=True,
                score=-1,
                feedback="反馈",
                reference_answer="参考答案"
            )
        
        with pytest.raises(ValidationError, match="分数必须在0-100之间"):
            EvaluationResult(
                is_correct=True,
                score=101,
                feedback="反馈",
                reference_answer="参考答案"
            )
    
    def test_evaluation_result_validation_empty_feedback(self):
        """测试空反馈验证"""
        with pytest.raises(ValidationError, match="反馈内容不能为空"):
            EvaluationResult(
                is_correct=True,
                score=80,
                feedback="",
                reference_answer="参考答案"
            )
    
    def test_evaluation_result_validation_long_feedback(self):
        """测试反馈长度验证"""
        long_feedback = "a" * 2001
        with pytest.raises(ValidationError, match="反馈内容不能超过2000个字符"):
            EvaluationResult(
                is_correct=True,
                score=80,
                feedback=long_feedback,
                reference_answer="参考答案"
            )
    
    def test_evaluation_result_validation_empty_reference(self):
        """测试空参考答案验证"""
        with pytest.raises(ValidationError, match="参考答案不能为空"):
            EvaluationResult(
                is_correct=True,
                score=80,
                feedback="反馈",
                reference_answer=""
            )
    
    def test_evaluation_result_validation_long_reference(self):
        """测试参考答案长度验证"""
        long_reference = "a" * 2001
        with pytest.raises(ValidationError, match="参考答案不能超过2000个字符"):
            EvaluationResult(
                is_correct=True,
                score=80,
                feedback="反馈",
                reference_answer=long_reference
            )
    
    def test_evaluation_result_validation_invalid_is_correct(self):
        """测试无效is_correct类型验证"""
        result = EvaluationResult(
            is_correct=True,
            score=80,
            feedback="反馈",
            reference_answer="参考答案"
        )
        # 直接设置无效类型来触发验证错误
        result.is_correct = "invalid"
        with pytest.raises(ValidationError, match="is_correct必须是布尔值"):
            result.validate()
    
    def test_evaluation_result_validation_invalid_status(self):
        """测试无效状态验证"""
        result = EvaluationResult(
            is_correct=True,
            score=80,
            feedback="反馈",
            reference_answer="参考答案"
        )
        # 直接设置无效状态来触发验证错误
        result.status = "invalid"
        with pytest.raises(ValidationError, match="评估状态必须是有效的枚举值"):
            result.validate()
    
    def test_evaluation_result_json_serialization(self):
        """测试JSON序列化和反序列化"""
        result = EvaluationResult(
            is_correct=True,
            score=85.5,
            feedback="答案基本正确。",
            reference_answer="参考答案",
            missing_points=["缺少细节"],
            strengths=["理解正确"],
            status=EvaluationStatus.PARTIAL
        )
        
        json_str = result.to_json()
        result_restored = EvaluationResult.from_json(json_str)
        
        assert result_restored.is_correct == result.is_correct
        assert result_restored.score == result.score
        assert result_restored.feedback == result.feedback
        assert result_restored.reference_answer == result.reference_answer
        assert result_restored.missing_points == result.missing_points
        assert result_restored.strengths == result.strengths
        assert result_restored.status == result.status


# ============================================================================
# QARecord Tests 问答记录模型测试
# ============================================================================

class TestQARecord:
    """测试问答记录数据模型"""
    
    def test_qa_record_creation_valid(self):
        """测试有效的问答记录创建"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="答案正确。",
            reference_answer="参考答案"
        )
        
        record = QARecord(
            kb_name="AI知识库",
            question="什么是机器学习？",
            user_answer="机器学习是AI的一个分支。",
            evaluation=evaluation
        )
        
        assert record.kb_name == "AI知识库"
        assert record.question == "什么是机器学习？"
        assert record.user_answer == "机器学习是AI的一个分支。"
        assert record.evaluation == evaluation
        assert record.id is None
        assert isinstance(record.created_at, datetime)
    
    def test_qa_record_validation_empty_kb_name(self):
        """测试空知识库名称验证"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="答案正确。",
            reference_answer="参考答案"
        )
        
        with pytest.raises(ValidationError, match="知识库名称不能为空"):
            QARecord(
                kb_name="",
                question="问题",
                user_answer="答案",
                evaluation=evaluation
            )
    
    def test_qa_record_validation_empty_question(self):
        """测试空问题验证"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="答案正确。",
            reference_answer="参考答案"
        )
        
        with pytest.raises(ValidationError, match="问题内容不能为空"):
            QARecord(
                kb_name="知识库",
                question="",
                user_answer="答案",
                evaluation=evaluation
            )
    
    def test_qa_record_validation_long_question(self):
        """测试问题长度验证"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="答案正确。",
            reference_answer="参考答案"
        )
        
        long_question = "a" * 1001
        with pytest.raises(ValidationError, match="问题内容不能超过1000个字符"):
            QARecord(
                kb_name="知识库",
                question=long_question,
                user_answer="答案",
                evaluation=evaluation
            )
    
    def test_qa_record_validation_empty_answer(self):
        """测试空答案验证"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="答案正确。",
            reference_answer="参考答案"
        )
        
        with pytest.raises(ValidationError, match="用户答案不能为空"):
            QARecord(
                kb_name="知识库",
                question="问题",
                user_answer="",
                evaluation=evaluation
            )
    
    def test_qa_record_validation_long_answer(self):
        """测试答案长度验证"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="答案正确。",
            reference_answer="参考答案"
        )
        
        long_answer = "a" * 2001
        with pytest.raises(ValidationError, match="用户答案不能超过2000个字符"):
            QARecord(
                kb_name="知识库",
                question="问题",
                user_answer=long_answer,
                evaluation=evaluation
            )
    
    def test_qa_record_validation_invalid_evaluation_type(self):
        """测试无效评估结果类型验证"""
        record = QARecord(
            kb_name="知识库",
            question="问题",
            user_answer="答案",
            evaluation=EvaluationResult(
                is_correct=True,
                score=90.0,
                feedback="反馈",
                reference_answer="参考答案"
            )
        )
        # 直接设置无效类型来触发验证错误
        record.evaluation = "invalid"
        with pytest.raises(ValidationError, match="评估结果必须是EvaluationResult类型"):
            record.validate()
    
    def test_qa_record_json_serialization(self):
        """测试JSON序列化和反序列化"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="答案正确。",
            reference_answer="参考答案"
        )
        
        record = QARecord(
            kb_name="AI知识库",
            question="什么是机器学习？",
            user_answer="机器学习是AI的一个分支。",
            evaluation=evaluation,
            id=123
        )
        
        json_str = record.to_json()
        record_restored = QARecord.from_json(json_str)
        
        assert record_restored.kb_name == record.kb_name
        assert record_restored.question == record.question
        assert record_restored.user_answer == record.user_answer
        assert record_restored.id == record.id
        assert record_restored.created_at == record.created_at
        
        # 验证评估结果
        assert record_restored.evaluation.is_correct == evaluation.is_correct
        assert record_restored.evaluation.score == evaluation.score
        assert record_restored.evaluation.feedback == evaluation.feedback
        assert record_restored.evaluation.reference_answer == evaluation.reference_answer
    
    def test_qa_record_get_summary(self):
        """测试获取记录摘要"""
        evaluation_correct = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="答案正确。",
            reference_answer="参考答案"
        )
        
        evaluation_incorrect = EvaluationResult(
            is_correct=False,
            score=60.0,
            feedback="答案不正确。",
            reference_answer="参考答案"
        )
        
        # 测试正确答案摘要
        record_correct = QARecord(
            kb_name="知识库",
            question="问题",
            user_answer="答案",
            evaluation=evaluation_correct
        )
        
        summary = record_correct.get_summary()
        assert "正确" in summary
        assert "90.0" in summary
        
        # 测试错误答案摘要
        record_incorrect = QARecord(
            kb_name="知识库",
            question="问题",
            user_answer="答案",
            evaluation=evaluation_incorrect
        )
        
        summary = record_incorrect.get_summary()
        assert "错误" in summary
        assert "60.0" in summary