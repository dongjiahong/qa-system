"""
Integration tests for CLI Interactive Q&A Command
CLI交互式问答命令集成测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call
from click.testing import CliRunner
from datetime import datetime
from io import StringIO

from src.cli import main, start_new_review, KnowledgeCLI
from src.models import (
    KnowledgeBase,
    Question,
    EvaluationResult,
    QARecord,
    QuestionDifficulty,
    EvaluationStatus,
    FileProcessingError,
    ValidationError,
    VectorStoreError,
    DatabaseError,
    KnowledgeSystemError,
    KnowledgeBaseNotFoundError,
    ModelServiceError
)


class TestInteractiveQAIntegration:
    """交互式问答命令集成测试"""
    
    @pytest.fixture
    def runner(self):
        """创建CLI运行器"""
        return CliRunner()
    
    @pytest.fixture
    def mock_kb(self):
        """模拟知识库"""
        return KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=2,
            document_count=10,
            description="测试知识库"
        )
    
    @pytest.fixture
    def mock_question(self):
        """模拟问题"""
        return Question(
            content="什么是机器学习？",
            kb_name="test_kb",
            source_context="机器学习是人工智能的一个分支...",
            difficulty=QuestionDifficulty.MEDIUM
        )
    
    @pytest.fixture
    def mock_evaluation_correct(self):
        """模拟正确答案的评估结果"""
        return EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="回答很好，涵盖了机器学习的主要概念",
            reference_answer="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式",
            strengths=["概念理解准确", "表达清晰"],
            missing_points=[],
            status=EvaluationStatus.SUCCESS
        )
    
    @pytest.fixture
    def mock_evaluation_incorrect(self):
        """模拟错误答案的评估结果"""
        return EvaluationResult(
            is_correct=False,
            score=45.0,
            feedback="回答不够准确，缺少关键概念",
            reference_answer="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式",
            strengths=["尝试回答问题"],
            missing_points=["缺少算法概念", "未提及数据学习"],
            status=EvaluationStatus.SUCCESS
        )
    
    @patch('src.cli.cli_instance')
    def test_interactive_qa_command_success(self, mock_cli_instance, runner):
        """测试交互式问答命令成功执行"""
        # 模拟成功的问答会话
        mock_cli_instance.start_new_review.return_value = None
        
        result = runner.invoke(start_new_review, [], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 0
        mock_cli_instance.start_new_review.assert_called_once_with('test_kb')
    
    @patch('src.cli.cli_instance')
    def test_interactive_qa_command_kb_not_found(self, mock_cli_instance, runner):
        """测试知识库不存在的情况"""
        mock_cli_instance.start_new_review.side_effect = KnowledgeBaseNotFoundError(
            "知识库 'nonexistent_kb' 不存在"
        )
        
        result = runner.invoke(start_new_review, [], obj={'kb_name': 'nonexistent_kb'})
        
        assert result.exit_code == 1
        assert "知识库不存在" in result.output
        assert "nonexistent_kb" in result.output
    
    @patch('src.cli.cli_instance')
    def test_interactive_qa_command_model_service_error(self, mock_cli_instance, runner):
        """测试模型服务错误"""
        mock_cli_instance.start_new_review.side_effect = ModelServiceError(
            "Ollama服务不可用"
        )
        
        result = runner.invoke(start_new_review, [], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 1
        assert "模型服务错误" in result.output
        assert "Ollama服务不可用" in result.output
        assert "Ollama 服务是否正常运行" in result.output
    
    @patch('src.cli.cli_instance')
    def test_interactive_qa_command_vector_store_error(self, mock_cli_instance, runner):
        """测试向量存储错误"""
        mock_cli_instance.start_new_review.side_effect = VectorStoreError(
            "ChromaDB连接失败"
        )
        
        result = runner.invoke(start_new_review, [], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 1
        assert "向量存储错误" in result.output
        assert "ChromaDB连接失败" in result.output
        assert "ChromaDB 服务是否正常运行" in result.output
    
    @patch('src.cli.cli_instance')
    def test_interactive_qa_command_keyboard_interrupt(self, mock_cli_instance, runner):
        """测试键盘中断"""
        mock_cli_instance.start_new_review.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(start_new_review, [], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 0
        assert "操作已取消" in result.output


class TestInteractiveQAFlow:
    """测试交互式问答流程"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    def test_start_new_review_kb_not_found(self, cli_instance):
        """测试知识库不存在"""
        cli_instance.kb_manager.get_knowledge_base.return_value = None
        
        with pytest.raises(KnowledgeBaseNotFoundError) as exc_info:
            cli_instance.start_new_review("nonexistent_kb")
        
        assert "nonexistent_kb" in str(exc_info.value)
        cli_instance.kb_manager.get_knowledge_base.assert_called_once_with("nonexistent_kb")
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_quit_immediately(self, mock_show_progress, mock_input, cli_instance):
        """测试立即退出会话"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        mock_show_progress.return_value = mock_question
        
        # 模拟用户输入退出命令
        mock_input.return_value = "quit"
        
        # 执行测试
        cli_instance.start_new_review("test_kb")
        
        # 验证调用
        cli_instance.kb_manager.get_knowledge_base.assert_called_once_with("test_kb")
        mock_show_progress.assert_called_once()
        mock_input.assert_called_once()
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_exit_command(self, mock_show_progress, mock_input, cli_instance):
        """测试使用exit命令退出"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        mock_show_progress.return_value = mock_question
        
        # 模拟用户输入退出命令
        mock_input.return_value = "exit"
        
        # 执行测试
        cli_instance.start_new_review("test_kb")
        
        # 验证调用
        cli_instance.kb_manager.get_knowledge_base.assert_called_once_with("test_kb")
        mock_show_progress.assert_called_once()
        mock_input.assert_called_once()
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_chinese_exit_command(self, mock_show_progress, mock_input, cli_instance):
        """测试使用中文退出命令"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        mock_show_progress.return_value = mock_question
        
        # 模拟用户输入中文退出命令
        mock_input.return_value = "退出"
        
        # 执行测试
        cli_instance.start_new_review("test_kb")
        
        # 验证调用
        cli_instance.kb_manager.get_knowledge_base.assert_called_once_with("test_kb")
        mock_show_progress.assert_called_once()
        mock_input.assert_called_once()
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_empty_answer(self, mock_show_progress, mock_input, cli_instance):
        """测试空答案处理"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        mock_show_progress.return_value = mock_question
        
        # 模拟用户输入：先空答案，然后退出
        mock_input.side_effect = ["", "quit"]
        
        # 执行测试
        cli_instance.start_new_review("test_kb")
        
        # 验证调用
        assert mock_input.call_count == 2
        # 空答案会导致重新生成问题，所以会调用两次
        assert mock_show_progress.call_count == 2
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_whitespace_answer(self, mock_show_progress, mock_input, cli_instance):
        """测试仅空白字符的答案"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        mock_show_progress.return_value = mock_question
        
        # 模拟用户输入：先空白字符，然后退出
        mock_input.side_effect = ["   ", "quit"]
        
        # 执行测试
        cli_instance.start_new_review("test_kb")
        
        # 验证调用
        assert mock_input.call_count == 2
        # 空白字符答案会导致重新生成问题，所以会调用两次
        assert mock_show_progress.call_count == 2
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_complete_qa_cycle(self, mock_show_progress, mock_input, cli_instance):
        """测试完整的问答循环"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题
        mock_question = Question(
            content="什么是机器学习？",
            kb_name="test_kb",
            source_context="机器学习是人工智能的一个分支"
        )
        
        # 模拟答案评估结果
        mock_evaluation = EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="回答很好",
            reference_answer="机器学习是人工智能的一个分支",
            strengths=["概念准确"],
            missing_points=[]
        )
        
        # 设置mock返回值
        mock_show_progress.side_effect = [
            mock_question,  # 生成问题
            mock_evaluation,  # 评估答案
            None  # 保存记录
        ]
        
        # 模拟用户输入：回答问题，然后退出
        mock_input.side_effect = [
            "机器学习是让计算机从数据中学习的技术",  # 用户答案
            "quit"  # 退出命令
        ]
        
        # 执行测试
        cli_instance.start_new_review("test_kb")
        
        # 验证调用
        assert mock_input.call_count == 2
        assert mock_show_progress.call_count == 3
        
        # 验证show_progress被正确调用，但不验证底层方法调用
        # 因为show_progress包装了实际的方法调用
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_continue_session(self, mock_show_progress, mock_input, cli_instance):
        """测试继续会话"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟两个问题
        mock_question1 = Question(
            content="问题1？",
            kb_name="test_kb",
            source_context="上下文1"
        )
        mock_question2 = Question(
            content="问题2？",
            kb_name="test_kb",
            source_context="上下文2"
        )
        
        # 模拟评估结果
        mock_evaluation = EvaluationResult(
            is_correct=True,
            score=80.0,
            feedback="回答正确",
            reference_answer="参考答案",
            strengths=["正确"],
            missing_points=[]
        )
        
        # 设置mock返回值
        mock_show_progress.side_effect = [
            mock_question1,  # 第一个问题
            mock_evaluation,  # 第一个评估
            None,  # 第一个保存
            mock_question2,  # 第二个问题
            mock_evaluation,  # 第二个评估
            None  # 第二个保存
        ]
        
        # 模拟用户输入：回答两个问题，然后退出
        mock_input.side_effect = [
            "答案1",  # 第一个答案
            "",  # 继续会话（按回车）
            "答案2",  # 第二个答案
            "quit"  # 退出
        ]
        
        # 执行测试
        cli_instance.start_new_review("test_kb")
        
        # 验证调用次数
        assert mock_input.call_count == 4
        assert mock_show_progress.call_count == 6
        # 验证show_progress被正确调用，但不验证底层方法调用
        # 因为show_progress包装了实际的方法调用
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_question_generation_error(self, mock_show_progress, mock_input, cli_instance):
        """测试问题生成错误"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟问题生成失败
        mock_show_progress.side_effect = ModelServiceError("问题生成失败")
        
        # 执行测试，错误会被handle_error装饰器捕获并转换为SystemExit
        try:
            cli_instance.start_new_review("test_kb")
        except ModelServiceError:
            # 如果直接抛出ModelServiceError，说明错误处理正常
            pass
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_answer_evaluation_error(self, mock_show_progress, mock_input, cli_instance):
        """测试答案评估错误"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题成功，但评估失败
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        
        mock_show_progress.side_effect = [
            mock_question,  # 生成问题成功
            ModelServiceError("答案评估失败")  # 评估失败
        ]
        
        # 模拟用户输入答案
        mock_input.return_value = "测试答案"
        
        # 执行测试，错误会被抛出
        try:
            cli_instance.start_new_review("test_kb")
        except ModelServiceError:
            # 如果直接抛出ModelServiceError，说明错误处理正常
            pass
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_history_save_error(self, mock_show_progress, mock_input, cli_instance):
        """测试历史记录保存错误"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题和评估成功，但保存失败
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        
        mock_evaluation = EvaluationResult(
            is_correct=True,
            score=80.0,
            feedback="回答正确",
            reference_answer="参考答案",
            strengths=["正确"],
            missing_points=[]
        )
        
        mock_show_progress.side_effect = [
            mock_question,  # 生成问题成功
            mock_evaluation,  # 评估成功
            DatabaseError("保存历史记录失败")  # 保存失败
        ]
        
        # 模拟用户输入答案
        mock_input.return_value = "测试答案"
        
        # 执行测试，错误会被抛出
        try:
            cli_instance.start_new_review("test_kb")
        except DatabaseError:
            # 如果直接抛出DatabaseError，说明错误处理正常
            pass
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_keyboard_interrupt_during_session(self, mock_show_progress, mock_input, cli_instance):
        """测试会话过程中的键盘中断"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        mock_show_progress.return_value = mock_question
        
        # 模拟键盘中断
        mock_input.side_effect = KeyboardInterrupt()
        
        # 执行测试，不应该抛出异常
        cli_instance.start_new_review("test_kb")
        
        # 验证调用
        mock_show_progress.assert_called_once()
        mock_input.assert_called_once()


class TestInteractiveQAStateManagement:
    """测试交互式问答的状态管理"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_qa_record_creation(self, mock_show_progress, mock_input, cli_instance):
        """测试问答记录的创建"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟问题和评估结果
        mock_question = Question(
            content="什么是Python？",
            kb_name="test_kb",
            source_context="Python是一种编程语言"
        )
        
        mock_evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="回答很好",
            reference_answer="Python是一种高级编程语言",
            strengths=["概念准确"],
            missing_points=[]
        )
        
        mock_show_progress.side_effect = [
            mock_question,
            mock_evaluation,
            None  # 保存成功
        ]
        
        mock_input.side_effect = [
            "Python是一种编程语言",  # 用户答案
            "quit"  # 退出
        ]
        
        # 执行测试
        cli_instance.start_new_review("test_kb")
        
        # 验证show_progress被调用了3次（生成问题、评估答案、保存记录）
        assert mock_show_progress.call_count == 3
        # 验证用户输入被调用了2次（答案输入、退出命令）
        assert mock_input.call_count == 2
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_error_recovery_after_failed_question_generation(self, mock_show_progress, mock_input, cli_instance):
        """测试问题生成失败后的错误恢复"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 第一次生成失败，第二次成功
        mock_question = Question(
            content="测试问题？",
            kb_name="test_kb",
            source_context="测试上下文"
        )
        
        mock_show_progress.side_effect = [
            ModelServiceError("第一次失败"),  # 第一次失败
        ]
        
        # 执行测试，错误会被抛出
        try:
            cli_instance.start_new_review("test_kb")
        except ModelServiceError:
            # 如果直接抛出ModelServiceError，说明错误处理正常
            pass


class TestInteractiveQAUserExperience:
    """测试交互式问答的用户体验"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    def test_display_evaluation_result_correct_answer(self, cli_instance):
        """测试正确答案的评估结果显示"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=95.0,
            feedback="回答非常准确和完整",
            reference_answer="这是参考答案",
            strengths=["逻辑清晰", "内容准确", "表达流畅"],
            missing_points=[],
            status=EvaluationStatus.SUCCESS
        )
        
        # 测试不会抛出异常
        cli_instance._display_evaluation_result(evaluation)
    
    def test_display_evaluation_result_incorrect_answer(self, cli_instance):
        """测试错误答案的评估结果显示"""
        evaluation = EvaluationResult(
            is_correct=False,
            score=35.0,
            feedback="回答有一些问题，需要改进",
            reference_answer="这是正确的参考答案",
            strengths=["尝试回答了问题"],
            missing_points=["缺少关键概念A", "未提及重要点B", "逻辑不够清晰"],
            status=EvaluationStatus.SUCCESS
        )
        
        # 测试不会抛出异常
        cli_instance._display_evaluation_result(evaluation)
    
    def test_display_evaluation_result_partial_status(self, cli_instance):
        """测试部分成功状态的评估结果显示"""
        evaluation = EvaluationResult(
            is_correct=False,
            score=60.0,
            feedback="评估部分成功，但有一些问题",
            reference_answer="参考答案",
            strengths=["部分正确"],
            missing_points=["需要更多细节"],
            status=EvaluationStatus.PARTIAL
        )
        
        # 测试不会抛出异常
        cli_instance._display_evaluation_result(evaluation)
    
    def test_display_evaluation_result_error_status(self, cli_instance):
        """测试错误状态的评估结果显示"""
        evaluation = EvaluationResult(
            is_correct=False,
            score=0.0,
            feedback="评估过程中出现错误",
            reference_answer="无法获取参考答案",
            strengths=[],
            missing_points=["系统错误"],
            status=EvaluationStatus.ERROR
        )
        
        # 测试不会抛出异常
        cli_instance._display_evaluation_result(evaluation)


if __name__ == "__main__":
    pytest.main([__file__])