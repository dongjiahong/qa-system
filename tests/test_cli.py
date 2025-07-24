"""
Unit tests for CLI module
CLI模块单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
from click.testing import CliRunner
from rich.console import Console

from src.cli import (
    KnowledgeCLI,
    handle_error,
    validate_file_paths,
    show_progress,
    main,
    create_knowledge_base,
    list_knowledge_bases,
    delete_knowledge_base,
    show_system_status,
    start_new_review,
    show_history
)
from src.models import (
    KnowledgeBase,
    Question,
    EvaluationResult,
    QARecord,
    QuestionDifficulty,
    EvaluationStatus,
    KnowledgeSystemError,
    FileProcessingError,
    ModelServiceError,
    DatabaseError,
    ValidationError,
    KnowledgeBaseNotFoundError,
    VectorStoreError
)


class TestErrorHandling:
    """测试错误处理装饰器"""
    
    def test_handle_error_file_processing_error(self, capsys):
        """测试文件处理错误处理"""
        @handle_error
        def test_func():
            raise FileProcessingError("测试文件错误", {"file": "test.pdf"})
        
        with pytest.raises(SystemExit) as exc_info:
            test_func()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "文件处理错误" in captured.out
        assert "测试文件错误" in captured.out
    
    def test_handle_error_model_service_error(self, capsys):
        """测试模型服务错误处理"""
        @handle_error
        def test_func():
            raise ModelServiceError("Ollama服务不可用")
        
        with pytest.raises(SystemExit) as exc_info:
            test_func()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "模型服务错误" in captured.out
        assert "Ollama服务不可用" in captured.out
        assert "Ollama 服务是否正常运行" in captured.out
    
    def test_handle_error_database_error(self, capsys):
        """测试数据库错误处理"""
        @handle_error
        def test_func():
            raise DatabaseError("数据库连接失败")
        
        with pytest.raises(SystemExit) as exc_info:
            test_func()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "数据库错误" in captured.out
        assert "数据库连接失败" in captured.out
    
    def test_handle_error_vector_store_error(self, capsys):
        """测试向量存储错误处理"""
        @handle_error
        def test_func():
            raise VectorStoreError("ChromaDB连接失败")
        
        with pytest.raises(SystemExit) as exc_info:
            test_func()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "向量存储错误" in captured.out
        assert "ChromaDB连接失败" in captured.out
    
    def test_handle_error_knowledge_base_not_found(self, capsys):
        """测试知识库不存在错误处理"""
        @handle_error
        def test_func():
            raise KnowledgeBaseNotFoundError("知识库不存在")
        
        with pytest.raises(SystemExit) as exc_info:
            test_func()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "知识库不存在" in captured.out
    
    def test_handle_error_validation_error(self, capsys):
        """测试参数验证错误处理"""
        @handle_error
        def test_func():
            raise ValidationError("参数无效")
        
        with pytest.raises(SystemExit) as exc_info:
            test_func()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "参数验证错误" in captured.out
        assert "参数无效" in captured.out
    
    def test_handle_error_keyboard_interrupt(self, capsys):
        """测试键盘中断处理"""
        @handle_error
        def test_func():
            raise KeyboardInterrupt()
        
        with pytest.raises(SystemExit) as exc_info:
            test_func()
        
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "操作已取消" in captured.out
    
    def test_handle_error_generic_exception(self, capsys):
        """测试通用异常处理"""
        @handle_error
        def test_func():
            raise Exception("未知错误")
        
        with pytest.raises(SystemExit) as exc_info:
            test_func()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "未知错误" in captured.out


class TestFileValidation:
    """测试文件路径验证"""
    
    @patch('src.cli.get_config')
    def test_validate_file_paths_success(self, mock_get_config, tmp_path):
        """测试文件路径验证成功"""
        # 创建测试文件
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")
        
        # 模拟配置
        mock_config = Mock()
        mock_config.supported_file_extensions = ['.pdf', '.txt', '.md']
        mock_get_config.return_value = mock_config
        
        result = validate_file_paths([str(test_file)])
        assert len(result) == 1
        assert Path(result[0]).name == "test.pdf"
    
    @patch('src.cli.get_config')
    def test_validate_file_paths_file_not_exists(self, mock_get_config):
        """测试文件不存在"""
        mock_config = Mock()
        mock_config.supported_file_extensions = ['.pdf', '.txt', '.md']
        mock_get_config.return_value = mock_config
        
        with pytest.raises(ValidationError) as exc_info:
            validate_file_paths(["/nonexistent/file.pdf"])
        
        assert "文件不存在" in str(exc_info.value)
    
    @patch('src.cli.get_config')
    def test_validate_file_paths_not_file(self, mock_get_config, tmp_path):
        """测试路径不是文件"""
        # 创建目录
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        mock_config = Mock()
        mock_config.supported_file_extensions = ['.pdf', '.txt', '.md']
        mock_get_config.return_value = mock_config
        
        with pytest.raises(ValidationError) as exc_info:
            validate_file_paths([str(test_dir)])
        
        assert "路径不是文件" in str(exc_info.value)
    
    @patch('src.cli.get_config')
    def test_validate_file_paths_unsupported_format(self, mock_get_config, tmp_path):
        """测试不支持的文件格式"""
        # 创建不支持格式的文件
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")
        
        mock_config = Mock()
        mock_config.supported_file_extensions = ['.pdf', '.txt', '.md']
        mock_get_config.return_value = mock_config
        
        with pytest.raises(ValidationError) as exc_info:
            validate_file_paths([str(test_file)])
        
        assert "不支持的文件格式" in str(exc_info.value)
        assert "支持的格式" in str(exc_info.value)


class TestProgressDisplay:
    """测试进度显示功能"""
    
    def test_show_progress_success(self):
        """测试进度显示成功"""
        def test_task(x, y):
            return x + y
        
        result = show_progress("测试任务", test_task, 1, 2)
        assert result == 3
    
    def test_show_progress_with_exception(self):
        """测试进度显示异常处理"""
        def test_task():
            raise ValueError("测试错误")
        
        with pytest.raises(ValueError):
            show_progress("测试任务", test_task)


class TestKnowledgeCLI:
    """测试KnowledgeCLI类"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    @patch('src.cli.validate_file_paths')
    @patch('src.cli.show_progress')
    def test_create_knowledge_base_success(self, mock_show_progress, mock_validate, cli_instance):
        """测试创建知识库成功"""
        # 模拟验证文件
        mock_validate.return_value = ["/path/to/file.pdf"]
        
        # 模拟创建知识库
        mock_kb = KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=1,
            document_count=5,
            description="测试知识库"
        )
        mock_show_progress.return_value = mock_kb
        
        # 执行测试
        cli_instance.create_knowledge_base("test_kb", ["/path/to/file.pdf"], "测试知识库")
        
        # 验证调用
        mock_validate.assert_called_once_with(["/path/to/file.pdf"])
        mock_show_progress.assert_called_once()
    
    def test_start_new_review_kb_not_found(self, cli_instance):
        """测试开始问答会话 - 知识库不存在"""
        cli_instance.kb_manager.get_knowledge_base.return_value = None
        
        with pytest.raises(KnowledgeBaseNotFoundError):
            cli_instance.start_new_review("nonexistent_kb")
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_start_new_review_quit_immediately(self, mock_show_progress, mock_input, cli_instance):
        """测试开始问答会话 - 立即退出"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟生成问题
        mock_question = Question(
            content="测试问题",
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
    
    def test_show_history_kb_not_found(self, cli_instance):
        """测试显示历史记录 - 知识库不存在"""
        cli_instance.kb_manager.get_knowledge_base.return_value = None
        
        with pytest.raises(KnowledgeBaseNotFoundError):
            cli_instance.show_history("nonexistent_kb")
    
    def test_show_history_no_records(self, cli_instance):
        """测试显示历史记录 - 无记录"""
        from src.history_manager import HistoryPage, PaginationInfo
        
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟无历史记录
        empty_page = HistoryPage(
            records=[],
            pagination=PaginationInfo(page=1, page_size=10, total_count=0)
        )
        cli_instance.history_manager.get_history_page.return_value = empty_page
        cli_instance.history_manager.get_statistics.return_value = {'total_count': 0}
        
        # 执行测试
        cli_instance.show_history("test_kb")
        
        # 验证调用
        cli_instance.history_manager.get_history_page.assert_called_once()
    
    def test_show_history_with_records(self, cli_instance):
        """测试显示历史记录 - 有记录"""
        from src.history_manager import HistoryPage, PaginationInfo
        
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟历史记录
        mock_evaluation = EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="回答正确",
            reference_answer="参考答案"
        )
        mock_record = QARecord(
            id=1,
            kb_name="test_kb",
            question="测试问题",
            user_answer="用户答案",
            evaluation=mock_evaluation,
            created_at=datetime.now()
        )
        
        # 模拟分页结果
        history_page = HistoryPage(
            records=[mock_record],
            pagination=PaginationInfo(page=1, page_size=10, total_count=1, total_pages=1)
        )
        cli_instance.history_manager.get_history_page.return_value = history_page
        cli_instance.history_manager.get_statistics.return_value = {
            'total_count': 1,
            'accuracy_rate': 100.0,
            'average_score': 85.0,
            'recent_activity_count': 1
        }
        
        # 执行测试
        cli_instance.show_history("test_kb")
        
        # 验证调用
        cli_instance.history_manager.get_history_page.assert_called_once()
        cli_instance.history_manager.get_statistics.assert_called_once_with("test_kb")
    
    def test_list_knowledge_bases_empty(self, cli_instance):
        """测试列出知识库 - 空列表"""
        cli_instance.kb_manager.get_knowledge_base_details.return_value = []
        
        # 执行测试
        cli_instance.list_knowledge_bases()
        
        # 验证调用
        cli_instance.kb_manager.get_knowledge_base_details.assert_called_once()
    
    def test_list_knowledge_bases_with_data(self, cli_instance):
        """测试列出知识库 - 有数据"""
        mock_kb = KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=2,
            document_count=10,
            description="测试知识库"
        )
        cli_instance.kb_manager.get_knowledge_base_details.return_value = [mock_kb]
        
        # 执行测试
        cli_instance.list_knowledge_bases()
        
        # 验证调用
        cli_instance.kb_manager.get_knowledge_base_details.assert_called_once()
    
    def test_delete_knowledge_base_not_found(self, cli_instance):
        """测试删除知识库 - 不存在"""
        cli_instance.kb_manager.get_knowledge_base.return_value = None
        
        with pytest.raises(KnowledgeBaseNotFoundError):
            cli_instance.delete_knowledge_base("nonexistent_kb")
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_delete_knowledge_base_cancelled(self, mock_show_progress, mock_input, cli_instance):
        """测试删除知识库 - 取消删除"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟用户取消删除
        mock_input.return_value = "n"
        
        # 执行测试
        cli_instance.delete_knowledge_base("test_kb")
        
        # 验证没有调用删除
        mock_show_progress.assert_not_called()
    
    @patch('src.cli.console.input')
    @patch('src.cli.show_progress')
    def test_delete_knowledge_base_confirmed(self, mock_show_progress, mock_input, cli_instance):
        """测试删除知识库 - 确认删除"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 模拟用户确认删除
        mock_input.return_value = "y"
        mock_show_progress.return_value = True
        
        # 执行测试
        cli_instance.delete_knowledge_base("test_kb")
        
        # 验证调用删除
        mock_show_progress.assert_called_once()
    
    def test_delete_knowledge_base_force(self, cli_instance):
        """测试删除知识库 - 强制删除"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        with patch('src.cli.show_progress') as mock_show_progress:
            mock_show_progress.return_value = True
            
            # 执行测试
            cli_instance.delete_knowledge_base("test_kb", force=True)
            
            # 验证调用删除
            mock_show_progress.assert_called_once()
    
    @patch('src.cli.show_progress')
    def test_show_system_status(self, mock_show_progress, cli_instance):
        """测试显示系统状态"""
        mock_health_info = {
            "status": "healthy",
            "components": {
                "database": {
                    "status": "healthy",
                    "knowledge_base_count": 3
                },
                "vector_store": {
                    "status": "healthy",
                    "embedding_model": "shaw/dmeta-embedding-zh-small-q4"
                },
                "document_processor": {
                    "status": "healthy",
                    "supported_formats": [".pdf", ".txt", ".md"]
                }
            },
            "timestamp": "2024-01-01T12:00:00"
        }
        mock_show_progress.return_value = mock_health_info
        
        # 执行测试
        cli_instance.show_system_status()
        
        # 验证调用
        mock_show_progress.assert_called_once()


class TestClickCommands:
    """测试Click命令"""
    
    def test_main_command_help(self):
        """测试主命令帮助"""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "知识库问答系统" in result.output
    
    def test_main_command_version(self):
        """测试版本信息"""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert "0.1.0" in result.output
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command(self, mock_cli_instance):
        """测试创建知识库命令"""
        runner = CliRunner()
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', 'test1.pdf',
            '--file', 'test2.txt',
            '--description', '测试知识库'
        ])
        
        if result.exit_code != 0:
            print(f"Command output: {result.output}")
            print(f"Exception: {result.exception}")
        
        mock_cli_instance.create_knowledge_base.assert_called_once_with(
            'test_kb', ['test1.pdf', 'test2.txt'], '测试知识库'
        )
    
    @patch('src.cli.cli_instance')
    def test_list_knowledge_bases_command(self, mock_cli_instance):
        """测试列出知识库命令"""
        runner = CliRunner()
        result = runner.invoke(list_knowledge_bases)
        
        if result.exit_code != 0:
            print(f"Command output: {result.output}")
            print(f"Exception: {result.exception}")
        
        mock_cli_instance.list_knowledge_bases.assert_called_once()
    
    @patch('src.cli.cli_instance')
    def test_delete_knowledge_base_command(self, mock_cli_instance):
        """测试删除知识库命令"""
        runner = CliRunner()
        result = runner.invoke(delete_knowledge_base, ['test_kb', '--force'])
        
        if result.exit_code != 0:
            print(f"Command output: {result.output}")
            print(f"Exception: {result.exception}")
        
        mock_cli_instance.delete_knowledge_base.assert_called_once_with('test_kb', True)
    
    @patch('src.cli.cli_instance')
    def test_show_system_status_command(self, mock_cli_instance):
        """测试显示系统状态命令"""
        runner = CliRunner()
        result = runner.invoke(show_system_status)
        
        if result.exit_code != 0:
            print(f"Command output: {result.output}")
            print(f"Exception: {result.exception}")
        
        mock_cli_instance.show_system_status.assert_called_once()
    
    @patch('src.cli.cli_instance')
    def test_start_new_review_command(self, mock_cli_instance):
        """测试开始问答会话命令"""
        runner = CliRunner()
        result = runner.invoke(start_new_review, [], obj={'kb_name': 'test_kb'})
        
        if result.exit_code != 0:
            print(f"Command output: {result.output}")
            print(f"Exception: {result.exception}")
        
        mock_cli_instance.start_new_review.assert_called_once_with('test_kb')
    
    @patch('src.cli.cli_instance')
    def test_show_history_command(self, mock_cli_instance):
        """测试显示历史记录命令"""
        runner = CliRunner()
        result = runner.invoke(show_history, ['--limit', '5', '--page', '2'], obj={'kb_name': 'test_kb'})
        
        if result.exit_code != 0:
            print(f"Command output: {result.output}")
            print(f"Exception: {result.exception}")
        
        mock_cli_instance.show_history.assert_called_once_with(
            'test_kb', 5, 2, False, None, None, None, 'time', 'desc', False
        )


class TestEvaluationResultDisplay:
    """测试评估结果显示"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    def test_display_evaluation_result_correct(self, cli_instance):
        """测试显示正确答案的评估结果"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="回答很好",
            reference_answer="参考答案",
            strengths=["逻辑清晰", "内容准确"],
            missing_points=[]
        )
        
        # 执行测试（不会抛出异常即为成功）
        cli_instance._display_evaluation_result(evaluation)
    
    def test_display_evaluation_result_incorrect(self, cli_instance):
        """测试显示错误答案的评估结果"""
        evaluation = EvaluationResult(
            is_correct=False,
            score=60.0,
            feedback="回答有误",
            reference_answer="参考答案",
            strengths=["部分正确"],
            missing_points=["缺少关键点1", "缺少关键点2"]
        )
        
        # 执行测试（不会抛出异常即为成功）
        cli_instance._display_evaluation_result(evaluation)


if __name__ == "__main__":
    pytest.main([__file__])