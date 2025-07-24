"""
Enhanced History Command Tests for Knowledge QA System CLI
测试增强的历史查看命令功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from click.testing import CliRunner

from src.cli import KnowledgeCLI, main
from src.models import (
    KnowledgeBase, QARecord, EvaluationResult, EvaluationStatus,
    KnowledgeBaseNotFoundError
)
from src.history_manager import HistoryPage, PaginationInfo, HistoryFilter, SortField, SortOrder


class TestEnhancedHistoryCommand:
    """测试增强的历史查看命令"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例的fixture"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    @pytest.fixture
    def mock_kb(self):
        """创建模拟知识库"""
        return KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=1,
            document_count=5
        )
    
    @pytest.fixture
    def mock_qa_record(self):
        """创建模拟问答记录"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="回答正确",
            reference_answer="参考答案",
            missing_points=[],
            strengths=["理解准确"],
            status=EvaluationStatus.SUCCESS
        )
        return QARecord(
            id=1,
            kb_name="test_kb",
            question="测试问题",
            user_answer="用户答案",
            evaluation=evaluation,
            created_at=datetime.now()
        )
    
    @pytest.fixture
    def mock_history_page(self, mock_qa_record):
        """创建模拟历史分页结果"""
        pagination = PaginationInfo(
            page=1,
            page_size=10,
            total_count=1,
            total_pages=1,
            has_next=False,
            has_prev=False
        )
        return HistoryPage(
            records=[mock_qa_record],
            pagination=pagination
        )
    
    def test_show_history_basic(self, cli_instance, mock_kb, mock_history_page):
        """测试基本历史记录显示"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        cli_instance.history_manager.get_history_page.return_value = mock_history_page
        cli_instance.history_manager.get_statistics.return_value = {
            'total_count': 1,
            'accuracy_rate': 100.0,
            'average_score': 85.0,
            'recent_activity_count': 1
        }
        
        # 执行测试
        cli_instance.show_history("test_kb")
        
        # 验证调用
        cli_instance.kb_manager.get_knowledge_base.assert_called_once_with("test_kb")
        cli_instance.history_manager.get_history_page.assert_called_once_with(
            "test_kb", 1, 10, SortField.CREATED_AT, SortOrder.DESC
        )
        cli_instance.history_manager.get_statistics.assert_called_once_with("test_kb")
    
    def test_show_history_with_filters(self, cli_instance, mock_kb, mock_history_page):
        """测试带过滤条件的历史记录显示"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        cli_instance.history_manager.get_filtered_history.return_value = mock_history_page
        cli_instance.history_manager.get_statistics.return_value = {
            'total_count': 1,
            'accuracy_rate': 100.0,
            'average_score': 85.0,
            'recent_activity_count': 1
        }
        
        # 执行测试
        cli_instance.show_history(
            "test_kb", 
            filter_correct=True, 
            min_score=80.0,
            search="测试"
        )
        
        # 验证调用
        cli_instance.history_manager.get_filtered_history.assert_called_once()
        call_args = cli_instance.history_manager.get_filtered_history.call_args
        filter_criteria = call_args[0][0]
        
        assert filter_criteria.kb_name == "test_kb"
        assert filter_criteria.is_correct == True
        assert filter_criteria.min_score == 80.0
        assert filter_criteria.question_contains == "测试"
    
    def test_show_history_detailed_view(self, cli_instance, mock_kb, mock_history_page):
        """测试详细视图显示"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        cli_instance.history_manager.get_history_page.return_value = mock_history_page
        cli_instance.history_manager.get_statistics.return_value = {
            'total_count': 1,
            'accuracy_rate': 100.0,
            'average_score': 85.0,
            'recent_activity_count': 1
        }
        
        # 执行测试
        cli_instance.show_history("test_kb", detailed=True)
        
        # 验证调用
        cli_instance.history_manager.get_history_page.assert_called_once()
    
    def test_show_history_kb_not_found(self, cli_instance):
        """测试知识库不存在"""
        cli_instance.kb_manager.get_knowledge_base.return_value = None
        
        with pytest.raises(KnowledgeBaseNotFoundError):
            cli_instance.show_history("nonexistent_kb")
    
    def test_show_history_no_records(self, cli_instance, mock_kb):
        """测试无历史记录"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        empty_page = HistoryPage(
            records=[],
            pagination=PaginationInfo(page=1, page_size=10, total_count=0)
        )
        cli_instance.history_manager.get_history_page.return_value = empty_page
        
        # 执行测试
        cli_instance.show_history("test_kb")
        
        # 验证调用
        cli_instance.history_manager.get_history_page.assert_called_once()
    
    def test_show_history_detail(self, cli_instance, mock_kb, mock_qa_record):
        """测试显示单个记录详情"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        cli_instance.history_manager.get_record_by_id.return_value = mock_qa_record
        
        # 执行测试
        cli_instance.show_history_detail("test_kb", 1)
        
        # 验证调用
        cli_instance.history_manager.get_record_by_id.assert_called_once_with(1)
    
    def test_show_history_detail_not_found(self, cli_instance, mock_kb):
        """测试记录不存在"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        cli_instance.history_manager.get_record_by_id.return_value = None
        
        # 执行测试
        cli_instance.show_history_detail("test_kb", 999)
        
        # 验证调用
        cli_instance.history_manager.get_record_by_id.assert_called_once_with(999)
    
    def test_show_history_detail_wrong_kb(self, cli_instance, mock_kb, mock_qa_record):
        """测试记录属于其他知识库"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        mock_qa_record.kb_name = "other_kb"
        cli_instance.history_manager.get_record_by_id.return_value = mock_qa_record
        
        # 执行测试
        cli_instance.show_history_detail("test_kb", 1)
        
        # 验证调用
        cli_instance.history_manager.get_record_by_id.assert_called_once_with(1)
    
    def test_export_history_json(self, cli_instance, mock_kb):
        """测试导出JSON格式历史记录"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        cli_instance.history_manager.export_history.return_value = '{"data": "test"}'
        
        # 执行测试
        cli_instance.export_history("test_kb", "json")
        
        # 验证调用
        cli_instance.history_manager.export_history.assert_called_once_with("test_kb", "json")
    
    def test_export_history_csv_to_file(self, cli_instance, mock_kb, tmp_path):
        """测试导出CSV格式到文件"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        cli_instance.history_manager.export_history.return_value = "id,question,answer\n1,test,test"
        
        # 创建输出文件路径
        output_file = tmp_path / "export.csv"
        
        # 执行测试
        cli_instance.export_history("test_kb", "csv", str(output_file))
        
        # 验证文件创建
        assert output_file.exists()
        assert "id,question,answer" in output_file.read_text()
    
    def test_sort_options(self, cli_instance, mock_kb, mock_history_page):
        """测试排序选项"""
        # 设置模拟
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        cli_instance.history_manager.get_history_page.return_value = mock_history_page
        cli_instance.history_manager.get_statistics.return_value = {
            'total_count': 1,
            'accuracy_rate': 100.0,
            'average_score': 85.0,
            'recent_activity_count': 1
        }
        
        # 测试按分数排序
        cli_instance.show_history("test_kb", sort_by="score", sort_order="asc")
        
        # 验证调用
        cli_instance.history_manager.get_history_page.assert_called_with(
            "test_kb", 1, 10, SortField.SCORE, SortOrder.ASC
        )


class TestHistoryClickCommands:
    """测试历史记录Click命令"""
    
    @patch('src.cli.cli_instance')
    def test_history_command_basic(self, mock_cli_instance):
        """测试基本历史命令"""
        from src.cli import show_history
        runner = CliRunner()
        result = runner.invoke(show_history, [], obj={'kb_name': 'test_kb'})
        
        if result.exit_code != 0:
            print(f"Command output: {result.output}")
            print(f"Exception: {result.exception}")
        
        assert result.exit_code == 0
        mock_cli_instance.show_history.assert_called_once_with(
            'test_kb', 10, 1, False, None, None, None, 'time', 'desc', False
        )
    
    @patch('src.cli.cli_instance')
    def test_history_command_with_filters(self, mock_cli_instance):
        """测试带过滤条件的历史命令"""
        from src.cli import show_history
        runner = CliRunner()
        result = runner.invoke(show_history, [
            '--correct',
            '--min-score', '80',
            '--search', '测试',
            '--detailed'
        ], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 0
        mock_cli_instance.show_history.assert_called_once_with(
            'test_kb', 10, 1, True, 80.0, None, '测试', 'time', 'desc', True
        )
    
    @patch('src.cli.cli_instance')
    def test_history_command_with_pagination(self, mock_cli_instance):
        """测试分页参数"""
        from src.cli import show_history
        runner = CliRunner()
        result = runner.invoke(show_history, [
            '--limit', '20',
            '--page', '2'
        ], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 0
        mock_cli_instance.show_history.assert_called_once_with(
            'test_kb', 20, 2, False, None, None, None, 'time', 'desc', False
        )
    
    @patch('src.cli.cli_instance')
    def test_history_command_with_sorting(self, mock_cli_instance):
        """测试排序参数"""
        from src.cli import show_history
        runner = CliRunner()
        result = runner.invoke(show_history, [
            '--sort-by', 'score',
            '--sort-order', 'asc'
        ], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 0
        mock_cli_instance.show_history.assert_called_once_with(
            'test_kb', 10, 1, False, None, None, None, 'score', 'asc', False
        )
    
    @patch('src.cli.cli_instance')
    def test_detail_command(self, mock_cli_instance):
        """测试详情命令"""
        from src.cli import show_history_detail
        runner = CliRunner()
        result = runner.invoke(show_history_detail, ['123'], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 0
        mock_cli_instance.show_history_detail.assert_called_once_with('test_kb', 123)
    
    @patch('src.cli.cli_instance')
    def test_export_command_json(self, mock_cli_instance):
        """测试导出JSON命令"""
        from src.cli import export_history
        runner = CliRunner()
        result = runner.invoke(export_history, [], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 0
        mock_cli_instance.export_history.assert_called_once_with('test_kb', 'json', None)
    
    @patch('src.cli.cli_instance')
    def test_export_command_csv_with_output(self, mock_cli_instance):
        """测试导出CSV到文件命令"""
        from src.cli import export_history
        runner = CliRunner()
        result = runner.invoke(export_history, [
            '--format', 'csv',
            '--output', 'export.csv'
        ], obj={'kb_name': 'test_kb'})
        
        assert result.exit_code == 0
        mock_cli_instance.export_history.assert_called_once_with('test_kb', 'csv', 'export.csv')


class TestHistoryDisplayMethods:
    """测试历史记录显示方法"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例的fixture"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    @pytest.fixture
    def mock_qa_record(self):
        """创建模拟问答记录"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="回答正确",
            reference_answer="参考答案",
            missing_points=["需要更详细"],
            strengths=["理解准确", "逻辑清晰"],
            status=EvaluationStatus.SUCCESS
        )
        return QARecord(
            id=1,
            kb_name="test_kb",
            question="这是一个测试问题？",
            user_answer="这是用户的答案",
            evaluation=evaluation,
            created_at=datetime.now()
        )
    
    @patch('src.cli.console')
    def test_display_history_stats(self, mock_console, cli_instance):
        """测试显示历史统计信息"""
        stats = {
            'total_count': 100,
            'accuracy_rate': 85.5,
            'average_score': 78.2,
            'recent_activity_count': 15
        }
        
        cli_instance._display_history_stats("test_kb", stats)
        
        # 验证console.print被调用
        assert mock_console.print.call_count >= 1
    
    @patch('src.cli.console')
    def test_display_history_table(self, mock_console, cli_instance, mock_qa_record):
        """测试显示历史记录表格"""
        pagination = PaginationInfo(page=1, page_size=10, total_count=1, total_pages=1)
        
        cli_instance._display_history_table("test_kb", [mock_qa_record], pagination, 1)
        
        # 验证console.print被调用
        assert mock_console.print.call_count >= 1
    
    @patch('src.cli.console')
    def test_display_single_record_detail(self, mock_console, cli_instance, mock_qa_record):
        """测试显示单个记录详情"""
        cli_instance._display_single_record_detail(mock_qa_record, show_header=True)
        
        # 验证console.print被调用多次（问题、答案、评估、参考答案）
        assert mock_console.print.call_count >= 4
    
    @patch('src.cli.console')
    def test_display_pagination_info(self, mock_console, cli_instance):
        """测试显示分页信息"""
        pagination = PaginationInfo(
            page=2, 
            page_size=10, 
            total_count=25, 
            total_pages=3,
            has_next=True,
            has_prev=True
        )
        
        cli_instance._display_pagination_info(pagination)
        
        # 验证console.print被调用
        assert mock_console.print.call_count >= 1
    
    @patch('src.cli.console')
    def test_display_history_help(self, mock_console, cli_instance):
        """测试显示历史记录帮助"""
        cli_instance._display_history_help("test_kb")
        
        # 验证console.print被调用
        assert mock_console.print.call_count >= 1