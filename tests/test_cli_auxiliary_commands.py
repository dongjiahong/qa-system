"""
Integration tests for auxiliary management commands
辅助管理命令集成测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from click.testing import CliRunner

from src.cli import (
    main,
    list_knowledge_bases,
    delete_knowledge_base,
    show_system_status,
    KnowledgeCLI
)
from src.models import (
    KnowledgeBase,
    KnowledgeBaseNotFoundError,
    DatabaseError,
    VectorStoreError
)


class TestAuxiliaryCommandsIntegration:
    """辅助管理命令集成测试"""
    
    def test_list_empty_knowledge_bases_command(self):
        """测试列出空知识库列表命令"""
        runner = CliRunner()
        
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.list_knowledge_bases.return_value = None
            
            result = runner.invoke(list_knowledge_bases)
            
            assert result.exit_code == 0
            mock_cli.list_knowledge_bases.assert_called_once()
    
    def test_list_knowledge_bases_with_multiple_entries(self):
        """测试列出多个知识库"""
        runner = CliRunner()
        
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.list_knowledge_bases.return_value = None
            
            result = runner.invoke(list_knowledge_bases)
            
            assert result.exit_code == 0
            mock_cli.list_knowledge_bases.assert_called_once()
    
    def test_delete_knowledge_base_with_confirmation(self):
        """测试删除知识库需要确认"""
        runner = CliRunner()
        
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.delete_knowledge_base.return_value = None
            
            result = runner.invoke(delete_knowledge_base, ['test_kb'])
            
            assert result.exit_code == 0
            mock_cli.delete_knowledge_base.assert_called_once_with('test_kb', False)
    
    def test_delete_knowledge_base_force_flag(self):
        """测试强制删除知识库"""
        runner = CliRunner()
        
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.delete_knowledge_base.return_value = None
            
            result = runner.invoke(delete_knowledge_base, ['test_kb', '--force'])
            
            assert result.exit_code == 0
            mock_cli.delete_knowledge_base.assert_called_once_with('test_kb', True)
    
    def test_system_status_healthy(self):
        """测试系统状态检查 - 健康状态"""
        runner = CliRunner()
        
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.show_system_status.return_value = None
            
            result = runner.invoke(show_system_status)
            
            assert result.exit_code == 0
            mock_cli.show_system_status.assert_called_once()
    
    def test_command_help_messages(self):
        """测试命令帮助信息"""
        runner = CliRunner()
        
        # 测试主命令帮助
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "知识库问答系统" in result.output
        
        # 测试list命令帮助
        result = runner.invoke(list_knowledge_bases, ['--help'])
        assert result.exit_code == 0
        assert "列出所有知识库" in result.output
        
        # 测试delete命令帮助
        result = runner.invoke(delete_knowledge_base, ['--help'])
        assert result.exit_code == 0
        assert "删除知识库" in result.output
        
        # 测试status命令帮助
        result = runner.invoke(show_system_status, ['--help'])
        assert result.exit_code == 0
        assert "显示系统状态" in result.output


class TestAuxiliaryCommandsErrorHandling:
    """辅助管理命令错误处理测试"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    def test_list_knowledge_bases_database_error(self, cli_instance):
        """测试列出知识库时数据库错误"""
        cli_instance.kb_manager.get_knowledge_base_details.side_effect = DatabaseError("数据库连接失败")
        
        with pytest.raises(DatabaseError):
            cli_instance.list_knowledge_bases()
    
    def test_delete_nonexistent_knowledge_base(self, cli_instance):
        """测试删除不存在的知识库"""
        cli_instance.kb_manager.get_knowledge_base.return_value = None
        
        with pytest.raises(KnowledgeBaseNotFoundError):
            cli_instance.delete_knowledge_base("nonexistent_kb")
    
    def test_system_status_with_unhealthy_components(self, cli_instance):
        """测试系统状态检查 - 组件不健康"""
        unhealthy_status = {
            "status": "unhealthy",
            "components": {
                "vector_store": {
                    "status": "unhealthy",
                    "error": "ChromaDB连接失败"
                },
                "database": {
                    "status": "healthy",
                    "knowledge_base_count": 0
                },
                "document_processor": {
                    "status": "healthy",
                    "supported_formats": [".pdf", ".txt", ".md"]
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with patch('src.cli.show_progress') as mock_progress:
            mock_progress.return_value = unhealthy_status
            
            # 执行测试（不应抛出异常）
            cli_instance.show_system_status()
            
            mock_progress.assert_called_once()
    
    def test_delete_knowledge_base_partial_failure(self, cli_instance):
        """测试删除知识库部分失败"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        with patch('src.cli.show_progress') as mock_progress:
            # 模拟部分删除成功
            mock_progress.return_value = False
            
            # 执行测试
            cli_instance.delete_knowledge_base("test_kb", force=True)
            
            mock_progress.assert_called_once()


class TestKnowledgeBaseStatistics:
    """知识库统计信息测试"""
    
    @pytest.fixture
    def cli_instance(self):
        """创建CLI实例"""
        with patch('src.cli.KnowledgeBaseManager'), \
             patch('src.cli.QuestionGenerator'), \
             patch('src.cli.AnswerEvaluator'), \
             patch('src.cli.HistoryManager'), \
             patch('src.cli.get_config'):
            return KnowledgeCLI()
    
    def test_list_knowledge_bases_with_statistics(self, cli_instance):
        """测试列出知识库包含统计信息"""
        mock_kbs = [
            KnowledgeBase(
                name="kb1",
                created_at=datetime.now(),
                file_count=3,
                document_count=15,
                description="第一个知识库"
            ),
            KnowledgeBase(
                name="kb2",
                created_at=datetime.now(),
                file_count=1,
                document_count=5,
                description="第二个知识库"
            )
        ]
        
        cli_instance.kb_manager.get_knowledge_base_details.return_value = mock_kbs
        
        # 执行测试
        cli_instance.list_knowledge_bases()
        
        # 验证调用
        cli_instance.kb_manager.get_knowledge_base_details.assert_called_once()
    
    def test_system_status_with_detailed_components(self, cli_instance):
        """测试系统状态检查包含详细组件信息"""
        detailed_status = {
            "status": "healthy",
            "components": {
                "vector_store": {
                    "status": "healthy",
                    "embedding_model": "shaw/dmeta-embedding-zh-small-q4",
                    "collections_count": 3
                },
                "database": {
                    "status": "healthy",
                    "knowledge_base_count": 3,
                    "total_qa_records": 150
                },
                "document_processor": {
                    "status": "healthy",
                    "supported_formats": [".pdf", ".txt", ".md", ".epub"]
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with patch('src.cli.show_progress') as mock_progress:
            mock_progress.return_value = detailed_status
            
            # 执行测试
            cli_instance.show_system_status()
            
            mock_progress.assert_called_once()


class TestSafetyMechanisms:
    """安全机制测试"""
    
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
    def test_delete_confirmation_mechanism(self, mock_input, cli_instance):
        """测试删除确认机制"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(
            name="important_kb",
            created_at=datetime.now(),
            file_count=10,
            document_count=100,
            description="重要知识库"
        )
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        # 测试用户拒绝删除
        mock_input.return_value = "n"
        
        with patch('src.cli.show_progress') as mock_progress:
            cli_instance.delete_knowledge_base("important_kb")
            
            # 验证没有调用删除
            mock_progress.assert_not_called()
        
        # 测试用户确认删除
        mock_input.return_value = "y"
        
        with patch('src.cli.show_progress') as mock_progress:
            mock_progress.return_value = True
            
            cli_instance.delete_knowledge_base("important_kb")
            
            # 验证调用了删除
            mock_progress.assert_called_once()
    
    def test_force_delete_bypasses_confirmation(self, cli_instance):
        """测试强制删除绕过确认"""
        # 模拟知识库存在
        mock_kb = KnowledgeBase(name="test_kb", created_at=datetime.now())
        cli_instance.kb_manager.get_knowledge_base.return_value = mock_kb
        
        with patch('src.cli.show_progress') as mock_progress, \
             patch('src.cli.console.input') as mock_input:
            mock_progress.return_value = True
            
            # 执行强制删除
            cli_instance.delete_knowledge_base("test_kb", force=True)
            
            # 验证没有调用输入确认
            mock_input.assert_not_called()
            # 验证调用了删除
            mock_progress.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])