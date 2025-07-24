"""
Integration tests for CLI Knowledge Base Creation Command
CLI知识库创建命令集成测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner
from datetime import datetime

from src.cli import main, create_knowledge_base
from src.models import (
    KnowledgeBase,
    FileProcessingError,
    ValidationError,
    VectorStoreError,
    DatabaseError,
    KnowledgeSystemError
)


class TestKnowledgeBaseCreationIntegration:
    """知识库创建命令集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """创建示例文件"""
        files = []
        
        # 创建PDF文件（模拟）
        pdf_file = Path(temp_dir) / "sample.pdf"
        pdf_file.write_text("PDF content", encoding='utf-8')
        files.append(str(pdf_file))
        
        # 创建文本文件
        txt_file = Path(temp_dir) / "sample.txt"
        txt_file.write_text("这是一个测试文本文件。\n包含中文内容。", encoding='utf-8')
        files.append(str(txt_file))
        
        # 创建Markdown文件
        md_file = Path(temp_dir) / "sample.md"
        md_file.write_text("# 测试标题\n\n这是测试内容。", encoding='utf-8')
        files.append(str(md_file))
        
        return files
    
    @pytest.fixture
    def runner(self):
        """创建CLI运行器"""
        return CliRunner()
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_success(self, mock_cli_instance, runner, sample_files):
        """测试知识库创建命令成功场景"""
        # 模拟成功创建知识库
        mock_kb = KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=3,
            document_count=10,
            description="测试知识库"
        )
        mock_cli_instance.create_knowledge_base.return_value = None
        
        # 执行命令
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', sample_files[0],
            '--file', sample_files[1],
            '--file', sample_files[2],
            '--description', '测试知识库'
        ])
        
        # 验证结果
        assert result.exit_code == 0
        mock_cli_instance.create_knowledge_base.assert_called_once_with(
            'test_kb', sample_files, '测试知识库'
        )
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_single_file(self, mock_cli_instance, runner, sample_files):
        """测试单文件知识库创建"""
        mock_cli_instance.create_knowledge_base.return_value = None
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'single_file_kb',
            '--file', sample_files[0]
        ])
        
        assert result.exit_code == 0
        mock_cli_instance.create_knowledge_base.assert_called_once_with(
            'single_file_kb', [sample_files[0]], None
        )
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_missing_name(self, mock_cli_instance, runner, sample_files):
        """测试缺少名称参数"""
        result = runner.invoke(create_knowledge_base, [
            '--file', sample_files[0]
        ])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
        mock_cli_instance.create_knowledge_base.assert_not_called()
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_missing_files(self, mock_cli_instance, runner):
        """测试缺少文件参数"""
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb'
        ])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
        mock_cli_instance.create_knowledge_base.assert_not_called()
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_file_processing_error(self, mock_cli_instance, runner, sample_files):
        """测试文件处理错误"""
        mock_cli_instance.create_knowledge_base.side_effect = FileProcessingError(
            "文件格式不支持", {"file": sample_files[0]}
        )
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', sample_files[0]
        ])
        
        assert result.exit_code == 1
        assert "文件处理错误" in result.output
        assert "文件格式不支持" in result.output
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_validation_error(self, mock_cli_instance, runner, sample_files):
        """测试参数验证错误"""
        mock_cli_instance.create_knowledge_base.side_effect = ValidationError(
            "知识库名称已存在"
        )
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'existing_kb',
            '--file', sample_files[0]
        ])
        
        assert result.exit_code == 1
        assert "参数验证错误" in result.output
        assert "知识库名称已存在" in result.output
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_vector_store_error(self, mock_cli_instance, runner, sample_files):
        """测试向量存储错误"""
        mock_cli_instance.create_knowledge_base.side_effect = VectorStoreError(
            "ChromaDB连接失败"
        )
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', sample_files[0]
        ])
        
        assert result.exit_code == 1
        assert "向量存储错误" in result.output
        assert "ChromaDB连接失败" in result.output
        assert "ChromaDB 服务是否正常运行" in result.output
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_database_error(self, mock_cli_instance, runner, sample_files):
        """测试数据库错误"""
        mock_cli_instance.create_knowledge_base.side_effect = DatabaseError(
            "数据库写入失败"
        )
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', sample_files[0]
        ])
        
        assert result.exit_code == 1
        assert "数据库错误" in result.output
        assert "数据库写入失败" in result.output
        assert "数据库文件权限和磁盘空间" in result.output
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_system_error(self, mock_cli_instance, runner, sample_files):
        """测试系统错误"""
        mock_cli_instance.create_knowledge_base.side_effect = KnowledgeSystemError(
            "系统内部错误"
        )
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', sample_files[0]
        ])
        
        assert result.exit_code == 1
        assert "系统错误" in result.output
        assert "系统内部错误" in result.output
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_keyboard_interrupt(self, mock_cli_instance, runner, sample_files):
        """测试键盘中断"""
        mock_cli_instance.create_knowledge_base.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', sample_files[0]
        ])
        
        assert result.exit_code == 0
        assert "操作已取消" in result.output
    
    def test_create_knowledge_base_command_nonexistent_file(self, runner):
        """测试不存在的文件"""
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', '/nonexistent/file.pdf'
        ])
        
        assert result.exit_code == 1
        assert "文件不存在" in result.output or "参数验证错误" in result.output
    
    def test_create_knowledge_base_command_directory_as_file(self, runner, temp_dir):
        """测试将目录作为文件传入"""
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', temp_dir
        ])
        
        assert result.exit_code == 1
        assert "路径不是文件" in result.output or "参数验证错误" in result.output
    
    def test_create_knowledge_base_command_unsupported_format(self, runner, temp_dir):
        """测试不支持的文件格式"""
        # 创建不支持格式的文件
        unsupported_file = Path(temp_dir) / "test.xyz"
        unsupported_file.write_text("test content")
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', str(unsupported_file)
        ])
        
        assert result.exit_code == 1
        assert "不支持的文件格式" in result.output or "参数验证错误" in result.output
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_long_description(self, mock_cli_instance, runner, sample_files):
        """测试长描述"""
        long_description = "这是一个很长的描述" * 50
        mock_cli_instance.create_knowledge_base.return_value = None
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', sample_files[0],
            '--description', long_description
        ])
        
        assert result.exit_code == 0
        mock_cli_instance.create_knowledge_base.assert_called_once_with(
            'test_kb', [sample_files[0]], long_description
        )
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_special_characters_in_name(self, mock_cli_instance, runner, sample_files):
        """测试名称中包含特殊字符"""
        special_name = "测试-知识库_2024.v1"
        mock_cli_instance.create_knowledge_base.return_value = None
        
        result = runner.invoke(create_knowledge_base, [
            '--name', special_name,
            '--file', sample_files[0]
        ])
        
        assert result.exit_code == 0
        mock_cli_instance.create_knowledge_base.assert_called_once_with(
            special_name, [sample_files[0]], None
        )
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_multiple_files_same_type(self, mock_cli_instance, runner, temp_dir):
        """测试多个相同类型文件"""
        # 创建多个文本文件
        files = []
        for i in range(5):
            txt_file = Path(temp_dir) / f"sample_{i}.txt"
            txt_file.write_text(f"这是第{i}个测试文件", encoding='utf-8')
            files.append(str(txt_file))
        
        mock_cli_instance.create_knowledge_base.return_value = None
        
        # 构建命令参数
        cmd_args = ['--name', 'multi_file_kb']
        for file_path in files:
            cmd_args.extend(['--file', file_path])
        
        result = runner.invoke(create_knowledge_base, cmd_args)
        
        assert result.exit_code == 0
        mock_cli_instance.create_knowledge_base.assert_called_once_with(
            'multi_file_kb', files, None
        )
    
    @patch('src.cli.cli_instance')
    def test_create_knowledge_base_command_help(self, mock_cli_instance, runner):
        """测试命令帮助信息"""
        result = runner.invoke(create_knowledge_base, ['--help'])
        
        assert result.exit_code == 0
        assert "知识库名称" in result.output
        assert "--name" in result.output
        assert "--file" in result.output
        assert "--description" in result.output
        assert "文档文件路径" in result.output
        mock_cli_instance.create_knowledge_base.assert_not_called()


class TestKnowledgeBaseCreationProgressAndFeedback:
    """测试知识库创建过程中的进度显示和状态反馈"""
    
    @pytest.fixture
    def runner(self):
        """创建CLI运行器"""
        return CliRunner()
    
    @pytest.fixture
    def temp_file(self):
        """创建临时文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("测试内容")
            temp_file_path = f.name
        
        yield temp_file_path
        
        # 清理
        try:
            Path(temp_file_path).unlink()
        except FileNotFoundError:
            pass
    
    @patch('src.cli.KnowledgeBaseManager')
    @patch('src.cli.validate_file_paths')
    @patch('src.cli.show_progress')
    def test_progress_display_during_creation(self, mock_show_progress, mock_validate, mock_kb_manager, runner, temp_file):
        """测试创建过程中的进度显示"""
        # 模拟文件验证
        mock_validate.return_value = [temp_file]
        
        # 模拟知识库创建
        mock_kb = KnowledgeBase(
            name="progress_test_kb",
            created_at=datetime.now(),
            file_count=1,
            document_count=5
        )
        mock_show_progress.return_value = mock_kb
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'progress_test_kb',
            '--file', temp_file
        ])
        
        assert result.exit_code == 0
        
        # 验证进度显示被调用
        mock_show_progress.assert_called_once()
        call_args = mock_show_progress.call_args
        assert "创建知识库" in call_args[0][0]
        assert "progress_test_kb" in call_args[0][0]
    
    @patch('src.cli.cli_instance')
    def test_success_feedback_display(self, mock_cli_instance, runner, temp_file):
        """测试成功反馈显示"""
        # 模拟成功创建知识库
        mock_cli_instance.create_knowledge_base.return_value = None
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'feedback_test_kb',
            '--file', temp_file,
            '--description', '测试描述'
        ])
        
        assert result.exit_code == 0
        
        # 验证CLI方法被调用
        mock_cli_instance.create_knowledge_base.assert_called_once_with(
            'feedback_test_kb', [temp_file], '测试描述'
        )
    
    @patch('src.cli.KnowledgeBaseManager')
    @patch('src.cli.validate_file_paths')
    def test_error_feedback_display(self, mock_validate, mock_kb_manager, runner, temp_file):
        """测试错误反馈显示"""
        # 模拟文件验证失败
        mock_validate.side_effect = ValidationError("文件验证失败")
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'error_test_kb',
            '--file', temp_file
        ])
        
        assert result.exit_code == 1
        assert "参数验证错误" in result.output
        assert "文件验证失败" in result.output


class TestKnowledgeBaseCreationParameterValidation:
    """测试知识库创建命令的参数验证"""
    
    @pytest.fixture
    def runner(self):
        """创建CLI运行器"""
        return CliRunner()
    
    def test_empty_name_validation(self, runner):
        """测试空名称验证"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            result = runner.invoke(create_knowledge_base, [
                '--name', '',
                '--file', temp_file.name
            ])
            
            assert result.exit_code == 1
    
    def test_whitespace_only_name_validation(self, runner):
        """测试仅空白字符名称验证"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            result = runner.invoke(create_knowledge_base, [
                '--name', '   ',
                '--file', temp_file.name
            ])
            
            assert result.exit_code == 1
    
    def test_very_long_name_validation(self, runner):
        """测试超长名称验证"""
        long_name = "a" * 200  # 超过100字符限制
        
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            result = runner.invoke(create_knowledge_base, [
                '--name', long_name,
                '--file', temp_file.name
            ])
            
            # 可能在参数验证或知识库创建时失败
            assert result.exit_code == 1
    
    def test_empty_file_list_validation(self, runner):
        """测试空文件列表验证"""
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb'
            # 没有 --file 参数
        ])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
    
    @patch('src.cli.validate_file_paths')
    def test_file_validation_called(self, mock_validate, runner):
        """测试文件验证被调用"""
        mock_validate.side_effect = ValidationError("测试验证错误")
        
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', 'test.txt'
        ])
        
        assert result.exit_code == 1
        mock_validate.assert_called_once_with(['test.txt'])


if __name__ == "__main__":
    pytest.main([__file__])