"""
Simple Integration Tests for Knowledge QA System
知识库问答系统简单集成测试

This module contains basic integration tests that verify the core
functionality works together properly.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

# Import CLI commands
from src.cli import main


class TestBasicIntegration:
    """基础集成测试"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        temp_dir = tempfile.mkdtemp()
        original_data_dir = os.environ.get('KNOWLEDGE_DATA_DIR')
        os.environ['KNOWLEDGE_DATA_DIR'] = temp_dir
        
        yield temp_dir
        
        # 清理
        if original_data_dir:
            os.environ['KNOWLEDGE_DATA_DIR'] = original_data_dir
        else:
            os.environ.pop('KNOWLEDGE_DATA_DIR', None)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_document(self, temp_data_dir):
        """创建示例文档"""
        doc_path = Path(temp_data_dir) / "test_doc.md"
        doc_path.write_text("""
# 测试文档

## Python基础
Python是一种编程语言，具有以下特点：
1. 简洁易读
2. 跨平台支持
3. 丰富的库

## 基本语法
```python
def hello():
    print("Hello, World!")
```
""", encoding='utf-8')
        return str(doc_path)
    
    @pytest.fixture
    def runner(self):
        """CLI测试运行器"""
        return CliRunner()
    
    def test_help_commands(self, runner):
        """测试帮助命令"""
        # 测试主帮助
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "知识库问答系统" in result.output or "Knowledge QA System" in result.output
        
        # 测试快速开始
        result = runner.invoke(main, ['--quick-start'])
        assert result.exit_code == 0
        
        # 测试环境检查
        result = runner.invoke(main, ['--check-env'])
        assert result.exit_code == 0
    
    def test_list_empty_knowledge_bases(self, temp_data_dir, runner):
        """测试列出空知识库列表"""
        result = runner.invoke(main, ['list'])
        assert result.exit_code == 0
        assert "知识库列表" in result.output or "共 0 个知识库" in result.output
    
    def test_system_status(self, temp_data_dir, runner):
        """测试系统状态检查"""
        result = runner.invoke(main, ['status'])
        assert result.exit_code == 0
        assert "系统状态" in result.output or "状态检查" in result.output
    
    @patch('src.vector_store.VectorStore')
    @patch('src.llm_client.OllamaClient')
    def test_create_knowledge_base_mocked(self, mock_llm_client, mock_vector_store, 
                                        temp_data_dir, sample_document, runner):
        """测试创建知识库（使用模拟）"""
        # 配置模拟对象
        mock_vs_instance = MagicMock()
        mock_vector_store.return_value = mock_vs_instance
        mock_vs_instance.create_collection.return_value = None
        mock_vs_instance.add_documents.return_value = None
        
        mock_llm_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.health_check.return_value = True
        
        # 创建知识库
        result = runner.invoke(main, [
            'new', '--name', 'test-kb', '--file', sample_document
        ])
        
        # 验证结果
        assert result.exit_code == 0
        assert "创建" in result.output or "成功" in result.output
        
        # 验证知识库被创建
        result = runner.invoke(main, ['list'])
        assert result.exit_code == 0
        assert "test-kb" in result.output
    
    def test_error_handling(self, temp_data_dir, runner):
        """测试错误处理"""
        # 测试文件不存在
        result = runner.invoke(main, [
            'new', '--name', 'error-test', '--file', '/nonexistent/file.txt'
        ])
        assert result.exit_code != 0
        
        # 测试访问不存在的知识库
        result = runner.invoke(main, ['nonexistent-kb', 'review', 'new'])
        assert result.exit_code != 0
    
    def test_configuration_commands(self, temp_data_dir, runner):
        """测试配置相关命令"""
        # 测试配置显示
        result = runner.invoke(main, ['config', 'show'])
        assert result.exit_code == 0
        
        # 测试配置验证
        result = runner.invoke(main, ['config', 'validate'])
        assert result.exit_code == 0
    
    def test_troubleshooting_commands(self, temp_data_dir, runner):
        """测试故障排除命令"""
        # 测试故障排除帮助
        result = runner.invoke(main, ['--troubleshoot-all'])
        assert result.exit_code == 0
        
        # 测试特定故障排除
        result = runner.invoke(main, ['--troubleshoot', 'ollama_connection'])
        assert result.exit_code == 0
    
    def test_examples_command(self, temp_data_dir, runner):
        """测试示例命令"""
        result = runner.invoke(main, ['--examples', 'getting_started'])
        assert result.exit_code == 0
        assert "示例" in result.output or "example" in result.output.lower()


class TestDocumentationIntegration:
    """文档集成测试"""
    
    def test_readme_examples_work(self):
        """测试README中的示例是否有效"""
        # 这里可以测试README中提到的命令是否真的存在
        from src.cli import main
        runner = CliRunner()
        
        # 测试README中提到的主要命令
        commands_to_test = [
            ['--help'],
            ['--version'],
            ['list'],
            ['status'],
            ['config', 'show'],
            ['--quick-start'],
            ['--check-env']
        ]
        
        for cmd in commands_to_test:
            result = runner.invoke(main, cmd)
            # 命令应该存在（不应该返回"未知命令"错误）
            assert "Unknown command" not in result.output
            assert "No such command" not in result.output
    
    def test_sample_documents_exist(self):
        """测试示例文档是否存在"""
        sample_docs = [
            "examples/sample_documents/python_basics.md",
            "examples/sample_documents/machine_learning_intro.md"
        ]
        
        for doc_path in sample_docs:
            assert Path(doc_path).exists(), f"示例文档不存在: {doc_path}"
            
            # 检查文档内容不为空
            content = Path(doc_path).read_text(encoding='utf-8')
            assert len(content.strip()) > 100, f"示例文档内容太少: {doc_path}"
    
    def test_demo_scripts_exist(self):
        """测试演示脚本是否存在"""
        demo_scripts = [
            "examples/demo_complete_workflow.py",
            "demo_interactive_qa.py"
        ]
        
        for script_path in demo_scripts:
            assert Path(script_path).exists(), f"演示脚本不存在: {script_path}"
    
    def test_documentation_files_exist(self):
        """测试文档文件是否存在"""
        doc_files = [
            "README.md",
            "docs/usage_guide.md",
            "docs/user_experience_implementation.md"
        ]
        
        for doc_path in doc_files:
            assert Path(doc_path).exists(), f"文档文件不存在: {doc_path}"
            
            # 检查文档内容
            content = Path(doc_path).read_text(encoding='utf-8')
            assert len(content.strip()) > 500, f"文档内容太少: {doc_path}"


if __name__ == "__main__":
    # 运行简单集成测试
    pytest.main([__file__, "-v", "--tb=short"])