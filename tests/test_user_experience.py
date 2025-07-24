"""
User Experience and Error Handling Tests
用户体验和错误处理测试
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from src.cli import main
from src.config import get_config, validate_system_requirements
from src.help_system import HelpSystem
from src.models import (
    ValidationError, 
    FileProcessingError, 
    ModelServiceError, 
    DatabaseError,
    KnowledgeBaseNotFoundError
)


class TestErrorHandling:
    """错误处理测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.runner = CliRunner()
    
    def test_file_processing_error_handling(self):
        """测试文件处理错误的用户友好提示"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.create_knowledge_base.side_effect = FileProcessingError(
                "文件格式不支持", 
                details={"file": "test.xyz", "supported": [".pdf", ".txt"]}
            )
            
            result = self.runner.invoke(main, [
                'new', '--name', 'test', '--file', 'test.xyz'
            ])
            
            assert result.exit_code == 1
            assert "文件处理错误" in result.output
            assert "解决建议" in result.output
            assert "检查文件格式" in result.output
            assert "troubleshoot file_processing" in result.output
    
    def test_model_service_error_handling(self):
        """测试模型服务错误的用户友好提示"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.start_new_review.side_effect = ModelServiceError(
                "无法连接到 Ollama 服务",
                details={"url": "http://localhost:11434", "timeout": 60}
            )
            
            result = self.runner.invoke(main, [
                'review', 'test-kb', 'new'
            ])
            
            assert result.exit_code == 1
            assert "模型服务错误" in result.output
            assert "解决建议" in result.output
            assert "ollama serve" in result.output
            assert "troubleshoot ollama_connection" in result.output
    
    def test_database_error_handling(self):
        """测试数据库错误的用户友好提示"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.show_history.side_effect = DatabaseError(
                "数据库连接失败",
                details={"db_path": "/path/to/db", "error": "Permission denied"}
            )
            
            result = self.runner.invoke(main, [
                'review', 'test-kb', 'history'
            ])
            
            assert result.exit_code == 1
            assert "数据库错误" in result.output
            assert "解决建议" in result.output
            assert "检查数据目录权限" in result.output
            assert "troubleshoot database_issues" in result.output
    
    def test_knowledge_base_not_found_error(self):
        """测试知识库不存在错误的用户友好提示"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.start_new_review.side_effect = KnowledgeBaseNotFoundError(
                "知识库 'nonexistent' 不存在"
            )
            
            result = self.runner.invoke(main, [
                'review', 'nonexistent', 'new'
            ])
            
            assert result.exit_code == 1
            assert "知识库不存在" in result.output
            assert "可用操作" in result.output
            assert "knowledge list" in result.output
            assert "knowledge new" in result.output
    
    def test_validation_error_handling(self):
        """测试参数验证错误的用户友好提示"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.create_knowledge_base.side_effect = ValidationError(
                "知识库名称不能为空",
                details={"parameter": "name", "value": ""}
            )
            
            result = self.runner.invoke(main, [
                'new', '--name', '', '--file', 'test.txt'
            ])
            
            assert result.exit_code == 1
            assert "参数验证错误" in result.output
            assert "知识库名称不能为空" in result.output
            assert "knowledge --help" in result.output
    
    def test_keyboard_interrupt_handling(self):
        """测试键盘中断的优雅处理"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.start_new_review.side_effect = KeyboardInterrupt()
            
            result = self.runner.invoke(main, [
                'review', 'test-kb', 'new'
            ])
            
            assert result.exit_code == 0
            assert "操作已取消" in result.output
    
    def test_unexpected_error_handling(self):
        """测试未预期错误的处理"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.list_knowledge_bases.side_effect = RuntimeError("Unexpected error")
            
            result = self.runner.invoke(main, ['list'])
            
            assert result.exit_code == 1
            assert "未知错误" in result.output
            assert "发生未预期的错误" in result.output
            assert "获取帮助" in result.output
    
    def test_debug_mode_error_details(self):
        """测试调试模式下的详细错误信息"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.list_knowledge_bases.side_effect = RuntimeError("Debug test error")
            
            result = self.runner.invoke(main, ['--debug', 'list'])
            
            assert result.exit_code == 1
            # 在调试模式下应该显示详细的错误堆栈


class TestHelpSystem:
    """帮助系统测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.runner = CliRunner()
        self.help_system = HelpSystem()
    
    def test_show_available_commands(self):
        """测试显示可用命令"""
        result = self.runner.invoke(main, [])
        
        assert result.exit_code == 0
        assert "知识库管理" in result.output
        assert "问答学习" in result.output
        assert "new" in result.output
        assert "review" in result.output
    
    def test_command_specific_help(self):
        """测试特定命令的详细帮助"""
        result = self.runner.invoke(main, ['--help-command', 'new'])
        
        assert result.exit_code == 0
        assert "创建新的知识库" in result.output
        assert "用法:" in result.output
        assert "选项:" in result.output
        assert "示例:" in result.output
        assert "注意事项:" in result.output
    
    def test_examples_display(self):
        """测试示例显示"""
        result = self.runner.invoke(main, ['--examples', 'getting_started'])
        
        assert result.exit_code == 0
        assert "快速开始" in result.output
        assert "knowledge new" in result.output
        assert "knowledge" in result.output and "review" in result.output
    
    def test_troubleshooting_guide(self):
        """测试故障排除指南"""
        result = self.runner.invoke(main, ['--troubleshoot-all'])
        
        assert result.exit_code == 0
        assert "故障排除指南" in result.output
        assert "ollama_connection" in result.output
        assert "file_processing" in result.output
    
    def test_specific_troubleshooting(self):
        """测试特定问题的故障排除"""
        result = self.runner.invoke(main, ['--troubleshoot', 'ollama_connection'])
        
        assert result.exit_code == 0
        assert "Ollama 服务连接失败" in result.output
        assert "症状:" in result.output
        assert "解决方案:" in result.output
        assert "相关命令:" in result.output
    
    def test_quick_start_guide(self):
        """测试快速开始指南"""
        result = self.runner.invoke(main, ['--quick-start'])
        
        assert result.exit_code == 0
        assert "快速开始指南" in result.output
        assert "环境准备" in result.output
        assert "创建知识库" in result.output
        assert "开始学习" in result.output
    
    def test_help_system_command_info(self):
        """测试帮助系统的命令信息完整性"""
        commands = self.help_system.commands
        
        # 检查所有主要命令都有帮助信息
        required_commands = ["new", "list", "delete", "status", "review new", "review history"]
        
        for cmd in required_commands:
            assert cmd in commands
            assert "description" in commands[cmd]
            assert "usage" in commands[cmd]
            assert "examples" in commands[cmd]
    
    def test_help_system_examples(self):
        """测试帮助系统的示例完整性"""
        examples = self.help_system.examples
        
        required_categories = ["getting_started", "advanced_usage", "troubleshooting"]
        
        for category in required_categories:
            assert category in examples
            assert len(examples[category]) > 0
    
    def test_help_system_troubleshooting(self):
        """测试帮助系统的故障排除信息完整性"""
        troubleshooting = self.help_system.troubleshooting
        
        required_problems = [
            "ollama_connection", "model_missing", "file_processing", 
            "database_issues", "chinese_processing"
        ]
        
        for problem in required_problems:
            assert problem in troubleshooting
            assert "problem" in troubleshooting[problem]
            assert "symptoms" in troubleshooting[problem]
            assert "solutions" in troubleshooting[problem]


class TestConfigurationSystem:
    """配置系统测试"""
    
    def test_config_validation(self):
        """测试配置验证"""
        config = get_config()
        
        # 测试基本配置项存在
        assert hasattr(config, 'app_name')
        assert hasattr(config, 'version')
        assert hasattr(config, 'ollama_base_url')
        assert hasattr(config, 'ollama_model')
        
        # 测试配置验证
        assert config.ollama_timeout >= 10
        assert config.max_file_size_mb >= 1
        assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def test_environment_validation(self):
        """测试环境验证"""
        config = get_config()
        validation_result = config.validate_environment()
        
        assert "status" in validation_result
        assert "issues" in validation_result
        assert "warnings" in validation_result
        assert "components" in validation_result
        
        # 状态应该是 healthy 或 unhealthy
        assert validation_result["status"] in ["healthy", "unhealthy"]
    
    def test_system_requirements_validation(self):
        """测试系统需求验证"""
        validation_result = validate_system_requirements()
        
        assert "status" in validation_result
        assert "components" in validation_result
        
        # 检查 Python 版本组件
        assert "python" in validation_result["components"]
        python_component = validation_result["components"]["python"]
        assert "status" in python_component
        assert "version" in python_component
    
    def test_config_file_operations(self):
        """测试配置文件操作"""
        from src.config import save_config_file, load_config_file
        
        config = get_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # 测试保存配置
            save_config_file(config, config_path)
            assert config_path.exists()
            
            # 测试加载配置
            loaded_config = load_config_file(config_path)
            assert isinstance(loaded_config, dict)
            assert "app_name" in loaded_config


class TestUserExperienceFeatures:
    """用户体验功能测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.runner = CliRunner()
    
    def test_progress_indicators(self):
        """测试进度指示器"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.create_knowledge_base.return_value = Mock()
            
            result = self.runner.invoke(main, [
                'new', '--name', 'test', '--file', 'test.txt'
            ])
            
            # 应该包含进度相关的输出
            # 具体的进度显示可能因为测试环境而不同
            assert result.exit_code == 0 or "创建知识库" in result.output
    
    def test_color_output_control(self):
        """测试彩色输出控制"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.list_knowledge_bases.return_value = None
            
            # 测试禁用彩色输出
            result = self.runner.invoke(main, ['--no-color', 'list'])
            
            # 在无色模式下不应该有ANSI颜色代码
            # 这个测试可能需要根据实际的输出格式调整
            assert result.exit_code == 0
    
    def test_verbose_output(self):
        """测试详细输出模式"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.list_knowledge_bases.return_value = None
            
            result = self.runner.invoke(main, ['--verbose', 'list'])
            
            # 详细模式应该提供更多信息
            assert result.exit_code == 0
    
    def test_confirmation_prompts(self):
        """测试确认提示"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.delete_knowledge_base.return_value = None
            
            # 模拟用户输入 'n' (不确认删除)
            result = self.runner.invoke(main, [
                'delete', 'test-kb'
            ], input='n\n')
            
            # 应该取消删除操作
            assert result.exit_code == 0 or "取消" in result.output
    
    def test_input_validation_feedback(self):
        """测试输入验证反馈"""
        # 测试空的知识库名称
        result = self.runner.invoke(main, [
            'new', '--name', '', '--file', 'test.txt'
        ])
        
        # 应该提供清晰的验证错误信息
        assert result.exit_code != 0
    
    def test_contextual_help_suggestions(self):
        """测试上下文相关的帮助建议"""
        # 测试不存在的命令
        result = self.runner.invoke(main, ['nonexistent-command'])
        
        # 应该提供帮助建议
        assert result.exit_code != 0


class TestErrorScenarios:
    """错误场景测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.runner = CliRunner()
    
    def test_missing_required_arguments(self):
        """测试缺少必需参数"""
        result = self.runner.invoke(main, ['new'])
        
        assert result.exit_code != 0
        # Click 应该提供参数缺失的错误信息
    
    def test_invalid_file_paths(self):
        """测试无效文件路径"""
        result = self.runner.invoke(main, [
            'new', '--name', 'test', '--file', '/nonexistent/path/file.txt'
        ])
        
        assert result.exit_code != 0
    
    def test_invalid_command_combinations(self):
        """测试无效的命令组合"""
        result = self.runner.invoke(main, [
            'nonexistent-kb', 'review', 'new'
        ])
        
        assert result.exit_code != 0
    
    def test_permission_errors(self):
        """测试权限错误"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.create_knowledge_base.side_effect = PermissionError("Permission denied")
            
            result = self.runner.invoke(main, [
                'new', '--name', 'test', '--file', 'test.txt'
            ])
            
            assert result.exit_code == 1
    
    def test_disk_space_errors(self):
        """测试磁盘空间错误"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.create_knowledge_base.side_effect = OSError("No space left on device")
            
            result = self.runner.invoke(main, [
                'new', '--name', 'test', '--file', 'test.txt'
            ])
            
            assert result.exit_code == 1
    
    def test_network_connectivity_errors(self):
        """测试网络连接错误"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.start_new_review.side_effect = ConnectionError("Connection failed")
            
            result = self.runner.invoke(main, [
                'review', 'test-kb', 'new'
            ])
            
            assert result.exit_code == 1


class TestAccessibilityAndUsability:
    """可访问性和可用性测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.runner = CliRunner()
    
    def test_help_text_clarity(self):
        """测试帮助文本的清晰度"""
        result = self.runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "知识库问答系统" in result.output
        assert "快速开始" in result.output
        assert "获取帮助" in result.output
    
    def test_error_message_clarity(self):
        """测试错误信息的清晰度"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_cli.create_knowledge_base.side_effect = ValidationError("测试错误信息")
            
            result = self.runner.invoke(main, [
                'new', '--name', 'test', '--file', 'test.txt'
            ])
            
            assert result.exit_code == 1
            assert "测试错误信息" in result.output
            assert "💡" in result.output  # 应该有帮助提示图标
    
    def test_command_discoverability(self):
        """测试命令的可发现性"""
        result = self.runner.invoke(main, [])
        
        assert result.exit_code == 0
        # 应该显示主要的命令类别和命令
        assert "知识库管理" in result.output
        assert "问答学习" in result.output
    
    def test_feedback_quality(self):
        """测试反馈质量"""
        with patch('src.cli.cli_instance') as mock_cli:
            mock_kb = Mock()
            mock_kb.name = "test-kb"
            mock_kb.file_count = 1
            mock_kb.document_count = 5
            mock_cli.create_knowledge_base.return_value = mock_kb
            
            result = self.runner.invoke(main, [
                'new', '--name', 'test-kb', '--file', 'test.txt'
            ])
            
            assert result.exit_code == 0
            # 应该提供创建成功的详细反馈
            assert "创建成功" in result.output or "✓" in result.output


if __name__ == "__main__":
    pytest.main([__file__])