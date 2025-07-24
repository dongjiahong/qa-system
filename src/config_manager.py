"""
Configuration Management System
配置管理系统
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import get_config, save_config_file, Settings
from .models import ValidationError

console = Console()


@dataclass
class ConfigTemplate:
    """配置模板"""
    name: str
    description: str
    settings: Dict[str, Any]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".knowledge_qa"
        self.config_file = self.config_dir / "config.json"
        self.templates_dir = self.config_dir / "templates"
        self.backup_dir = self.config_dir / "backups"
        
        # 确保目录存在
        for directory in [self.config_dir, self.templates_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, ConfigTemplate]:
        """加载默认配置模板"""
        return {
            "development": ConfigTemplate(
                name="development",
                description="开发环境配置",
                settings={
                    "debug": True,
                    "log_level": "DEBUG",
                    "verbose_output": True,
                    "ollama_timeout": 30,
                    "question_max_retries": 1,
                    "evaluation_max_retries": 1
                }
            ),
            "production": ConfigTemplate(
                name="production",
                description="生产环境配置",
                settings={
                    "debug": False,
                    "log_level": "INFO",
                    "verbose_output": False,
                    "ollama_timeout": 60,
                    "question_max_retries": 3,
                    "evaluation_max_retries": 3
                }
            ),
            "performance": ConfigTemplate(
                name="performance",
                description="性能优化配置",
                settings={
                    "progress_bars": False,
                    "cli_colors": False,
                    "vector_search_k": 3,
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "ollama_timeout": 30
                }
            ),
            "chinese_optimized": ConfigTemplate(
                name="chinese_optimized",
                description="中文优化配置",
                settings={
                    "embedding_model": "shaw/dmeta-embedding-zh-small-q4",
                    "chunk_size": 800,
                    "chunk_overlap": 100,
                    "max_context_length": 3000,
                    "question_generation_temperature": 0.8,
                    "evaluation_temperature": 0.4
                }
            )
        }
    
    def get_current_config(self) -> Settings:
        """获取当前配置"""
        return get_config()
    
    def save_current_config(self, config_path: Optional[Path] = None) -> None:
        """保存当前配置"""
        if config_path is None:
            config_path = self.config_file
        
        current_config = self.get_current_config()
        save_config_file(current_config, config_path)
        
        console.print(f"[green]✅ 配置已保存到: {config_path}[/green]")
    
    def load_config(self, config_path: Path) -> Settings:
        """加载配置文件"""
        if not config_path.exists():
            raise ValidationError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 创建新的配置实例
            new_config = Settings(**config_data)
            console.print(f"[green]✅ 配置已从 {config_path} 加载[/green]")
            return new_config
            
        except Exception as e:
            raise ValidationError(f"加载配置文件失败: {e}")
    
    def backup_config(self, backup_name: Optional[str] = None) -> Path:
        """备份当前配置"""
        if backup_name is None:
            from datetime import datetime
            backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        backup_path = self.backup_dir / backup_name
        self.save_current_config(backup_path)
        
        console.print(f"[blue]📦 配置已备份到: {backup_path}[/blue]")
        return backup_path
    
    def restore_config(self, backup_path: Path) -> None:
        """恢复配置"""
        if not backup_path.exists():
            raise ValidationError(f"备份文件不存在: {backup_path}")
        
        # 先备份当前配置
        self.backup_config("before_restore")
        
        # 恢复配置
        restored_config = self.load_config(backup_path)
        save_config_file(restored_config, self.config_file)
        
        console.print(f"[green]🔄 配置已从 {backup_path} 恢复[/green]")
    
    def apply_template(self, template_name: str) -> None:
        """应用配置模板"""
        if template_name not in self.templates:
            available = ", ".join(self.templates.keys())
            raise ValidationError(f"模板 '{template_name}' 不存在。可用模板: {available}")
        
        template = self.templates[template_name]
        
        # 备份当前配置
        self.backup_config(f"before_template_{template_name}")
        
        # 获取当前配置并应用模板
        current_config = self.get_current_config()
        
        # 更新配置
        for key, value in template.settings.items():
            if hasattr(current_config, key):
                setattr(current_config, key, value)
        
        # 保存更新后的配置
        save_config_file(current_config, self.config_file)
        
        console.print(f"[green]✅ 已应用模板 '{template_name}': {template.description}[/green]")
    
    def create_template(self, name: str, description: str, settings: Dict[str, Any]) -> None:
        """创建自定义模板"""
        template = ConfigTemplate(name, description, settings)
        self.templates[name] = template
        
        # 保存模板到文件
        template_path = self.templates_dir / f"{name}.json"
        template_data = {
            "name": name,
            "description": description,
            "settings": settings
        }
        
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✅ 模板 '{name}' 已创建[/green]")
    
    def delete_template(self, name: str) -> None:
        """删除模板"""
        if name in ["development", "production", "performance", "chinese_optimized"]:
            raise ValidationError(f"不能删除内置模板: {name}")
        
        if name not in self.templates:
            raise ValidationError(f"模板 '{name}' 不存在")
        
        # 删除模板文件
        template_path = self.templates_dir / f"{name}.json"
        if template_path.exists():
            template_path.unlink()
        
        # 从内存中删除
        del self.templates[name]
        
        console.print(f"[yellow]🗑️ 模板 '{name}' 已删除[/yellow]")
    
    def list_templates(self) -> None:
        """列出所有模板"""
        console.print(Panel(
            "配置模板列表",
            title="[bold cyan]可用模板[/bold cyan]",
            border_style="cyan"
        ))
        
        table = Table()
        table.add_column("模板名称", style="cyan", width=20)
        table.add_column("描述", style="white", width=30)
        table.add_column("类型", style="yellow", width=10)
        table.add_column("主要设置", style="dim", width=40)
        
        for name, template in self.templates.items():
            template_type = "内置" if name in ["development", "production", "performance", "chinese_optimized"] else "自定义"
            
            # 显示主要设置
            key_settings = []
            for key, value in list(template.settings.items())[:3]:
                key_settings.append(f"{key}={value}")
            
            if len(template.settings) > 3:
                key_settings.append("...")
            
            table.add_row(
                name,
                template.description,
                template_type,
                ", ".join(key_settings)
            )
        
        console.print(table)
        console.print(f"\n[dim]使用 'knowledge config apply <模板名称>' 应用模板[/dim]")
    
    def show_config(self, show_sensitive: bool = False) -> None:
        """显示当前配置"""
        config = self.get_current_config()
        
        console.print(Panel(
            f"当前配置 (v{config.version})",
            title="[bold green]系统配置[/bold green]",
            border_style="green"
        ))
        
        # 按类别组织配置
        categories = {
            "应用设置": ["app_name", "version", "debug", "log_level", "verbose_output"],
            "Ollama 设置": ["ollama_base_url", "ollama_model", "ollama_timeout", "ollama_max_retries"],
            "文件处理": ["supported_file_extensions", "max_file_size_mb", "max_files_per_kb"],
            "向量存储": ["embedding_model", "vector_search_k", "chunk_size", "chunk_overlap"],
            "问答设置": ["max_context_length", "question_generation_temperature", "evaluation_temperature"],
            "界面设置": ["cli_colors", "progress_bars"]
        }
        
        for category, keys in categories.items():
            console.print(f"\n[bold yellow]{category}:[/bold yellow]")
            
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Setting", style="cyan", width=25)
            table.add_column("Value", style="white", width=30)
            table.add_column("Type", style="dim", width=15)
            
            for key in keys:
                if hasattr(config, key):
                    value = getattr(config, key)
                    
                    # 隐藏敏感信息
                    if not show_sensitive and "password" in key.lower():
                        value = "***"
                    
                    # 格式化值
                    if isinstance(value, list):
                        value_str = ", ".join(str(v) for v in value[:3])
                        if len(value) > 3:
                            value_str += "..."
                    else:
                        value_str = str(value)
                    
                    table.add_row(key, value_str, type(value).__name__)
            
            console.print(table)
    
    def validate_config(self) -> Dict[str, Any]:
        """验证配置"""
        config = self.get_current_config()
        validation_result = {
            "status": "valid",
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # 验证 Ollama 设置
        if config.ollama_timeout < 10:
            validation_result["warnings"].append("Ollama 超时时间过短，可能导致请求失败")
        
        if config.ollama_timeout > 300:
            validation_result["warnings"].append("Ollama 超时时间过长，可能影响用户体验")
        
        # 验证文件处理设置
        if config.max_file_size_mb > 500:
            validation_result["warnings"].append("最大文件大小设置过大，可能导致内存问题")
        
        if config.chunk_overlap >= config.chunk_size:
            validation_result["errors"].append("chunk_overlap 必须小于 chunk_size")
            validation_result["status"] = "invalid"
        
        # 验证向量搜索设置
        if config.vector_search_k > 20:
            validation_result["warnings"].append("向量搜索结果数量过多，可能影响性能")
        
        # 验证温度设置
        if config.question_generation_temperature > 1.5:
            validation_result["warnings"].append("问题生成温度过高，可能导致问题质量下降")
        
        if config.evaluation_temperature > 1.0:
            validation_result["warnings"].append("评估温度过高，可能导致评估不准确")
        
        # 提供优化建议
        if config.debug and not config.verbose_output:
            validation_result["suggestions"].append("调试模式下建议启用详细输出")
        
        if config.cli_colors and not config.progress_bars:
            validation_result["suggestions"].append("启用彩色输出时建议同时启用进度条")
        
        return validation_result
    
    def show_validation_result(self, validation_result: Dict[str, Any]) -> None:
        """显示验证结果"""
        status = validation_result["status"]
        status_color = "green" if status == "valid" else "red"
        status_text = "有效" if status == "valid" else "无效"
        
        console.print(Panel(
            f"配置状态: [{status_color}]{status_text}[/{status_color}]",
            title="[bold]配置验证结果[/bold]",
            border_style=status_color
        ))
        
        # 显示错误
        if validation_result["errors"]:
            console.print(f"\n[bold red]❌ 错误:[/bold red]")
            for error in validation_result["errors"]:
                console.print(f"  • {error}")
        
        # 显示警告
        if validation_result["warnings"]:
            console.print(f"\n[bold yellow]⚠️ 警告:[/bold yellow]")
            for warning in validation_result["warnings"]:
                console.print(f"  • {warning}")
        
        # 显示建议
        if validation_result["suggestions"]:
            console.print(f"\n[bold blue]💡 建议:[/bold blue]")
            for suggestion in validation_result["suggestions"]:
                console.print(f"  • {suggestion}")
        
        if status == "valid" and not validation_result["warnings"]:
            console.print(f"\n[green]✅ 配置完全正常![/green]")
    
    def reset_to_defaults(self) -> None:
        """重置为默认配置"""
        # 备份当前配置
        self.backup_config("before_reset")
        
        # 删除配置文件，让系统使用默认值
        if self.config_file.exists():
            self.config_file.unlink()
        
        console.print(f"[green]🔄 配置已重置为默认值[/green]")
    
    def export_config(self, export_path: Path, include_sensitive: bool = False) -> None:
        """导出配置"""
        config = self.get_current_config()
        
        # 转换为字典
        config_dict = {}
        for key in dir(config):
            if not key.startswith('_') and not callable(getattr(config, key)):
                value = getattr(config, key)
                
                # 跳过敏感信息
                if not include_sensitive and "password" in key.lower():
                    continue
                
                # 跳过复杂对象
                if isinstance(value, (str, int, float, bool, list, dict)):
                    config_dict[key] = value
        
        # 添加元数据
        export_data = {
            "metadata": {
                "exported_at": str(datetime.now()),
                "version": config.version,
                "app_name": config.app_name
            },
            "config": config_dict
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]📤 配置已导出到: {export_path}[/green]")
    
    def import_config(self, import_path: Path) -> None:
        """导入配置"""
        if not import_path.exists():
            raise ValidationError(f"导入文件不存在: {import_path}")
        
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 检查格式
            if "config" not in import_data:
                raise ValidationError("无效的配置导入文件格式")
            
            # 备份当前配置
            self.backup_config("before_import")
            
            # 应用导入的配置
            config_data = import_data["config"]
            new_config = Settings(**config_data)
            save_config_file(new_config, self.config_file)
            
            console.print(f"[green]📥 配置已从 {import_path} 导入[/green]")
            
            # 显示导入的元数据
            if "metadata" in import_data:
                metadata = import_data["metadata"]
                console.print(f"[dim]导出时间: {metadata.get('exported_at', 'Unknown')}[/dim]")
                console.print(f"[dim]原版本: {metadata.get('version', 'Unknown')}[/dim]")
            
        except Exception as e:
            raise ValidationError(f"导入配置失败: {e}")


# 全局配置管理器实例
config_manager = ConfigManager()