"""
Enhanced help system and user guidance for Knowledge QA System
增强的帮助系统和用户指导
"""

from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.markdown import Markdown

console = Console()


class HelpSystem:
    """Enhanced help system with detailed documentation and examples"""
    
    def __init__(self):
        self.commands = self._build_command_help()
        self.examples = self._build_examples()
        self.troubleshooting = self._build_troubleshooting()
    
    def _build_command_help(self) -> Dict[str, Dict]:
        """Build comprehensive command help documentation"""
        return {
            "new": {
                "description": "创建新的知识库并导入文档",
                "usage": "knowledge new --name <知识库名称> --file <文件路径> [选项]",
                "options": [
                    ("--name, -n", "知识库名称 (必需)", "字符串"),
                    ("--file, -f", "文档文件路径 (可多次使用)", "文件路径"),
                    ("--description, -d", "知识库描述", "字符串"),
                ],
                "examples": [
                    "knowledge new -n python-tutorial -f tutorial.pdf",
                    "knowledge new -n literature -f book1.epub -f book2.pdf -d '文学作品集'",
                    "knowledge new --name tech-docs --file manual.md --file guide.txt"
                ],
                "notes": [
                    "支持的文件格式: PDF, TXT, MD, EPUB",
                    "单个文件最大 100MB",
                    "每个知识库最多 100 个文件",
                    "知识库名称不能重复"
                ]
            },
            "list": {
                "description": "列出所有已创建的知识库",
                "usage": "knowledge list",
                "options": [],
                "examples": ["knowledge list"],
                "notes": [
                    "显示知识库名称、描述、文件数量等信息",
                    "按创建时间排序显示"
                ]
            },
            "delete": {
                "description": "删除指定的知识库",
                "usage": "knowledge delete <知识库名称> [选项]",
                "options": [
                    ("--force, -f", "强制删除，不询问确认", "标志")
                ],
                "examples": [
                    "knowledge delete python-tutorial",
                    "knowledge delete old-kb --force"
                ],
                "notes": [
                    "删除操作不可恢复",
                    "会同时删除向量数据和历史记录",
                    "默认会要求用户确认"
                ]
            },
            "status": {
                "description": "显示系统状态和健康检查",
                "usage": "knowledge status",
                "options": [],
                "examples": ["knowledge status"],
                "notes": [
                    "检查 Ollama 服务连接",
                    "验证数据库和向量存储状态",
                    "显示系统组件健康状况"
                ]
            },
            "review new": {
                "description": "开始新的问答学习会话",
                "usage": "knowledge <知识库名称> review new",
                "options": [],
                "examples": [
                    "knowledge python-tutorial review new",
                    "knowledge literature review new"
                ],
                "notes": [
                    "系统会基于知识库内容生成问题",
                    "输入答案后会得到评估和反馈",
                    "输入 'quit' 或 'exit' 退出会话",
                    "问答记录会自动保存到历史"
                ]
            },
            "review history": {
                "description": "查看问答历史记录",
                "usage": "knowledge <知识库名称> review history [选项]",
                "options": [
                    ("--limit, -l", "显示记录数量 (默认: 10)", "整数"),
                    ("--page, -p", "页码 (默认: 1)", "整数"),
                    ("--correct", "只显示正确的记录", "标志"),
                    ("--incorrect", "只显示错误的记录", "标志"),
                    ("--min-score", "最低分数过滤", "浮点数"),
                    ("--max-score", "最高分数过滤", "浮点数"),
                    ("--search, -s", "搜索问题内容", "字符串"),
                    ("--sort-by", "排序字段 (time/score/result)", "字符串"),
                    ("--sort-order", "排序顺序 (asc/desc)", "字符串"),
                    ("--detailed, -d", "显示详细信息", "标志")
                ],
                "examples": [
                    "knowledge python-tutorial review history",
                    "knowledge literature review history --limit 20 --page 2",
                    "knowledge tech-docs review history --correct --min-score 80",
                    "knowledge python-tutorial review history --search '函数' --detailed"
                ],
                "notes": [
                    "支持分页浏览历史记录",
                    "可按正确性、分数等条件过滤",
                    "支持关键词搜索问题内容",
                    "详细模式显示完整问答内容"
                ]
            },
            "review detail": {
                "description": "查看单个历史记录的详细信息",
                "usage": "knowledge <知识库名称> review detail <记录ID>",
                "options": [],
                "examples": [
                    "knowledge python-tutorial review detail 123",
                    "knowledge literature review detail 456"
                ],
                "notes": [
                    "显示完整的问题、答案和评估结果",
                    "包含参考答案和改进建议",
                    "记录ID可从历史列表中获取"
                ]
            },
            "review export": {
                "description": "导出历史记录",
                "usage": "knowledge <知识库名称> review export [选项]",
                "options": [
                    ("--format, -f", "导出格式 (json/csv)", "字符串"),
                    ("--output, -o", "输出文件路径", "文件路径")
                ],
                "examples": [
                    "knowledge python-tutorial review export",
                    "knowledge literature review export --format csv --output history.csv"
                ],
                "notes": [
                    "支持 JSON 和 CSV 格式导出",
                    "不指定输出文件时显示到控制台",
                    "包含完整的问答和评估数据"
                ]
            }
        }
    
    def _build_examples(self) -> Dict[str, List[str]]:
        """Build usage examples for different scenarios"""
        return {
            "getting_started": [
                "# 1. 创建第一个知识库",
                "knowledge new --name python-basics --file python_tutorial.pdf",
                "",
                "# 2. 开始学习",
                "knowledge python-basics review new",
                "",
                "# 3. 查看学习历史",
                "knowledge python-basics review history",
                "",
                "# 4. 列出所有知识库",
                "knowledge list"
            ],
            "advanced_usage": [
                "# 创建包含多个文件的知识库",
                "knowledge new -n advanced-python \\",
                "  -f advanced_concepts.pdf \\",
                "  -f design_patterns.md \\",
                "  -f best_practices.txt \\",
                "  -d '高级Python编程资料'",
                "",
                "# 过滤查看错误的记录",
                "knowledge advanced-python review history --incorrect --min-score 0 --max-score 60",
                "",
                "# 搜索特定主题的问答",
                "knowledge advanced-python review history --search '装饰器' --detailed",
                "",
                "# 导出学习记录进行分析",
                "knowledge advanced-python review export --format csv --output learning_progress.csv"
            ],
            "troubleshooting": [
                "# 检查系统状态",
                "knowledge status",
                "",
                "# 查看详细错误信息",
                "knowledge --verbose <command>",
                "",
                "# 强制删除有问题的知识库",
                "knowledge delete problematic-kb --force"
            ]
        }
    
    def _build_troubleshooting(self) -> Dict[str, Dict]:
        """Build troubleshooting guide"""
        return {
            "ollama_connection": {
                "problem": "Ollama 服务连接失败",
                "symptoms": [
                    "ModelServiceError: 无法连接到 Ollama 服务",
                    "Connection refused 错误",
                    "Timeout 错误"
                ],
                "solutions": [
                    "1. 检查 Ollama 服务是否运行: ollama serve",
                    "2. 验证服务地址配置: 默认 http://localhost:11434",
                    "3. 检查防火墙设置",
                    "4. 重启 Ollama 服务"
                ],
                "commands": [
                    "ollama serve",
                    "ollama list",
                    "curl http://localhost:11434/api/tags"
                ]
            },
            "model_missing": {
                "problem": "模型未找到或加载失败",
                "symptoms": [
                    "Model not found 错误",
                    "模型加载超时",
                    "生成问题或评估失败"
                ],
                "solutions": [
                    "1. 拉取所需模型: ollama pull qwen3:1.7b",
                    "2. 拉取嵌入模型: ollama pull shaw/dmeta-embedding-zh-small-q4",
                    "3. 检查模型列表: ollama list",
                    "4. 验证模型名称配置"
                ],
                "commands": [
                    "ollama pull qwen3:1.7b",
                    "ollama pull shaw/dmeta-embedding-zh-small-q4",
                    "ollama list"
                ]
            },
            "file_processing": {
                "problem": "文件处理失败",
                "symptoms": [
                    "FileProcessingError 错误",
                    "不支持的文件格式",
                    "文件过大或损坏"
                ],
                "solutions": [
                    "1. 检查文件格式: 支持 PDF, TXT, MD, EPUB",
                    "2. 验证文件大小: 最大 100MB",
                    "3. 确认文件未损坏且可正常打开",
                    "4. 检查文件编码: 推荐 UTF-8"
                ],
                "commands": [
                    "file <filename>",
                    "ls -lh <filename>",
                    "head -n 5 <filename>"
                ]
            },
            "database_issues": {
                "problem": "数据库操作失败",
                "symptoms": [
                    "DatabaseError 错误",
                    "权限拒绝",
                    "磁盘空间不足"
                ],
                "solutions": [
                    "1. 检查数据目录权限: data/ 目录",
                    "2. 验证磁盘空间",
                    "3. 检查数据库文件完整性",
                    "4. 重新初始化数据库"
                ],
                "commands": [
                    "ls -la data/",
                    "df -h",
                    "sqlite3 data/qa_history.db '.schema'"
                ]
            },
            "chinese_processing": {
                "problem": "中文处理异常",
                "symptoms": [
                    "中文显示乱码",
                    "分词错误",
                    "语义理解不准确"
                ],
                "solutions": [
                    "1. 确认终端支持 UTF-8 编码",
                    "2. 验证中文嵌入模型已安装",
                    "3. 检查文档编码格式",
                    "4. 更新模型到最新版本"
                ],
                "commands": [
                    "locale",
                    "ollama list | grep dmeta",
                    "file -i <chinese_file>"
                ]
            }
        }
    
    def show_command_help(self, command: str) -> None:
        """Show detailed help for a specific command"""
        if command not in self.commands:
            console.print(f"[red]未知命令: {command}[/red]")
            self.show_available_commands()
            return
        
        cmd_info = self.commands[command]
        
        # Command title and description
        console.print(Panel(
            f"[bold]{cmd_info['description']}[/bold]",
            title=f"[cyan]命令: {command}[/cyan]",
            border_style="cyan"
        ))
        
        # Usage
        console.print(f"\n[bold green]用法:[/bold green]")
        console.print(f"  {cmd_info['usage']}")
        
        # Options
        if cmd_info['options']:
            console.print(f"\n[bold yellow]选项:[/bold yellow]")
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Option", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Type", style="dim")
            
            for option, desc, type_info in cmd_info['options']:
                table.add_row(option, desc, type_info)
            
            console.print(table)
        
        # Examples
        if cmd_info['examples']:
            console.print(f"\n[bold blue]示例:[/bold blue]")
            for example in cmd_info['examples']:
                console.print(f"  [dim]$[/dim] {example}")
        
        # Notes
        if cmd_info['notes']:
            console.print(f"\n[bold magenta]注意事项:[/bold magenta]")
            for note in cmd_info['notes']:
                console.print(f"  • {note}")
    
    def show_available_commands(self) -> None:
        """Show all available commands"""
        console.print(Panel(
            "知识库问答系统 - 可用命令",
            title="[bold cyan]帮助[/bold cyan]",
            border_style="cyan"
        ))
        
        # Group commands by category
        categories = {
            "知识库管理": ["new", "list", "delete", "status"],
            "问答学习": ["review new", "review history", "review detail", "review export"]
        }
        
        for category, commands in categories.items():
            console.print(f"\n[bold yellow]{category}:[/bold yellow]")
            
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Command", style="cyan", width=20)
            table.add_column("Description", style="white")
            
            for cmd in commands:
                if cmd in self.commands:
                    table.add_row(cmd, self.commands[cmd]['description'])
            
            console.print(table)
        
        console.print(f"\n[dim]使用 'knowledge <command> --help' 查看具体命令的详细帮助[/dim]")
    
    def show_examples(self, category: str = "getting_started") -> None:
        """Show usage examples"""
        if category not in self.examples:
            console.print(f"[red]未知示例类别: {category}[/red]")
            console.print(f"可用类别: {', '.join(self.examples.keys())}")
            return
        
        title_map = {
            "getting_started": "快速开始",
            "advanced_usage": "高级用法",
            "troubleshooting": "故障排除"
        }
        
        title = title_map.get(category, category)
        
        console.print(Panel(
            "\n".join(self.examples[category]),
            title=f"[bold green]{title} 示例[/bold green]",
            border_style="green"
        ))
    
    def show_troubleshooting(self, problem: Optional[str] = None) -> None:
        """Show troubleshooting guide"""
        if problem and problem in self.troubleshooting:
            # Show specific problem
            issue = self.troubleshooting[problem]
            
            console.print(Panel(
                issue['problem'],
                title="[bold red]问题[/bold red]",
                border_style="red"
            ))
            
            # Symptoms
            console.print(f"\n[bold yellow]症状:[/bold yellow]")
            for symptom in issue['symptoms']:
                console.print(f"  • {symptom}")
            
            # Solutions
            console.print(f"\n[bold green]解决方案:[/bold green]")
            for solution in issue['solutions']:
                console.print(f"  {solution}")
            
            # Commands
            if issue.get('commands'):
                console.print(f"\n[bold blue]相关命令:[/bold blue]")
                for command in issue['commands']:
                    console.print(f"  [dim]$[/dim] {command}")
        
        else:
            # Show all problems
            console.print(Panel(
                "常见问题和解决方案",
                title="[bold red]故障排除指南[/bold red]",
                border_style="red"
            ))
            
            table = Table()
            table.add_column("问题类型", style="cyan", width=20)
            table.add_column("问题描述", style="white", width=40)
            table.add_column("查看详情", style="blue", width=20)
            
            for key, issue in self.troubleshooting.items():
                table.add_row(
                    key.replace("_", " ").title(),
                    issue['problem'],
                    f"--troubleshoot {key}"
                )
            
            console.print(table)
            
            console.print(f"\n[dim]使用 'knowledge --troubleshoot <问题类型>' 查看详细解决方案[/dim]")
    
    def show_quick_start(self) -> None:
        """Show quick start guide"""
        quick_start = """
# 🚀 快速开始指南

## 1. 环境准备
确保已安装并启动 Ollama 服务，并拉取所需模型：

```bash
# 启动 Ollama 服务
ollama serve

# 拉取模型（在另一个终端）
ollama pull qwen3:1.7b
ollama pull shaw/dmeta-embedding-zh-small-q4
```

## 2. 创建知识库
选择你的学习资料（PDF、TXT、MD、EPUB 格式）：

```bash
knowledge new --name my-study --file textbook.pdf --file notes.md
```

## 3. 开始学习
系统会基于你的资料生成问题：

```bash
knowledge my-study review new
```

## 4. 查看进度
回顾你的学习历史：

```bash
knowledge my-study review history
```

## 5. 管理知识库
查看和管理你的知识库：

```bash
knowledge list          # 列出所有知识库
knowledge status        # 检查系统状态
```

🎉 现在你可以开始智能学习了！
        """
        
        console.print(Markdown(quick_start))


# Global help system instance
help_system = HelpSystem()