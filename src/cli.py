"""
Command Line Interface for Knowledge QA System
知识库问答系统命令行界面
"""

import sys
import traceback
import signal
from pathlib import Path
from typing import List, Optional, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.live import Live
from rich.status import Status
from rich import print as rich_print

from .models import (
    KnowledgeSystemError,
    FileProcessingError,
    ModelServiceError,
    DatabaseError,
    ValidationError,
    KnowledgeBaseNotFoundError,
    VectorStoreError,
)
from .knowledge_base_manager import KnowledgeBaseManager
from .question_generator import QuestionGenerator
from .answer_evaluator import AnswerEvaluator
from .history_manager import HistoryManager
from .config import get_config, validate_system_requirements, save_config_file
from .help_system import help_system

# 初始化Rich控制台
console = Console()


def handle_error(func):
    """
    增强的错误处理装饰器

    统一处理各种异常，提供用户友好的错误信息和解决建议
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileProcessingError as e:
            console.print(
                Panel(
                    f"[red]{e.message}[/red]",
                    title="[bold red]文件处理错误[/bold red]",
                    border_style="red",
                )
            )

            # 提供解决建议
            suggestions = [
                "检查文件格式是否支持 (PDF, TXT, MD, EPUB)",
                "确认文件大小不超过 100MB",
                "验证文件未损坏且可正常打开",
                "检查文件路径是否正确",
            ]

            console.print("\n[yellow]💡 解决建议:[/yellow]")
            for suggestion in suggestions:
                console.print(f"  • {suggestion}")

            if e.details:
                console.print(f"\n[dim]详细信息: {e.details}[/dim]")

            console.print(
                f"\n[dim]获取更多帮助: knowledge --troubleshoot file_processing[/dim]"
            )
            console.print(f"[dim]环境检查: knowledge --check-env[/dim]")
            sys.exit(1)

        except ModelServiceError as e:
            console.print(
                Panel(
                    f"[red]{e.message}[/red]",
                    title="[bold red]模型服务错误[/bold red]",
                    border_style="red",
                )
            )

            suggestions = [
                "检查 Ollama 服务是否运行: ollama serve",
                "验证模型是否已安装: ollama list",
                "拉取所需模型: ollama pull qwen3:1.7b",
                "检查服务地址配置是否正确",
            ]

            console.print("\n[yellow]💡 解决建议:[/yellow]")
            for suggestion in suggestions:
                console.print(f"  • {suggestion}")

            if e.details:
                console.print(f"\n[dim]详细信息: {e.details}[/dim]")

            console.print(
                f"\n[dim]获取更多帮助: knowledge --troubleshoot ollama_connection[/dim]"
            )
            console.print(f"[dim]环境检查: knowledge --check-env[/dim]")
            sys.exit(1)

        except DatabaseError as e:
            console.print(
                Panel(
                    f"[red]{e.message}[/red]",
                    title="[bold red]数据库错误[/bold red]",
                    border_style="red",
                )
            )

            suggestions = [
                "检查数据目录权限: ls -la data/",
                "验证磁盘空间: df -h",
                "检查数据库文件完整性",
                "重启应用程序",
            ]

            console.print("\n[yellow]💡 解决建议:[/yellow]")
            for suggestion in suggestions:
                console.print(f"  • {suggestion}")

            if e.details:
                console.print(f"\n[dim]详细信息: {e.details}[/dim]")

            console.print(
                f"\n[dim]获取更多帮助: knowledge --troubleshoot database_issues[/dim]"
            )
            console.print(f"[dim]环境检查: knowledge --check-env[/dim]")
            sys.exit(1)

        except VectorStoreError as e:
            console.print(
                Panel(
                    f"[red]{e.message}[/red]",
                    title="[bold red]向量存储错误[/bold red]",
                    border_style="red",
                )
            )

            suggestions = [
                "检查 ChromaDB 数据目录权限",
                "验证磁盘空间是否充足",
                "重启应用程序",
                "清理损坏的向量数据",
            ]

            console.print("\n[yellow]💡 解决建议:[/yellow]")
            for suggestion in suggestions:
                console.print(f"  • {suggestion}")

            if e.details:
                console.print(f"\n[dim]详细信息: {e.details}[/dim]")

            sys.exit(1)

        except KnowledgeBaseNotFoundError as e:
            console.print(
                Panel(
                    f"[yellow]{e.message}[/yellow]",
                    title="[bold yellow]知识库不存在[/bold yellow]",
                    border_style="yellow",
                )
            )

            console.print("\n[blue]💡 可用操作:[/blue]")
            console.print("  • 查看所有知识库: [cyan]knowledge list[/cyan]")
            console.print(
                "  • 创建新知识库: [cyan]knowledge new --name <名称> --file <文件>[/cyan]"
            )

            sys.exit(1)

        except ValidationError as e:
            console.print(
                Panel(
                    f"[yellow]{e.message}[/yellow]",
                    title="[bold yellow]参数验证错误[/bold yellow]",
                    border_style="yellow",
                )
            )

            if e.details:
                console.print(f"\n[dim]详细信息: {e.details}[/dim]")

            console.print(f"\n[blue]💡 获取帮助:[/blue] [cyan]knowledge --help[/cyan]")
            sys.exit(1)

        except KnowledgeSystemError as e:
            console.print(
                Panel(
                    f"[red]{e.message}[/red]",
                    title="[bold red]系统错误[/bold red]",
                    border_style="red",
                )
            )

            if e.details:
                console.print(f"\n[dim]详细信息: {e.details}[/dim]")

            console.print(f"\n[blue]💡 获取帮助:[/blue]")
            console.print("  • 检查系统状态: [cyan]knowledge status[/cyan]")
            console.print("  • 查看故障排除: [cyan]knowledge --troubleshoot[/cyan]")
            sys.exit(1)

        except click.ClickException:
            # Click异常直接抛出，由Click处理
            raise

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ 操作已取消[/yellow]")
            sys.exit(0)

        except Exception as e:
            console.print(
                Panel(
                    f"[red]发生未预期的错误: {str(e)}[/red]",
                    title="[bold red]未知错误[/bold red]",
                    border_style="red",
                )
            )

            config = get_config()
            if config.debug:
                console.print("\n[dim]详细错误信息:[/dim]")
                console.print(traceback.format_exc())
            else:
                console.print("\n[dim]使用 --debug 选项查看详细错误信息[/dim]")

            console.print(f"\n[blue]💡 获取帮助:[/blue]")
            console.print("  • 检查系统状态: [cyan]knowledge status[/cyan]")
            console.print("  • 报告问题: 请提供错误信息和操作步骤")
            sys.exit(1)

    return wrapper


def setup_signal_handlers():
    """设置信号处理器，优雅处理中断"""

    def signal_handler(signum, frame):
        console.print("\n[yellow]⚠ 收到中断信号，正在清理...[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_file_paths(files: List[str]) -> List[str]:
    """
    验证文件路径

    Args:
        files: 文件路径列表

    Returns:
        List[str]: 验证通过的文件路径列表

    Raises:
        ValidationError: 文件验证失败
    """
    validated_files = []

    for file_path in files:
        path = Path(file_path)

        if not path.exists():
            raise ValidationError(f"文件不存在: {file_path}")

        if not path.is_file():
            raise ValidationError(f"路径不是文件: {file_path}")

        # 检查文件扩展名
        config = get_config()
        if path.suffix.lower() not in config.supported_file_extensions:
            supported = ", ".join(config.supported_file_extensions)
            raise ValidationError(
                f"不支持的文件格式: {path.suffix}\n" f"支持的格式: {supported}"
            )

        validated_files.append(str(path.resolve()))

    return validated_files


def show_progress(description: str, task_func, *args, **kwargs):
    """
    增强的进度指示器

    Args:
        description: 任务描述
        task_func: 要执行的任务函数
        *args, **kwargs: 传递给任务函数的参数

    Returns:
        任务函数的返回值
    """
    config = get_config()

    if not config.progress_bars:
        # 简单文本提示
        console.print(f"[dim]{description}...[/dim]")
        return task_func(*args, **kwargs)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=None)
        try:
            result = task_func(*args, **kwargs)
            progress.update(task, description=f"✅ {description}")
            return result
        except Exception as e:
            progress.update(task, description=f"❌ {description}")
            raise


def show_status(message: str):
    """显示状态信息"""
    config = get_config()
    if config.progress_bars:
        return Status(message, console=console)
    else:
        console.print(f"[dim]{message}...[/dim]")
        return None


def confirm_action(message: str, default: bool = False) -> bool:
    """确认用户操作"""
    return Confirm.ask(message, default=default, console=console)


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """获取用户输入"""
    return Prompt.ask(prompt, default=default, console=console)


def show_environment_check():
    """显示环境检查结果"""
    console.print(
        Panel(
            "环境和依赖检查",
            title="[bold cyan]系统诊断[/bold cyan]",
            border_style="cyan",
        )
    )

    # 检查系统需求
    sys_validation = validate_system_requirements()

    # 显示 Python 环境
    python_info = sys_validation["components"].get("python", {})
    python_status = python_info.get("status", "unknown")
    python_version = python_info.get("version", "unknown")

    status_color = "green" if python_status == "healthy" else "red"
    console.print(f"Python 版本: [{status_color}]{python_version}[/{status_color}]")

    # 检查依赖包
    console.print(f"\n[bold yellow]依赖包状态:[/bold yellow]")

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("包名", style="cyan", width=25)
    table.add_column("状态", style="white", width=10)
    table.add_column("说明", style="dim", width=30)

    for package, info in sys_validation["components"].items():
        if package == "python":
            continue

        status = info.get("status", "unknown")
        status_color = "green" if status == "healthy" else "red"
        status_text = "✓ 正常" if status == "healthy" else "✗ 缺失"

        description = ""
        if package == "chromadb":
            description = "向量数据库"
        elif package == "llama-index":
            description = "文档处理框架"
        elif package == "click":
            description = "命令行界面"
        elif package == "rich":
            description = "终端美化"

        table.add_row(package, Text(status_text, style=status_color), description)

    console.print(table)

    # 检查配置
    config = get_config()
    config_validation = config.validate_environment()

    console.print(f"\n[bold yellow]配置检查:[/bold yellow]")

    config_status = config_validation["status"]
    status_color = "green" if config_status == "healthy" else "red"
    status_text = "正常" if config_status == "healthy" else "异常"

    console.print(f"配置状态: [{status_color}]{status_text}[/{status_color}]")

    # 显示问题和警告
    if config_validation["issues"]:
        console.print(f"\n[bold red]发现问题:[/bold red]")
        for issue in config_validation["issues"]:
            console.print(f"  • {issue}")

    if config_validation["warnings"]:
        console.print(f"\n[bold yellow]警告:[/bold yellow]")
        for warning in config_validation["warnings"]:
            console.print(f"  • {warning}")

    # 显示建议
    console.print(f"\n[bold blue]建议操作:[/bold blue]")

    if sys_validation["status"] != "healthy":
        console.print("  • 安装缺失的依赖包: pip install -r requirements.txt")

    if config_validation["status"] != "healthy":
        console.print("  • 检查数据目录权限")
        console.print("  • 验证磁盘空间")

    console.print("  • 启动 Ollama 服务: ollama serve")
    console.print("  • 拉取所需模型: ollama pull qwen3:1.7b")
    console.print("  • 检查系统状态: knowledge status")

    # 显示配置文件位置
    config_file = Path.home() / ".knowledge_qa" / "config.json"
    console.print(f"\n[dim]配置文件位置: {config_file}[/dim]")
    console.print(f"[dim]日志文件位置: {config.log_file}[/dim]")


class KnowledgeCLI:
    """知识库命令行界面主类"""

    def __init__(self):
        """初始化CLI"""
        self.kb_manager = KnowledgeBaseManager()
        self.question_generator = QuestionGenerator()
        self.answer_evaluator = AnswerEvaluator()
        self.history_manager = HistoryManager()
        self.config = get_config()

    def create_knowledge_base(
        self, name: str, files: List[str], description: Optional[str] = None
    ):
        """创建知识库"""
        # 验证文件
        validated_files = validate_file_paths(files)

        # 创建知识库
        kb = show_progress(
            f"创建知识库 '{name}'...",
            self.kb_manager.create_knowledge_base,
            name,
            validated_files,
            description,
        )

        # 显示成功信息
        console.print(f"[green]✓[/green] 知识库 '{name}' 创建成功!")
        console.print(f"  - 文件数量: {kb.file_count}")
        console.print(f"  - 文档数量: {kb.document_count}")
        console.print(f"  - 创建时间: {kb.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if kb.description:
            console.print(f"  - 描述: {kb.description}")

    def start_new_review(self, kb_name: str):
        """开始新的问答会话"""
        # 检查知识库是否存在
        kb = self.kb_manager.get_knowledge_base(kb_name)
        if not kb:
            raise KnowledgeBaseNotFoundError(f"知识库 '{kb_name}' 不存在")

        console.print(f"[blue]开始 '{kb_name}' 知识库问答会话[/blue]")
        console.print("输入 'quit' 或 'exit' 退出会话\n")

        while True:
            try:
                # 生成问题
                question = show_progress(
                    "生成问题中...", self.question_generator.generate_question, kb_name
                )

                # 显示问题
                console.print(
                    Panel(
                        question.content,
                        title="[bold blue]问题[/bold blue]",
                        border_style="blue",
                    )
                )

                # 获取用户答案
                user_answer = console.input(
                    "\n[bold green]请输入您的答案:[/bold green] "
                )

                # 检查退出命令
                if user_answer.lower() in ["quit", "exit", "退出"]:
                    console.print("[yellow]会话已结束[/yellow]")
                    break

                if not user_answer.strip():
                    console.print("[yellow]答案不能为空，请重新输入[/yellow]")
                    continue

                # 评估答案
                evaluation = show_progress(
                    "评估答案中...",
                    self.answer_evaluator.evaluate_answer,
                    question.content,
                    user_answer,
                    kb_name,
                )

                # 显示评估结果
                self._display_evaluation_result(evaluation)

                # 保存问答记录
                from .models import QARecord

                qa_record = QARecord(
                    kb_name=kb_name,
                    question=question.content,
                    user_answer=user_answer,
                    evaluation=evaluation,
                )

                show_progress(
                    "保存记录中...", self.history_manager.save_qa_record, qa_record
                )

                # 询问是否继续
                console.print()
                continue_session = console.input(
                    "[dim]按回车继续，输入 'quit' 退出:[/dim] "
                )
                if continue_session.lower() in ["quit", "exit", "退出"]:
                    console.print("[yellow]会话已结束[/yellow]")
                    break

                console.print("\n" + "=" * 50 + "\n")

            except KeyboardInterrupt:
                console.print("\n[yellow]会话已取消[/yellow]")
                break

    def show_history(
        self,
        kb_name: str,
        limit: int = 10,
        page: int = 1,
        filter_correct: Optional[bool] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        search: Optional[str] = None,
        sort_by: str = "time",
        sort_order: str = "desc",
        detailed: bool = False,
    ):
        """显示历史记录"""
        from .history_manager import HistoryFilter, SortField, SortOrder

        # 检查知识库是否存在
        kb = self.kb_manager.get_knowledge_base(kb_name)
        if not kb:
            raise KnowledgeBaseNotFoundError(f"知识库 '{kb_name}' 不存在")

        # 构建过滤条件
        filter_criteria = HistoryFilter(
            kb_name=kb_name,
            is_correct=filter_correct,
            min_score=min_score,
            max_score=max_score,
            question_contains=search,
        )

        # 设置排序
        sort_field_map = {
            "time": SortField.CREATED_AT,
            "score": SortField.SCORE,
            "result": SortField.IS_CORRECT,
        }
        sort_field = sort_field_map.get(sort_by, SortField.CREATED_AT)
        sort_order_enum = (
            SortOrder.DESC if sort_order.lower() == "desc" else SortOrder.ASC
        )

        # 获取历史记录
        if any(
            [
                filter_correct is not None,
                min_score is not None,
                max_score is not None,
                search,
            ]
        ):
            # 使用过滤功能
            history_page = self.history_manager.get_filtered_history(
                filter_criteria, page, limit, sort_field, sort_order_enum
            )
        else:
            # 使用普通分页
            history_page = self.history_manager.get_history_page(
                kb_name, page, limit, sort_field, sort_order_enum
            )

        if not history_page.records:
            if any(
                [
                    filter_correct is not None,
                    min_score is not None,
                    max_score is not None,
                    search,
                ]
            ):
                console.print(f"[yellow]没有找到符合条件的历史记录[/yellow]")
            else:
                console.print(f"[yellow]知识库 '{kb_name}' 暂无历史记录[/yellow]")
            return

        # 显示统计信息
        stats = self.history_manager.get_statistics(kb_name)
        self._display_history_stats(kb_name, stats)

        if detailed:
            # 详细视图
            self._display_detailed_history(
                history_page.records, history_page.pagination
            )
        else:
            # 表格视图
            self._display_history_table(
                kb_name, history_page.records, history_page.pagination, page
            )

        # 显示分页信息和操作提示
        self._display_pagination_info(history_page.pagination)
        self._display_history_help(kb_name)

    def show_history_detail(self, kb_name: str, record_id: int):
        """显示单个历史记录的详细信息"""
        # 检查知识库是否存在
        kb = self.kb_manager.get_knowledge_base(kb_name)
        if not kb:
            raise KnowledgeBaseNotFoundError(f"知识库 '{kb_name}' 不存在")

        # 获取记录
        record = self.history_manager.get_record_by_id(record_id)
        if not record:
            console.print(f"[red]记录 ID {record_id} 不存在[/red]")
            return

        if record.kb_name != kb_name:
            console.print(f"[red]记录 ID {record_id} 不属于知识库 '{kb_name}'[/red]")
            return

        # 显示详细信息
        self._display_single_record_detail(record)

    def export_history(
        self, kb_name: str, format: str = "json", output_file: Optional[str] = None
    ):
        """导出历史记录"""
        # 检查知识库是否存在
        kb = self.kb_manager.get_knowledge_base(kb_name)
        if not kb:
            raise KnowledgeBaseNotFoundError(f"知识库 '{kb_name}' 不存在")

        # 导出数据
        data = show_progress(
            f"导出 '{kb_name}' 历史记录...",
            self.history_manager.export_history,
            kb_name,
            format,
        )

        # 保存到文件或显示
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(data)
            console.print(f"[green]✓[/green] 历史记录已导出到: {output_file}")
        else:
            console.print(data)

    def _display_history_stats(self, kb_name: str, stats: Dict[str, Any]):
        """显示历史统计信息"""
        if stats["total_count"] == 0:
            return

        accuracy = stats.get("accuracy_rate", 0)
        avg_score = stats.get("average_score", 0)
        recent_activity = stats.get("recent_activity_count", 0)

        stats_text = []
        stats_text.append(f"总记录: {stats['total_count']}")
        stats_text.append(f"正确率: {accuracy:.1f}%")
        stats_text.append(f"平均分: {avg_score:.1f}")
        stats_text.append(f"近7天: {recent_activity}条")

        console.print(
            Panel(
                " | ".join(stats_text),
                title=f"[bold cyan]{kb_name} 统计信息[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        console.print()

    def _display_history_table(
        self, kb_name: str, records: List, pagination, page: int
    ):
        """显示历史记录表格"""
        # 创建表格
        title_parts = [f"知识库 '{kb_name}' 历史记录"]
        if pagination.total_pages > 1:
            title_parts.append(f"(第 {page}/{pagination.total_pages} 页)")

        table = Table(title=" ".join(title_parts))
        table.add_column("ID", style="dim", width=6)
        table.add_column("时间", style="cyan", width=16)
        table.add_column("问题", style="white", width=45)
        table.add_column("结果", style="green", width=8)
        table.add_column("分数", style="yellow", width=8)
        table.add_column("操作", style="blue", width=12)

        for record in records:
            result_text = "✓ 正确" if record.evaluation.is_correct else "✗ 错误"
            result_style = "green" if record.evaluation.is_correct else "red"

            # 截断长问题
            question_text = record.question
            if len(question_text) > 40:
                question_text = question_text[:37] + "..."

            table.add_row(
                str(record.id),
                record.created_at.strftime("%m-%d %H:%M"),
                question_text,
                Text(result_text, style=result_style),
                f"{record.evaluation.score:.1f}",
                f"[blue]详情[/blue]",
            )

        console.print(table)

    def _display_detailed_history(self, records: List, pagination):
        """显示详细历史记录"""
        for i, record in enumerate(records, 1):
            if i > 1:
                console.print("\n" + "─" * 80 + "\n")

            self._display_single_record_detail(record, show_header=True)

    def _display_single_record_detail(self, record, show_header: bool = False):
        """显示单个记录的详细信息"""
        if show_header:
            header = (
                f"记录 #{record.id} - {record.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            console.print(f"[bold cyan]{header}[/bold cyan]")
            console.print()

        # 问题
        console.print(
            Panel(
                record.question,
                title="[bold blue]问题[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # 用户答案
        console.print(
            Panel(
                record.user_answer,
                title="[bold green]您的答案[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # 评估结果
        result_color = "green" if record.evaluation.is_correct else "red"
        result_text = "正确" if record.evaluation.is_correct else "错误"

        eval_content = []
        eval_content.append(
            f"[bold]结果:[/bold] [{result_color}]{result_text}[/{result_color}]"
        )
        eval_content.append(f"[bold]分数:[/bold] {record.evaluation.score:.1f}/10")
        eval_content.append("")
        eval_content.append(f"[bold]反馈:[/bold]\n{record.evaluation.feedback}")

        if record.evaluation.strengths:
            eval_content.append("")
            eval_content.append("[bold green]优点:[/bold green]")
            for strength in record.evaluation.strengths:
                eval_content.append(f"  • {strength}")

        if record.evaluation.missing_points:
            eval_content.append("")
            eval_content.append("[bold yellow]需要补充:[/bold yellow]")
            for point in record.evaluation.missing_points:
                eval_content.append(f"  • {point}")

        console.print(
            Panel(
                "\n".join(eval_content),
                title="[bold yellow]评估结果[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        # 参考答案
        console.print(
            Panel(
                record.evaluation.reference_answer,
                title="[bold magenta]参考答案[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )

    def _display_pagination_info(self, pagination):
        """显示分页信息"""
        if pagination.total_pages <= 1:
            return

        info_parts = []
        info_parts.append(f"第 {pagination.page}/{pagination.total_pages} 页")
        info_parts.append(f"共 {pagination.total_count} 条记录")

        console.print(f"\n[dim]{' | '.join(info_parts)}[/dim]")

        # 导航提示
        nav_parts = []
        if pagination.has_prev:
            nav_parts.append(f"--page {pagination.page - 1} (上一页)")
        if pagination.has_next:
            nav_parts.append(f"--page {pagination.page + 1} (下一页)")

        if nav_parts:
            console.print(f"[dim]使用 {' 或 '.join(nav_parts)}[/dim]")

    def _display_history_help(self, kb_name: str):
        """显示历史记录操作帮助"""
        console.print(f"\n[dim]💡 提示:[/dim]")
        console.print(
            f"[dim]  • 查看详情: knowledge review {kb_name} history --detailed[/dim]"
        )
        console.print(
            f"[dim]  • 过滤记录: knowledge review {kb_name} history --correct/--incorrect[/dim]"
        )
        console.print(
            f"[dim]  • 搜索问题: knowledge review {kb_name} history --search '关键词'[/dim]"
        )
        console.print(f"[dim]  • 导出记录: knowledge review {kb_name} export[/dim]")

    def list_knowledge_bases(self):
        """列出所有知识库"""
        knowledge_bases = self.kb_manager.get_knowledge_base_details()

        if not knowledge_bases:
            console.print("[yellow]暂无知识库[/yellow]")
            console.print("使用 'knowledge new' 创建新的知识库")
            return

        # 创建表格
        table = Table(title="知识库列表")
        table.add_column("名称", style="cyan", width=20)
        table.add_column("描述", style="white", width=30)
        table.add_column("文件数", style="green", width=8)
        table.add_column("文档数", style="green", width=8)
        table.add_column("创建时间", style="dim", width=16)

        for kb in knowledge_bases:
            description = kb.description or "[dim]无描述[/dim]"
            if len(description) > 25:
                description = description[:22] + "..."

            table.add_row(
                kb.name,
                description,
                str(kb.file_count),
                str(kb.document_count),
                kb.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        console.print(f"\n[dim]共 {len(knowledge_bases)} 个知识库[/dim]")

    def delete_knowledge_base(self, name: str, force: bool = False):
        """删除知识库"""
        # 检查知识库是否存在
        kb = self.kb_manager.get_knowledge_base(name)
        if not kb:
            raise KnowledgeBaseNotFoundError(f"知识库 '{name}' 不存在")

        # 确认删除
        if not force:
            console.print(f"[yellow]警告: 即将删除知识库 '{name}'[/yellow]")
            console.print(f"  - 文件数量: {kb.file_count}")
            console.print(f"  - 文档数量: {kb.document_count}")
            console.print(
                f"  - 创建时间: {kb.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            confirm = console.input("\n[red]此操作不可恢复，确认删除吗? (y/N):[/red] ")
            if confirm.lower() not in ["y", "yes", "是"]:
                console.print("[yellow]删除已取消[/yellow]")
                return

        # 删除知识库
        success = show_progress(
            f"删除知识库 '{name}'...", self.kb_manager.delete_knowledge_base, name
        )

        if success:
            console.print(f"[green]✓[/green] 知识库 '{name}' 删除成功")
        else:
            console.print(f"[yellow]⚠[/yellow] 知识库 '{name}' 部分删除成功")

    def show_system_status(self):
        """显示系统状态"""
        health_info = show_progress("检查系统状态...", self.kb_manager.health_check)

        # 显示整体状态
        status_color = "green" if health_info["status"] == "healthy" else "red"
        status_text = "正常" if health_info["status"] == "healthy" else "异常"

        console.print(
            Panel(
                f"系统状态: [{status_color}]{status_text}[/{status_color}]",
                title="[bold]系统健康检查[/bold]",
                border_style=status_color,
            )
        )

        # 显示组件状态
        components = health_info.get("components", {})

        table = Table(title="组件状态")
        table.add_column("组件", style="cyan", width=20)
        table.add_column("状态", style="white", width=10)
        table.add_column("详细信息", style="dim", width=40)

        for component_name, component_info in components.items():
            status = component_info.get("status", "unknown")
            status_color = "green" if status == "healthy" else "red"
            status_text = "正常" if status == "healthy" else "异常"

            details = []
            if component_name == "database":
                details.append(
                    f"知识库数量: {component_info.get('knowledge_base_count', 0)}"
                )
            elif component_name == "document_processor":
                formats = component_info.get("supported_formats", [])
                details.append(f"支持格式: {', '.join(formats)}")
            elif component_name == "vector_store":
                if "embedding_model" in component_info:
                    details.append(f"嵌入模型: {component_info['embedding_model']}")

            error = component_info.get("error")
            if error:
                details.append(f"错误: {error}")

            table.add_row(
                component_name,
                Text(status_text, style=status_color),
                " | ".join(details) if details else "-",
            )

        console.print(table)

        # 显示时间戳
        timestamp = health_info.get("timestamp", "")
        if timestamp:
            console.print(f"\n[dim]检查时间: {timestamp}[/dim]")

    def _display_evaluation_result(self, evaluation):
        """显示评估结果"""
        # 结果标题
        if evaluation.is_correct:
            title = "[bold green]✓ 回答正确![/bold green]"
            border_style = "green"
        else:
            title = "[bold red]✗ 回答有误[/bold red]"
            border_style = "red"

        # 构建结果内容
        content = []
        content.append(f"[bold]分数:[/bold] {evaluation.score:.1f}/10")
        content.append("")
        content.append(f"[bold]反馈:[/bold]\n{evaluation.feedback}")

        if evaluation.strengths:
            content.append("")
            content.append("[bold green]优点:[/bold green]")
            for strength in evaluation.strengths:
                content.append(f"  • {strength}")

        if evaluation.missing_points:
            content.append("")
            content.append("[bold yellow]需要补充:[/bold yellow]")
            for point in evaluation.missing_points:
                content.append(f"  • {point}")

        content.append("")
        content.append(f"[bold]参考答案:[/bold]\n{evaluation.reference_answer}")

        console.print(Panel("\n".join(content), title=title, border_style=border_style))


# 全局CLI实例
cli_instance = KnowledgeCLI()


# ============================================================================
# 配置管理命令
# ============================================================================
# ============================================================================
# Click命令定义
# ============================================================================


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="knowledge")
@click.option("--debug", is_flag=True, help="启用调试模式")
@click.option("--verbose", "-v", is_flag=True, help="显示详细输出")
@click.option("--config", type=click.Path(exists=True), help="指定配置文件路径")
@click.option("--no-color", is_flag=True, help="禁用彩色输出")
@click.option("--help-command", help="显示特定命令的详细帮助")
@click.option(
    "--examples",
    type=click.Choice(["getting_started", "advanced_usage", "troubleshooting"]),
    help="显示使用示例",
)
@click.option("--troubleshoot", help="显示特定问题的故障排除指南")
@click.option("--troubleshoot-all", is_flag=True, help="显示所有故障排除指南")
@click.option("--quick-start", is_flag=True, help="显示快速开始指南")
@click.option("--check-env", is_flag=True, help="检查环境和依赖")
@click.help_option("--help", "-h")
@click.pass_context
def main(
    ctx,
    debug,
    verbose,
    config,
    no_color,
    help_command,
    examples,
    troubleshoot,
    troubleshoot_all,
    quick_start,
    check_env,
):
    """
    🧠 知识库问答系统 (Knowledge QA System)

    基于向量检索和大语言模型的智能学习工具

    \b
    快速开始:
      knowledge new -n my-kb -f document.pdf    # 创建知识库
      knowledge review my-kb new                # 开始学习
      knowledge review my-kb history            # 查看历史

    \b
    获取帮助:
      knowledge --help-command new             # 命令详细帮助
      knowledge --examples getting_started     # 使用示例
      knowledge --troubleshoot                 # 故障排除
      knowledge --quick-start                  # 快速开始指南
    """
    # 设置信号处理器
    setup_signal_handlers()

    # 处理配置
    if config:
        from pathlib import Path

        get_config(Path(config), force_reload=True)

    config_obj = get_config()

    # 设置调试和详细模式
    if debug:
        config_obj.debug = True
        config_obj.log_level = "DEBUG"

    if verbose:
        config_obj.verbose_output = True

    if no_color:
        config_obj.cli_colors = False
        console._color_system = None

    # Debug: print received options
    if config_obj.debug:
        console.print(
            f"[dim]Debug: quick_start={quick_start}, help_command={help_command}[/dim]"
        )

    # 处理帮助选项
    if help_command:
        help_system.show_command_help(help_command)
        ctx.exit()

    if examples:
        help_system.show_examples(examples)
        ctx.exit()

    if troubleshoot:
        help_system.show_troubleshooting(troubleshoot)
        ctx.exit()

    if troubleshoot_all:
        help_system.show_troubleshooting()
        ctx.exit()

    if quick_start:
        help_system.show_quick_start()
        ctx.exit()

    if check_env:
        show_environment_check()
        ctx.exit()

    # 如果没有子命令，显示帮助
    if ctx.invoked_subcommand is None:
        help_system.show_available_commands()

        # 显示系统状态摘要
        try:
            validation = validate_system_requirements()
            if validation["status"] != "healthy":
                console.print(
                    f"\n[yellow]⚠ 系统检查发现问题，使用 'knowledge status' 查看详情[/yellow]"
                )
        except Exception:
            pass

        ctx.exit()


@main.command("new", help="创建新的知识库")
@click.option("--name", "-n", required=True, help="知识库名称")
@click.option(
    "--file",
    "-f",
    "files",
    multiple=True,
    required=True,
    help="文档文件路径 (可多次使用)",
)
@click.option("--description", "-d", help="知识库描述")
@handle_error
def create_knowledge_base(name: str, files: tuple, description: Optional[str]):
    """创建新的知识库"""
    cli_instance.create_knowledge_base(name, list(files), description)


@main.command("list", help="列出所有知识库")
@handle_error
def list_knowledge_bases():
    """列出所有知识库"""
    cli_instance.list_knowledge_bases()


@main.command("delete", help="删除知识库")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="强制删除，不询问确认")
@handle_error
def delete_knowledge_base(name: str, force: bool):
    """删除知识库"""
    cli_instance.delete_knowledge_base(name, force)


@main.command("status", help="显示系统状态")
@handle_error
def show_system_status():
    """显示系统状态"""
    cli_instance.show_system_status()


@main.group("config", help="配置管理")
def config_group():
    """配置管理命令组"""
    pass


@config_group.command("show", help="显示当前配置")
@click.option("--sensitive", is_flag=True, help="显示敏感信息")
@handle_error
def show_config(sensitive: bool):
    """显示当前配置"""
    from .config_manager import config_manager

    config_manager.show_config(show_sensitive=sensitive)


@config_group.command("validate", help="验证配置")
@handle_error
def validate_config():
    """验证配置"""
    from .config_manager import config_manager

    validation_result = config_manager.validate_config()
    config_manager.show_validation_result(validation_result)


@config_group.command("save", help="保存当前配置")
@click.option("--path", type=click.Path(), help="保存路径")
@handle_error
def save_config(path: Optional[str]):
    """保存当前配置"""
    from .config_manager import config_manager

    save_path = Path(path) if path else None
    config_manager.save_current_config(save_path)


@config_group.command("load", help="加载配置文件")
@click.argument("config_path", type=click.Path(exists=True))
@handle_error
def load_config(config_path: str):
    """加载配置文件"""
    from .config_manager import config_manager

    config_manager.load_config(Path(config_path))


@config_group.command("backup", help="备份当前配置")
@click.option("--name", help="备份名称")
@handle_error
def backup_config(name: Optional[str]):
    """备份当前配置"""
    from .config_manager import config_manager

    config_manager.backup_config(name)


@config_group.command("restore", help="恢复配置")
@click.argument("backup_path", type=click.Path(exists=True))
@handle_error
def restore_config(backup_path: str):
    """恢复配置"""
    from .config_manager import config_manager

    if confirm_action(f"确认从 {backup_path} 恢复配置吗？"):
        config_manager.restore_config(Path(backup_path))
    else:
        console.print("[yellow]恢复已取消[/yellow]")


@config_group.command("reset", help="重置为默认配置")
@click.option("--force", is_flag=True, help="强制重置，不询问确认")
@handle_error
def reset_config(force: bool):
    """重置为默认配置"""
    from .config_manager import config_manager

    if force or confirm_action("确认重置配置为默认值吗？此操作会备份当前配置"):
        config_manager.reset_to_defaults()
    else:
        console.print("[yellow]重置已取消[/yellow]")


@config_group.group("template", help="配置模板管理")
def template_group():
    """配置模板管理命令组"""
    pass


@template_group.command("list", help="列出所有模板")
@handle_error
def list_templates():
    """列出所有模板"""
    from .config_manager import config_manager

    config_manager.list_templates()


@template_group.command("apply", help="应用配置模板")
@click.argument("template_name")
@handle_error
def apply_template(template_name: str):
    """应用配置模板"""
    from .config_manager import config_manager

    if confirm_action(f"确认应用模板 '{template_name}' 吗？当前配置会被备份"):
        config_manager.apply_template(template_name)
    else:
        console.print("[yellow]应用已取消[/yellow]")


@template_group.command("create", help="创建自定义模板")
@click.argument("name")
@click.option("--description", "-d", required=True, help="模板描述")
@click.option("--from-current", is_flag=True, help="基于当前配置创建")
@handle_error
def create_template(name: str, description: str, from_current: bool):
    """创建自定义模板"""
    from .config_manager import config_manager

    if from_current:
        # 基于当前配置创建模板
        current_config = config_manager.get_current_config()
        settings = {}

        # 选择要包含的设置
        console.print("[blue]选择要包含在模板中的设置类别:[/blue]")
        categories = {
            "ollama": ["ollama_timeout", "ollama_max_retries", "ollama_retry_delay"],
            "performance": ["vector_search_k", "chunk_size", "chunk_overlap"],
            "ui": ["cli_colors", "progress_bars", "verbose_output"],
            "logging": ["log_level", "debug"],
            "generation": ["question_generation_temperature", "evaluation_temperature"],
        }

        for category, keys in categories.items():
            if confirm_action(f"包含 {category} 设置？"):
                for key in keys:
                    if hasattr(current_config, key):
                        settings[key] = getattr(current_config, key)

        config_manager.create_template(name, description, settings)
    else:
        # 交互式创建模板
        console.print("[blue]交互式创建模板 (输入空值跳过):[/blue]")
        settings = {}

        # 常用设置
        common_settings = {
            "debug": ("调试模式", bool),
            "log_level": ("日志级别", str),
            "ollama_timeout": ("Ollama 超时时间", int),
            "cli_colors": ("彩色输出", bool),
            "progress_bars": ("进度条", bool),
        }

        for key, (desc, type_func) in common_settings.items():
            value = get_user_input(f"{desc} ({key})")
            if value:
                try:
                    if type_func == bool:
                        settings[key] = value.lower() in ["true", "1", "yes", "y"]
                    else:
                        settings[key] = type_func(value)
                except ValueError:
                    console.print(f"[yellow]跳过无效值: {value}[/yellow]")

        if settings:
            config_manager.create_template(name, description, settings)
        else:
            console.print("[yellow]未提供任何设置，模板创建已取消[/yellow]")


@template_group.command("delete", help="删除自定义模板")
@click.argument("name")
@click.option("--force", is_flag=True, help="强制删除，不询问确认")
@handle_error
def delete_template(name: str, force: bool):
    """删除自定义模板"""
    from .config_manager import config_manager

    if force or confirm_action(f"确认删除模板 '{name}' 吗？"):
        config_manager.delete_template(name)
    else:
        console.print("[yellow]删除已取消[/yellow]")


@config_group.command("export", help="导出配置")
@click.argument("export_path", type=click.Path())
@click.option("--include-sensitive", is_flag=True, help="包含敏感信息")
@handle_error
def export_config(export_path: str, include_sensitive: bool):
    """导出配置"""
    from .config_manager import config_manager

    config_manager.export_config(Path(export_path), include_sensitive)


@config_group.command("import", help="导入配置")
@click.argument("import_path", type=click.Path(exists=True))
@handle_error
def import_config(import_path: str):
    """导入配置"""
    from .config_manager import config_manager

    if confirm_action(f"确认从 {import_path} 导入配置吗？当前配置会被备份"):
        config_manager.import_config(Path(import_path))
    else:
        console.print("[yellow]导入已取消[/yellow]")


@main.group()
@click.argument("kb_name")
@click.pass_context
def review(ctx, kb_name: str):
    """知识库问答功能"""
    ctx.ensure_object(dict)
    ctx.obj["kb_name"] = kb_name


@review.command("new")
@click.pass_context
@handle_error
def start_new_review(ctx):
    """开始新的问答会话"""
    kb_name = ctx.obj["kb_name"]
    cli_instance.start_new_review(kb_name)


@review.command("history")
@click.option("--limit", "-l", default=10, help="显示记录数量 (默认: 10)")
@click.option("--page", "-p", default=1, help="页码 (默认: 1)")
@click.option("--correct", "filter_correct", flag_value=True, help="只显示正确的记录")
@click.option(
    "--incorrect", "filter_correct", flag_value=False, help="只显示错误的记录"
)
@click.option("--min-score", type=float, help="最低分数过滤")
@click.option("--max-score", type=float, help="最高分数过滤")
@click.option("--search", "-s", help="搜索问题内容")
@click.option(
    "--sort-by",
    type=click.Choice(["time", "score", "result"]),
    default="time",
    help="排序字段 (默认: time)",
)
@click.option(
    "--sort-order",
    type=click.Choice(["asc", "desc"]),
    default="desc",
    help="排序顺序 (默认: desc)",
)
@click.option("--detailed", "-d", is_flag=True, help="显示详细信息")
@click.pass_context
@handle_error
def show_history(
    ctx,
    limit: int,
    page: int,
    filter_correct: Optional[bool],
    min_score: Optional[float],
    max_score: Optional[float],
    search: Optional[str],
    sort_by: str,
    sort_order: str,
    detailed: bool,
):
    """查看问答历史记录"""
    kb_name = ctx.obj["kb_name"]
    cli_instance.show_history(
        kb_name,
        limit,
        page,
        filter_correct,
        min_score,
        max_score,
        search,
        sort_by,
        sort_order,
        detailed,
    )


@review.command("detail")
@click.argument("record_id", type=int)
@click.pass_context
@handle_error
def show_history_detail(ctx, record_id: int):
    """查看单个历史记录的详细信息"""
    kb_name = ctx.obj["kb_name"]
    cli_instance.show_history_detail(kb_name, record_id)


@review.command("export")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "csv"]),
    default="json",
    help="导出格式 (默认: json)",
)
@click.option("--output", "-o", help="输出文件路径")
@click.pass_context
@handle_error
def export_history(ctx, format: str, output: Optional[str]):
    """导出历史记录"""
    kb_name = ctx.obj["kb_name"]
    cli_instance.export_history(kb_name, format, output)


if __name__ == "__main__":
    main()
