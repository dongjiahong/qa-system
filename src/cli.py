"""
Command Line Interface for Knowledge QA System
知识库问答系统命令行界面
"""

import sys
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rich_print

from .models import (
    KnowledgeSystemError,
    FileProcessingError,
    ModelServiceError,
    DatabaseError,
    ValidationError,
    KnowledgeBaseNotFoundError,
    VectorStoreError
)
from .knowledge_base_manager import KnowledgeBaseManager
from .question_generator import QuestionGenerator
from .answer_evaluator import AnswerEvaluator
from .history_manager import HistoryManager
from .config import get_config

# 初始化Rich控制台
console = Console()


def handle_error(func):
    """
    错误处理装饰器
    
    统一处理各种异常，提供用户友好的错误信息
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileProcessingError as e:
            console.print(f"[red]文件处理错误:[/red] {e.message}", style="red")
            if e.details:
                console.print(f"详细信息: {e.details}")
            sys.exit(1)
        except ModelServiceError as e:
            console.print(f"[red]模型服务错误:[/red] {e.message}", style="red")
            console.print("请检查 Ollama 服务是否正常运行")
            if e.details:
                console.print(f"详细信息: {e.details}")
            sys.exit(1)
        except DatabaseError as e:
            console.print(f"[red]数据库错误:[/red] {e.message}", style="red")
            console.print("请检查数据库文件权限和磁盘空间")
            if e.details:
                console.print(f"详细信息: {e.details}")
            sys.exit(1)
        except VectorStoreError as e:
            console.print(f"[red]向量存储错误:[/red] {e.message}", style="red")
            console.print("请检查 ChromaDB 服务是否正常运行")
            if e.details:
                console.print(f"详细信息: {e.details}")
            sys.exit(1)
        except KnowledgeBaseNotFoundError as e:
            console.print(f"[yellow]知识库不存在:[/yellow] {e.message}", style="yellow")
            console.print("使用 'knowledge list' 查看可用的知识库")
            sys.exit(1)
        except ValidationError as e:
            console.print(f"[yellow]参数验证错误:[/yellow] {e.message}", style="yellow")
            if e.details:
                console.print(f"详细信息: {e.details}")
            sys.exit(1)
        except KnowledgeSystemError as e:
            console.print(f"[red]系统错误:[/red] {e.message}", style="red")
            if e.details:
                console.print(f"详细信息: {e.details}")
            sys.exit(1)
        except click.ClickException:
            # Click异常直接抛出，由Click处理
            raise
        except KeyboardInterrupt:
            console.print("\n[yellow]操作已取消[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[red]未知错误:[/red] {str(e)}", style="red")
            console.print("\n[dim]详细错误信息:[/dim]")
            console.print(traceback.format_exc())
            sys.exit(1)
    
    return wrapper


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
                f"不支持的文件格式: {path.suffix}\n"
                f"支持的格式: {supported}"
            )
        
        validated_files.append(str(path.resolve()))
    
    return validated_files


def show_progress(description: str, task_func, *args, **kwargs):
    """
    显示进度指示器
    
    Args:
        description: 任务描述
        task_func: 要执行的任务函数
        *args, **kwargs: 传递给任务函数的参数
        
    Returns:
        任务函数的返回值
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(description, total=None)
        try:
            result = task_func(*args, **kwargs)
            progress.update(task, description=f"✓ {description}")
            return result
        except Exception as e:
            progress.update(task, description=f"✗ {description}")
            raise


class KnowledgeCLI:
    """知识库命令行界面主类"""
    
    def __init__(self):
        """初始化CLI"""
        self.kb_manager = KnowledgeBaseManager()
        self.question_generator = QuestionGenerator()
        self.answer_evaluator = AnswerEvaluator()
        self.history_manager = HistoryManager()
        self.config = get_config()
    
    def create_knowledge_base(self, name: str, files: List[str], description: Optional[str] = None):
        """创建知识库"""
        # 验证文件
        validated_files = validate_file_paths(files)
        
        # 创建知识库
        kb = show_progress(
            f"创建知识库 '{name}'...",
            self.kb_manager.create_knowledge_base,
            name, validated_files, description
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
                    "生成问题中...",
                    self.question_generator.generate_question,
                    kb_name
                )
                
                # 显示问题
                console.print(Panel(
                    question.content,
                    title="[bold blue]问题[/bold blue]",
                    border_style="blue"
                ))
                
                # 获取用户答案
                user_answer = console.input("\n[bold green]请输入您的答案:[/bold green] ")
                
                # 检查退出命令
                if user_answer.lower() in ['quit', 'exit', '退出']:
                    console.print("[yellow]会话已结束[/yellow]")
                    break
                
                if not user_answer.strip():
                    console.print("[yellow]答案不能为空，请重新输入[/yellow]")
                    continue
                
                # 评估答案
                evaluation = show_progress(
                    "评估答案中...",
                    self.answer_evaluator.evaluate_answer,
                    question.content, user_answer, kb_name
                )
                
                # 显示评估结果
                self._display_evaluation_result(evaluation)
                
                # 保存问答记录
                from .models import QARecord
                qa_record = QARecord(
                    kb_name=kb_name,
                    question=question.content,
                    user_answer=user_answer,
                    evaluation=evaluation
                )
                
                show_progress(
                    "保存记录中...",
                    self.history_manager.save_qa_record,
                    qa_record
                )
                
                # 询问是否继续
                console.print()
                continue_session = console.input("[dim]按回车继续，输入 'quit' 退出:[/dim] ")
                if continue_session.lower() in ['quit', 'exit', '退出']:
                    console.print("[yellow]会话已结束[/yellow]")
                    break
                
                console.print("\n" + "="*50 + "\n")
                
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
        detailed: bool = False
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
            question_contains=search
        )
        
        # 设置排序
        sort_field_map = {
            "time": SortField.CREATED_AT,
            "score": SortField.SCORE,
            "result": SortField.IS_CORRECT
        }
        sort_field = sort_field_map.get(sort_by, SortField.CREATED_AT)
        sort_order_enum = SortOrder.DESC if sort_order.lower() == "desc" else SortOrder.ASC
        
        # 获取历史记录
        if any([filter_correct is not None, min_score is not None, max_score is not None, search]):
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
            if any([filter_correct is not None, min_score is not None, max_score is not None, search]):
                console.print(f"[yellow]没有找到符合条件的历史记录[/yellow]")
            else:
                console.print(f"[yellow]知识库 '{kb_name}' 暂无历史记录[/yellow]")
            return
        
        # 显示统计信息
        stats = self.history_manager.get_statistics(kb_name)
        self._display_history_stats(kb_name, stats)
        
        if detailed:
            # 详细视图
            self._display_detailed_history(history_page.records, history_page.pagination)
        else:
            # 表格视图
            self._display_history_table(kb_name, history_page.records, history_page.pagination, page)
        
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
    
    def export_history(self, kb_name: str, format: str = "json", output_file: Optional[str] = None):
        """导出历史记录"""
        # 检查知识库是否存在
        kb = self.kb_manager.get_knowledge_base(kb_name)
        if not kb:
            raise KnowledgeBaseNotFoundError(f"知识库 '{kb_name}' 不存在")
        
        # 导出数据
        data = show_progress(
            f"导出 '{kb_name}' 历史记录...",
            self.history_manager.export_history,
            kb_name, format
        )
        
        # 保存到文件或显示
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(data)
            console.print(f"[green]✓[/green] 历史记录已导出到: {output_file}")
        else:
            console.print(data)
    
    def _display_history_stats(self, kb_name: str, stats: Dict[str, Any]):
        """显示历史统计信息"""
        if stats['total_count'] == 0:
            return
        
        accuracy = stats.get('accuracy_rate', 0)
        avg_score = stats.get('average_score', 0)
        recent_activity = stats.get('recent_activity_count', 0)
        
        stats_text = []
        stats_text.append(f"总记录: {stats['total_count']}")
        stats_text.append(f"正确率: {accuracy:.1f}%")
        stats_text.append(f"平均分: {avg_score:.1f}")
        stats_text.append(f"近7天: {recent_activity}条")
        
        console.print(Panel(
            " | ".join(stats_text),
            title=f"[bold cyan]{kb_name} 统计信息[/bold cyan]",
            border_style="cyan",
            padding=(0, 1)
        ))
        console.print()
    
    def _display_history_table(self, kb_name: str, records: List, pagination, page: int):
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
                f"[blue]详情[/blue]"
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
            header = f"记录 #{record.id} - {record.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            console.print(f"[bold cyan]{header}[/bold cyan]")
            console.print()
        
        # 问题
        console.print(Panel(
            record.question,
            title="[bold blue]问题[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # 用户答案
        console.print(Panel(
            record.user_answer,
            title="[bold green]您的答案[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # 评估结果
        result_color = "green" if record.evaluation.is_correct else "red"
        result_text = "正确" if record.evaluation.is_correct else "错误"
        
        eval_content = []
        eval_content.append(f"[bold]结果:[/bold] [{result_color}]{result_text}[/{result_color}]")
        eval_content.append(f"[bold]分数:[/bold] {record.evaluation.score:.1f}/100")
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
        
        console.print(Panel(
            "\n".join(eval_content),
            title="[bold yellow]评估结果[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        ))
        
        # 参考答案
        console.print(Panel(
            record.evaluation.reference_answer,
            title="[bold magenta]参考答案[/bold magenta]",
            border_style="magenta",
            padding=(1, 2)
        ))
    
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
        console.print(f"[dim]  • 查看详情: knowledge {kb_name} review history --detailed[/dim]")
        console.print(f"[dim]  • 过滤记录: knowledge {kb_name} review history --correct/--incorrect[/dim]")
        console.print(f"[dim]  • 搜索问题: knowledge {kb_name} review history --search '关键词'[/dim]")
        console.print(f"[dim]  • 导出记录: knowledge {kb_name} review export[/dim]")
    
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
                kb.created_at.strftime("%Y-%m-%d %H:%M")
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
            console.print(f"  - 创建时间: {kb.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            confirm = console.input("\n[red]此操作不可恢复，确认删除吗? (y/N):[/red] ")
            if confirm.lower() not in ['y', 'yes', '是']:
                console.print("[yellow]删除已取消[/yellow]")
                return
        
        # 删除知识库
        success = show_progress(
            f"删除知识库 '{name}'...",
            self.kb_manager.delete_knowledge_base,
            name
        )
        
        if success:
            console.print(f"[green]✓[/green] 知识库 '{name}' 删除成功")
        else:
            console.print(f"[yellow]⚠[/yellow] 知识库 '{name}' 部分删除成功")
    
    def show_system_status(self):
        """显示系统状态"""
        health_info = show_progress(
            "检查系统状态...",
            self.kb_manager.health_check
        )
        
        # 显示整体状态
        status_color = "green" if health_info["status"] == "healthy" else "red"
        status_text = "正常" if health_info["status"] == "healthy" else "异常"
        
        console.print(Panel(
            f"系统状态: [{status_color}]{status_text}[/{status_color}]",
            title="[bold]系统健康检查[/bold]",
            border_style=status_color
        ))
        
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
                details.append(f"知识库数量: {component_info.get('knowledge_base_count', 0)}")
            elif component_name == "document_processor":
                formats = component_info.get('supported_formats', [])
                details.append(f"支持格式: {', '.join(formats)}")
            elif component_name == "vector_store":
                if 'embedding_model' in component_info:
                    details.append(f"嵌入模型: {component_info['embedding_model']}")
            
            error = component_info.get("error")
            if error:
                details.append(f"错误: {error}")
            
            table.add_row(
                component_name,
                Text(status_text, style=status_color),
                " | ".join(details) if details else "-"
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
        content.append(f"[bold]分数:[/bold] {evaluation.score:.1f}/100")
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
        
        console.print(Panel(
            "\n".join(content),
            title=title,
            border_style=border_style
        ))


# 全局CLI实例
cli_instance = KnowledgeCLI()


# ============================================================================
# Click命令定义
# ============================================================================

@click.group()
@click.version_option(version="0.1.0", prog_name="knowledge")
@click.help_option("--help", "-h")
def main():
    """
    知识库问答系统
    
    基于向量检索和大语言模型的智能学习工具
    """
    pass


@main.command("new", help="创建新的知识库")
@click.option("--name", "-n", required=True, help="知识库名称")
@click.option("--file", "-f", "files", multiple=True, required=True, help="文档文件路径 (可多次使用)")
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


@main.group()
@click.argument("kb_name")
@click.pass_context
def review(ctx, kb_name: str):
    """知识库问答功能"""
    ctx.ensure_object(dict)
    ctx.obj['kb_name'] = kb_name


@review.command("new")
@click.pass_context
@handle_error
def start_new_review(ctx):
    """开始新的问答会话"""
    kb_name = ctx.obj['kb_name']
    cli_instance.start_new_review(kb_name)


@review.command("history")
@click.option("--limit", "-l", default=10, help="显示记录数量 (默认: 10)")
@click.option("--page", "-p", default=1, help="页码 (默认: 1)")
@click.option("--correct", "filter_correct", flag_value=True, help="只显示正确的记录")
@click.option("--incorrect", "filter_correct", flag_value=False, help="只显示错误的记录")
@click.option("--min-score", type=float, help="最低分数过滤")
@click.option("--max-score", type=float, help="最高分数过滤")
@click.option("--search", "-s", help="搜索问题内容")
@click.option("--sort-by", type=click.Choice(['time', 'score', 'result']), default='time', help="排序字段 (默认: time)")
@click.option("--sort-order", type=click.Choice(['asc', 'desc']), default='desc', help="排序顺序 (默认: desc)")
@click.option("--detailed", "-d", is_flag=True, help="显示详细信息")
@click.pass_context
@handle_error
def show_history(ctx, limit: int, page: int, filter_correct: Optional[bool], min_score: Optional[float], 
                max_score: Optional[float], search: Optional[str], sort_by: str, sort_order: str, detailed: bool):
    """查看问答历史记录"""
    kb_name = ctx.obj['kb_name']
    cli_instance.show_history(
        kb_name, limit, page, filter_correct, min_score, max_score, 
        search, sort_by, sort_order, detailed
    )


@review.command("detail")
@click.argument("record_id", type=int)
@click.pass_context
@handle_error
def show_history_detail(ctx, record_id: int):
    """查看单个历史记录的详细信息"""
    kb_name = ctx.obj['kb_name']
    cli_instance.show_history_detail(kb_name, record_id)


@review.command("export")
@click.option("--format", "-f", type=click.Choice(['json', 'csv']), default='json', help="导出格式 (默认: json)")
@click.option("--output", "-o", help="输出文件路径")
@click.pass_context
@handle_error
def export_history(ctx, format: str, output: Optional[str]):
    """导出历史记录"""
    kb_name = ctx.obj['kb_name']
    cli_instance.export_history(kb_name, format, output)


if __name__ == "__main__":
    main()