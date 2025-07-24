"""
History Manager for Knowledge QA System
历史记录管理器，管理问答历史的存储和检索
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import QARecord, KnowledgeSystemError, DatabaseError, ValidationError
from .database import get_qa_record_repository, get_knowledge_base_repository

logger = logging.getLogger(__name__)


class SortOrder(Enum):
    """排序顺序"""
    ASC = "asc"
    DESC = "desc"


class SortField(Enum):
    """排序字段"""
    CREATED_AT = "created_at"
    SCORE = "score"
    IS_CORRECT = "is_correct"


@dataclass
class HistoryFilter:
    """历史记录过滤条件"""
    kb_name: Optional[str] = None
    is_correct: Optional[bool] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    question_contains: Optional[str] = None


@dataclass
class PaginationInfo:
    """分页信息"""
    page: int = 1
    page_size: int = 20
    total_count: int = 0
    total_pages: int = 0
    has_next: bool = False
    has_prev: bool = False
    
    def __post_init__(self):
        """计算分页信息"""
        if self.page_size > 0:
            self.total_pages = (self.total_count + self.page_size - 1) // self.page_size
            self.has_next = self.page < self.total_pages
            self.has_prev = self.page > 1


@dataclass
class HistoryPage:
    """历史记录分页结果"""
    records: List[QARecord]
    pagination: PaginationInfo


class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self):
        """初始化历史记录管理器"""
        self.qa_repo = get_qa_record_repository()
        self.kb_repo = get_knowledge_base_repository()
        logger.info("历史记录管理器初始化完成")
    
    def save_qa_record(self, qa_record: QARecord) -> int:
        """
        保存问答记录
        
        Args:
            qa_record: 问答记录对象
            
        Returns:
            记录ID
            
        Raises:
            ValidationError: 数据验证失败
            DatabaseError: 数据库操作失败
        """
        try:
            # 验证知识库是否存在
            if not self.kb_repo.exists(qa_record.kb_name):
                raise ValidationError(f"知识库 '{qa_record.kb_name}' 不存在")
            
            # 保存记录
            record_id = self.qa_repo.create(qa_record)
            logger.info(f"保存问答记录成功: {record_id}")
            return record_id
            
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"保存问答记录失败: {e}")
            raise KnowledgeSystemError(f"保存问答记录失败: {e}")
    
    def get_record_by_id(self, record_id: int) -> Optional[QARecord]:
        """
        根据ID获取问答记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            问答记录对象或None
        """
        try:
            return self.qa_repo.get_by_id(record_id)
        except Exception as e:
            logger.error(f"获取问答记录失败: {e}")
            raise KnowledgeSystemError(f"获取问答记录失败: {e}")
    
    def get_history_page(
        self,
        kb_name: str,
        page: int = 1,
        page_size: int = 20,
        sort_field: SortField = SortField.CREATED_AT,
        sort_order: SortOrder = SortOrder.DESC
    ) -> HistoryPage:
        """
        获取分页的历史记录
        
        Args:
            kb_name: 知识库名称
            page: 页码（从1开始）
            page_size: 每页大小
            sort_field: 排序字段
            sort_order: 排序顺序
            
        Returns:
            历史记录分页结果
            
        Raises:
            ValidationError: 参数验证失败
            DatabaseError: 数据库操作失败
        """
        try:
            # 参数验证
            if page < 1:
                raise ValidationError("页码必须大于0")
            if page_size < 1 or page_size > 100:
                raise ValidationError("每页大小必须在1-100之间")
            
            # 验证知识库是否存在
            if not self.kb_repo.exists(kb_name):
                raise ValidationError(f"知识库 '{kb_name}' 不存在")
            
            # 获取总数
            total_count = self.qa_repo.count_by_knowledge_base(kb_name)
            
            # 计算偏移量
            offset = (page - 1) * page_size
            
            # 获取记录（目前数据库层只支持按创建时间排序）
            records = self.qa_repo.get_by_knowledge_base(kb_name, page_size, offset)
            
            # 如果需要其他排序方式，在内存中排序
            if sort_field != SortField.CREATED_AT:
                records = self._sort_records(records, sort_field, sort_order)
            elif sort_order == SortOrder.ASC:
                records = list(reversed(records))
            
            # 创建分页信息
            pagination = PaginationInfo(
                page=page,
                page_size=page_size,
                total_count=total_count
            )
            
            return HistoryPage(records=records, pagination=pagination)
            
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"获取历史记录分页失败: {e}")
            raise KnowledgeSystemError(f"获取历史记录分页失败: {e}")
    
    def get_filtered_history(
        self,
        filter_criteria: HistoryFilter,
        page: int = 1,
        page_size: int = 20,
        sort_field: SortField = SortField.CREATED_AT,
        sort_order: SortOrder = SortOrder.DESC
    ) -> HistoryPage:
        """
        获取过滤后的历史记录
        
        Args:
            filter_criteria: 过滤条件
            page: 页码
            page_size: 每页大小
            sort_field: 排序字段
            sort_order: 排序顺序
            
        Returns:
            过滤后的历史记录分页结果
        """
        try:
            if not filter_criteria.kb_name:
                raise ValidationError("必须指定知识库名称")
            
            # 获取所有记录（简单实现，实际应该在数据库层实现过滤）
            all_records = self.qa_repo.get_by_knowledge_base(
                filter_criteria.kb_name, 
                limit=1000,  # 临时限制
                offset=0
            )
            
            # 应用过滤条件
            filtered_records = self._apply_filters(all_records, filter_criteria)
            
            # 排序
            sorted_records = self._sort_records(filtered_records, sort_field, sort_order)
            
            # 分页
            total_count = len(sorted_records)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_records = sorted_records[start_idx:end_idx]
            
            pagination = PaginationInfo(
                page=page,
                page_size=page_size,
                total_count=total_count
            )
            
            return HistoryPage(records=page_records, pagination=pagination)
            
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"获取过滤历史记录失败: {e}")
            raise KnowledgeSystemError(f"获取过滤历史记录失败: {e}")
    
    def get_recent_history(
        self,
        kb_name: str,
        days: int = 7,
        limit: int = 50
    ) -> List[QARecord]:
        """
        获取最近的历史记录
        
        Args:
            kb_name: 知识库名称
            days: 最近天数
            limit: 限制数量
            
        Returns:
            最近的问答记录列表
        """
        try:
            if days < 1:
                raise ValidationError("天数必须大于0")
            if limit < 1 or limit > 200:
                raise ValidationError("限制数量必须在1-200之间")
            
            # 验证知识库是否存在
            if not self.kb_repo.exists(kb_name):
                raise ValidationError(f"知识库 '{kb_name}' 不存在")
            
            # 获取记录
            records = self.qa_repo.get_by_knowledge_base(kb_name, limit, 0)
            
            # 过滤最近的记录
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_records = [
                record for record in records 
                if record.created_at >= cutoff_date
            ]
            
            return recent_records
            
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"获取最近历史记录失败: {e}")
            raise KnowledgeSystemError(f"获取最近历史记录失败: {e}")
    
    def get_statistics(self, kb_name: str) -> Dict[str, Any]:
        """
        获取知识库问答统计信息
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            统计信息字典
        """
        try:
            # 验证知识库是否存在
            if not self.kb_repo.exists(kb_name):
                raise ValidationError(f"知识库 '{kb_name}' 不存在")
            
            stats = self.qa_repo.get_statistics(kb_name)
            
            # 添加额外的统计信息
            if stats['total_count'] > 0:
                # 获取最近的记录来计算趋势
                recent_records = self.get_recent_history(kb_name, days=7, limit=100)
                stats['recent_activity_count'] = len(recent_records)
                
                if recent_records:
                    recent_correct = sum(1 for r in recent_records if r.evaluation.is_correct)
                    stats['recent_accuracy_rate'] = (recent_correct / len(recent_records)) * 100
                else:
                    stats['recent_accuracy_rate'] = 0
            else:
                stats['recent_activity_count'] = 0
                stats['recent_accuracy_rate'] = 0
            
            return stats
            
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            raise KnowledgeSystemError(f"获取统计信息失败: {e}")
    
    def delete_record(self, record_id: int) -> bool:
        """
        删除单个问答记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否删除成功
        """
        try:
            return self.qa_repo.delete_by_id(record_id)
        except Exception as e:
            logger.error(f"删除问答记录失败: {e}")
            raise KnowledgeSystemError(f"删除问答记录失败: {e}")
    
    def delete_knowledge_base_history(self, kb_name: str) -> int:
        """
        删除知识库的所有历史记录
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            删除的记录数量
        """
        try:
            # 验证知识库是否存在
            if not self.kb_repo.exists(kb_name):
                raise ValidationError(f"知识库 '{kb_name}' 不存在")
            
            deleted_count = self.qa_repo.delete_by_knowledge_base(kb_name)
            logger.info(f"删除知识库 {kb_name} 的 {deleted_count} 条历史记录")
            return deleted_count
            
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"删除知识库历史记录失败: {e}")
            raise KnowledgeSystemError(f"删除知识库历史记录失败: {e}")
    
    def cleanup_old_records(self, kb_name: str, days: int = 90) -> int:
        """
        清理旧的历史记录
        
        Args:
            kb_name: 知识库名称
            days: 保留天数
            
        Returns:
            清理的记录数量
        """
        try:
            if days < 1:
                raise ValidationError("保留天数必须大于0")
            
            # 验证知识库是否存在
            if not self.kb_repo.exists(kb_name):
                raise ValidationError(f"知识库 '{kb_name}' 不存在")
            
            # 获取所有记录
            all_records = self.qa_repo.get_by_knowledge_base(kb_name, limit=10000, offset=0)
            
            # 找出需要删除的旧记录
            cutoff_date = datetime.now() - timedelta(days=days)
            old_records = [
                record for record in all_records 
                if record.created_at < cutoff_date
            ]
            
            # 删除旧记录
            deleted_count = 0
            for record in old_records:
                if self.qa_repo.delete_by_id(record.id):
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"清理知识库 {kb_name} 的 {deleted_count} 条旧记录")
            
            return deleted_count
            
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")
            raise KnowledgeSystemError(f"清理旧记录失败: {e}")
    
    def export_history(self, kb_name: str, format: str = "json") -> str:
        """
        导出历史记录
        
        Args:
            kb_name: 知识库名称
            format: 导出格式 (json, csv)
            
        Returns:
            导出的数据字符串
        """
        try:
            # 验证知识库是否存在
            if not self.kb_repo.exists(kb_name):
                raise ValidationError(f"知识库 '{kb_name}' 不存在")
            
            # 获取所有记录
            records = self.qa_repo.get_by_knowledge_base(kb_name, limit=10000, offset=0)
            
            if format.lower() == "json":
                import json
                data = [record.to_dict() for record in records]
                return json.dumps(data, ensure_ascii=False, indent=2)
            elif format.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # 写入标题行
                writer.writerow([
                    "ID", "知识库", "问题", "用户答案", "是否正确", "分数", 
                    "反馈", "参考答案", "创建时间"
                ])
                
                # 写入数据行
                for record in records:
                    writer.writerow([
                        record.id,
                        record.kb_name,
                        record.question,
                        record.user_answer,
                        "是" if record.evaluation.is_correct else "否",
                        record.evaluation.score,
                        record.evaluation.feedback,
                        record.evaluation.reference_answer,
                        record.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    ])
                
                return output.getvalue()
            else:
                raise ValidationError(f"不支持的导出格式: {format}")
                
        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"导出历史记录失败: {e}")
            raise KnowledgeSystemError(f"导出历史记录失败: {e}")
    
    def _apply_filters(self, records: List[QARecord], filter_criteria: HistoryFilter) -> List[QARecord]:
        """应用过滤条件"""
        filtered_records = records
        
        if filter_criteria.is_correct is not None:
            filtered_records = [
                r for r in filtered_records 
                if r.evaluation.is_correct == filter_criteria.is_correct
            ]
        
        if filter_criteria.min_score is not None:
            filtered_records = [
                r for r in filtered_records 
                if r.evaluation.score >= filter_criteria.min_score
            ]
        
        if filter_criteria.max_score is not None:
            filtered_records = [
                r for r in filtered_records 
                if r.evaluation.score <= filter_criteria.max_score
            ]
        
        if filter_criteria.start_date is not None:
            filtered_records = [
                r for r in filtered_records 
                if r.created_at >= filter_criteria.start_date
            ]
        
        if filter_criteria.end_date is not None:
            filtered_records = [
                r for r in filtered_records 
                if r.created_at <= filter_criteria.end_date
            ]
        
        if filter_criteria.question_contains is not None:
            search_term = filter_criteria.question_contains.lower()
            filtered_records = [
                r for r in filtered_records 
                if search_term in r.question.lower()
            ]
        
        return filtered_records
    
    def _sort_records(
        self, 
        records: List[QARecord], 
        sort_field: SortField, 
        sort_order: SortOrder
    ) -> List[QARecord]:
        """对记录进行排序"""
        reverse = (sort_order == SortOrder.DESC)
        
        if sort_field == SortField.CREATED_AT:
            return sorted(records, key=lambda r: r.created_at, reverse=reverse)
        elif sort_field == SortField.SCORE:
            return sorted(records, key=lambda r: r.evaluation.score, reverse=reverse)
        elif sort_field == SortField.IS_CORRECT:
            return sorted(records, key=lambda r: r.evaluation.is_correct, reverse=reverse)
        else:
            return records