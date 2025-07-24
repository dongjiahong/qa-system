"""
Unit tests for History Manager
历史记录管理器的单元测试
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.history_manager import (
    HistoryManager, SortOrder, SortField, HistoryFilter, 
    PaginationInfo, HistoryPage
)
from src.models import (
    QARecord, EvaluationResult, EvaluationStatus, KnowledgeBase,
    ValidationError, DatabaseError, KnowledgeSystemError
)


class TestPaginationInfo:
    """测试分页信息类"""
    
    def test_pagination_info_calculation(self):
        """测试分页信息计算"""
        # 测试基本分页计算
        pagination = PaginationInfo(page=1, page_size=10, total_count=25)
        assert pagination.total_pages == 3
        assert pagination.has_next is True
        assert pagination.has_prev is False
        
        # 测试中间页
        pagination = PaginationInfo(page=2, page_size=10, total_count=25)
        assert pagination.total_pages == 3
        assert pagination.has_next is True
        assert pagination.has_prev is True
        
        # 测试最后一页
        pagination = PaginationInfo(page=3, page_size=10, total_count=25)
        assert pagination.total_pages == 3
        assert pagination.has_next is False
        assert pagination.has_prev is True
        
        # 测试空结果
        pagination = PaginationInfo(page=1, page_size=10, total_count=0)
        assert pagination.total_pages == 0
        assert pagination.has_next is False
        assert pagination.has_prev is False
    
    def test_pagination_info_edge_cases(self):
        """测试分页信息边界情况"""
        # 测试总数刚好整除页大小
        pagination = PaginationInfo(page=1, page_size=10, total_count=20)
        assert pagination.total_pages == 2
        
        # 测试页大小为0的情况
        pagination = PaginationInfo(page=1, page_size=0, total_count=10)
        assert pagination.total_pages == 0


class TestHistoryManager:
    """测试历史记录管理器"""
    
    @pytest.fixture
    def mock_qa_repo(self):
        """模拟问答记录仓库"""
        return Mock()
    
    @pytest.fixture
    def mock_kb_repo(self):
        """模拟知识库仓库"""
        return Mock()
    
    @pytest.fixture
    def history_manager(self, mock_qa_repo, mock_kb_repo):
        """创建历史记录管理器实例"""
        with patch('src.history_manager.get_qa_record_repository', return_value=mock_qa_repo), \
             patch('src.history_manager.get_knowledge_base_repository', return_value=mock_kb_repo):
            return HistoryManager()
    
    @pytest.fixture
    def sample_evaluation(self):
        """创建示例评估结果"""
        return EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="答案基本正确，但可以更详细",
            reference_answer="这是参考答案",
            missing_points=["缺少细节1"],
            strengths=["理解正确"],
            status=EvaluationStatus.SUCCESS
        )
    
    @pytest.fixture
    def sample_qa_record(self, sample_evaluation):
        """创建示例问答记录"""
        return QARecord(
            id=1,
            kb_name="test_kb",
            question="这是一个测试问题？",
            user_answer="这是用户的答案",
            evaluation=sample_evaluation,
            created_at=datetime.now()
        )
    
    def test_save_qa_record_success(self, history_manager, mock_qa_repo, mock_kb_repo, sample_qa_record):
        """测试成功保存问答记录"""
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.create.return_value = 123
        
        # 执行测试
        record_id = history_manager.save_qa_record(sample_qa_record)
        
        # 验证结果
        assert record_id == 123
        mock_kb_repo.exists.assert_called_once_with("test_kb")
        mock_qa_repo.create.assert_called_once_with(sample_qa_record)
    
    def test_save_qa_record_kb_not_exists(self, history_manager, mock_kb_repo, sample_qa_record):
        """测试保存记录时知识库不存在"""
        # 设置模拟
        mock_kb_repo.exists.return_value = False
        
        # 执行测试并验证异常
        with pytest.raises(ValidationError, match="知识库 'test_kb' 不存在"):
            history_manager.save_qa_record(sample_qa_record)
    
    def test_save_qa_record_database_error(self, history_manager, mock_qa_repo, mock_kb_repo, sample_qa_record):
        """测试保存记录时数据库错误"""
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.create.side_effect = DatabaseError("数据库连接失败")
        
        # 执行测试并验证异常
        with pytest.raises(DatabaseError):
            history_manager.save_qa_record(sample_qa_record)
    
    def test_get_record_by_id_success(self, history_manager, mock_qa_repo, sample_qa_record):
        """测试成功根据ID获取记录"""
        # 设置模拟
        mock_qa_repo.get_by_id.return_value = sample_qa_record
        
        # 执行测试
        result = history_manager.get_record_by_id(1)
        
        # 验证结果
        assert result == sample_qa_record
        mock_qa_repo.get_by_id.assert_called_once_with(1)
    
    def test_get_record_by_id_not_found(self, history_manager, mock_qa_repo):
        """测试根据ID获取记录时记录不存在"""
        # 设置模拟
        mock_qa_repo.get_by_id.return_value = None
        
        # 执行测试
        result = history_manager.get_record_by_id(999)
        
        # 验证结果
        assert result is None
    
    def test_get_history_page_success(self, history_manager, mock_qa_repo, mock_kb_repo, sample_qa_record):
        """测试成功获取历史记录分页"""
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.count_by_knowledge_base.return_value = 25
        mock_qa_repo.get_by_knowledge_base.return_value = [sample_qa_record] * 10
        
        # 执行测试
        result = history_manager.get_history_page("test_kb", page=1, page_size=10)
        
        # 验证结果
        assert isinstance(result, HistoryPage)
        assert len(result.records) == 10
        assert result.pagination.total_count == 25
        assert result.pagination.page == 1
        assert result.pagination.page_size == 10
        assert result.pagination.total_pages == 3
        
        # 验证调用
        mock_kb_repo.exists.assert_called_once_with("test_kb")
        mock_qa_repo.count_by_knowledge_base.assert_called_once_with("test_kb")
        mock_qa_repo.get_by_knowledge_base.assert_called_once_with("test_kb", 10, 0)
    
    def test_get_history_page_invalid_params(self, history_manager):
        """测试获取历史记录分页时参数无效"""
        # 测试页码小于1
        with pytest.raises(ValidationError, match="页码必须大于0"):
            history_manager.get_history_page("test_kb", page=0)
        
        # 测试页大小无效
        with pytest.raises(ValidationError, match="每页大小必须在1-100之间"):
            history_manager.get_history_page("test_kb", page_size=0)
        
        with pytest.raises(ValidationError, match="每页大小必须在1-100之间"):
            history_manager.get_history_page("test_kb", page_size=101)
    
    def test_get_history_page_kb_not_exists(self, history_manager, mock_kb_repo):
        """测试获取历史记录分页时知识库不存在"""
        # 设置模拟
        mock_kb_repo.exists.return_value = False
        
        # 执行测试并验证异常
        with pytest.raises(ValidationError, match="知识库 'test_kb' 不存在"):
            history_manager.get_history_page("test_kb")
    
    def test_get_filtered_history(self, history_manager, mock_qa_repo, mock_kb_repo):
        """测试获取过滤后的历史记录"""
        # 创建测试数据
        records = []
        for i in range(5):
            evaluation = EvaluationResult(
                is_correct=(i % 2 == 0),
                score=80.0 + i * 5,
                feedback=f"反馈{i}",
                reference_answer=f"参考答案{i}"
            )
            record = QARecord(
                id=i,
                kb_name="test_kb",
                question=f"问题{i}",
                user_answer=f"答案{i}",
                evaluation=evaluation,
                created_at=datetime.now() - timedelta(days=i)
            )
            records.append(record)
        
        # 设置模拟
        mock_qa_repo.get_by_knowledge_base.return_value = records
        
        # 创建过滤条件
        filter_criteria = HistoryFilter(
            kb_name="test_kb",
            is_correct=True,
            min_score=85.0
        )
        
        # 执行测试
        result = history_manager.get_filtered_history(filter_criteria, page=1, page_size=10)
        
        # 验证结果 - 应该只有分数>=85且正确的记录
        assert len(result.records) == 2  # 记录2和4满足条件
        for record in result.records:
            assert record.evaluation.is_correct is True
            assert record.evaluation.score >= 85.0
    
    def test_get_recent_history(self, history_manager, mock_qa_repo, mock_kb_repo):
        """测试获取最近的历史记录"""
        # 创建测试数据
        now = datetime.now()
        records = [
            QARecord(
                id=1,
                kb_name="test_kb",
                question="最近问题1",
                user_answer="答案1",
                evaluation=EvaluationResult(
                    is_correct=True,
                    score=90.0,
                    feedback="很好",
                    reference_answer="参考"
                ),
                created_at=now - timedelta(days=1)  # 1天前
            ),
            QARecord(
                id=2,
                kb_name="test_kb",
                question="最近问题2",
                user_answer="答案2",
                evaluation=EvaluationResult(
                    is_correct=True,
                    score=85.0,
                    feedback="不错",
                    reference_answer="参考"
                ),
                created_at=now - timedelta(days=10)  # 10天前，应该被过滤掉
            )
        ]
        
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.get_by_knowledge_base.return_value = records
        
        # 执行测试
        result = history_manager.get_recent_history("test_kb", days=7, limit=50)
        
        # 验证结果 - 只有1天前的记录应该被返回
        assert len(result) == 1
        assert result[0].id == 1
    
    def test_get_recent_history_invalid_params(self, history_manager):
        """测试获取最近历史记录时参数无效"""
        # 测试天数无效
        with pytest.raises(ValidationError, match="天数必须大于0"):
            history_manager.get_recent_history("test_kb", days=0)
        
        # 测试限制数量无效
        with pytest.raises(ValidationError, match="限制数量必须在1-200之间"):
            history_manager.get_recent_history("test_kb", limit=0)
        
        with pytest.raises(ValidationError, match="限制数量必须在1-200之间"):
            history_manager.get_recent_history("test_kb", limit=201)
    
    def test_get_statistics(self, history_manager, mock_qa_repo, mock_kb_repo):
        """测试获取统计信息"""
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.get_statistics.return_value = {
            'total_count': 100,
            'correct_count': 80,
            'incorrect_count': 20,
            'accuracy_rate': 80.0,
            'avg_score': 85.5,
            'last_activity': '2024-01-01T10:00:00'
        }
        
        # 模拟最近记录
        recent_records = [
            QARecord(
                id=i,
                kb_name="test_kb",
                question=f"问题{i}",
                user_answer=f"答案{i}",
                evaluation=EvaluationResult(
                    is_correct=(i % 2 == 0),
                    score=80.0,
                    feedback="反馈",
                    reference_answer="参考"
                ),
                created_at=datetime.now()
            ) for i in range(10)
        ]
        
        with patch.object(history_manager, 'get_recent_history', return_value=recent_records):
            # 执行测试
            result = history_manager.get_statistics("test_kb")
            
            # 验证结果
            assert result['total_count'] == 100
            assert result['correct_count'] == 80
            assert result['accuracy_rate'] == 80.0
            assert result['avg_score'] == 85.5
            assert result['recent_activity_count'] == 10
            assert result['recent_accuracy_rate'] == 50.0  # 10个记录中5个正确
    
    def test_delete_record(self, history_manager, mock_qa_repo):
        """测试删除单个记录"""
        # 设置模拟
        mock_qa_repo.delete_by_id.return_value = True
        
        # 执行测试
        result = history_manager.delete_record(123)
        
        # 验证结果
        assert result is True
        mock_qa_repo.delete_by_id.assert_called_once_with(123)
    
    def test_delete_knowledge_base_history(self, history_manager, mock_qa_repo, mock_kb_repo):
        """测试删除知识库历史记录"""
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.delete_by_knowledge_base.return_value = 50
        
        # 执行测试
        result = history_manager.delete_knowledge_base_history("test_kb")
        
        # 验证结果
        assert result == 50
        mock_kb_repo.exists.assert_called_once_with("test_kb")
        mock_qa_repo.delete_by_knowledge_base.assert_called_once_with("test_kb")
    
    def test_cleanup_old_records(self, history_manager, mock_qa_repo, mock_kb_repo):
        """测试清理旧记录"""
        # 创建测试数据
        now = datetime.now()
        old_record = QARecord(
            id=1,
            kb_name="test_kb",
            question="旧问题",
            user_answer="旧答案",
            evaluation=EvaluationResult(
                is_correct=True,
                score=80.0,
                feedback="反馈",
                reference_answer="参考"
            ),
            created_at=now - timedelta(days=100)  # 100天前
        )
        
        new_record = QARecord(
            id=2,
            kb_name="test_kb",
            question="新问题",
            user_answer="新答案",
            evaluation=EvaluationResult(
                is_correct=True,
                score=85.0,
                feedback="反馈",
                reference_answer="参考"
            ),
            created_at=now - timedelta(days=30)  # 30天前
        )
        
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.get_by_knowledge_base.return_value = [old_record, new_record]
        mock_qa_repo.delete_by_id.return_value = True
        
        # 执行测试 - 保留90天内的记录
        result = history_manager.cleanup_old_records("test_kb", days=90)
        
        # 验证结果 - 只有100天前的记录应该被删除
        assert result == 1
        mock_qa_repo.delete_by_id.assert_called_once_with(1)
    
    def test_export_history_json(self, history_manager, mock_qa_repo, mock_kb_repo, sample_qa_record):
        """测试导出历史记录为JSON格式"""
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.get_by_knowledge_base.return_value = [sample_qa_record]
        
        # 执行测试
        result = history_manager.export_history("test_kb", format="json")
        
        # 验证结果
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['kb_name'] == "test_kb"
        assert data[0]['question'] == "这是一个测试问题？"
    
    def test_export_history_csv(self, history_manager, mock_qa_repo, mock_kb_repo, sample_qa_record):
        """测试导出历史记录为CSV格式"""
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        mock_qa_repo.get_by_knowledge_base.return_value = [sample_qa_record]
        
        # 执行测试
        result = history_manager.export_history("test_kb", format="csv")
        
        # 验证结果
        assert isinstance(result, str)
        lines = result.strip().split('\n')
        assert len(lines) == 2  # 标题行 + 数据行
        assert "ID" in lines[0]  # 检查标题行
        assert "test_kb" in lines[1]  # 检查数据行
    
    def test_export_history_unsupported_format(self, history_manager, mock_kb_repo):
        """测试导出不支持的格式"""
        # 设置模拟
        mock_kb_repo.exists.return_value = True
        
        # 执行测试并验证异常
        with pytest.raises(ValidationError, match="不支持的导出格式"):
            history_manager.export_history("test_kb", format="xml")
    
    def test_sort_records(self, history_manager):
        """测试记录排序功能"""
        # 创建测试数据
        now = datetime.now()
        records = []
        for i in range(3):
            evaluation = EvaluationResult(
                is_correct=(i == 1),  # 只有中间的记录是正确的
                score=70.0 + i * 10,  # 分数递增
                feedback=f"反馈{i}",
                reference_answer=f"参考{i}"
            )
            record = QARecord(
                id=i,
                kb_name="test_kb",
                question=f"问题{i}",
                user_answer=f"答案{i}",
                evaluation=evaluation,
                created_at=now - timedelta(hours=i)  # 时间递减
            )
            records.append(record)
        
        # 测试按分数降序排序
        sorted_records = history_manager._sort_records(records, SortField.SCORE, SortOrder.DESC)
        assert sorted_records[0].evaluation.score == 90.0
        assert sorted_records[1].evaluation.score == 80.0
        assert sorted_records[2].evaluation.score == 70.0
        
        # 测试按分数升序排序
        sorted_records = history_manager._sort_records(records, SortField.SCORE, SortOrder.ASC)
        assert sorted_records[0].evaluation.score == 70.0
        assert sorted_records[1].evaluation.score == 80.0
        assert sorted_records[2].evaluation.score == 90.0
        
        # 测试按正确性排序
        sorted_records = history_manager._sort_records(records, SortField.IS_CORRECT, SortOrder.DESC)
        assert sorted_records[0].evaluation.is_correct is True
        assert sorted_records[1].evaluation.is_correct is False
        assert sorted_records[2].evaluation.is_correct is False
    
    def test_apply_filters(self, history_manager):
        """测试过滤条件应用"""
        # 创建测试数据
        now = datetime.now()
        records = []
        for i in range(5):
            evaluation = EvaluationResult(
                is_correct=(i % 2 == 0),
                score=70.0 + i * 5,
                feedback=f"反馈{i}",
                reference_answer=f"参考{i}"
            )
            record = QARecord(
                id=i,
                kb_name="test_kb",
                question=f"测试问题{i}",
                user_answer=f"答案{i}",
                evaluation=evaluation,
                created_at=now - timedelta(days=i)
            )
            records.append(record)
        
        # 测试正确性过滤
        filter_criteria = HistoryFilter(is_correct=True)
        filtered = history_manager._apply_filters(records, filter_criteria)
        assert len(filtered) == 3  # 索引0,2,4的记录是正确的
        
        # 测试分数范围过滤
        filter_criteria = HistoryFilter(min_score=80.0, max_score=85.0)
        filtered = history_manager._apply_filters(records, filter_criteria)
        assert len(filtered) == 2  # 分数80和85的记录
        
        # 测试日期范围过滤
        filter_criteria = HistoryFilter(
            start_date=now - timedelta(days=2),
            end_date=now
        )
        filtered = history_manager._apply_filters(records, filter_criteria)
        assert len(filtered) == 3  # 0,1,2天前的记录
        
        # 测试问题内容过滤
        filter_criteria = HistoryFilter(question_contains="问题1")
        filtered = history_manager._apply_filters(records, filter_criteria)
        assert len(filtered) == 1
        assert "问题1" in filtered[0].question
        
        # 测试组合过滤
        filter_criteria = HistoryFilter(
            is_correct=True,
            min_score=80.0
        )
        filtered = history_manager._apply_filters(records, filter_criteria)
        assert len(filtered) == 2  # 正确且分数>=80的记录


class TestHistoryManagerIntegration:
    """历史记录管理器集成测试"""
    
    @pytest.fixture
    def temp_db(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # 清理
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_full_workflow(self, temp_db):
        """测试完整的工作流程"""
        # 这个测试需要真实的数据库连接
        # 由于当前的设计使用单例模式，这里只做基本的接口测试
        
        with patch('src.history_manager.get_qa_record_repository') as mock_qa_repo, \
             patch('src.history_manager.get_knowledge_base_repository') as mock_kb_repo:
            
            # 设置模拟
            mock_kb_repo.return_value.exists.return_value = True
            mock_qa_repo.return_value.create.return_value = 1
            mock_qa_repo.return_value.get_by_id.return_value = None
            mock_qa_repo.return_value.count_by_knowledge_base.return_value = 0
            mock_qa_repo.return_value.get_by_knowledge_base.return_value = []
            
            # 创建管理器
            manager = HistoryManager()
            
            # 测试基本操作
            evaluation = EvaluationResult(
                is_correct=True,
                score=90.0,
                feedback="很好",
                reference_answer="参考答案"
            )
            
            record = QARecord(
                kb_name="test_kb",
                question="测试问题",
                user_answer="测试答案",
                evaluation=evaluation
            )
            
            # 保存记录
            record_id = manager.save_qa_record(record)
            assert record_id == 1
            
            # 获取分页
            page = manager.get_history_page("test_kb")
            assert isinstance(page, HistoryPage)
            assert len(page.records) == 0
            assert page.pagination.total_count == 0


if __name__ == "__main__":
    pytest.main([__file__])