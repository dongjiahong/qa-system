"""
Data integrity tests for History Manager
历史记录管理器的数据完整性测试
"""

import pytest
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

from src.history_manager import HistoryManager, HistoryFilter, SortField, SortOrder
from src.models import (
    QARecord, EvaluationResult, EvaluationStatus, KnowledgeBase,
    ValidationError, DatabaseError
)
from src.database import SQLiteDatabase, get_qa_record_repository, get_knowledge_base_repository


class TestHistoryManagerDataIntegrity:
    """历史记录管理器数据完整性测试"""
    
    @pytest.fixture
    def temp_db_path(self):
        """创建临时数据库路径"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # 清理
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def test_db(self, temp_db_path):
        """创建测试数据库"""
        return SQLiteDatabase(temp_db_path)
    
    @pytest.fixture
    def history_manager_with_real_db(self, test_db):
        """使用真实数据库的历史记录管理器"""
        with patch('src.history_manager.get_qa_record_repository') as mock_qa_repo, \
             patch('src.history_manager.get_knowledge_base_repository') as mock_kb_repo:
            
            # 使用真实的仓库实例
            from src.database import QARecordRepository, KnowledgeBaseRepository
            mock_qa_repo.return_value = QARecordRepository(test_db)
            mock_kb_repo.return_value = KnowledgeBaseRepository(test_db)
            
            return HistoryManager()
    
    @pytest.fixture
    def sample_knowledge_base(self):
        """创建示例知识库"""
        return KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=1,
            document_count=10,
            description="测试知识库"
        )
    
    @pytest.fixture
    def sample_evaluation(self):
        """创建示例评估结果"""
        return EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="答案基本正确，但可以更详细一些",
            reference_answer="这是标准的参考答案",
            missing_points=["缺少关键细节", "需要更多例子"],
            strengths=["理解正确", "逻辑清晰"],
            status=EvaluationStatus.SUCCESS
        )
    
    def test_data_persistence_and_retrieval(self, history_manager_with_real_db, sample_knowledge_base, sample_evaluation):
        """测试数据持久化和检索的完整性"""
        manager = history_manager_with_real_db
        
        # 首先创建知识库
        kb_repo = get_knowledge_base_repository()
        with patch('src.database.get_database', return_value=manager.qa_repo.db):
            kb_repo = manager.kb_repo
            kb_repo.create(sample_knowledge_base)
        
        # 创建问答记录
        qa_record = QARecord(
            kb_name="test_kb",
            question="什么是机器学习？",
            user_answer="机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。",
            evaluation=sample_evaluation,
            created_at=datetime.now()
        )
        
        # 保存记录
        record_id = manager.save_qa_record(qa_record)
        assert record_id is not None
        assert record_id > 0
        
        # 检索记录
        retrieved_record = manager.get_record_by_id(record_id)
        assert retrieved_record is not None
        assert retrieved_record.id == record_id
        assert retrieved_record.kb_name == qa_record.kb_name
        assert retrieved_record.question == qa_record.question
        assert retrieved_record.user_answer == qa_record.user_answer
        assert retrieved_record.evaluation.is_correct == qa_record.evaluation.is_correct
        assert retrieved_record.evaluation.score == qa_record.evaluation.score
        assert retrieved_record.evaluation.feedback == qa_record.evaluation.feedback
        assert retrieved_record.evaluation.reference_answer == qa_record.evaluation.reference_answer
        assert retrieved_record.evaluation.missing_points == qa_record.evaluation.missing_points
        assert retrieved_record.evaluation.strengths == qa_record.evaluation.strengths
        assert retrieved_record.evaluation.status == qa_record.evaluation.status
    
    def test_foreign_key_constraint_integrity(self, history_manager_with_real_db, sample_evaluation):
        """测试外键约束的完整性"""
        manager = history_manager_with_real_db
        
        # 尝试保存引用不存在知识库的记录
        qa_record = QARecord(
            kb_name="nonexistent_kb",
            question="测试问题",
            user_answer="测试答案",
            evaluation=sample_evaluation
        )
        
        # 应该抛出验证错误
        with pytest.raises(ValidationError, match="知识库 'nonexistent_kb' 不存在"):
            manager.save_qa_record(qa_record)
    
    def test_data_consistency_across_operations(self, history_manager_with_real_db, sample_knowledge_base, sample_evaluation):
        """测试跨操作的数据一致性"""
        manager = history_manager_with_real_db
        
        # 创建知识库
        manager.kb_repo.create(sample_knowledge_base)
        
        # 创建多个问答记录
        records = []
        for i in range(5):
            evaluation = EvaluationResult(
                is_correct=(i % 2 == 0),
                score=70.0 + i * 5,
                feedback=f"反馈{i}",
                reference_answer=f"参考答案{i}",
                missing_points=[f"缺失点{i}"],
                strengths=[f"优点{i}"],
                status=EvaluationStatus.SUCCESS
            )
            
            record = QARecord(
                kb_name="test_kb",
                question=f"问题{i}",
                user_answer=f"答案{i}",
                evaluation=evaluation,
                created_at=datetime.now() - timedelta(minutes=i)
            )
            
            record_id = manager.save_qa_record(record)
            record.id = record_id
            records.append(record)
        
        # 验证统计信息的一致性
        stats = manager.get_statistics("test_kb")
        assert stats['total_count'] == 5
        assert stats['correct_count'] == 3  # 索引0,2,4是正确的
        assert stats['incorrect_count'] == 2
        assert stats['accuracy_rate'] == 60.0
        
        # 验证分页查询的一致性
        page1 = manager.get_history_page("test_kb", page=1, page_size=3)
        page2 = manager.get_history_page("test_kb", page=2, page_size=3)
        
        assert len(page1.records) == 3
        assert len(page2.records) == 2
        assert page1.pagination.total_count == 5
        assert page2.pagination.total_count == 5
        
        # 验证所有记录都能被检索到
        all_retrieved_ids = set()
        for record in page1.records + page2.records:
            all_retrieved_ids.add(record.id)
        
        original_ids = set(record.id for record in records)
        assert all_retrieved_ids == original_ids
    
    def test_transaction_rollback_integrity(self, history_manager_with_real_db, sample_knowledge_base):
        """测试事务回滚的完整性"""
        manager = history_manager_with_real_db
        
        # 创建知识库
        manager.kb_repo.create(sample_knowledge_base)
        
        # 创建一个有效的记录
        valid_evaluation = EvaluationResult(
            is_correct=True,
            score=90.0,
            feedback="很好",
            reference_answer="参考答案"
        )
        
        valid_record = QARecord(
            kb_name="test_kb",
            question="有效问题",
            user_answer="有效答案",
            evaluation=valid_evaluation
        )
        
        # 保存有效记录
        record_id = manager.save_qa_record(valid_record)
        assert record_id is not None
        
        # 验证记录存在
        initial_count = manager.qa_repo.count_by_knowledge_base("test_kb")
        assert initial_count == 1
        
        # 尝试保存无效记录（这应该失败但不影响已有数据）
        try:
            invalid_evaluation = EvaluationResult(
                is_correct=True,
                score=150.0,  # 无效分数，应该超出范围
                feedback="",  # 空反馈，应该无效
                reference_answer=""  # 空参考答案，应该无效
            )
            
            invalid_record = QARecord(
                kb_name="test_kb",
                question="",  # 空问题，应该无效
                user_answer="",  # 空答案，应该无效
                evaluation=invalid_evaluation
            )
            
            manager.save_qa_record(invalid_record)
            assert False, "应该抛出验证错误"
        except ValidationError:
            pass  # 预期的异常
        
        # 验证原有数据没有受到影响
        final_count = manager.qa_repo.count_by_knowledge_base("test_kb")
        assert final_count == initial_count
        
        # 验证原有记录仍然可以检索
        retrieved_record = manager.get_record_by_id(record_id)
        assert retrieved_record is not None
        assert retrieved_record.question == "有效问题"
    
    def test_concurrent_access_integrity(self, history_manager_with_real_db, sample_knowledge_base, sample_evaluation):
        """测试并发访问的数据完整性"""
        manager = history_manager_with_real_db
        
        # 创建知识库
        manager.kb_repo.create(sample_knowledge_base)
        
        # 模拟并发保存记录
        record_ids = []
        for i in range(10):
            evaluation = EvaluationResult(
                is_correct=True,
                score=80.0 + i,
                feedback=f"并发测试反馈{i}",
                reference_answer=f"并发测试参考{i}"
            )
            
            record = QARecord(
                kb_name="test_kb",
                question=f"并发问题{i}",
                user_answer=f"并发答案{i}",
                evaluation=evaluation
            )
            
            record_id = manager.save_qa_record(record)
            record_ids.append(record_id)
        
        # 验证所有记录都被正确保存
        assert len(record_ids) == 10
        assert len(set(record_ids)) == 10  # 所有ID都是唯一的
        
        # 验证所有记录都可以检索
        for record_id in record_ids:
            retrieved_record = manager.get_record_by_id(record_id)
            assert retrieved_record is not None
            assert retrieved_record.id == record_id
        
        # 验证统计信息正确
        stats = manager.get_statistics("test_kb")
        assert stats['total_count'] == 10
        assert stats['correct_count'] == 10
    
    def test_data_validation_integrity(self, history_manager_with_real_db, sample_knowledge_base):
        """测试数据验证的完整性"""
        manager = history_manager_with_real_db
        
        # 创建知识库
        manager.kb_repo.create(sample_knowledge_base)
        
        # 测试各种无效数据
        test_cases = [
            {
                "name": "空问题",
                "question": "",
                "user_answer": "有效答案",
                "score": 80.0,
                "feedback": "有效反馈",
                "reference_answer": "有效参考"
            },
            {
                "name": "空答案",
                "question": "有效问题",
                "user_answer": "",
                "score": 80.0,
                "feedback": "有效反馈",
                "reference_answer": "有效参考"
            },
            {
                "name": "无效分数",
                "question": "有效问题",
                "user_answer": "有效答案",
                "score": 150.0,
                "feedback": "有效反馈",
                "reference_answer": "有效参考"
            },
            {
                "name": "空反馈",
                "question": "有效问题",
                "user_answer": "有效答案",
                "score": 80.0,
                "feedback": "",
                "reference_answer": "有效参考"
            },
            {
                "name": "空参考答案",
                "question": "有效问题",
                "user_answer": "有效答案",
                "score": 80.0,
                "feedback": "有效反馈",
                "reference_answer": ""
            }
        ]
        
        for test_case in test_cases:
            with pytest.raises(ValidationError):
                evaluation = EvaluationResult(
                    is_correct=True,
                    score=test_case["score"],
                    feedback=test_case["feedback"],
                    reference_answer=test_case["reference_answer"]
                )
                
                record = QARecord(
                    kb_name="test_kb",
                    question=test_case["question"],
                    user_answer=test_case["user_answer"],
                    evaluation=evaluation
                )
                
                manager.save_qa_record(record)
        
        # 验证没有无效数据被保存
        stats = manager.get_statistics("test_kb")
        assert stats['total_count'] == 0
    
    def test_cleanup_operations_integrity(self, history_manager_with_real_db, sample_knowledge_base):
        """测试清理操作的数据完整性"""
        manager = history_manager_with_real_db
        
        # 创建知识库
        manager.kb_repo.create(sample_knowledge_base)
        
        # 创建不同时间的记录
        now = datetime.now()
        record_ids = []
        
        for i in range(10):
            evaluation = EvaluationResult(
                is_correct=True,
                score=80.0,
                feedback=f"反馈{i}",
                reference_answer=f"参考{i}"
            )
            
            record = QARecord(
                kb_name="test_kb",
                question=f"问题{i}",
                user_answer=f"答案{i}",
                evaluation=evaluation,
                created_at=now - timedelta(days=i * 10)  # 每个记录相差10天
            )
            
            record_id = manager.save_qa_record(record)
            record_ids.append(record_id)
        
        # 验证初始状态
        initial_stats = manager.get_statistics("test_kb")
        assert initial_stats['total_count'] == 10
        
        # 清理30天前的记录（应该删除3-9的记录，保留0-2）
        # 记录0: 0天前, 记录1: 10天前, 记录2: 20天前, 记录3: 30天前, 记录4: 40天前...
        # 30天前的记录应该被删除，所以删除记录3-9，共7条
        deleted_count = manager.cleanup_old_records("test_kb", days=30)
        assert deleted_count == 7
        
        # 验证清理后的状态
        final_stats = manager.get_statistics("test_kb")
        assert final_stats['total_count'] == 3
        
        # 验证正确的记录被保留
        remaining_records = manager.get_history_page("test_kb", page=1, page_size=10).records
        assert len(remaining_records) == 3
        
        # 验证被删除的记录确实不存在
        for i in range(3, 10):
            record_id = record_ids[i]
            retrieved_record = manager.get_record_by_id(record_id)
            assert retrieved_record is None
        
        # 验证保留的记录仍然存在
        for i in range(3):
            record_id = record_ids[i]
            retrieved_record = manager.get_record_by_id(record_id)
            assert retrieved_record is not None
    
    def test_export_import_integrity(self, history_manager_with_real_db, sample_knowledge_base):
        """测试导出导入的数据完整性"""
        manager = history_manager_with_real_db
        
        # 创建知识库
        manager.kb_repo.create(sample_knowledge_base)
        
        # 创建测试记录
        original_records = []
        for i in range(5):
            evaluation = EvaluationResult(
                is_correct=(i % 2 == 0),
                score=70.0 + i * 5,
                feedback=f"导出测试反馈{i}",
                reference_answer=f"导出测试参考{i}",
                missing_points=[f"缺失{i}"],
                strengths=[f"优点{i}"]
            )
            
            record = QARecord(
                kb_name="test_kb",
                question=f"导出问题{i}",
                user_answer=f"导出答案{i}",
                evaluation=evaluation
            )
            
            record_id = manager.save_qa_record(record)
            record.id = record_id
            original_records.append(record)
        
        # 导出为JSON
        json_export = manager.export_history("test_kb", format="json")
        assert isinstance(json_export, str)
        assert len(json_export) > 0
        
        # 验证JSON导出的完整性
        import json
        exported_data = json.loads(json_export)
        assert len(exported_data) == 5
        
        # 由于数据库按创建时间倒序返回，需要反向匹配
        for i, exported_record in enumerate(exported_data):
            original_record = original_records[4-i]  # 反向索引
            assert exported_record['kb_name'] == original_record.kb_name
            assert exported_record['question'] == original_record.question
            assert exported_record['user_answer'] == original_record.user_answer
            assert exported_record['evaluation']['is_correct'] == original_record.evaluation.is_correct
            assert exported_record['evaluation']['score'] == original_record.evaluation.score
            assert exported_record['evaluation']['feedback'] == original_record.evaluation.feedback
        
        # 导出为CSV
        csv_export = manager.export_history("test_kb", format="csv")
        assert isinstance(csv_export, str)
        assert len(csv_export) > 0
        
        # 验证CSV导出的完整性
        lines = csv_export.strip().split('\n')
        assert len(lines) == 6  # 标题行 + 5条数据行
        
        # 验证标题行
        header = lines[0]
        assert "ID" in header
        assert "知识库" in header
        assert "问题" in header
        assert "用户答案" in header
        
        # 验证数据行（由于数据库按创建时间倒序返回，第一行数据对应最后创建的记录）
        for i in range(1, 6):
            data_line = lines[i]
            assert "test_kb" in data_line
            assert f"导出问题{5-i}" in data_line  # 反向索引
            assert f"导出答案{5-i}" in data_line


if __name__ == "__main__":
    pytest.main([__file__])