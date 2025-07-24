"""
Unit tests for database layer
数据库层单元测试
"""

import pytest
import tempfile
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.database import (
    SQLiteDatabase, KnowledgeBaseRepository, QARecordRepository,
    get_database, get_knowledge_base_repository, get_qa_record_repository
)
from src.models import (
    KnowledgeBase, QARecord, EvaluationResult, EvaluationStatus,
    DatabaseError, ValidationError
)


class TestSQLiteDatabase:
    """SQLite数据库测试"""
    
    @pytest.fixture
    def temp_db(self):
        """临时数据库fixture"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = SQLiteDatabase(str(db_path))
            yield db
    
    def test_database_initialization(self, temp_db):
        """测试数据库初始化"""
        # 验证数据库文件存在
        assert temp_db.db_path.exists()
        
        # 验证表结构
        with temp_db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('knowledge_bases', 'qa_records')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            assert 'knowledge_bases' in tables
            assert 'qa_records' in tables
    
    def test_connection_context_manager(self, temp_db):
        """测试连接上下文管理器"""
        with temp_db.get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_connection_error_handling(self, temp_db):
        """测试连接错误处理"""
        with patch('src.database.sqlite3.connect', side_effect=Exception("Connection failed")):
            with pytest.raises(DatabaseError, match="数据库操作失败"):
                with temp_db.get_connection() as conn:
                    pass
    
    def test_database_migration(self, temp_db):
        """测试数据库迁移"""
        temp_db.migrate_database()
        
        # 验证版本表存在
        with temp_db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            """)
            assert cursor.fetchone() is not None
            
            # 验证版本记录
            cursor = conn.execute("SELECT version FROM schema_version ORDER BY version")
            versions = [row[0] for row in cursor.fetchall()]
            assert 1 in versions


class TestKnowledgeBaseRepository:
    """知识库仓库测试"""
    
    @pytest.fixture
    def repo(self):
        """知识库仓库fixture"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = SQLiteDatabase(str(db_path))
            yield KnowledgeBaseRepository(db)
    
    @pytest.fixture
    def sample_kb(self):
        """示例知识库"""
        return KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=2,
            document_count=10,
            description="测试知识库"
        )
    
    def test_create_knowledge_base(self, repo, sample_kb):
        """测试创建知识库"""
        repo.create(sample_kb)
        
        # 验证创建成功
        kb = repo.get_by_name("test_kb")
        assert kb is not None
        assert kb.name == "test_kb"
        assert kb.file_count == 2
        assert kb.document_count == 10
        assert kb.description == "测试知识库"
    
    def test_create_duplicate_knowledge_base(self, repo, sample_kb):
        """测试创建重复知识库"""
        repo.create(sample_kb)
        
        # 尝试创建重复知识库
        with pytest.raises(ValidationError, match="已存在"):
            repo.create(sample_kb)
    
    def test_get_knowledge_base_not_found(self, repo):
        """测试获取不存在的知识库"""
        kb = repo.get_by_name("nonexistent")
        assert kb is None
    
    def test_get_all_knowledge_bases(self, repo):
        """测试获取所有知识库"""
        # 创建多个知识库
        kb1 = KnowledgeBase("kb1", datetime.now(), 1, 5)
        kb2 = KnowledgeBase("kb2", datetime.now(), 2, 10)
        
        repo.create(kb1)
        repo.create(kb2)
        
        # 获取所有知识库
        all_kbs = repo.get_all()
        assert len(all_kbs) == 2
        
        # 验证按创建时间倒序排列
        names = [kb.name for kb in all_kbs]
        assert "kb2" in names
        assert "kb1" in names
    
    def test_update_knowledge_base(self, repo, sample_kb):
        """测试更新知识库"""
        repo.create(sample_kb)
        
        # 更新知识库信息
        sample_kb.file_count = 5
        sample_kb.document_count = 20
        sample_kb.description = "更新后的描述"
        
        result = repo.update(sample_kb)
        assert result is True
        
        # 验证更新成功
        updated_kb = repo.get_by_name("test_kb")
        assert updated_kb.file_count == 5
        assert updated_kb.document_count == 20
        assert updated_kb.description == "更新后的描述"
    
    def test_update_nonexistent_knowledge_base(self, repo, sample_kb):
        """测试更新不存在的知识库"""
        result = repo.update(sample_kb)
        assert result is False
    
    def test_delete_knowledge_base(self, repo, sample_kb):
        """测试删除知识库"""
        repo.create(sample_kb)
        
        # 删除知识库
        result = repo.delete("test_kb")
        assert result is True
        
        # 验证删除成功
        kb = repo.get_by_name("test_kb")
        assert kb is None
    
    def test_delete_nonexistent_knowledge_base(self, repo):
        """测试删除不存在的知识库"""
        result = repo.delete("nonexistent")
        assert result is False
    
    def test_knowledge_base_exists(self, repo, sample_kb):
        """测试检查知识库存在性"""
        assert repo.exists("test_kb") is False
        
        repo.create(sample_kb)
        assert repo.exists("test_kb") is True
    
    def test_invalid_knowledge_base_validation(self, repo):
        """测试无效知识库验证"""
        # 测试在创建时验证失败
        with pytest.raises(ValidationError, match="知识库名称不能为空"):
            invalid_kb = KnowledgeBase(
                name="",  # 空名称
                created_at=datetime.now()
            )


class TestQARecordRepository:
    """问答记录仓库测试"""
    
    @pytest.fixture
    def repo_with_kb(self):
        """带知识库的问答记录仓库fixture"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = SQLiteDatabase(str(db_path))
            
            # 创建知识库
            kb_repo = KnowledgeBaseRepository(db)
            kb = KnowledgeBase("test_kb", datetime.now(), 1, 5)
            kb_repo.create(kb)
            
            qa_repo = QARecordRepository(db)
            yield qa_repo, kb_repo
    
    @pytest.fixture
    def sample_qa_record(self):
        """示例问答记录"""
        evaluation = EvaluationResult(
            is_correct=True,
            score=85.0,
            feedback="回答正确，但可以更详细",
            reference_answer="参考答案内容",
            missing_points=["缺少细节1", "缺少细节2"],
            strengths=["理解正确", "逻辑清晰"],
            status=EvaluationStatus.SUCCESS
        )
        
        return QARecord(
            kb_name="test_kb",
            question="什么是机器学习？",
            user_answer="机器学习是人工智能的一个分支",
            evaluation=evaluation,
            created_at=datetime.now()
        )
    
    def test_create_qa_record(self, repo_with_kb, sample_qa_record):
        """测试创建问答记录"""
        qa_repo, _ = repo_with_kb
        
        record_id = qa_repo.create(sample_qa_record)
        assert record_id > 0
        
        # 验证创建成功
        record = qa_repo.get_by_id(record_id)
        assert record is not None
        assert record.question == "什么是机器学习？"
        assert record.user_answer == "机器学习是人工智能的一个分支"
        assert record.evaluation.is_correct is True
        assert record.evaluation.score == 85.0
    
    def test_get_qa_record_not_found(self, repo_with_kb):
        """测试获取不存在的问答记录"""
        qa_repo, _ = repo_with_kb
        
        record = qa_repo.get_by_id(999)
        assert record is None
    
    def test_get_qa_records_by_knowledge_base(self, repo_with_kb, sample_qa_record):
        """测试根据知识库获取问答记录"""
        qa_repo, _ = repo_with_kb
        
        # 创建多个问答记录
        record1 = sample_qa_record
        record2 = QARecord(
            kb_name="test_kb",
            question="什么是深度学习？",
            user_answer="深度学习是机器学习的子集",
            evaluation=EvaluationResult(
                is_correct=False,
                score=60.0,
                feedback="回答不够准确",
                reference_answer="深度学习参考答案"
            )
        )
        
        qa_repo.create(record1)
        qa_repo.create(record2)
        
        # 获取知识库的问答记录
        records = qa_repo.get_by_knowledge_base("test_kb")
        assert len(records) == 2
        
        # 验证按时间倒序排列
        assert records[0].question == "什么是深度学习？"  # 后创建的在前
        assert records[1].question == "什么是机器学习？"
    
    def test_get_qa_records_with_pagination(self, repo_with_kb, sample_qa_record):
        """测试分页获取问答记录"""
        qa_repo, _ = repo_with_kb
        
        # 创建多个记录
        for i in range(5):
            record = QARecord(
                kb_name="test_kb",
                question=f"问题 {i}",
                user_answer=f"答案 {i}",
                evaluation=EvaluationResult(
                    is_correct=True,
                    score=80.0,
                    feedback="反馈",
                    reference_answer="参考答案"
                )
            )
            qa_repo.create(record)
        
        # 测试分页
        page1 = qa_repo.get_by_knowledge_base("test_kb", limit=2, offset=0)
        page2 = qa_repo.get_by_knowledge_base("test_kb", limit=2, offset=2)
        
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].question != page2[0].question
    
    def test_count_qa_records(self, repo_with_kb, sample_qa_record):
        """测试统计问答记录数量"""
        qa_repo, _ = repo_with_kb
        
        # 初始数量为0
        count = qa_repo.count_by_knowledge_base("test_kb")
        assert count == 0
        
        # 创建记录后数量增加
        qa_repo.create(sample_qa_record)
        count = qa_repo.count_by_knowledge_base("test_kb")
        assert count == 1
    
    def test_delete_qa_records_by_knowledge_base(self, repo_with_kb, sample_qa_record):
        """测试删除知识库的所有问答记录"""
        qa_repo, _ = repo_with_kb
        
        # 创建记录
        qa_repo.create(sample_qa_record)
        qa_repo.create(sample_qa_record)
        
        # 删除所有记录
        deleted_count = qa_repo.delete_by_knowledge_base("test_kb")
        assert deleted_count == 2
        
        # 验证删除成功
        count = qa_repo.count_by_knowledge_base("test_kb")
        assert count == 0
    
    def test_delete_qa_record_by_id(self, repo_with_kb, sample_qa_record):
        """测试根据ID删除问答记录"""
        qa_repo, _ = repo_with_kb
        
        record_id = qa_repo.create(sample_qa_record)
        
        # 删除记录
        result = qa_repo.delete_by_id(record_id)
        assert result is True
        
        # 验证删除成功
        record = qa_repo.get_by_id(record_id)
        assert record is None
    
    def test_delete_nonexistent_qa_record(self, repo_with_kb):
        """测试删除不存在的问答记录"""
        qa_repo, _ = repo_with_kb
        
        result = qa_repo.delete_by_id(999)
        assert result is False
    
    def test_get_qa_statistics(self, repo_with_kb):
        """测试获取问答统计信息"""
        qa_repo, _ = repo_with_kb
        
        # 创建不同类型的记录
        correct_record = QARecord(
            kb_name="test_kb",
            question="问题1",
            user_answer="答案1",
            evaluation=EvaluationResult(
                is_correct=True,
                score=90.0,
                feedback="正确",
                reference_answer="参考答案"
            )
        )
        
        incorrect_record = QARecord(
            kb_name="test_kb",
            question="问题2",
            user_answer="答案2",
            evaluation=EvaluationResult(
                is_correct=False,
                score=60.0,
                feedback="错误",
                reference_answer="参考答案"
            )
        )
        
        qa_repo.create(correct_record)
        qa_repo.create(incorrect_record)
        
        # 获取统计信息
        stats = qa_repo.get_statistics("test_kb")
        
        assert stats['total_count'] == 2
        assert stats['correct_count'] == 1
        assert stats['incorrect_count'] == 1
        assert stats['accuracy_rate'] == 50.0
        assert stats['avg_score'] == 75.0
        assert stats['last_activity'] is not None
    
    def test_get_empty_statistics(self, repo_with_kb):
        """测试获取空知识库的统计信息"""
        qa_repo, _ = repo_with_kb
        
        stats = qa_repo.get_statistics("test_kb")
        
        assert stats['total_count'] == 0
        assert stats['correct_count'] == 0
        assert stats['incorrect_count'] == 0
        assert stats['accuracy_rate'] == 0
        assert stats['avg_score'] == 0
        assert stats['last_activity'] is None
    
    def test_invalid_qa_record_validation(self, repo_with_kb):
        """测试无效问答记录验证"""
        qa_repo, _ = repo_with_kb
        
        # 测试在创建时验证失败
        with pytest.raises(ValidationError, match="知识库名称不能为空"):
            invalid_record = QARecord(
                kb_name="",  # 空知识库名称
                question="问题",
                user_answer="答案",
                evaluation=EvaluationResult(
                    is_correct=True,
                    score=80.0,
                    feedback="反馈",
                    reference_answer="参考答案"
                )
            )
    
    def test_json_serialization_in_database(self, repo_with_kb, sample_qa_record):
        """测试数据库中JSON序列化"""
        qa_repo, _ = repo_with_kb
        
        # 创建包含复杂数据的记录
        sample_qa_record.evaluation.missing_points = ["缺失点1", "缺失点2"]
        sample_qa_record.evaluation.strengths = ["优点1", "优点2"]
        
        record_id = qa_repo.create(sample_qa_record)
        
        # 验证反序列化正确
        record = qa_repo.get_by_id(record_id)
        assert record.evaluation.missing_points == ["缺失点1", "缺失点2"]
        assert record.evaluation.strengths == ["优点1", "优点2"]


class TestDatabaseSingletons:
    """测试数据库单例模式"""
    
    def test_get_database_singleton(self):
        """测试数据库单例"""
        db1 = get_database()
        db2 = get_database()
        assert db1 is db2
    
    def test_get_repository_instances(self):
        """测试仓库实例获取"""
        kb_repo = get_knowledge_base_repository()
        qa_repo = get_qa_record_repository()
        
        assert isinstance(kb_repo, KnowledgeBaseRepository)
        assert isinstance(qa_repo, QARecordRepository)
        
        # 验证使用相同的数据库实例
        assert kb_repo.db is qa_repo.db


class TestDatabaseErrorHandling:
    """测试数据库错误处理"""
    
    @pytest.fixture
    def temp_db(self):
        """临时数据库fixture"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = SQLiteDatabase(str(db_path))
            yield db
    
    def test_database_connection_error(self, temp_db):
        """测试数据库连接错误"""
        with patch.object(temp_db, 'get_connection', side_effect=Exception("Connection error")):
            repo = KnowledgeBaseRepository(temp_db)
            
            with pytest.raises(Exception):
                repo.get_all()
    
    def test_sql_execution_error(self, temp_db):
        """测试SQL执行错误"""
        repo = KnowledgeBaseRepository(temp_db)
        
        # 模拟数据库连接错误
        with patch('src.database.sqlite3.connect', side_effect=Exception("Connection error")):
            with pytest.raises(DatabaseError):
                repo.get_all()
    
    def test_json_parsing_error(self, temp_db):
        """测试JSON解析错误"""
        qa_repo = QARecordRepository(temp_db)
        
        # 直接插入无效JSON数据
        with temp_db.get_connection() as conn:
            conn.execute("""
                INSERT INTO knowledge_bases (name, created_at, file_count, document_count)
                VALUES ('test_kb', '2024-01-01T00:00:00', 0, 0)
            """)
            
            conn.execute("""
                INSERT INTO qa_records 
                (kb_name, question, user_answer, is_correct, score, feedback, 
                 reference_answer, missing_points, strengths, evaluation_status, created_at)
                VALUES ('test_kb', '问题', '答案', 1, 80.0, '反馈', '参考答案',
                        'invalid_json', '[]', 'success', '2024-01-01T00:00:00')
            """)
            conn.commit()
        
        with pytest.raises(DatabaseError, match="解析问答记录失败"):
            qa_repo.get_by_knowledge_base("test_kb")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])