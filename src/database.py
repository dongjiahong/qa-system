"""
Database layer for Knowledge QA System
数据库层实现，使用 SQLite 存储知识库元数据和问答历史记录
"""

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

from .models import (
    KnowledgeBase, QARecord, EvaluationResult, EvaluationStatus,
    DatabaseError, ValidationError, KnowledgeSystemError
)

logger = logging.getLogger(__name__)


class SQLiteDatabase:
    """SQLite 数据库管理器"""
    
    def __init__(self, db_path: str = "data/knowledge_qa.db"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._initialize_database()
        logger.info(f"数据库初始化完成: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = None
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # 启用字典式访问
            conn.execute("PRAGMA foreign_keys = ON")  # 启用外键约束
            yield conn
        except (ValidationError, KnowledgeSystemError):
            # Re-raise our custom exceptions without wrapping
            if conn:
                conn.rollback()
            raise
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"数据库操作错误: {e}")
            raise DatabaseError(f"数据库操作失败: {e}")
        finally:
            if conn:
                conn.close()
    
    def _initialize_database(self) -> None:
        """初始化数据库表结构"""
        try:
            with self.get_connection() as conn:
                # 创建知识库元数据表
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_bases (
                        name TEXT PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL,
                        file_count INTEGER DEFAULT 0,
                        document_count INTEGER DEFAULT 0,
                        description TEXT
                    )
                """)
                
                # 创建问答历史记录表
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS qa_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        kb_name TEXT NOT NULL,
                        question TEXT NOT NULL,
                        user_answer TEXT NOT NULL,
                        is_correct BOOLEAN NOT NULL,
                        score REAL NOT NULL,
                        feedback TEXT NOT NULL,
                        reference_answer TEXT NOT NULL,
                        missing_points TEXT,  -- JSON array
                        strengths TEXT,       -- JSON array
                        evaluation_status TEXT DEFAULT 'success',
                        created_at TIMESTAMP NOT NULL,
                        FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE
                    )
                """)
                
                # 创建索引以提高查询性能
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_qa_records_kb_name 
                    ON qa_records(kb_name)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_qa_records_created_at 
                    ON qa_records(created_at DESC)
                """)
                
                conn.commit()
                logger.info("数据库表结构初始化完成")
                
        except sqlite3.Error as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(f"数据库初始化失败: {e}")
    
    def migrate_database(self) -> None:
        """数据库迁移脚本"""
        try:
            with self.get_connection() as conn:
                # 检查当前数据库版本
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='schema_version'
                """)
                
                if not cursor.fetchone():
                    # 创建版本表
                    conn.execute("""
                        CREATE TABLE schema_version (
                            version INTEGER PRIMARY KEY,
                            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
                    logger.info("创建数据库版本表")
                
                # 获取当前版本
                cursor = conn.execute("SELECT MAX(version) FROM schema_version")
                current_version = cursor.fetchone()[0] or 0
                
                # 执行迁移
                if current_version < 2:
                    self._migrate_to_v2(conn)
                    conn.execute("INSERT INTO schema_version (version) VALUES (2)")
                    logger.info("数据库迁移到版本 2")
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"数据库迁移失败: {e}")
            raise DatabaseError(f"数据库迁移失败: {e}")
    
    def _migrate_to_v2(self, conn: sqlite3.Connection) -> None:
        """迁移到版本 2 - 添加评估状态字段"""
        try:
            # 检查是否已存在 evaluation_status 列
            cursor = conn.execute("PRAGMA table_info(qa_records)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'evaluation_status' not in columns:
                conn.execute("""
                    ALTER TABLE qa_records 
                    ADD COLUMN evaluation_status TEXT DEFAULT 'success'
                """)
                logger.info("添加 evaluation_status 列")
                
        except sqlite3.Error as e:
            logger.error(f"版本 2 迁移失败: {e}")
            raise


class KnowledgeBaseRepository:
    """知识库元数据仓库"""
    
    def __init__(self, db: SQLiteDatabase):
        self.db = db
    
    def create(self, knowledge_base: KnowledgeBase) -> None:
        """
        创建知识库记录
        
        Args:
            knowledge_base: 知识库对象
            
        Raises:
            DatabaseError: 数据库操作失败
            ValidationError: 数据验证失败
        """
        try:
            knowledge_base.validate()
            
            with self.db.get_connection() as conn:
                try:
                    conn.execute("""
                        INSERT INTO knowledge_bases 
                        (name, created_at, file_count, document_count, description)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        knowledge_base.name,
                        knowledge_base.created_at,
                        knowledge_base.file_count,
                        knowledge_base.document_count,
                        knowledge_base.description
                    ))
                    conn.commit()
                    logger.info(f"创建知识库记录: {knowledge_base.name}")
                except sqlite3.IntegrityError as e:
                    if "UNIQUE constraint failed" in str(e):
                        raise ValidationError(f"知识库 '{knowledge_base.name}' 已存在")
                    raise DatabaseError(f"创建知识库失败: {e}")
                
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            logger.error(f"创建知识库失败: {e}")
            raise DatabaseError(f"创建知识库失败: {e}")
    
    def get_by_name(self, name: str) -> Optional[KnowledgeBase]:
        """
        根据名称获取知识库
        
        Args:
            name: 知识库名称
            
        Returns:
            知识库对象或 None
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT name, created_at, file_count, document_count, description
                    FROM knowledge_bases WHERE name = ?
                """, (name,))
                
                row = cursor.fetchone()
                if row:
                    return KnowledgeBase(
                        name=row['name'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        file_count=row['file_count'],
                        document_count=row['document_count'],
                        description=row['description']
                    )
                return None
                
        except sqlite3.Error as e:
            logger.error(f"获取知识库失败: {e}")
            raise DatabaseError(f"获取知识库失败: {e}")
    
    def get_all(self) -> List[KnowledgeBase]:
        """
        获取所有知识库
        
        Returns:
            知识库列表
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT name, created_at, file_count, document_count, description
                    FROM knowledge_bases ORDER BY created_at DESC
                """)
                
                knowledge_bases = []
                for row in cursor.fetchall():
                    kb = KnowledgeBase(
                        name=row['name'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        file_count=row['file_count'],
                        document_count=row['document_count'],
                        description=row['description']
                    )
                    knowledge_bases.append(kb)
                
                return knowledge_bases
                
        except sqlite3.Error as e:
            logger.error(f"获取知识库列表失败: {e}")
            raise DatabaseError(f"获取知识库列表失败: {e}")
    
    def update(self, knowledge_base: KnowledgeBase) -> bool:
        """
        更新知识库信息
        
        Args:
            knowledge_base: 知识库对象
            
        Returns:
            是否更新成功
        """
        try:
            knowledge_base.validate()
            
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE knowledge_bases 
                    SET file_count = ?, document_count = ?, description = ?
                    WHERE name = ?
                """, (
                    knowledge_base.file_count,
                    knowledge_base.document_count,
                    knowledge_base.description,
                    knowledge_base.name
                ))
                
                conn.commit()
                updated = cursor.rowcount > 0
                if updated:
                    logger.info(f"更新知识库: {knowledge_base.name}")
                return updated
                
        except sqlite3.Error as e:
            logger.error(f"更新知识库失败: {e}")
            raise DatabaseError(f"更新知识库失败: {e}")
    
    def delete(self, name: str) -> bool:
        """
        删除知识库
        
        Args:
            name: 知识库名称
            
        Returns:
            是否删除成功
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM knowledge_bases WHERE name = ?
                """, (name,))
                
                conn.commit()
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info(f"删除知识库: {name}")
                return deleted
                
        except sqlite3.Error as e:
            logger.error(f"删除知识库失败: {e}")
            raise DatabaseError(f"删除知识库失败: {e}")
    
    def exists(self, name: str) -> bool:
        """
        检查知识库是否存在
        
        Args:
            name: 知识库名称
            
        Returns:
            是否存在
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 1 FROM knowledge_bases WHERE name = ? LIMIT 1
                """, (name,))
                
                return cursor.fetchone() is not None
                
        except sqlite3.Error as e:
            logger.error(f"检查知识库存在性失败: {e}")
            raise DatabaseError(f"检查知识库存在性失败: {e}")


class QARecordRepository:
    """问答记录仓库"""
    
    def __init__(self, db: SQLiteDatabase):
        self.db = db
    
    def create(self, qa_record: QARecord) -> int:
        """
        创建问答记录
        
        Args:
            qa_record: 问答记录对象
            
        Returns:
            记录ID
        """
        try:
            qa_record.validate()
            
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO qa_records 
                    (kb_name, question, user_answer, is_correct, score, feedback, 
                     reference_answer, missing_points, strengths, evaluation_status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    qa_record.kb_name,
                    qa_record.question,
                    qa_record.user_answer,
                    qa_record.evaluation.is_correct,
                    qa_record.evaluation.score,
                    qa_record.evaluation.feedback,
                    qa_record.evaluation.reference_answer,
                    json.dumps(qa_record.evaluation.missing_points, ensure_ascii=False),
                    json.dumps(qa_record.evaluation.strengths, ensure_ascii=False),
                    qa_record.evaluation.status.value,
                    qa_record.created_at
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"创建问答记录: {record_id}")
                return record_id
                
        except sqlite3.Error as e:
            logger.error(f"创建问答记录失败: {e}")
            raise DatabaseError(f"创建问答记录失败: {e}")
    
    def get_by_id(self, record_id: int) -> Optional[QARecord]:
        """
        根据ID获取问答记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            问答记录对象或 None
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, kb_name, question, user_answer, is_correct, score, 
                           feedback, reference_answer, missing_points, strengths, 
                           evaluation_status, created_at
                    FROM qa_records WHERE id = ?
                """, (record_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_qa_record(row)
                return None
                
        except sqlite3.Error as e:
            logger.error(f"获取问答记录失败: {e}")
            raise DatabaseError(f"获取问答记录失败: {e}")
    
    def get_by_knowledge_base(
        self, 
        kb_name: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[QARecord]:
        """
        根据知识库获取问答记录
        
        Args:
            kb_name: 知识库名称
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            问答记录列表
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, kb_name, question, user_answer, is_correct, score, 
                           feedback, reference_answer, missing_points, strengths, 
                           evaluation_status, created_at
                    FROM qa_records 
                    WHERE kb_name = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (kb_name, limit, offset))
                
                records = []
                for row in cursor.fetchall():
                    records.append(self._row_to_qa_record(row))
                
                return records
                
        except sqlite3.Error as e:
            logger.error(f"获取知识库问答记录失败: {e}")
            raise DatabaseError(f"获取知识库问答记录失败: {e}")
    
    def count_by_knowledge_base(self, kb_name: str) -> int:
        """
        统计知识库的问答记录数量
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            记录数量
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM qa_records WHERE kb_name = ?
                """, (kb_name,))
                
                return cursor.fetchone()[0]
                
        except sqlite3.Error as e:
            logger.error(f"统计问答记录失败: {e}")
            raise DatabaseError(f"统计问答记录失败: {e}")
    
    def delete_by_knowledge_base(self, kb_name: str) -> int:
        """
        删除知识库的所有问答记录
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            删除的记录数量
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM qa_records WHERE kb_name = ?
                """, (kb_name,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"删除知识库 {kb_name} 的 {deleted_count} 条问答记录")
                
                return deleted_count
                
        except sqlite3.Error as e:
            logger.error(f"删除问答记录失败: {e}")
            raise DatabaseError(f"删除问答记录失败: {e}")
    
    def delete_by_id(self, record_id: int) -> bool:
        """
        根据ID删除问答记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否删除成功
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM qa_records WHERE id = ?
                """, (record_id,))
                
                conn.commit()
                deleted = cursor.rowcount > 0
                
                if deleted:
                    logger.info(f"删除问答记录: {record_id}")
                
                return deleted
                
        except sqlite3.Error as e:
            logger.error(f"删除问答记录失败: {e}")
            raise DatabaseError(f"删除问答记录失败: {e}")
    
    def get_statistics(self, kb_name: str) -> Dict[str, Any]:
        """
        获取知识库问答统计信息
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            统计信息字典
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_count,
                        SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_count,
                        AVG(score) as avg_score,
                        MAX(created_at) as last_activity
                    FROM qa_records 
                    WHERE kb_name = ?
                """, (kb_name,))
                
                row = cursor.fetchone()
                if row:
                    total_count = row['total_count'] or 0
                    correct_count = row['correct_count'] or 0
                    return {
                        'total_count': total_count,
                        'correct_count': correct_count,
                        'incorrect_count': total_count - correct_count,
                        'accuracy_rate': (correct_count / total_count * 100) if total_count > 0 else 0,
                        'avg_score': round(row['avg_score'], 2) if row['avg_score'] else 0,
                        'last_activity': row['last_activity']
                    }
                
                return {
                    'total_count': 0,
                    'correct_count': 0,
                    'incorrect_count': 0,
                    'accuracy_rate': 0,
                    'avg_score': 0,
                    'last_activity': None
                }
                
        except sqlite3.Error as e:
            logger.error(f"获取统计信息失败: {e}")
            raise DatabaseError(f"获取统计信息失败: {e}")
    
    def _row_to_qa_record(self, row: sqlite3.Row) -> QARecord:
        """将数据库行转换为QARecord对象"""
        try:
            missing_points = json.loads(row['missing_points']) if row['missing_points'] else []
            strengths = json.loads(row['strengths']) if row['strengths'] else []
            
            evaluation = EvaluationResult(
                is_correct=bool(row['is_correct']),
                score=row['score'],
                feedback=row['feedback'],
                reference_answer=row['reference_answer'],
                missing_points=missing_points,
                strengths=strengths,
                status=EvaluationStatus(row['evaluation_status'] or 'success')
            )
            
            return QARecord(
                id=row['id'],
                kb_name=row['kb_name'],
                question=row['question'],
                user_answer=row['user_answer'],
                evaluation=evaluation,
                created_at=datetime.fromisoformat(row['created_at'])
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"解析问答记录失败: {e}")
            raise DatabaseError(f"解析问答记录失败: {e}")


# 数据库管理器单例
_db_instance: Optional[SQLiteDatabase] = None


def get_database() -> SQLiteDatabase:
    """获取数据库实例（单例模式）"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SQLiteDatabase()
    return _db_instance


def get_knowledge_base_repository() -> KnowledgeBaseRepository:
    """获取知识库仓库实例"""
    return KnowledgeBaseRepository(get_database())


def get_qa_record_repository() -> QARecordRepository:
    """获取问答记录仓库实例"""
    return QARecordRepository(get_database())