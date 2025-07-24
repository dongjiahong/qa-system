#!/usr/bin/env python3
"""
Database initialization script for Knowledge QA System
数据库初始化脚本
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .database import SQLiteDatabase, get_database
    from .config import Settings
except ImportError:
    from database import SQLiteDatabase, get_database
    from config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def initialize_database(db_path: str = None) -> None:
    """
    初始化数据库
    
    Args:
        db_path: 数据库文件路径，如果为None则使用配置文件中的路径
    """
    try:
        if db_path is None:
            config = Settings()
            db_path = config.db_path
        
        logger.info(f"正在初始化数据库: {db_path}")
        
        # 创建数据库实例
        db = SQLiteDatabase(db_path)
        
        # 运行迁移
        db.migrate_database()
        
        logger.info("数据库初始化完成")
        
        # 验证表结构
        with db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"已创建的表: {', '.join(tables)}")
            
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        sys.exit(1)


def reset_database(db_path: str = None) -> None:
    """
    重置数据库（删除所有数据）
    
    Args:
        db_path: 数据库文件路径
    """
    try:
        if db_path is None:
            config = Settings()
            db_path = config.db_path
        
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()
            logger.info(f"已删除数据库文件: {db_path}")
        
        # 重新初始化
        initialize_database(db_path)
        
    except Exception as e:
        logger.error(f"数据库重置失败: {e}")
        sys.exit(1)


def check_database_status(db_path: str = None) -> None:
    """
    检查数据库状态
    
    Args:
        db_path: 数据库文件路径
    """
    try:
        if db_path is None:
            config = Settings()
            db_path = config.db_path
        
        db_file = Path(db_path)
        if not db_file.exists():
            logger.warning(f"数据库文件不存在: {db_path}")
            return
        
        db = SQLiteDatabase(db_path)
        
        with db.get_connection() as conn:
            # 检查表结构
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"数据库表: {', '.join(tables)}")
            
            # 检查知识库数量
            cursor = conn.execute("SELECT COUNT(*) FROM knowledge_bases")
            kb_count = cursor.fetchone()[0]
            logger.info(f"知识库数量: {kb_count}")
            
            # 检查问答记录数量
            cursor = conn.execute("SELECT COUNT(*) FROM qa_records")
            qa_count = cursor.fetchone()[0]
            logger.info(f"问答记录数量: {qa_count}")
            
            # 检查数据库版本
            try:
                cursor = conn.execute("SELECT MAX(version) FROM schema_version")
                version = cursor.fetchone()[0]
                logger.info(f"数据库版本: {version}")
            except:
                logger.info("数据库版本: 未知（可能是旧版本）")
        
        logger.info("数据库状态检查完成")
        
    except Exception as e:
        logger.error(f"数据库状态检查失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge QA System Database Management")
    parser.add_argument(
        "action", 
        choices=["init", "reset", "status"],
        help="要执行的操作"
    )
    parser.add_argument(
        "--db-path",
        help="数据库文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    if args.action == "init":
        initialize_database(args.db_path)
    elif args.action == "reset":
        confirm = input("确定要重置数据库吗？这将删除所有数据 (y/N): ")
        if confirm.lower() == 'y':
            reset_database(args.db_path)
        else:
            logger.info("操作已取消")
    elif args.action == "status":
        check_database_status(args.db_path)


if __name__ == "__main__":
    main()