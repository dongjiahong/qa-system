"""
Integration tests for Knowledge Base Manager
知识库管理器集成测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.knowledge_base_manager import KnowledgeBaseManager
from src.models import (
    KnowledgeBase, ValidationError, FileProcessingError, 
    VectorStoreError, DatabaseError, KnowledgeBaseNotFoundError
)


class TestKnowledgeBaseManager:
    """知识库管理器测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """创建示例文件"""
        files = []
        
        # 创建文本文件
        txt_file = Path(temp_dir) / "sample.txt"
        txt_file.write_text("这是一个测试文本文件。\n包含中文内容。", encoding='utf-8')
        files.append(str(txt_file))
        
        # 创建Markdown文件
        md_file = Path(temp_dir) / "sample.md"
        md_file.write_text("# 测试标题\n\n这是一个测试Markdown文件。", encoding='utf-8')
        files.append(str(md_file))
        
        return files
    
    @pytest.fixture
    def kb_manager(self):
        """创建知识库管理器实例"""
        with patch('src.knowledge_base_manager.get_knowledge_base_repository'), \
             patch('src.knowledge_base_manager.get_qa_record_repository'), \
             patch('src.knowledge_base_manager.VectorStore'), \
             patch('src.knowledge_base_manager.DocumentProcessor'):
            
            manager = KnowledgeBaseManager()
            
            # 模拟仓库
            manager.kb_repository = Mock()
            manager.qa_repository = Mock()
            manager.vector_store = Mock()
            manager.document_processor = Mock()
            
            return manager
    
    def test_init(self):
        """测试初始化"""
        with patch('src.knowledge_base_manager.get_knowledge_base_repository'), \
             patch('src.knowledge_base_manager.get_qa_record_repository'), \
             patch('src.knowledge_base_manager.VectorStore'), \
             patch('src.knowledge_base_manager.DocumentProcessor'):
            
            manager = KnowledgeBaseManager()
            
            assert manager.document_processor is not None
            assert manager.vector_store is not None
            assert manager.kb_repository is not None
            assert manager.qa_repository is not None
    
    def test_create_knowledge_base_success(self, kb_manager, sample_files):
        """测试成功创建知识库"""
        # 设置模拟
        kb_manager.kb_repository.exists.return_value = False
        kb_manager.document_processor.process_multiple_files.return_value = [
            Mock(text="文档内容1", metadata={"source": "file1"}),
            Mock(text="文档内容2", metadata={"source": "file2"})
        ]
        kb_manager.document_processor.preprocess_documents.return_value = [
            Mock(text="处理后内容1", metadata={"source": "file1"}),
            Mock(text="处理后内容2", metadata={"source": "file2"})
        ]
        kb_manager.document_processor.chunk_documents.return_value = [
            Mock(text="块1", metadata={"source": "file1", "chunk_id": 0}),
            Mock(text="块2", metadata={"source": "file2", "chunk_id": 1})
        ]
        
        # 执行测试
        kb = kb_manager.create_knowledge_base(
            name="test_kb",
            files=sample_files,
            description="测试知识库"
        )
        
        # 验证结果
        assert isinstance(kb, KnowledgeBase)
        assert kb.name == "test_kb"
        assert kb.description == "测试知识库"
        assert kb.file_count == len(sample_files)
        assert kb.document_count == 2
        
        # 验证调用
        kb_manager.kb_repository.exists.assert_called_once_with("test_kb")
        kb_manager.vector_store.create_collection.assert_called_once_with("test_kb", reset_if_exists=False)
        kb_manager.vector_store.add_documents.assert_called_once()
        kb_manager.kb_repository.create.assert_called_once()
    
    def test_create_knowledge_base_already_exists(self, kb_manager, sample_files):
        """测试创建已存在的知识库"""
        kb_manager.kb_repository.exists.return_value = True
        
        with pytest.raises(ValidationError, match="知识库 'test_kb' 已存在"):
            kb_manager.create_knowledge_base("test_kb", sample_files)
    
    def test_create_knowledge_base_empty_name(self, kb_manager, sample_files):
        """测试创建空名称知识库"""
        with pytest.raises(ValidationError, match="知识库名称不能为空"):
            kb_manager.create_knowledge_base("", sample_files)
    
    def test_create_knowledge_base_no_files(self, kb_manager):
        """测试创建无文件知识库"""
        with pytest.raises(ValidationError, match="必须提供至少一个文件"):
            kb_manager.create_knowledge_base("test_kb", [])
    
    def test_create_knowledge_base_nonexistent_file(self, kb_manager):
        """测试创建包含不存在文件的知识库"""
        with pytest.raises(ValidationError, match="文件不存在"):
            kb_manager.create_knowledge_base("test_kb", ["/nonexistent/file.txt"])
    
    def test_create_knowledge_base_processing_failure(self, kb_manager, sample_files):
        """测试文档处理失败"""
        kb_manager.kb_repository.exists.return_value = False
        kb_manager.document_processor.process_multiple_files.side_effect = FileProcessingError("处理失败")
        
        with pytest.raises(FileProcessingError):
            kb_manager.create_knowledge_base("test_kb", sample_files)
        
        # 验证清理调用
        kb_manager.vector_store.collection_exists.assert_called()
    
    def test_create_knowledge_base_vector_failure(self, kb_manager, sample_files):
        """测试向量存储失败"""
        kb_manager.kb_repository.exists.return_value = False
        kb_manager.document_processor.process_multiple_files.return_value = [Mock()]
        kb_manager.document_processor.preprocess_documents.return_value = [Mock()]
        kb_manager.document_processor.chunk_documents.return_value = [Mock()]
        kb_manager.vector_store.create_collection.side_effect = VectorStoreError("向量存储失败")
        
        with pytest.raises(VectorStoreError):
            kb_manager.create_knowledge_base("test_kb", sample_files)
    
    def test_delete_knowledge_base_success(self, kb_manager):
        """测试成功删除知识库"""
        kb_manager.kb_repository.exists.return_value = True
        kb_manager.qa_repository.delete_by_knowledge_base.return_value = 5
        kb_manager.vector_store.delete_collection.return_value = True
        kb_manager.kb_repository.delete.return_value = True
        
        result = kb_manager.delete_knowledge_base("test_kb")
        
        assert result is True
        kb_manager.qa_repository.delete_by_knowledge_base.assert_called_once_with("test_kb")
        kb_manager.vector_store.delete_collection.assert_called_once_with("test_kb")
        kb_manager.kb_repository.delete.assert_called_once_with("test_kb")
    
    def test_delete_knowledge_base_not_exists(self, kb_manager):
        """测试删除不存在的知识库"""
        kb_manager.kb_repository.exists.return_value = False
        
        with pytest.raises(KnowledgeBaseNotFoundError, match="知识库 'test_kb' 不存在"):
            kb_manager.delete_knowledge_base("test_kb")
    
    def test_delete_knowledge_base_partial_failure(self, kb_manager):
        """测试部分删除失败"""
        kb_manager.kb_repository.exists.return_value = True
        kb_manager.qa_repository.delete_by_knowledge_base.return_value = 0
        kb_manager.vector_store.delete_collection.return_value = True
        kb_manager.kb_repository.delete.return_value = False  # 数据库删除失败
        
        result = kb_manager.delete_knowledge_base("test_kb")
        
        assert result is False
    
    def test_get_knowledge_base_success(self, kb_manager):
        """测试成功获取知识库"""
        expected_kb = KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=2,
            document_count=5
        )
        kb_manager.kb_repository.get_by_name.return_value = expected_kb
        
        result = kb_manager.get_knowledge_base("test_kb")
        
        assert result == expected_kb
        kb_manager.kb_repository.get_by_name.assert_called_once_with("test_kb")
    
    def test_get_knowledge_base_not_found(self, kb_manager):
        """测试获取不存在的知识库"""
        kb_manager.kb_repository.get_by_name.return_value = None
        
        result = kb_manager.get_knowledge_base("test_kb")
        
        assert result is None
    
    def test_list_knowledge_bases(self, kb_manager):
        """测试列出知识库"""
        mock_kbs = [
            KnowledgeBase("kb1", datetime.now()),
            KnowledgeBase("kb2", datetime.now()),
            KnowledgeBase("kb3", datetime.now())
        ]
        kb_manager.kb_repository.get_all.return_value = mock_kbs
        
        result = kb_manager.list_knowledge_bases()
        
        assert result == ["kb1", "kb2", "kb3"]
        kb_manager.kb_repository.get_all.assert_called_once()
    
    def test_get_knowledge_base_details(self, kb_manager):
        """测试获取知识库详细信息"""
        mock_kbs = [
            KnowledgeBase("kb1", datetime.now()),
            KnowledgeBase("kb2", datetime.now())
        ]
        kb_manager.kb_repository.get_all.return_value = mock_kbs
        
        result = kb_manager.get_knowledge_base_details()
        
        assert result == mock_kbs
        kb_manager.kb_repository.get_all.assert_called_once()
    
    def test_add_documents_success(self, kb_manager, sample_files):
        """测试成功添加文档"""
        existing_kb = KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=1,
            document_count=2
        )
        kb_manager.kb_repository.get_by_name.return_value = existing_kb
        kb_manager.document_processor.process_multiple_files.return_value = [
            Mock(text="新文档1", metadata={}),
            Mock(text="新文档2", metadata={})
        ]
        kb_manager.document_processor.preprocess_documents.return_value = [
            Mock(text="新文档1", metadata={}),
            Mock(text="新文档2", metadata={})
        ]
        kb_manager.document_processor.chunk_documents.return_value = [
            Mock(text="新文档1", metadata={}),
            Mock(text="新文档2", metadata={})
        ]
        
        result = kb_manager.add_documents("test_kb", sample_files)
        
        assert result == 2
        kb_manager.vector_store.add_documents.assert_called_once()
        kb_manager.kb_repository.update.assert_called_once()
        
        # 验证统计信息更新
        updated_kb = kb_manager.kb_repository.update.call_args[0][0]
        assert updated_kb.file_count == 3  # 1 + 2
        assert updated_kb.document_count == 4  # 2 + 2
    
    def test_add_documents_kb_not_exists(self, kb_manager, sample_files):
        """测试向不存在的知识库添加文档"""
        kb_manager.kb_repository.get_by_name.return_value = None
        
        with pytest.raises(KnowledgeBaseNotFoundError, match="知识库 'test_kb' 不存在"):
            kb_manager.add_documents("test_kb", sample_files)
    
    def test_get_knowledge_base_stats(self, kb_manager):
        """测试获取知识库统计信息"""
        mock_kb = KnowledgeBase(
            name="test_kb",
            created_at=datetime.now(),
            file_count=3,
            document_count=10,
            description="测试知识库"
        )
        kb_manager.kb_repository.get_by_name.return_value = mock_kb
        kb_manager.vector_store.get_collection_stats.return_value = {
            "document_count": 10,
            "embedding_model": "shaw/dmeta-embedding-zh-small-q4"
        }
        kb_manager.qa_repository.get_statistics.return_value = {
            "total_count": 5,
            "correct_count": 3,
            "accuracy_rate": 60.0
        }
        
        result = kb_manager.get_knowledge_base_stats("test_kb")
        
        assert result["name"] == "test_kb"
        assert result["description"] == "测试知识库"
        assert result["file_count"] == 3
        assert result["document_count"] == 10
        assert result["vector_document_count"] == 10
        assert result["embedding_model"] == "shaw/dmeta-embedding-zh-small-q4"
        assert result["qa_statistics"]["total_count"] == 5
    
    def test_search_knowledge_base(self, kb_manager):
        """测试搜索知识库"""
        kb_manager.kb_repository.exists.return_value = True
        
        mock_search_results = [
            Mock(
                document=Mock(content="搜索结果1", metadata={"source": "file1"}),
                score=0.9,
                distance=0.1
            ),
            Mock(
                document=Mock(content="搜索结果2", metadata={"source": "file2"}),
                score=0.8,
                distance=0.2
            )
        ]
        kb_manager.vector_store.similarity_search.return_value = mock_search_results
        
        result = kb_manager.search_knowledge_base("test_kb", "测试查询", k=2)
        
        assert len(result) == 2
        assert result[0]["content"] == "搜索结果1"
        assert result[0]["score"] == 0.9
        assert result[1]["content"] == "搜索结果2"
        assert result[1]["score"] == 0.8
        
        kb_manager.vector_store.similarity_search.assert_called_once_with("test_kb", "测试查询", 2)
    
    def test_search_knowledge_base_not_exists(self, kb_manager):
        """测试搜索不存在的知识库"""
        kb_manager.kb_repository.exists.return_value = False
        
        with pytest.raises(KnowledgeBaseNotFoundError, match="知识库 'test_kb' 不存在"):
            kb_manager.search_knowledge_base("test_kb", "测试查询")
    
    def test_health_check_healthy(self, kb_manager):
        """测试健康检查 - 健康状态"""
        kb_manager.vector_store.health_check.return_value = {"status": "healthy"}
        kb_manager.kb_repository.get_all.return_value = [Mock(), Mock()]
        
        result = kb_manager.health_check()
        
        assert result["status"] == "healthy"
        assert result["components"]["vector_store"]["status"] == "healthy"
        assert result["components"]["database"]["status"] == "healthy"
        assert result["components"]["database"]["knowledge_base_count"] == 2
        assert result["components"]["document_processor"]["status"] == "healthy"
    
    def test_health_check_unhealthy(self, kb_manager):
        """测试健康检查 - 不健康状态"""
        kb_manager.vector_store.health_check.return_value = {"status": "unhealthy"}
        kb_manager.kb_repository.get_all.side_effect = DatabaseError("数据库连接失败")
        
        result = kb_manager.health_check()
        
        assert result["status"] == "unhealthy"
        assert result["components"]["vector_store"]["status"] == "unhealthy"
        assert result["components"]["database"]["status"] == "unhealthy"
        assert "数据库连接失败" in result["components"]["database"]["error"]
    
    def test_validate_create_params(self, kb_manager, sample_files):
        """测试创建参数验证"""
        # 测试正常参数
        kb_manager._validate_create_params("test_kb", sample_files)
        
        # 测试空名称
        with pytest.raises(ValidationError, match="知识库名称不能为空"):
            kb_manager._validate_create_params("", sample_files)
        
        # 测试名称过长
        long_name = "a" * 101
        with pytest.raises(ValidationError, match="知识库名称不能超过100个字符"):
            kb_manager._validate_create_params(long_name, sample_files)
        
        # 测试空文件列表
        with pytest.raises(ValidationError, match="必须提供至少一个文件"):
            kb_manager._validate_create_params("test_kb", [])
    
    def test_validate_files(self, kb_manager, sample_files):
        """测试文件验证"""
        # 测试正常文件
        kb_manager._validate_files(sample_files)
        
        # 测试空路径
        with pytest.raises(ValidationError, match="文件路径不能为空"):
            kb_manager._validate_files([""])
        
        # 测试不存在的文件
        with pytest.raises(ValidationError, match="文件不存在"):
            kb_manager._validate_files(["/nonexistent/file.txt"])
    
    def test_cleanup_failed_creation(self, kb_manager):
        """测试清理失败的创建"""
        kb_manager.vector_store.collection_exists.return_value = True
        kb_manager.kb_repository.exists.return_value = True
        
        # 不应该抛出异常
        kb_manager._cleanup_failed_creation("test_kb")
        
        kb_manager.vector_store.delete_collection.assert_called_once_with("test_kb")
        kb_manager.kb_repository.delete.assert_called_once_with("test_kb")
    
    def test_cleanup_failed_creation_with_errors(self, kb_manager):
        """测试清理时发生错误"""
        kb_manager.vector_store.collection_exists.return_value = True
        kb_manager.vector_store.delete_collection.side_effect = Exception("清理失败")
        kb_manager.kb_repository.exists.return_value = True
        kb_manager.kb_repository.delete.side_effect = Exception("清理失败")
        
        # 不应该抛出异常，只记录警告
        kb_manager._cleanup_failed_creation("test_kb")


class TestKnowledgeBaseManagerIntegration:
    """知识库管理器集成测试（使用真实组件）"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dir):
        """创建示例文件"""
        files = []
        
        # 创建文本文件
        txt_file = Path(temp_dir) / "sample.txt"
        txt_file.write_text("这是一个测试文本文件。\n包含中文内容。", encoding='utf-8')
        files.append(str(txt_file))
        
        return files
    
    @pytest.mark.integration
    def test_full_workflow(self, sample_files):
        """测试完整工作流程"""
        # 注意：这个测试需要真实的数据库和向量存储
        # 在实际环境中运行时需要确保依赖服务可用
        
        # 这里只是示例，实际测试需要根据环境配置
        pytest.skip("需要真实的数据库和向量存储服务")
        
        manager = KnowledgeBaseManager()
        
        try:
            # 创建知识库
            kb = manager.create_knowledge_base(
                name="integration_test_kb",
                files=sample_files,
                description="集成测试知识库"
            )
            
            assert kb.name == "integration_test_kb"
            assert kb.file_count == 1
            
            # 获取知识库
            retrieved_kb = manager.get_knowledge_base("integration_test_kb")
            assert retrieved_kb is not None
            assert retrieved_kb.name == "integration_test_kb"
            
            # 列出知识库
            kb_list = manager.list_knowledge_bases()
            assert "integration_test_kb" in kb_list
            
            # 获取统计信息
            stats = manager.get_knowledge_base_stats("integration_test_kb")
            assert stats["name"] == "integration_test_kb"
            assert stats["file_count"] == 1
            
            # 搜索知识库
            results = manager.search_knowledge_base("integration_test_kb", "测试", k=1)
            assert len(results) > 0
            
            # 健康检查
            health = manager.health_check()
            assert health["status"] == "healthy"
            
        finally:
            # 清理
            try:
                manager.delete_knowledge_base("integration_test_kb")
            except:
                pass