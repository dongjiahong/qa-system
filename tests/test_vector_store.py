"""
Unit tests for vector_store module
向量存储模块单元测试
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.vector_store import (
    VectorStore,
    DocumentChunk,
    SearchResult,
    create_document_chunk,
    chunk_text
)
from src.models import VectorStoreError


class TestDocumentChunk:
    """DocumentChunk 数据模型测试"""
    
    def test_document_chunk_creation(self):
        """测试文档块创建"""
        chunk = DocumentChunk(
            id="test-id",
            content="测试内容",
            metadata={"source": "test.txt"}
        )
        
        assert chunk.id == "test-id"
        assert chunk.content == "测试内容"
        assert chunk.metadata == {"source": "test.txt"}
        assert chunk.embedding is None
    
    def test_document_chunk_auto_id(self):
        """测试自动生成ID"""
        chunk = DocumentChunk(
            id="",
            content="测试内容",
            metadata={}
        )
        
        assert chunk.id != ""
        assert len(chunk.id) == 36  # UUID4 length


class TestSearchResult:
    """SearchResult 数据模型测试"""
    
    def test_search_result_creation(self):
        """测试搜索结果创建"""
        document = DocumentChunk(
            id="test-id",
            content="测试内容",
            metadata={}
        )
        
        result = SearchResult(
            document=document,
            score=0.85,
            distance=0.15
        )
        
        assert result.document == document
        assert result.score == 0.85
        assert result.distance == 0.15


class TestVectorStore:
    """VectorStore 主要功能测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        config = Mock()
        config.chroma_db_path = "test_data/chroma_db"
        config.ollama_base_url = "http://localhost:11434"
        return config
    
    @pytest.fixture
    def mock_client(self):
        """模拟 ChromaDB 客户端"""
        client = Mock()
        client.list_collections.return_value = []
        return client
    
    @pytest.fixture
    def mock_embedding_function(self):
        """模拟嵌入函数"""
        embedding_func = Mock()
        embedding_func.return_value = [[0.1, 0.2, 0.3]]
        return embedding_func
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_vector_store_initialization(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试向量存储初始化"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        vector_store = VectorStore(persist_directory=temp_dir)
        
        assert vector_store.persist_directory == Path(temp_dir)
        assert vector_store.client == mock_client
        assert vector_store.embedding_function == mock_embedding
        
        # 验证客户端初始化参数
        mock_client_class.assert_called_once()
        
        # 验证嵌入函数初始化参数
        mock_embedding_func.assert_called_once_with(
            url=mock_config.ollama_base_url,
            model_name="shaw/dmeta-embedding-zh-small-q4"
        )
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_create_collection_success(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试成功创建集合"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合不存在
        mock_client.list_collections.return_value = []
        mock_collection = Mock()
        mock_client.create_collection.return_value = mock_collection
        
        vector_store = VectorStore(persist_directory=temp_dir)
        vector_store.create_collection("test_kb")
        
        # 验证创建集合调用
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["name"] == "test_kb"
        assert call_args[1]["embedding_function"] == mock_embedding
        assert "created_at" in call_args[1]["metadata"]
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_create_collection_already_exists(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试集合已存在的情况"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合已存在
        mock_collection_obj = Mock()
        mock_collection_obj.name = "test_kb"
        mock_client.list_collections.return_value = [mock_collection_obj]
        
        vector_store = VectorStore(persist_directory=temp_dir)
        vector_store.create_collection("test_kb")
        
        # 验证没有调用创建集合
        mock_client.create_collection.assert_not_called()
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_create_collection_reset_if_exists(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试重置已存在的集合"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合已存在
        mock_collection_obj = Mock()
        mock_collection_obj.name = "test_kb"
        mock_client.list_collections.return_value = [mock_collection_obj]
        
        vector_store = VectorStore(persist_directory=temp_dir)
        vector_store.create_collection("test_kb", reset_if_exists=True)
        
        # 验证删除和创建集合调用
        mock_client.delete_collection.assert_called_once_with(name="test_kb")
        mock_client.create_collection.assert_called_once()
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_delete_collection_success(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试成功删除集合"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合存在
        mock_collection_obj = Mock()
        mock_collection_obj.name = "test_kb"
        mock_client.list_collections.return_value = [mock_collection_obj]
        
        vector_store = VectorStore(persist_directory=temp_dir)
        result = vector_store.delete_collection("test_kb")
        
        assert result is True
        mock_client.delete_collection.assert_called_once_with(name="test_kb")
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_delete_collection_not_exists(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试删除不存在的集合"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合不存在
        mock_client.list_collections.return_value = []
        
        vector_store = VectorStore(persist_directory=temp_dir)
        result = vector_store.delete_collection("test_kb")
        
        assert result is False
        mock_client.delete_collection.assert_not_called()
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_collection_exists(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试检查集合是否存在"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合列表
        mock_collection_obj = Mock()
        mock_collection_obj.name = "existing_kb"
        mock_client.list_collections.return_value = [mock_collection_obj]
        
        vector_store = VectorStore(persist_directory=temp_dir)
        
        assert vector_store.collection_exists("existing_kb") is True
        assert vector_store.collection_exists("non_existing_kb") is False
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_list_collections(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试列出所有集合"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合列表
        mock_collection1 = Mock()
        mock_collection1.name = "kb1"
        mock_collection2 = Mock()
        mock_collection2.name = "kb2"
        mock_collection3 = Mock()
        mock_collection3.name = "kb3"
        mock_collections = [mock_collection1, mock_collection2, mock_collection3]
        mock_client.list_collections.return_value = mock_collections
        
        vector_store = VectorStore(persist_directory=temp_dir)
        collections = vector_store.list_collections()
        
        assert collections == ["kb1", "kb2", "kb3"]
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_add_documents(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试添加文档"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合存在
        mock_collection_obj = Mock()
        mock_collection_obj.name = "test_kb"
        mock_client.list_collections.return_value = [mock_collection_obj]
        
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        
        # 创建测试文档
        documents = [
            DocumentChunk(id="doc1", content="内容1", metadata={"source": "file1.txt"}),
            DocumentChunk(id="doc2", content="内容2", metadata={"source": "file2.txt"})
        ]
        
        vector_store = VectorStore(persist_directory=temp_dir)
        vector_store.add_documents("test_kb", documents)
        
        # 验证添加文档调用
        mock_collection.add.assert_called_once_with(
            ids=["doc1", "doc2"],
            documents=["内容1", "内容2"],
            metadatas=[{"source": "file1.txt"}, {"source": "file2.txt"}]
        )
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_similarity_search(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试相似性搜索"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合存在
        mock_collection_obj = Mock()
        mock_collection_obj.name = "test_kb"
        mock_client.list_collections.return_value = [mock_collection_obj]
        
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        
        # 模拟查询结果
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['内容1', '内容2']],
            'metadatas': [[{'source': 'file1.txt'}, {'source': 'file2.txt'}]],
            'distances': [[0.1, 0.3]]
        }
        
        vector_store = VectorStore(persist_directory=temp_dir)
        results = vector_store.similarity_search("test_kb", "测试查询", k=2)
        
        assert len(results) == 2
        assert results[0].document.id == "doc1"
        assert results[0].document.content == "内容1"
        assert results[0].score == 0.9  # 1.0 - 0.1
        assert results[0].distance == 0.1
        
        assert results[1].document.id == "doc2"
        assert results[1].document.content == "内容2"
        assert results[1].score == 0.7  # 1.0 - 0.3
        assert results[1].distance == 0.3
        
        # 验证查询调用
        mock_collection.query.assert_called_once_with(
            query_texts=["测试查询"],
            n_results=2,
            where=None
        )
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_similarity_search_empty_query(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试空查询的相似性搜索"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        vector_store = VectorStore(persist_directory=temp_dir)
        results = vector_store.similarity_search("test_kb", "", k=5)
        
        assert results == []
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_get_collection_stats(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试获取集合统计信息"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟集合存在
        mock_collection_obj = Mock()
        mock_collection_obj.name = "test_kb"
        mock_client.list_collections.return_value = [mock_collection_obj]
        
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_collection.metadata = {"created_at": "2024-01-01"}
        mock_client.get_collection.return_value = mock_collection
        
        vector_store = VectorStore(persist_directory=temp_dir)
        stats = vector_store.get_collection_stats("test_kb")
        
        assert stats["name"] == "test_kb"
        assert stats["document_count"] == 100
        assert stats["metadata"] == {"created_at": "2024-01-01"}
        assert stats["embedding_model"] == "shaw/dmeta-embedding-zh-small-q4"
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_validate_collection_name(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试集合名称验证"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        vector_store = VectorStore(persist_directory=temp_dir)
        
        # 测试空名称
        with pytest.raises(VectorStoreError, match="Collection name cannot be empty"):
            vector_store._validate_collection_name("")
        
        # 测试过长名称
        long_name = "a" * 101
        with pytest.raises(VectorStoreError, match="Collection name cannot exceed 100 characters"):
            vector_store._validate_collection_name(long_name)
        
        # 测试无效字符
        with pytest.raises(VectorStoreError, match="Collection name contains invalid characters"):
            vector_store._validate_collection_name("test/kb")
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_health_check_healthy(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试健康检查 - 健康状态"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟健康状态
        mock_client.list_collections.return_value = [Mock(), Mock()]
        mock_embedding.return_value = [[0.1, 0.2, 0.3]]
        
        vector_store = VectorStore(persist_directory=temp_dir)
        health = vector_store.health_check()
        
        assert health["status"] == "healthy"
        assert health["client_connected"] is True
        assert health["embedding_model_available"] is True
        assert health["collections_count"] == 2
        assert health["embedding_model"] == "shaw/dmeta-embedding-zh-small-q4"
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_health_check_unhealthy(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir,
        mock_config
    ):
        """测试健康检查 - 不健康状态"""
        mock_get_config.return_value = mock_config
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟错误状态
        mock_client.list_collections.side_effect = Exception("Connection failed")
        
        vector_store = VectorStore(persist_directory=temp_dir)
        health = vector_store.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["client_connected"] is False
        assert health["embedding_model_available"] is False
        assert "error" in health


class TestUtilityFunctions:
    """工具函数测试"""
    
    def test_create_document_chunk(self):
        """测试创建文档块"""
        chunk = create_document_chunk(
            content="测试内容",
            metadata={"source": "test.txt"},
            chunk_id="test-id"
        )
        
        assert chunk.id == "test-id"
        assert chunk.content == "测试内容"
        assert chunk.metadata == {"source": "test.txt"}
    
    def test_create_document_chunk_auto_id(self):
        """测试自动生成ID的文档块"""
        chunk = create_document_chunk(content="测试内容")
        
        assert chunk.id != ""
        assert len(chunk.id) == 36  # UUID4 length
        assert chunk.content == "测试内容"
        assert chunk.metadata == {}
    
    def test_chunk_text_small_text(self):
        """测试小文本分块"""
        text = "这是一个短文本。"
        chunks = chunk_text(text, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_large_text(self):
        """测试大文本分块"""
        # 创建一个长文本
        sentences = ["这是第{}句话。".format(i) for i in range(1, 101)]
        text = "".join(sentences)
        
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 1
        
        # 验证所有块都不为空
        for chunk in chunks:
            assert chunk.strip() != ""
        
        # 验证块大小大致正确（允许一些变化因为句号分割）
        for chunk in chunks:
            assert len(chunk) <= 60  # chunk_size + some tolerance
    
    def test_chunk_text_with_punctuation(self):
        """测试带标点符号的文本分块"""
        text = "第一句话。第二句话！第三句话？第四句话。" * 10
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        
        assert len(chunks) > 1
        
        # 验证大多数块以标点符号结尾（因为优先在标点符号处分割）
        punctuation_endings = 0
        for chunk in chunks[:-1]:  # 除了最后一块
            if chunk.rstrip().endswith(('。', '！', '？', '.', '!', '?')):
                punctuation_endings += 1
        
        # 至少一半的块应该以标点符号结尾
        assert punctuation_endings >= len(chunks) // 2


class TestErrorHandling:
    """错误处理测试"""
    
    @patch('src.vector_store.get_config')
    def test_init_client_failure(self, mock_get_config, temp_dir):
        """测试客户端初始化失败"""
        mock_config = Mock()
        mock_config.chroma_db_path = temp_dir
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_get_config.return_value = mock_config
        
        with patch('src.vector_store.chromadb.PersistentClient', side_effect=Exception("Client init failed")):
            with pytest.raises(VectorStoreError, match="Failed to initialize ChromaDB client"):
                VectorStore(persist_directory=temp_dir)
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    def test_init_embedding_failure(self, mock_client_class, mock_get_config, temp_dir):
        """测试嵌入函数初始化失败"""
        mock_config = Mock()
        mock_config.chroma_db_path = temp_dir
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_get_config.return_value = mock_config
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        with patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction', 
                  side_effect=Exception("Embedding init failed")):
            with pytest.raises(VectorStoreError, match="Failed to initialize embedding function"):
                VectorStore(persist_directory=temp_dir)
    
    @patch('src.vector_store.get_config')
    @patch('src.vector_store.chromadb.PersistentClient')
    @patch('src.vector_store.embedding_functions.OllamaEmbeddingFunction')
    def test_collection_operation_failure(
        self, 
        mock_embedding_func, 
        mock_client_class, 
        mock_get_config,
        temp_dir
    ):
        """测试集合操作失败"""
        mock_config = Mock()
        mock_config.chroma_db_path = temp_dir
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_get_config.return_value = mock_config
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_embedding = Mock()
        mock_embedding_func.return_value = mock_embedding
        
        # 模拟创建集合失败
        mock_client.list_collections.return_value = []
        mock_client.create_collection.side_effect = Exception("Create failed")
        
        vector_store = VectorStore(persist_directory=temp_dir)
        
        with pytest.raises(VectorStoreError, match="Failed to create collection"):
            vector_store.create_collection("test_kb")