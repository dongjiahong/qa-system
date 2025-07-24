"""
Vector storage module using ChromaDB for Knowledge QA System
使用 ChromaDB 的向量存储模块
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
from pathlib import Path

from .models import VectorStoreError, KnowledgeSystemError
from .config import get_config


# ============================================================================
# Configuration and Setup 配置和设置
# ============================================================================

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """文档块数据模型"""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class SearchResult:
    """搜索结果数据模型"""
    
    document: DocumentChunk
    score: float
    distance: float


# ============================================================================
# Vector Store Implementation 向量存储实现
# ============================================================================

class VectorStore:
    """
    ChromaDB 向量存储管理器
    
    负责管理向量数据库的连接、集合操作、文档向量化和相似性搜索
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        初始化向量存储
        
        Args:
            persist_directory: 持久化目录路径，默认使用配置文件中的路径
        """
        self.config = get_config()
        
        # 设置持久化目录
        if persist_directory is None:
            persist_directory = self.config.chroma_db_path
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB 客户端
        self._init_client()
        
        # 初始化嵌入模型
        self._init_embedding_function()
        
        logger.info(f"VectorStore initialized with persist_directory: {self.persist_directory}")
    
    def _init_client(self) -> None:
        """初始化 ChromaDB 客户端"""
        try:
            # 配置 ChromaDB 设置
            settings = Settings(
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False,
                allow_reset=True,
            )
            
            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=settings
            )
            
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB client: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"persist_directory": str(self.persist_directory)})
    
    def _init_embedding_function(self) -> None:
        """初始化嵌入模型函数"""
        try:
            # 使用 Ollama 嵌入函数，指定中文嵌入模型
            self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
                url=self.config.ollama_base_url,
                model_name="shaw/dmeta-embedding-zh-small-q4"
            )
            
            logger.info("Embedding function initialized with shaw/dmeta-embedding-zh-small-q4")
            
        except Exception as e:
            error_msg = f"Failed to initialize embedding function: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"model": "shaw/dmeta-embedding-zh-small-q4"})
    
    def create_collection(self, kb_name: str, reset_if_exists: bool = False) -> None:
        """
        创建向量集合
        
        Args:
            kb_name: 知识库名称，用作集合名称
            reset_if_exists: 如果集合已存在是否重置
            
        Raises:
            VectorStoreError: 集合创建失败
        """
        try:
            # 验证集合名称
            self._validate_collection_name(kb_name)
            
            # 检查集合是否已存在
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if kb_name in existing_collections:
                if reset_if_exists:
                    logger.warning(f"Collection '{kb_name}' exists, deleting and recreating")
                    self.client.delete_collection(name=kb_name)
                else:
                    logger.info(f"Collection '{kb_name}' already exists, skipping creation")
                    return
            
            # 创建新集合
            collection = self.client.create_collection(
                name=kb_name,
                embedding_function=self.embedding_function,
                metadata={"created_at": str(uuid.uuid4())}
            )
            
            logger.info(f"Collection '{kb_name}' created successfully")
            
        except Exception as e:
            error_msg = f"Failed to create collection '{kb_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"kb_name": kb_name, "reset_if_exists": reset_if_exists})
    
    def delete_collection(self, kb_name: str) -> bool:
        """
        删除向量集合
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            bool: 删除是否成功
            
        Raises:
            VectorStoreError: 集合删除失败
        """
        try:
            # 检查集合是否存在
            if not self.collection_exists(kb_name):
                logger.warning(f"Collection '{kb_name}' does not exist, nothing to delete")
                return False
            
            # 删除集合
            self.client.delete_collection(name=kb_name)
            
            logger.info(f"Collection '{kb_name}' deleted successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete collection '{kb_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"kb_name": kb_name})
    
    def collection_exists(self, kb_name: str) -> bool:
        """
        检查集合是否存在
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            bool: 集合是否存在
        """
        try:
            existing_collections = [col.name for col in self.client.list_collections()]
            return kb_name in existing_collections
        except Exception as e:
            logger.error(f"Failed to check collection existence: {str(e)}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        列出所有集合
        
        Returns:
            List[str]: 集合名称列表
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            error_msg = f"Failed to list collections: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    def add_documents(self, kb_name: str, documents: List[DocumentChunk]) -> None:
        """
        添加文档到向量集合
        
        Args:
            kb_name: 知识库名称
            documents: 文档块列表
            
        Raises:
            VectorStoreError: 文档添加失败
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return
            
            # 获取集合
            collection = self._get_collection(kb_name)
            
            # 准备数据
            ids = [doc.id for doc in documents]
            texts = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # 批量添加文档
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to collection '{kb_name}'")
            
        except Exception as e:
            error_msg = f"Failed to add documents to collection '{kb_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"kb_name": kb_name, "document_count": len(documents)})
    
    def similarity_search(
        self, 
        kb_name: str, 
        query: str, 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        相似性搜索
        
        Args:
            kb_name: 知识库名称
            query: 查询文本
            k: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            List[SearchResult]: 搜索结果列表
            
        Raises:
            VectorStoreError: 搜索失败
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided")
                return []
            
            # 获取集合
            collection = self._get_collection(kb_name)
            
            # 执行查询
            results = collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_metadata
            )
            
            # 处理结果
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    document = DocumentChunk(
                        id=doc_id,
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {}
                    )
                    
                    # ChromaDB 返回的是距离，需要转换为相似度分数
                    distance = results['distances'][0][i] if results['distances'] else 0.0
                    score = max(0.0, 1.0 - distance)  # 简单的距离到相似度转换
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        distance=distance
                    ))
            
            logger.info(f"Found {len(search_results)} results for query in collection '{kb_name}'")
            return search_results
            
        except Exception as e:
            error_msg = f"Failed to perform similarity search in collection '{kb_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"kb_name": kb_name, "query": query, "k": k})
    
    def get_collection_stats(self, kb_name: str) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            collection = self._get_collection(kb_name)
            
            # 获取文档数量
            count = collection.count()
            
            # 获取集合元数据
            metadata = collection.metadata or {}
            
            return {
                "name": kb_name,
                "document_count": count,
                "metadata": metadata,
                "embedding_model": "shaw/dmeta-embedding-zh-small-q4"
            }
            
        except Exception as e:
            error_msg = f"Failed to get stats for collection '{kb_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"kb_name": kb_name})
    
    def update_document(self, kb_name: str, document: DocumentChunk) -> None:
        """
        更新文档
        
        Args:
            kb_name: 知识库名称
            document: 文档块
        """
        try:
            collection = self._get_collection(kb_name)
            
            collection.update(
                ids=[document.id],
                documents=[document.content],
                metadatas=[document.metadata]
            )
            
            logger.info(f"Updated document '{document.id}' in collection '{kb_name}'")
            
        except Exception as e:
            error_msg = f"Failed to update document in collection '{kb_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"kb_name": kb_name, "document_id": document.id})
    
    def delete_documents(self, kb_name: str, document_ids: List[str]) -> None:
        """
        删除文档
        
        Args:
            kb_name: 知识库名称
            document_ids: 文档ID列表
        """
        try:
            if not document_ids:
                logger.warning("No document IDs provided for deletion")
                return
            
            collection = self._get_collection(kb_name)
            
            collection.delete(ids=document_ids)
            
            logger.info(f"Deleted {len(document_ids)} documents from collection '{kb_name}'")
            
        except Exception as e:
            error_msg = f"Failed to delete documents from collection '{kb_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"kb_name": kb_name, "document_ids": document_ids})
    
    def _get_collection(self, kb_name: str):
        """
        获取集合对象
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            Collection: ChromaDB 集合对象
            
        Raises:
            VectorStoreError: 集合不存在或获取失败
        """
        try:
            if not self.collection_exists(kb_name):
                raise VectorStoreError(f"Collection '{kb_name}' does not exist", {"kb_name": kb_name})
            
            return self.client.get_collection(
                name=kb_name,
                embedding_function=self.embedding_function
            )
            
        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = f"Failed to get collection '{kb_name}': {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, {"kb_name": kb_name})
    
    def _validate_collection_name(self, name: str) -> None:
        """
        验证集合名称
        
        Args:
            name: 集合名称
            
        Raises:
            VectorStoreError: 名称无效
        """
        if not name or not name.strip():
            raise VectorStoreError("Collection name cannot be empty")
        
        if len(name) > 100:
            raise VectorStoreError("Collection name cannot exceed 100 characters")
        
        # ChromaDB 集合名称限制
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in name for char in invalid_chars):
            raise VectorStoreError(f"Collection name contains invalid characters: {invalid_chars}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            # 检查客户端连接
            collections = self.client.list_collections()
            
            # 检查嵌入模型
            test_embedding = self.embedding_function(["测试文本"])
            
            return {
                "status": "healthy",
                "client_connected": True,
                "embedding_model_available": True,
                "collections_count": len(collections),
                "persist_directory": str(self.persist_directory),
                "embedding_model": "shaw/dmeta-embedding-zh-small-q4"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "client_connected": False,
                "embedding_model_available": False,
                "persist_directory": str(self.persist_directory)
            }
    
    def reset_all(self) -> None:
        """
        重置所有数据（危险操作，仅用于测试）
        """
        try:
            self.client.reset()
            logger.warning("All vector store data has been reset")
        except Exception as e:
            error_msg = f"Failed to reset vector store: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg)


# ============================================================================
# Utility Functions 工具函数
# ============================================================================

def create_document_chunk(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_id: Optional[str] = None
) -> DocumentChunk:
    """
    创建文档块
    
    Args:
        content: 文档内容
        metadata: 元数据
        chunk_id: 文档块ID
        
    Returns:
        DocumentChunk: 文档块对象
    """
    return DocumentChunk(
        id=chunk_id or str(uuid.uuid4()),
        content=content,
        metadata=metadata or {}
    )


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    文本分块
    
    Args:
        text: 原始文本
        chunk_size: 块大小
        overlap: 重叠大小
        
    Returns:
        List[str]: 文本块列表
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 如果不是最后一块，尝试在句号或换行符处分割
        if end < len(text):
            # 寻找最近的句号或换行符
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in ['。', '！', '？', '\n', '.', '!', '?']:
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 计算下一个开始位置，考虑重叠
        start = max(start + 1, end - overlap)
        
        # 避免无限循环
        if start >= len(text):
            break
    
    return chunks