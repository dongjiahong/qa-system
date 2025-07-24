"""
Knowledge Base Manager module for Knowledge QA System
知识库管理器模块，整合文档处理和向量存储
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import uuid

from loguru import logger
from llama_index.core import Document

from .models import (
    KnowledgeBase, KnowledgeSystemError, FileProcessingError, 
    VectorStoreError, DatabaseError, ValidationError, KnowledgeBaseNotFoundError
)
from .database import get_knowledge_base_repository, get_qa_record_repository
from .document_processor import DocumentProcessor
from .vector_store import VectorStore, DocumentChunk, create_document_chunk
from .config import get_config


class KnowledgeBaseManager:
    """
    知识库管理器
    
    负责整合文档处理和向量存储，提供知识库的创建、删除、列表和文档导入功能
    """
    
    def __init__(self):
        """初始化知识库管理器"""
        self.config = get_config()
        
        # 初始化组件
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.kb_repository = get_knowledge_base_repository()
        self.qa_repository = get_qa_record_repository()
        
        logger.info("KnowledgeBaseManager initialized")
    
    def create_knowledge_base(
        self, 
        name: str, 
        files: List[str], 
        description: Optional[str] = None
    ) -> KnowledgeBase:
        """
        创建知识库
        
        Args:
            name: 知识库名称
            files: 文件路径列表
            description: 知识库描述
            
        Returns:
            KnowledgeBase: 创建的知识库对象
            
        Raises:
            ValidationError: 参数验证失败
            FileProcessingError: 文件处理失败
            VectorStoreError: 向量存储失败
            DatabaseError: 数据库操作失败
        """
        try:
            logger.info(f"Creating knowledge base: {name} with {len(files)} files")
            
            # 验证参数
            self._validate_create_params(name, files)
            
            # 检查知识库是否已存在
            if self.kb_repository.exists(name):
                raise ValidationError(f"知识库 '{name}' 已存在")
            
            # 处理文档
            logger.info("Processing documents...")
            documents = self._process_documents(files)
            
            if not documents:
                raise FileProcessingError("没有成功处理的文档")
            
            # 创建向量集合
            logger.info("Creating vector collection...")
            self.vector_store.create_collection(name, reset_if_exists=False)
            
            # 向量化并存储文档
            logger.info("Vectorizing and storing documents...")
            document_chunks = self._convert_to_chunks(documents)
            self.vector_store.add_documents(name, document_chunks)
            
            # 创建知识库记录
            knowledge_base = KnowledgeBase(
                name=name,
                created_at=datetime.now(),
                file_count=len(files),
                document_count=len(documents),
                description=description
            )
            
            self.kb_repository.create(knowledge_base)
            
            logger.info(f"Successfully created knowledge base: {name}")
            return knowledge_base
            
        except (ValidationError, FileProcessingError, VectorStoreError, DatabaseError):
            # 清理失败的创建
            self._cleanup_failed_creation(name)
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating knowledge base {name}: {e}")
            self._cleanup_failed_creation(name)
            raise KnowledgeSystemError(f"创建知识库失败: {str(e)}")
    
    def delete_knowledge_base(self, name: str) -> bool:
        """
        删除知识库
        
        Args:
            name: 知识库名称
            
        Returns:
            bool: 删除是否成功
            
        Raises:
            KnowledgeBaseNotFoundError: 知识库不存在
            VectorStoreError: 向量存储删除失败
            DatabaseError: 数据库操作失败
        """
        try:
            logger.info(f"Deleting knowledge base: {name}")
            
            # 检查知识库是否存在
            if not self.kb_repository.exists(name):
                raise KnowledgeBaseNotFoundError(f"知识库 '{name}' 不存在")
            
            # 删除问答历史记录
            logger.info("Deleting QA history records...")
            deleted_records = self.qa_repository.delete_by_knowledge_base(name)
            logger.info(f"Deleted {deleted_records} QA records")
            
            # 删除向量集合
            logger.info("Deleting vector collection...")
            vector_deleted = self.vector_store.delete_collection(name)
            
            # 删除知识库记录
            logger.info("Deleting knowledge base record...")
            db_deleted = self.kb_repository.delete(name)
            
            success = vector_deleted and db_deleted
            if success:
                logger.info(f"Successfully deleted knowledge base: {name}")
            else:
                logger.warning(f"Partial deletion of knowledge base: {name}")
            
            return success
            
        except (KnowledgeBaseNotFoundError, VectorStoreError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting knowledge base {name}: {e}")
            raise KnowledgeSystemError(f"删除知识库失败: {str(e)}")
    
    def get_knowledge_base(self, name: str) -> Optional[KnowledgeBase]:
        """
        获取知识库信息
        
        Args:
            name: 知识库名称
            
        Returns:
            Optional[KnowledgeBase]: 知识库对象或None
            
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            return self.kb_repository.get_by_name(name)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting knowledge base {name}: {e}")
            raise KnowledgeSystemError(f"获取知识库失败: {str(e)}")
    
    def list_knowledge_bases(self) -> List[str]:
        """
        列出所有知识库名称
        
        Returns:
            List[str]: 知识库名称列表
            
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            knowledge_bases = self.kb_repository.get_all()
            return [kb.name for kb in knowledge_bases]
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing knowledge bases: {e}")
            raise KnowledgeSystemError(f"获取知识库列表失败: {str(e)}")
    
    def get_knowledge_base_details(self) -> List[KnowledgeBase]:
        """
        获取所有知识库的详细信息
        
        Returns:
            List[KnowledgeBase]: 知识库详细信息列表
            
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            return self.kb_repository.get_all()
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting knowledge base details: {e}")
            raise KnowledgeSystemError(f"获取知识库详细信息失败: {str(e)}")
    
    def add_documents(self, kb_name: str, files: List[str]) -> int:
        """
        向现有知识库添加文档
        
        Args:
            kb_name: 知识库名称
            files: 文件路径列表
            
        Returns:
            int: 添加的文档数量
            
        Raises:
            KnowledgeBaseNotFoundError: 知识库不存在
            FileProcessingError: 文件处理失败
            VectorStoreError: 向量存储失败
            DatabaseError: 数据库操作失败
        """
        try:
            logger.info(f"Adding {len(files)} documents to knowledge base: {kb_name}")
            
            # 检查知识库是否存在
            kb = self.kb_repository.get_by_name(kb_name)
            if not kb:
                raise KnowledgeBaseNotFoundError(f"知识库 '{kb_name}' 不存在")
            
            # 验证文件
            self._validate_files(files)
            
            # 处理文档
            logger.info("Processing new documents...")
            documents = self._process_documents(files)
            
            if not documents:
                raise FileProcessingError("没有成功处理的文档")
            
            # 向量化并存储文档
            logger.info("Vectorizing and storing new documents...")
            document_chunks = self._convert_to_chunks(documents)
            self.vector_store.add_documents(kb_name, document_chunks)
            
            # 更新知识库统计信息
            kb.file_count += len(files)
            kb.document_count += len(documents)
            self.kb_repository.update(kb)
            
            logger.info(f"Successfully added {len(documents)} documents to knowledge base: {kb_name}")
            return len(documents)
            
        except (KnowledgeBaseNotFoundError, FileProcessingError, VectorStoreError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding documents to {kb_name}: {e}")
            raise KnowledgeSystemError(f"添加文档失败: {str(e)}")
    
    def get_knowledge_base_stats(self, name: str) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        Args:
            name: 知识库名称
            
        Returns:
            Dict[str, Any]: 统计信息
            
        Raises:
            KnowledgeBaseNotFoundError: 知识库不存在
        """
        try:
            # 获取基本信息
            kb = self.kb_repository.get_by_name(name)
            if not kb:
                raise KnowledgeBaseNotFoundError(f"知识库 '{name}' 不存在")
            
            # 获取向量存储统计
            vector_stats = self.vector_store.get_collection_stats(name)
            
            # 获取问答统计
            qa_stats = self.qa_repository.get_statistics(name)
            
            return {
                "name": kb.name,
                "description": kb.description,
                "created_at": kb.created_at.isoformat(),
                "file_count": kb.file_count,
                "document_count": kb.document_count,
                "vector_document_count": vector_stats.get("document_count", 0),
                "embedding_model": vector_stats.get("embedding_model"),
                "qa_statistics": qa_stats,
            }
            
        except (KnowledgeBaseNotFoundError, VectorStoreError, DatabaseError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting stats for {name}: {e}")
            raise KnowledgeSystemError(f"获取知识库统计失败: {str(e)}")
    
    def search_knowledge_base(
        self, 
        kb_name: str, 
        query: str, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        在知识库中搜索相关内容
        
        Args:
            kb_name: 知识库名称
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 搜索结果列表
            
        Raises:
            KnowledgeBaseNotFoundError: 知识库不存在
            VectorStoreError: 向量搜索失败
        """
        try:
            # 检查知识库是否存在
            if not self.kb_repository.exists(kb_name):
                raise KnowledgeBaseNotFoundError(f"知识库 '{kb_name}' 不存在")
            
            # 执行相似性搜索
            search_results = self.vector_store.similarity_search(kb_name, query, k)
            
            # 格式化结果
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "content": result.document.content,
                    "metadata": result.document.metadata,
                    "score": result.score,
                    "distance": result.distance,
                })
            
            return formatted_results
            
        except (KnowledgeBaseNotFoundError, VectorStoreError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error searching knowledge base {kb_name}: {e}")
            raise KnowledgeSystemError(f"搜索知识库失败: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        系统健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            # 检查向量存储
            vector_health = self.vector_store.health_check()
            
            # 检查数据库连接
            try:
                kb_count = len(self.kb_repository.get_all())
                db_healthy = True
                db_error = None
            except Exception as e:
                db_healthy = False
                db_error = str(e)
                kb_count = 0
            
            # 检查文档处理器
            try:
                supported_formats = self.config.supported_file_extensions
                processor_healthy = True
                processor_error = None
            except Exception as e:
                processor_healthy = False
                processor_error = str(e)
                supported_formats = []
            
            overall_healthy = (
                vector_health.get("status") == "healthy" and
                db_healthy and
                processor_healthy
            )
            
            return {
                "status": "healthy" if overall_healthy else "unhealthy",
                "components": {
                    "vector_store": vector_health,
                    "database": {
                        "status": "healthy" if db_healthy else "unhealthy",
                        "error": db_error,
                        "knowledge_base_count": kb_count,
                    },
                    "document_processor": {
                        "status": "healthy" if processor_healthy else "unhealthy",
                        "error": processor_error,
                        "supported_formats": supported_formats,
                    }
                },
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    def _validate_create_params(self, name: str, files: List[str]) -> None:
        """验证创建参数"""
        if not name or not name.strip():
            raise ValidationError("知识库名称不能为空")
        
        if len(name) > 100:
            raise ValidationError("知识库名称不能超过100个字符")
        
        if not files:
            raise ValidationError("必须提供至少一个文件")
        
        self._validate_files(files)
    
    def _validate_files(self, files: List[str]) -> None:
        """验证文件列表"""
        for file_path in files:
            if not file_path or not file_path.strip():
                raise ValidationError("文件路径不能为空")
            
            path = Path(file_path)
            if not path.exists():
                raise ValidationError(f"文件不存在: {file_path}")
            
            if not path.is_file():
                raise ValidationError(f"路径不是文件: {file_path}")
    
    def _process_documents(self, files: List[str]) -> List[Document]:
        """处理文档列表"""
        try:
            # 批量处理文件
            documents = self.document_processor.process_multiple_files(files)
            
            # 预处理文档
            documents = self.document_processor.preprocess_documents(documents)
            
            # 分块处理
            documents = self.document_processor.chunk_documents(documents)
            
            return documents
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise FileProcessingError(f"文档处理失败: {str(e)}")
    
    def _convert_to_chunks(self, documents: List[Document]) -> List[DocumentChunk]:
        """将LlamaIndex文档转换为向量存储文档块"""
        chunks = []
        
        for doc in documents:
            chunk = create_document_chunk(
                content=doc.text,
                metadata=doc.metadata,
                chunk_id=str(uuid.uuid4())
            )
            chunks.append(chunk)
        
        return chunks
    
    def _cleanup_failed_creation(self, name: str) -> None:
        """清理失败的创建操作"""
        try:
            logger.info(f"Cleaning up failed creation for: {name}")
            
            # 尝试删除向量集合
            try:
                if self.vector_store.collection_exists(name):
                    self.vector_store.delete_collection(name)
                    logger.info(f"Cleaned up vector collection: {name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup vector collection {name}: {e}")
            
            # 尝试删除数据库记录
            try:
                if self.kb_repository.exists(name):
                    self.kb_repository.delete(name)
                    logger.info(f"Cleaned up database record: {name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup database record {name}: {e}")
                
        except Exception as e:
            logger.error(f"Cleanup failed for {name}: {e}")