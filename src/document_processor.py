"""
Document processor module for Knowledge QA System
文档处理器模块，集成 LlamaIndex 文档加载器
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import mimetypes
import re

from loguru import logger
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import (
    PDFReader,
    EpubReader,
)
from llama_index.core.node_parser import SentenceSplitter

from .models import FileProcessingError, ValidationError
from .config import get_config


class DocumentProcessor:
    """
    文档处理器类，负责解析不同格式的文档并进行预处理
    """
    
    def __init__(self):
        """初始化文档处理器"""
        self.config = get_config()
        
        # 初始化文档加载器
        self.loaders = {
            '.pdf': PDFReader(),
            '.txt': SimpleDirectoryReader,
            '.md': SimpleDirectoryReader,
            '.epub': EpubReader(),
        }
        
        # 初始化文本分割器
        self.text_splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        
        logger.info("DocumentProcessor initialized")
    
    def validate_file_format(self, file_path: str) -> bool:
        """
        验证文件格式是否支持
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否支持该格式
            
        Raises:
            ValidationError: 文件格式验证失败
        """
        try:
            path = Path(file_path)
            
            # 检查文件是否存在
            if not path.exists():
                raise ValidationError(f"文件不存在: {file_path}")
            
            # 检查是否为文件
            if not path.is_file():
                raise ValidationError(f"路径不是文件: {file_path}")
            
            # 获取文件扩展名
            file_extension = path.suffix.lower()
            
            # 检查扩展名是否支持
            if file_extension not in self.config.supported_file_extensions:
                supported_formats = ", ".join(self.config.supported_file_extensions)
                raise ValidationError(
                    f"不支持的文件格式: {file_extension}. "
                    f"支持的格式: {supported_formats}"
                )
            
            # 检查文件大小
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                raise ValidationError(
                    f"文件过大: {file_size_mb:.1f}MB. "
                    f"最大支持: {self.config.max_file_size_mb}MB"
                )
            
            # 验证MIME类型（可选）
            mime_type, _ = mimetypes.guess_type(file_path)
            logger.debug(f"File {file_path} MIME type: {mime_type}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"File validation error for {file_path}: {e}")
            raise ValidationError(f"文件验证失败: {str(e)}")
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        处理单个文件，返回文档列表
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Document]: 解析后的文档列表
            
        Raises:
            FileProcessingError: 文件处理失败
        """
        try:
            # 验证文件格式
            self.validate_file_format(file_path)
            
            path = Path(file_path)
            file_extension = path.suffix.lower()
            
            logger.info(f"Processing file: {file_path} (format: {file_extension})")
            
            # 根据文件类型选择加载器
            documents = []
            
            if file_extension == '.pdf':
                documents = self._process_pdf(file_path)
            elif file_extension in ['.txt', '.md']:
                documents = self._process_text_file(file_path)
            elif file_extension == '.epub':
                documents = self._process_epub(file_path)
            else:
                raise FileProcessingError(f"未实现的文件格式处理: {file_extension}")
            
            # 添加文件元数据
            for doc in documents:
                doc.metadata.update({
                    'source_file': str(path.absolute()),
                    'file_name': path.name,
                    'file_extension': file_extension,
                    'file_size': path.stat().st_size,
                })
            
            logger.info(f"Successfully processed {len(documents)} documents from {file_path}")
            return documents
            
        except (ValidationError, FileProcessingError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {e}")
            raise FileProcessingError(f"文件处理失败: {str(e)}")
    
    def _process_pdf(self, file_path: str) -> List[Document]:
        """处理PDF文件"""
        try:
            loader = self.loaders['.pdf']
            documents = loader.load_data(file=file_path)
            
            if not documents:
                raise FileProcessingError("PDF文件解析结果为空")
            
            return documents
            
        except Exception as e:
            logger.error(f"PDF processing error for {file_path}: {e}")
            raise FileProcessingError(f"PDF文件处理失败: {str(e)}")
    
    def _process_text_file(self, file_path: str) -> List[Document]:
        """处理文本文件（TXT, MD）"""
        try:
            # 使用SimpleDirectoryReader处理单个文件
            loader = SimpleDirectoryReader(
                input_files=[file_path],
                recursive=False,
            )
            documents = loader.load_data()
            
            if not documents:
                raise FileProcessingError("文本文件解析结果为空")
            
            return documents
            
        except Exception as e:
            logger.error(f"Text file processing error for {file_path}: {e}")
            raise FileProcessingError(f"文本文件处理失败: {str(e)}")
    
    def _process_epub(self, file_path: str) -> List[Document]:
        """处理EPUB文件"""
        try:
            loader = self.loaders['.epub']
            documents = loader.load_data(file=file_path)
            
            if not documents:
                raise FileProcessingError("EPUB文件解析结果为空")
            
            return documents
            
        except Exception as e:
            logger.error(f"EPUB processing error for {file_path}: {e}")
            raise FileProcessingError(f"EPUB文件处理失败: {str(e)}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档进行分块处理
        
        Args:
            documents: 原始文档列表
            
        Returns:
            List[Document]: 分块后的文档列表
            
        Raises:
            FileProcessingError: 分块处理失败
        """
        try:
            if not documents:
                return []
            
            logger.info(f"Chunking {len(documents)} documents")
            
            # 使用文本分割器进行分块
            nodes = self.text_splitter.get_nodes_from_documents(documents)
            
            # 将节点转换回文档格式
            chunked_documents = []
            for i, node in enumerate(nodes):
                # 创建新的文档对象
                doc = Document(
                    text=node.text,
                    metadata=node.metadata.copy(),
                )
                
                # 添加分块信息
                doc.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(node.text),
                    'total_chunks': len(nodes),
                })
                
                # 如果有节点ID，也添加到元数据中
                if hasattr(node, 'node_id') and node.node_id:
                    doc.metadata['node_id'] = node.node_id
                
                chunked_documents.append(doc)
            
            logger.info(f"Successfully chunked into {len(chunked_documents)} chunks")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Document chunking error: {e}")
            raise FileProcessingError(f"文档分块失败: {str(e)}")
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档进行预处理
        
        Args:
            documents: 原始文档列表
            
        Returns:
            List[Document]: 预处理后的文档列表
            
        Raises:
            FileProcessingError: 预处理失败
        """
        try:
            if not documents:
                return []
            
            logger.info(f"Preprocessing {len(documents)} documents")
            
            processed_documents = []
            
            for doc in documents:
                # 清理文本内容
                cleaned_text = self._clean_text(doc.text)
                
                # 跳过空文档
                if not cleaned_text.strip():
                    logger.warning("Skipping empty document after cleaning")
                    continue
                
                # 创建新的文档对象
                processed_doc = Document(
                    text=cleaned_text,
                    metadata=doc.metadata.copy(),
                )
                
                # 添加预处理信息
                processed_doc.metadata.update({
                    'original_length': len(doc.text),
                    'processed_length': len(cleaned_text),
                    'preprocessing_applied': True,
                })
                
                processed_documents.append(processed_doc)
            
            logger.info(f"Successfully preprocessed {len(processed_documents)} documents")
            return processed_documents
            
        except Exception as e:
            logger.error(f"Document preprocessing error: {e}")
            raise FileProcessingError(f"文档预处理失败: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        cleaned = " ".join(text.split())
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()\[\]{}\"\'\-！？。、，；：“”‘’]', ' ', cleaned)
        
        # 再次清理多余空格
        cleaned = " ".join(cleaned.split())
        
        return cleaned.strip()
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """
        批量处理多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            List[Document]: 所有文件的文档列表
            
        Raises:
            FileProcessingError: 批量处理失败
        """
        try:
            if not file_paths:
                return []
            
            logger.info(f"Processing {len(file_paths)} files")
            
            all_documents = []
            failed_files = []
            
            for file_path in file_paths:
                try:
                    documents = self.process_file(file_path)
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    failed_files.append((file_path, str(e)))
            
            # 如果有文件处理失败，记录警告
            if failed_files:
                failed_list = "\n".join([f"  - {path}: {error}" for path, error in failed_files])
                logger.warning(f"Failed to process {len(failed_files)} files:\n{failed_list}")
            
            # 如果所有文件都失败了，抛出异常
            if not all_documents and file_paths:
                raise FileProcessingError("所有文件处理都失败了")
            
            logger.info(f"Successfully processed {len(all_documents)} documents from {len(file_paths) - len(failed_files)} files")
            return all_documents
            
        except FileProcessingError:
            raise
        except Exception as e:
            logger.error(f"Batch file processing error: {e}")
            raise FileProcessingError(f"批量文件处理失败: {str(e)}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 文件信息字典
            
        Raises:
            ValidationError: 文件不存在或无法访问
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise ValidationError(f"文件不存在: {file_path}")
            
            stat = path.stat()
            mime_type, _ = mimetypes.guess_type(file_path)
            
            return {
                'path': str(path.absolute()),
                'name': path.name,
                'extension': path.suffix.lower(),
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'mime_type': mime_type,
                'is_supported': path.suffix.lower() in self.config.supported_file_extensions,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            raise ValidationError(f"获取文件信息失败: {str(e)}")