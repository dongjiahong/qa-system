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
from .chinese_text_processor import get_chinese_text_processor
from .metadata_extractors import get_metadata_extractor_manager, ExtractorType


class DocumentProcessor:
    """
    文档处理器类，负责解析不同格式的文档并进行预处理
    """
    
    def __init__(self, enable_metadata_extraction: bool = True):
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
        
        # 初始化中文文本处理器
        self.chinese_processor = get_chinese_text_processor()
        
        # 初始化元数据抽取器
        self.enable_metadata_extraction = enable_metadata_extraction
        if enable_metadata_extraction:
            self.metadata_manager = get_metadata_extractor_manager()
        else:
            self.metadata_manager = None
        
        logger.info(f"DocumentProcessor initialized with Chinese text optimization and metadata extraction: {enable_metadata_extraction}")
    
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
            
            # 如果启用了元数据抽取，为文档添加元数据
            if self.enable_metadata_extraction and self.metadata_manager:
                documents = self._enrich_documents_with_metadata(documents)
            
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
        对文档进行分块处理，优化中文文本分块
        
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
            
            logger.info(f"Chunking {len(documents)} documents with Chinese optimization")
            
            chunked_documents = []
            chunk_id = 0
            
            for doc in documents:
                # 分析文本的中文特征
                text_stats = self.chinese_processor.analyze_text(doc.text)
                
                # 根据中文比例选择分块策略
                if text_stats.chinese_ratio > 0.5:
                    # 使用优化的中文分块
                    chunks = self.chinese_processor.smart_chunk_chinese_text(
                        doc.text, 
                        chunk_size=1000, 
                        overlap=200
                    )
                    logger.debug(f"Used Chinese chunking for document with {text_stats.chinese_ratio:.2f} Chinese ratio")
                else:
                    # 使用标准分块
                    nodes = self.text_splitter.get_nodes_from_documents([doc])
                    chunks = [node.text for node in nodes]
                    logger.debug(f"Used standard chunking for document with {text_stats.chinese_ratio:.2f} Chinese ratio")
                
                # 创建分块文档
                for chunk_text in chunks:
                    chunk_doc = Document(
                        text=chunk_text,
                        metadata=doc.metadata.copy(),
                    )
                    
                    # 添加分块信息
                    chunk_doc.metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_size': len(chunk_text),
                        'total_chunks': len(chunks),
                        'chinese_ratio': text_stats.chinese_ratio,
                        'chunking_method': 'chinese_optimized' if text_stats.chinese_ratio > 0.5 else 'standard'
                    })
                    
                    chunked_documents.append(chunk_doc)
                    chunk_id += 1
            
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
        批量处理多个文件，支持并行处理优化
        
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
            
            logger.info(f"Processing {len(file_paths)} files with performance optimization")
            
            all_documents = []
            failed_files = []
            
            # 对于大量文件，使用并行处理
            if len(file_paths) > 5:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_path = {
                        executor.submit(self.process_file, file_path): file_path 
                        for file_path in file_paths
                    }
                    
                    for future in as_completed(future_to_path):
                        file_path = future_to_path[future]
                        try:
                            documents = future.result()
                            all_documents.extend(documents)
                        except Exception as e:
                            logger.error(f"Failed to process file {file_path}: {e}")
                            failed_files.append((file_path, str(e)))
            else:
                # 对于少量文件，使用串行处理
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
    
    async def async_process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """
        异步批量处理多个文件
        
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
            
            logger.info(f"Async processing {len(file_paths)} files")
            
            import asyncio
            
            async def process_single_file(file_path: str) -> List[Document]:
                """异步处理单个文件"""
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.process_file, file_path)
            
            # 创建异步任务
            tasks = [process_single_file(file_path) for file_path in file_paths]
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果和异常
            all_documents = []
            failed_files = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process file {file_paths[i]}: {str(result)}")
                    failed_files.append((file_paths[i], str(result)))
                else:
                    all_documents.extend(result)
            
            # 如果有文件处理失败，记录警告
            if failed_files:
                failed_list = "\n".join([f"  - {path}: {error}" for path, error in failed_files])
                logger.warning(f"Failed to process {len(failed_files)} files:\n{failed_list}")
            
            # 如果所有文件都失败了，抛出异常
            if not all_documents and file_paths:
                raise FileProcessingError("所有文件处理都失败了")
            
            logger.info(f"Successfully async processed {len(all_documents)} documents from {len(file_paths) - len(failed_files)} files")
            return all_documents
            
        except FileProcessingError:
            raise
        except Exception as e:
            logger.error(f"Async batch file processing error: {e}")
            raise FileProcessingError(f"异步批量文件处理失败: {str(e)}")
    
    def get_processing_performance_stats(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        获取文件处理性能统计
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        import time
        
        start_time = time.time()
        
        try:
            documents = self.process_multiple_files(file_paths)
            processing_time = time.time() - start_time
            
            # 统计文档信息
            total_chars = sum(len(doc.text) for doc in documents)
            total_chunks = len(documents)
            
            # 分析中文内容比例
            chinese_docs = 0
            total_chinese_chars = 0
            
            for doc in documents:
                stats = self.chinese_processor.analyze_text(doc.text)
                if stats.chinese_ratio > 0.5:
                    chinese_docs += 1
                total_chinese_chars += stats.chinese_chars
            
            return {
                "total_files": len(file_paths),
                "total_documents": len(documents),
                "total_chunks": total_chunks,
                "total_chars": total_chars,
                "chinese_documents": chinese_docs,
                "chinese_ratio": total_chinese_chars / total_chars if total_chars > 0 else 0.0,
                "processing_time": processing_time,
                "chars_per_second": total_chars / processing_time if processing_time > 0 else 0.0,
                "files_per_second": len(file_paths) / processing_time if processing_time > 0 else 0.0
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "total_files": len(file_paths),
                "error": str(e),
                "processing_time": processing_time,
                "success": False
            }
    
    def _enrich_documents_with_metadata(self, documents: List[Document]) -> List[Document]:
        """为文档添加抽取的元数据"""
        try:
            if not documents or not self.metadata_manager:
                return documents
            
            logger.info(f"Enriching {len(documents)} documents with metadata")
            
            enriched_documents = []
            
            for doc in documents:
                try:
                    # 抽取元数据
                    metadata_results = self.metadata_manager.extract_all_metadata(doc)
                    
                    # 创建增强的文档
                    enriched_doc = Document(
                        text=doc.text,
                        metadata=doc.metadata.copy()
                    )
                    
                    # 添加抽取的元数据
                    for extractor_type, extracted_metadata in metadata_results.items():
                        if extracted_metadata.confidence > 0.3:  # 只保留高置信度的元数据
                            prefix = extractor_type.value
                            
                            # 添加元数据内容
                            enriched_doc.metadata[prefix] = extracted_metadata.content
                            enriched_doc.metadata[f"{prefix}_confidence"] = extracted_metadata.confidence
                            
                            # 特殊处理问答对
                            if extractor_type == ExtractorType.QUESTIONS_ANSWERED:
                                qa_pairs = extracted_metadata.content.get('qa_pairs', [])
                                if qa_pairs:
                                    enriched_doc.metadata['has_qa_content'] = True
                                    enriched_doc.metadata['qa_count'] = len(qa_pairs)
                            
                            # 特殊处理关键概念
                            elif extractor_type == ExtractorType.KEY_CONCEPTS:
                                concepts = extracted_metadata.content.get('concepts', [])
                                if concepts:
                                    enriched_doc.metadata['has_key_concepts'] = True
                                    enriched_doc.metadata['concept_count'] = len(concepts)
                                    enriched_doc.metadata['concept_names'] = [
                                        c.get('name', '') for c in concepts[:5] if c.get('name')
                                    ]
                    
                    # 添加元数据抽取摘要
                    summary = self.metadata_manager.get_extraction_summary(metadata_results)
                    enriched_doc.metadata['metadata_extraction_summary'] = summary
                    
                    enriched_documents.append(enriched_doc)
                    
                except Exception as e:
                    logger.warning(f"Failed to enrich document with metadata: {str(e)}")
                    # 如果元数据抽取失败，仍然保留原文档
                    enriched_documents.append(doc)
            
            logger.info(f"Successfully enriched {len(enriched_documents)} documents with metadata")
            return enriched_documents
            
        except Exception as e:
            logger.error(f"Document metadata enrichment failed: {str(e)}")
            # 如果整个过程失败，返回原文档
            return documents
    
    def extract_metadata_from_documents(
        self, 
        documents: List[Document], 
        extractor_types: Optional[List[ExtractorType]] = None
    ) -> Dict[str, Any]:
        """从文档集合中抽取元数据统计"""
        try:
            if not documents or not self.metadata_manager:
                return {}
            
            logger.info(f"Extracting metadata statistics from {len(documents)} documents")
            
            stats = {
                "total_documents": len(documents),
                "documents_with_qa": 0,
                "documents_with_concepts": 0,
                "total_qa_pairs": 0,
                "total_concepts": 0,
                "average_qa_confidence": 0.0,
                "average_concept_confidence": 0.0,
                "top_concepts": [],
                "sample_questions": []
            }
            
            qa_confidences = []
            concept_confidences = []
            all_concepts = []
            all_questions = []
            
            for doc in documents:
                # 检查是否已有元数据
                if 'questions_answered' in doc.metadata:
                    stats["documents_with_qa"] += 1
                    qa_data = doc.metadata['questions_answered']
                    qa_pairs = qa_data.get('qa_pairs', [])
                    stats["total_qa_pairs"] += len(qa_pairs)
                    
                    if 'questions_answered_confidence' in doc.metadata:
                        qa_confidences.append(doc.metadata['questions_answered_confidence'])
                    
                    # 收集问题样本
                    for pair in qa_pairs[:2]:  # 每个文档最多取2个问题
                        if pair.get('question'):
                            all_questions.append(pair['question'])
                
                if 'key_concepts' in doc.metadata:
                    stats["documents_with_concepts"] += 1
                    concept_data = doc.metadata['key_concepts']
                    concepts = concept_data.get('concepts', [])
                    stats["total_concepts"] += len(concepts)
                    
                    if 'key_concepts_confidence' in doc.metadata:
                        concept_confidences.append(doc.metadata['key_concepts_confidence'])
                    
                    # 收集概念
                    for concept in concepts:
                        if concept.get('name'):
                            all_concepts.append(concept['name'])
            
            # 计算平均置信度
            if qa_confidences:
                stats["average_qa_confidence"] = sum(qa_confidences) / len(qa_confidences)
            
            if concept_confidences:
                stats["average_concept_confidence"] = sum(concept_confidences) / len(concept_confidences)
            
            # 统计最常见的概念
            from collections import Counter
            concept_counts = Counter(all_concepts)
            stats["top_concepts"] = [
                {"name": concept, "count": count} 
                for concept, count in concept_counts.most_common(10)
            ]
            
            # 随机选择问题样本
            import random
            if all_questions:
                sample_size = min(5, len(all_questions))
                stats["sample_questions"] = random.sample(all_questions, sample_size)
            
            logger.info(f"Metadata statistics: {stats['documents_with_qa']} docs with Q&A, {stats['documents_with_concepts']} docs with concepts")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to extract metadata statistics: {str(e)}")
            return {"error": str(e)}