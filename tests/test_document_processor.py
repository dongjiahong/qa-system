"""
Unit tests for document processor module
文档处理器模块的单元测试
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from llama_index.core import Document

from src.document_processor import DocumentProcessor
from src.models import FileProcessingError, ValidationError


class TestDocumentProcessor:
    """DocumentProcessor 类的测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.processor = DocumentProcessor()
    
    def test_init(self):
        """测试初始化"""
        assert self.processor is not None
        assert hasattr(self.processor, 'loaders')
        assert hasattr(self.processor, 'text_splitter')
        assert '.pdf' in self.processor.loaders
        assert '.txt' in self.processor.loaders
        assert '.md' in self.processor.loaders
        assert '.epub' in self.processor.loaders
    
    def test_validate_file_format_success(self):
        """测试文件格式验证成功"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'test content')
            tmp_path = tmp.name
        
        try:
            result = self.processor.validate_file_format(tmp_path)
            assert result is True
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_format_file_not_exists(self):
        """测试文件不存在的情况"""
        with pytest.raises(ValidationError, match="文件不存在"):
            self.processor.validate_file_format("nonexistent_file.txt")
    
    def test_validate_file_format_unsupported_extension(self):
        """测试不支持的文件格式"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
            tmp.write(b'test content')
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValidationError, match="不支持的文件格式"):
                self.processor.validate_file_format(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_format_file_too_large(self):
        """测试文件过大的情况"""
        # 模拟大文件
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 模拟文件大小
            mock_stat = Mock()
            mock_stat.st_mode = 0o100644  # 普通文件模式
            mock_stat.st_size = 200 * 1024 * 1024  # 200MB
                
            with patch('pathlib.Path.stat', return_value=mock_stat):
                with pytest.raises(ValidationError, match="文件过大"):
                    self.processor.validate_file_format(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_format_directory(self):
        """测试路径是目录的情况"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValidationError, match="路径不是文件"):
                self.processor.validate_file_format(tmp_dir)
    
    @patch('src.document_processor.DocumentProcessor._process_text_file')
    def test_process_file_txt_success(self, mock_process_text):
        """测试处理TXT文件成功"""
        # 创建临时TXT文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write('测试内容')
            tmp_path = tmp.name
        
        try:
            # 模拟返回值
            mock_doc = Document(text='测试内容', metadata={})
            mock_process_text.return_value = [mock_doc]
            
            result = self.processor.process_file(tmp_path)
            
            assert len(result) == 1
            assert result[0].text == '测试内容'
            assert 'source_file' in result[0].metadata
            assert 'file_name' in result[0].metadata
            assert 'file_extension' in result[0].metadata
            assert result[0].metadata['file_extension'] == '.txt'
            
            mock_process_text.assert_called_once_with(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @patch('src.document_processor.DocumentProcessor._process_pdf')
    def test_process_file_pdf_success(self, mock_process_pdf):
        """测试处理PDF文件成功"""
        # 创建临时PDF文件
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'fake pdf content')
            tmp_path = tmp.name
        
        try:
            # 模拟返回值
            mock_doc = Document(text='PDF内容', metadata={})
            mock_process_pdf.return_value = [mock_doc]
            
            result = self.processor.process_file(tmp_path)
            
            assert len(result) == 1
            assert result[0].text == 'PDF内容'
            assert result[0].metadata['file_extension'] == '.pdf'
            
            mock_process_pdf.assert_called_once_with(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @patch('src.document_processor.DocumentProcessor._process_epub')
    def test_process_file_epub_success(self, mock_process_epub):
        """测试处理EPUB文件成功"""
        # 创建临时EPUB文件
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as tmp:
            tmp.write(b'fake epub content')
            tmp_path = tmp.name
        
        try:
            # 模拟返回值
            mock_doc = Document(text='EPUB内容', metadata={})
            mock_process_epub.return_value = [mock_doc]
            
            result = self.processor.process_file(tmp_path)
            
            assert len(result) == 1
            assert result[0].text == 'EPUB内容'
            assert result[0].metadata['file_extension'] == '.epub'
            
            mock_process_epub.assert_called_once_with(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_process_file_validation_error(self):
        """测试文件验证失败"""
        with pytest.raises(ValidationError):
            self.processor.process_file("nonexistent_file.txt")
    
    @patch('src.document_processor.SimpleDirectoryReader')
    def test_process_text_file_success(self, mock_reader_class):
        """测试处理文本文件成功"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write('测试文本内容')
            tmp_path = tmp.name
        
        try:
            # 模拟SimpleDirectoryReader
            mock_reader = Mock()
            mock_reader.load_data.return_value = [Document(text='测试文本内容', metadata={})]
            mock_reader_class.return_value = mock_reader
            
            result = self.processor._process_text_file(tmp_path)
            
            assert len(result) == 1
            assert result[0].text == '测试文本内容'
            
            mock_reader_class.assert_called_once_with(
                input_files=[tmp_path],
                recursive=False,
            )
            mock_reader.load_data.assert_called_once()
        finally:
            os.unlink(tmp_path)
    
    @patch('src.document_processor.SimpleDirectoryReader')
    def test_process_text_file_empty_result(self, mock_reader_class):
        """测试文本文件解析结果为空"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 模拟空结果
            mock_reader = Mock()
            mock_reader.load_data.return_value = []
            mock_reader_class.return_value = mock_reader
            
            with pytest.raises(FileProcessingError, match="文本文件解析结果为空"):
                self.processor._process_text_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_process_pdf_success(self):
        """测试处理PDF文件成功"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 模拟PDF加载器
            mock_loader = Mock()
            mock_loader.load_data.return_value = [Document(text='PDF内容', metadata={})]
            self.processor.loaders['.pdf'] = mock_loader
            
            result = self.processor._process_pdf(tmp_path)
            
            assert len(result) == 1
            assert result[0].text == 'PDF内容'
            
            mock_loader.load_data.assert_called_once_with(file=tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_process_pdf_empty_result(self):
        """测试PDF文件解析结果为空"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 模拟空结果
            mock_loader = Mock()
            mock_loader.load_data.return_value = []
            self.processor.loaders['.pdf'] = mock_loader
            
            with pytest.raises(FileProcessingError, match="PDF文件解析结果为空"):
                self.processor._process_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_process_epub_success(self):
        """测试处理EPUB文件成功"""
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 模拟EPUB加载器
            mock_loader = Mock()
            mock_loader.load_data.return_value = [Document(text='EPUB内容', metadata={})]
            self.processor.loaders['.epub'] = mock_loader
            
            result = self.processor._process_epub(tmp_path)
            
            assert len(result) == 1
            assert result[0].text == 'EPUB内容'
            
            mock_loader.load_data.assert_called_once_with(file=tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_process_epub_empty_result(self):
        """测试EPUB文件解析结果为空"""
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 模拟空结果
            mock_loader = Mock()
            mock_loader.load_data.return_value = []
            self.processor.loaders['.epub'] = mock_loader
            
            with pytest.raises(FileProcessingError, match="EPUB文件解析结果为空"):
                self.processor._process_epub(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_chunk_documents_success(self):
        """测试文档分块成功"""
        # 创建测试文档
        documents = [
            Document(text='这是第一个文档的内容，包含一些测试文本。' * 50, metadata={'source': 'test1'}),
            Document(text='这是第二个文档的内容，也包含测试文本。' * 50, metadata={'source': 'test2'}),
        ]
        
        # 模拟节点
        mock_node1 = Mock()
        mock_node1.text = '分块1内容'
        mock_node1.metadata = {'source': 'test1', 'chunk': 1}
        mock_node1.node_id = 'node1'
        
        mock_node2 = Mock()
        mock_node2.text = '分块2内容'
        mock_node2.metadata = {'source': 'test1', 'chunk': 2}
        mock_node2.node_id = 'node2'
        
        with patch('src.document_processor.SentenceSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter.get_nodes_from_documents.return_value = [mock_node1, mock_node2]
            mock_splitter_class.return_value = mock_splitter
            
            # Create a new processor with mocked splitter
            processor = DocumentProcessor()
            result = processor.chunk_documents(documents)
            
            assert len(result) == 2
            assert result[0].text == '分块1内容'
            assert result[1].text == '分块2内容'
            assert result[0].metadata['chunk_id'] == 0
            assert result[1].metadata['chunk_id'] == 1
            assert result[0].metadata['total_chunks'] == 2
            assert result[1].metadata['total_chunks'] == 2
            assert result[0].metadata['node_id'] == 'node1'
            assert result[1].metadata['node_id'] == 'node2'
    
    def test_chunk_documents_empty_input(self):
        """测试空文档列表分块"""
        result = self.processor.chunk_documents([])
        assert result == []
    
    def test_preprocess_documents_success(self):
        """测试文档预处理成功"""
        documents = [
            Document(text='  这是测试文本  \n\n  包含多余空格  ', metadata={'source': 'test1'}),
            Document(text='另一个测试文档！@#$%^&*()', metadata={'source': 'test2'}),
        ]
        
        result = self.processor.preprocess_documents(documents)
        
        assert len(result) == 2
        assert result[0].text.strip() == '这是测试文本 包含多余空格'
        assert 'original_length' in result[0].metadata
        assert 'processed_length' in result[0].metadata
        assert 'preprocessing_applied' in result[0].metadata
        assert result[0].metadata['preprocessing_applied'] is True
    
    def test_preprocess_documents_empty_input(self):
        """测试空文档列表预处理"""
        result = self.processor.preprocess_documents([])
        assert result == []
    
    def test_preprocess_documents_skip_empty(self):
        """测试跳过空文档"""
        documents = [
            Document(text='有效内容', metadata={'source': 'test1'}),
            Document(text='   \n\n   ', metadata={'source': 'test2'}),  # 空文档
            Document(text='另一个有效内容', metadata={'source': 'test3'}),
        ]
        
        result = self.processor.preprocess_documents(documents)
        
        # 应该跳过空文档
        assert len(result) == 2
        assert result[0].text == '有效内容'
        assert result[1].text == '另一个有效内容'
    
    def test_clean_text(self):
        """测试文本清理功能"""
        # 测试多余空格
        result = self.processor._clean_text('  多余   空格  测试  ')
        assert result == '多余 空格 测试'
        
        # 测试特殊字符
        result = self.processor._clean_text('测试@#$%文本！？。')
        assert result == '测试 文本！？。'
        
        # 测试空字符串
        result = self.processor._clean_text('')
        assert result == ''
        
        # 测试None
        result = self.processor._clean_text(None)
        assert result == ''
    
    @patch('src.document_processor.DocumentProcessor.process_file')
    def test_process_multiple_files_success(self, mock_process_file):
        """测试批量处理文件成功"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp2:
            tmp1_path = tmp1.name
            tmp2_path = tmp2.name
        
        try:
            # 模拟处理结果
            mock_process_file.side_effect = [
                [Document(text='文档1', metadata={})],
                [Document(text='文档2', metadata={})],
            ]
            
            result = self.processor.process_multiple_files([tmp1_path, tmp2_path])
            
            assert len(result) == 2
            assert result[0].text == '文档1'
            assert result[1].text == '文档2'
            assert mock_process_file.call_count == 2
        finally:
            os.unlink(tmp1_path)
            os.unlink(tmp2_path)
    
    @patch('src.document_processor.DocumentProcessor.process_file')
    def test_process_multiple_files_partial_failure(self, mock_process_file):
        """测试批量处理部分失败"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp2:
            tmp1_path = tmp1.name
            tmp2_path = tmp2.name
        
        try:
            # 模拟部分失败
            mock_process_file.side_effect = [
                [Document(text='文档1', metadata={})],
                FileProcessingError('处理失败'),
            ]
            
            result = self.processor.process_multiple_files([tmp1_path, tmp2_path])
            
            # 应该返回成功处理的文档
            assert len(result) == 1
            assert result[0].text == '文档1'
        finally:
            os.unlink(tmp1_path)
            os.unlink(tmp2_path)
    
    @patch('src.document_processor.DocumentProcessor.process_file')
    def test_process_multiple_files_all_failed(self, mock_process_file):
        """测试批量处理全部失败"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # 模拟全部失败
            mock_process_file.side_effect = FileProcessingError('处理失败')
            
            with pytest.raises(FileProcessingError, match="所有文件处理都失败了"):
                self.processor.process_multiple_files([tmp_path])
        finally:
            os.unlink(tmp_path)
    
    def test_process_multiple_files_empty_input(self):
        """测试空文件列表"""
        result = self.processor.process_multiple_files([])
        assert result == []
    
    def test_get_file_info_success(self):
        """测试获取文件信息成功"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write('测试内容')
            tmp_path = tmp.name
        
        try:
            result = self.processor.get_file_info(tmp_path)
            
            assert 'path' in result
            assert 'name' in result
            assert 'extension' in result
            assert 'size_bytes' in result
            assert 'size_mb' in result
            assert 'mime_type' in result
            assert 'is_supported' in result
            assert 'created_time' in result
            assert 'modified_time' in result
            
            assert result['extension'] == '.txt'
            assert result['is_supported'] is True
            assert result['size_bytes'] > 0
        finally:
            os.unlink(tmp_path)
    
    def test_get_file_info_file_not_exists(self):
        """测试获取不存在文件的信息"""
        with pytest.raises(ValidationError, match="文件不存在"):
            self.processor.get_file_info("nonexistent_file.txt")


class TestDocumentProcessorIntegration:
    """DocumentProcessor 集成测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.processor = DocumentProcessor()
    
    def test_full_processing_pipeline_txt(self):
        """测试完整的TXT文件处理流程"""
        # 创建测试文件
        test_content = "这是一个测试文档。\n\n包含多行内容和一些特殊字符@#$%。\n\n还有更多内容用于测试分块功能。" * 20
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name
        
        try:
            # 1. 处理文件
            documents = self.processor.process_file(tmp_path)
            assert len(documents) > 0
            
            # 2. 预处理
            processed_docs = self.processor.preprocess_documents(documents)
            assert len(processed_docs) > 0
            
            # 3. 分块
            chunked_docs = self.processor.chunk_documents(processed_docs)
            assert len(chunked_docs) > 0
            
            # 验证元数据
            for doc in chunked_docs:
                assert 'source_file' in doc.metadata
                assert 'chunk_id' in doc.metadata
                assert 'preprocessing_applied' in doc.metadata
                
        finally:
            os.unlink(tmp_path)
    
    def test_batch_processing_mixed_formats(self):
        """测试混合格式的批量处理"""
        # 创建不同格式的测试文件
        files = []
        
        # TXT文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write('TXT文件内容')
            files.append(tmp.name)
        
        # MD文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
            tmp.write('# Markdown标题\n\n这是Markdown内容')
            files.append(tmp.name)
        
        try:
            # 批量处理
            documents = self.processor.process_multiple_files(files)
            assert len(documents) >= 2
            
            # 验证不同格式的文档都被处理
            extensions = {doc.metadata['file_extension'] for doc in documents}
            assert '.txt' in extensions
            assert '.md' in extensions
            
        finally:
            for file_path in files:
                os.unlink(file_path)