"""
Comprehensive Integration Tests for Knowledge QA System
知识库问答系统综合集成测试

This module contains end-to-end integration tests that verify the complete
functionality of the knowledge QA system, including:
- Knowledge base creation and management
- Document processing and vectorization
- Question generation and answer evaluation
- History management and retrieval
- CLI interface and user interactions
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

# Import CLI commands
from src.cli import (
    main, create_knowledge_base, list_knowledge_bases, 
    delete_knowledge_base, start_new_review, show_history
)

# Import core components
from src.knowledge_base_manager import KnowledgeBaseManager
from src.question_generator import QuestionGenerator
from src.answer_evaluator import AnswerEvaluator
from src.history_manager import HistoryManager
from src.models import KnowledgeBase, Question, EvaluationResult, QARecord


class TestEndToEndWorkflow:
    """端到端工作流程测试"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        temp_dir = tempfile.mkdtemp()
        original_data_dir = os.environ.get('KNOWLEDGE_DATA_DIR')
        os.environ['KNOWLEDGE_DATA_DIR'] = temp_dir
        
        yield temp_dir
        
        # 清理
        if original_data_dir:
            os.environ['KNOWLEDGE_DATA_DIR'] = original_data_dir
        else:
            os.environ.pop('KNOWLEDGE_DATA_DIR', None)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_documents(self, temp_data_dir):
        """创建示例文档"""
        docs_dir = Path(temp_data_dir) / "sample_docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Python基础文档
        python_doc = docs_dir / "python_basics.md"
        python_doc.write_text("""
# Python编程基础

## 什么是Python？
Python是一种高级编程语言，由Guido van Rossum在1989年发明。

## Python的特点
1. 简洁易读：Python的语法接近自然语言
2. 跨平台：可以在多种操作系统上运行
3. 丰富的库：拥有大量的第三方库

## 基本语法
```python
# 变量定义
name = "Python"
version = 3.12

# 函数定义
def greet(name):
    return f"Hello, {name}!"
```

## 数据结构
- 列表（List）：有序的可变序列
- 字典（Dictionary）：键值对映射
- 元组（Tuple）：有序的不可变序列
""", encoding='utf-8')
        
        # 机器学习文档
        ml_doc = docs_dir / "machine_learning.md"
        ml_doc.write_text("""
# 机器学习入门

## 什么是机器学习？
机器学习是人工智能的一个分支，使计算机能够从数据中学习。

## 机器学习类型
1. 监督学习：使用标记数据进行训练
2. 无监督学习：从未标记数据中发现模式
3. 强化学习：通过与环境交互学习

## 常用算法
- 线性回归：用于回归问题
- 决策树：可用于分类和回归
- 随机森林：集成学习方法
- 神经网络：深度学习基础

## Python库
- Scikit-learn：通用机器学习库
- TensorFlow：深度学习框架
- Pandas：数据处理
""", encoding='utf-8')
        
        return [str(python_doc), str(ml_doc)]
    
    @pytest.fixture
    def runner(self):
        """CLI测试运行器"""
        return CliRunner()
    
    def test_complete_workflow_success(self, temp_data_dir, sample_documents, runner):
        """测试完整的成功工作流程"""
        kb_name = "test_knowledge_base"
        
        with patch('src.vector_store.VectorStore') as mock_vector_store, \
             patch('src.llm_client.OllamaClient') as mock_llm_client:
            
            # 配置模拟对象
            mock_vs_instance = MagicMock()
            mock_vector_store.return_value = mock_vs_instance
            mock_vs_instance.create_collection.return_value = None
            mock_vs_instance.add_documents.return_value = None
            mock_vs_instance.similarity_search.return_value = [
                MagicMock(content="Python是一种编程语言", metadata={"source": "python_basics.md"})
            ]
            
            mock_llm_instance = MagicMock()
            mock_llm_client.return_value = mock_llm_instance
            mock_llm_instance.health_check.return_value = True
            mock_llm_instance.generate_question.return_value = "什么是Python？"
            mock_llm_instance.evaluate_answer.return_value = {
                "is_correct": True,
                "score": 85,
                "feedback": "回答正确且完整",
                "missing_points": [],
                "reference_answer": "Python是一种高级编程语言"
            }
            
            # 1. 创建知识库
            result = runner.invoke(create_knowledge_base, [
                '--name', kb_name,
                '--file', sample_documents[0],
                '--file', sample_documents[1]
            ])
            
            assert result.exit_code == 0
            assert "创建成功" in result.output or "成功创建" in result.output
            
            # 2. 列出知识库
            result = runner.invoke(list_knowledge_bases)
            assert result.exit_code == 0
            assert kb_name in result.output
            
            # 3. 模拟问答会话
            with patch('builtins.input', side_effect=['Python是一种编程语言', 'quit']):
                result = runner.invoke(start_new_review, [kb_name])
                assert result.exit_code == 0
            
            # 4. 查看历史记录
            result = runner.invoke(show_history, [kb_name])
            assert result.exit_code == 0
            
            # 5. 删除知识库
            with patch('builtins.input', return_value='y'):
                result = runner.invoke(delete_knowledge_base, [kb_name])
                assert result.exit_code == 0
    
    def test_error_handling_workflow(self, temp_data_dir, runner):
        """测试错误处理工作流程"""
        
        # 测试创建知识库时文件不存在
        result = runner.invoke(create_knowledge_base, [
            '--name', 'test_kb',
            '--file', '/nonexistent/file.txt'
        ])
        assert result.exit_code != 0
        assert "文件不存在" in result.output or "找不到文件" in result.output
        
        # 测试访问不存在的知识库
        result = runner.invoke(start_new_review, ['nonexistent_kb'])
        assert result.exit_code != 0
        assert "知识库不存在" in result.output or "未找到" in result.output
    
    def test_chinese_content_processing(self, temp_data_dir, sample_documents, runner):
        """测试中文内容处理"""
        kb_name = "chinese_test_kb"
        
        with patch('src.vector_store.VectorStore') as mock_vector_store, \
             patch('src.llm_client.OllamaClient') as mock_llm_client:
            
            # 配置模拟对象支持中文
            mock_vs_instance = MagicMock()
            mock_vector_store.return_value = mock_vs_instance
            mock_vs_instance.create_collection.return_value = None
            mock_vs_instance.add_documents.return_value = None
            mock_vs_instance.similarity_search.return_value = [
                MagicMock(content="Python是一种高级编程语言", metadata={"source": "test.md"})
            ]
            
            mock_llm_instance = MagicMock()
            mock_llm_client.return_value = mock_llm_instance
            mock_llm_instance.health_check.return_value = True
            mock_llm_instance.generate_question.return_value = "Python有哪些特点？"
            mock_llm_instance.evaluate_answer.return_value = {
                "is_correct": True,
                "score": 90,
                "feedback": "回答准确，涵盖了Python的主要特点",
                "missing_points": [],
                "reference_answer": "Python具有简洁易读、跨平台、丰富的库等特点"
            }
            
            # 创建知识库
            result = runner.invoke(create_knowledge_base, [
                '--name', kb_name,
                '--file', sample_documents[0]
            ])
            
            assert result.exit_code == 0
            
            # 测试中文问答
            with patch('builtins.input', side_effect=['Python具有简洁易读的语法', 'quit']):
                result = runner.invoke(start_new_review, [kb_name])
                assert result.exit_code == 0


class TestComponentIntegration:
    """组件集成测试"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        temp_dir = tempfile.mkdtemp()
        original_data_dir = os.environ.get('KNOWLEDGE_DATA_DIR')
        os.environ['KNOWLEDGE_DATA_DIR'] = temp_dir
        
        yield temp_dir
        
        # 清理
        if original_data_dir:
            os.environ['KNOWLEDGE_DATA_DIR'] = original_data_dir
        else:
            os.environ.pop('KNOWLEDGE_DATA_DIR', None)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_knowledge_base_manager_integration(self, temp_data_dir):
        """测试知识库管理器集成"""
        with patch('src.vector_store.VectorStore') as mock_vector_store, \
             patch('src.document_processor.DocumentProcessor') as mock_doc_processor:
            
            # 配置模拟对象
            mock_vs_instance = MagicMock()
            mock_vector_store.return_value = mock_vs_instance
            
            mock_dp_instance = MagicMock()
            mock_doc_processor.return_value = mock_dp_instance
            mock_dp_instance.process_multiple_files.return_value = [
                MagicMock(text="测试文档内容", metadata={"source": "test.md"})
            ]
            
            # 创建知识库管理器
            manager = KnowledgeBaseManager()
            
            # 创建知识库
            kb = manager.create_knowledge_base(
                name="integration_test",
                files=["test.md"],
                description="集成测试知识库"
            )
            
            assert kb.name == "integration_test"
            assert kb.file_count == 1
            
            # 验证调用
            mock_vs_instance.create_collection.assert_called_once()
            mock_dp_instance.process_multiple_files.assert_called_once()
    
    def test_question_answer_flow_integration(self, temp_data_dir):
        """测试问答流程集成"""
        with patch('src.vector_store.VectorStore') as mock_vector_store, \
             patch('src.llm_client.OllamaClient') as mock_llm_client:
            
            # 配置模拟对象
            mock_vs_instance = MagicMock()
            mock_vector_store.return_value = mock_vs_instance
            mock_vs_instance.similarity_search.return_value = [
                MagicMock(content="Python是编程语言", metadata={"source": "test.md"})
            ]
            
            mock_llm_instance = MagicMock()
            mock_llm_client.return_value = mock_llm_instance
            mock_llm_instance.generate_question.return_value = "什么是Python？"
            mock_llm_instance.evaluate_answer.return_value = {
                "is_correct": True,
                "score": 85,
                "feedback": "回答正确",
                "missing_points": [],
                "reference_answer": "Python是编程语言"
            }
            
            # 创建组件
            question_generator = QuestionGenerator()
            answer_evaluator = AnswerEvaluator()
            
            # 生成问题
            question = question_generator.generate_question("test_kb")
            assert question.content == "什么是Python？"
            
            # 评估答案
            evaluation = answer_evaluator.evaluate_answer(
                question="什么是Python？",
                user_answer="Python是一种编程语言",
                kb_name="test_kb"
            )
            
            assert evaluation.is_correct is True
            assert evaluation.score == 85
    
    def test_history_management_integration(self, temp_data_dir):
        """测试历史记录管理集成"""
        from datetime import datetime
        
        # 创建历史管理器
        history_manager = HistoryManager()
        
        # 创建测试记录
        evaluation = EvaluationResult(
            is_correct=True,
            score=90,
            feedback="回答正确",
            missing_points=[],
            reference_answer="参考答案"
        )
        
        record = QARecord(
            kb_name="test_kb",
            question="测试问题？",
            user_answer="测试答案",
            evaluation=evaluation,
            created_at=datetime.now()
        )
        
        # 保存记录
        history_manager.save_qa_record(record)
        
        # 检索记录
        history = history_manager.get_history_page("test_kb", page=1, limit=10)
        assert len(history["records"]) == 1
        assert history["records"][0].question == "测试问题？"
        
        # 获取统计信息
        stats = history_manager.get_statistics("test_kb")
        assert stats["total_questions"] == 1
        assert stats["correct_answers"] == 1


class TestPerformanceAndScalability:
    """性能和可扩展性测试"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        temp_dir = tempfile.mkdtemp()
        original_data_dir = os.environ.get('KNOWLEDGE_DATA_DIR')
        os.environ['KNOWLEDGE_DATA_DIR'] = temp_dir
        
        yield temp_dir
        
        # 清理
        if original_data_dir:
            os.environ['KNOWLEDGE_DATA_DIR'] = original_data_dir
        else:
            os.environ.pop('KNOWLEDGE_DATA_DIR', None)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_large_document_processing(self, temp_data_dir):
        """测试大文档处理性能"""
        import time
        
        # 创建大文档
        large_content = "这是测试内容。" * 10000  # 约150KB的中文内容
        large_doc = Path(temp_data_dir) / "large_doc.txt"
        large_doc.write_text(large_content, encoding='utf-8')
        
        with patch('src.vector_store.VectorStore') as mock_vector_store:
            mock_vs_instance = MagicMock()
            mock_vector_store.return_value = mock_vs_instance
            
            manager = KnowledgeBaseManager()
            
            # 测量处理时间
            start_time = time.time()
            kb = manager.create_knowledge_base(
                name="large_doc_test",
                files=[str(large_doc)],
                description="大文档测试"
            )
            processing_time = time.time() - start_time
            
            # 验证处理成功且时间合理（应该在30秒内完成）
            assert kb.name == "large_doc_test"
            assert processing_time < 30.0
    
    def test_multiple_knowledge_bases(self, temp_data_dir):
        """测试多知识库管理"""
        with patch('src.vector_store.VectorStore') as mock_vector_store:
            mock_vs_instance = MagicMock()
            mock_vector_store.return_value = mock_vs_instance
            
            manager = KnowledgeBaseManager()
            
            # 创建多个知识库
            kb_names = []
            for i in range(5):
                kb_name = f"test_kb_{i}"
                
                # 创建测试文档
                test_doc = Path(temp_data_dir) / f"doc_{i}.txt"
                test_doc.write_text(f"这是第{i}个测试文档的内容。", encoding='utf-8')
                
                kb = manager.create_knowledge_base(
                    name=kb_name,
                    files=[str(test_doc)],
                    description=f"测试知识库{i}"
                )
                
                kb_names.append(kb_name)
                assert kb.name == kb_name
            
            # 验证所有知识库都存在
            all_kbs = manager.list_knowledge_bases()
            for kb_name in kb_names:
                assert kb_name in all_kbs
    
    def test_concurrent_operations(self, temp_data_dir):
        """测试并发操作"""
        import threading
        import time
        
        with patch('src.vector_store.VectorStore') as mock_vector_store:
            mock_vs_instance = MagicMock()
            mock_vector_store.return_value = mock_vs_instance
            
            manager = KnowledgeBaseManager()
            results = []
            errors = []
            
            def create_kb(index):
                try:
                    # 创建测试文档
                    test_doc = Path(temp_data_dir) / f"concurrent_doc_{index}.txt"
                    test_doc.write_text(f"并发测试文档{index}", encoding='utf-8')
                    
                    kb = manager.create_knowledge_base(
                        name=f"concurrent_kb_{index}",
                        files=[str(test_doc)],
                        description=f"并发测试{index}"
                    )
                    results.append(kb.name)
                except Exception as e:
                    errors.append(str(e))
            
            # 创建多个线程同时创建知识库
            threads = []
            for i in range(3):
                thread = threading.Thread(target=create_kb, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=10)
            
            # 验证结果
            assert len(errors) == 0, f"并发操作出现错误: {errors}"
            assert len(results) == 3
            assert all(f"concurrent_kb_{i}" in results for i in range(3))


class TestErrorRecoveryAndResilience:
    """错误恢复和系统韧性测试"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        temp_dir = tempfile.mkdtemp()
        original_data_dir = os.environ.get('KNOWLEDGE_DATA_DIR')
        os.environ['KNOWLEDGE_DATA_DIR'] = temp_dir
        
        yield temp_dir
        
        # 清理
        if original_data_dir:
            os.environ['KNOWLEDGE_DATA_DIR'] = original_data_dir
        else:
            os.environ.pop('KNOWLEDGE_DATA_DIR', None)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_partial_failure_recovery(self, temp_data_dir):
        """测试部分失败后的恢复"""
        with patch('src.vector_store.VectorStore') as mock_vector_store, \
             patch('src.document_processor.DocumentProcessor') as mock_doc_processor:
            
            # 配置文档处理器部分失败
            mock_dp_instance = MagicMock()
            mock_doc_processor.return_value = mock_dp_instance
            mock_dp_instance.process_multiple_files.side_effect = Exception("处理失败")
            
            mock_vs_instance = MagicMock()
            mock_vector_store.return_value = mock_vs_instance
            
            manager = KnowledgeBaseManager()
            
            # 创建测试文档
            test_doc = Path(temp_data_dir) / "test_doc.txt"
            test_doc.write_text("测试内容", encoding='utf-8')
            
            # 尝试创建知识库（应该失败）
            with pytest.raises(Exception):
                manager.create_knowledge_base(
                    name="failure_test",
                    files=[str(test_doc)],
                    description="失败测试"
                )
            
            # 验证没有创建不完整的知识库
            all_kbs = manager.list_knowledge_bases()
            assert "failure_test" not in all_kbs
    
    def test_service_unavailable_handling(self, temp_data_dir, runner):
        """测试服务不可用时的处理"""
        
        # 模拟Ollama服务不可用
        with patch('src.llm_client.OllamaClient') as mock_llm_client:
            mock_llm_instance = MagicMock()
            mock_llm_client.return_value = mock_llm_instance
            mock_llm_instance.health_check.return_value = False
            
            # 尝试开始问答会话
            result = runner.invoke(start_new_review, ['test_kb'])
            
            # 应该优雅地处理服务不可用的情况
            assert result.exit_code != 0
            assert "服务" in result.output or "连接" in result.output
    
    def test_data_corruption_handling(self, temp_data_dir):
        """测试数据损坏时的处理"""
        with patch('src.database.SQLiteDatabase') as mock_db:
            # 模拟数据库损坏
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance
            mock_db_instance.execute.side_effect = Exception("数据库损坏")
            
            manager = KnowledgeBaseManager()
            
            # 尝试列出知识库（应该处理数据库错误）
            with pytest.raises(Exception):
                manager.list_knowledge_bases()


if __name__ == "__main__":
    # 运行集成测试
    pytest.main([__file__, "-v", "--tb=short"])