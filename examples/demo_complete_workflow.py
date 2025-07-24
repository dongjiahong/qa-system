#!/usr/bin/env python3
"""
Complete Workflow Demo for Knowledge QA System
知识库问答系统完整工作流程演示

This demo script showcases the complete functionality of the knowledge QA system,
including knowledge base creation, document processing, question generation,
answer evaluation, and history management.

Usage:
    python examples/demo_complete_workflow.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

def create_demo_documents():
    """创建演示文档"""
    print("📝 创建演示文档...")
    
    # 创建临时目录
    demo_dir = Path(tempfile.mkdtemp(prefix="knowledge_qa_demo_"))
    
    # Python基础文档
    python_doc = demo_dir / "python_basics.md"
    python_doc.write_text("""
# Python编程基础

## 什么是Python？
Python是一种高级编程语言，由Guido van Rossum在1989年发明。它具有简洁、易读的语法，广泛应用于Web开发、数据科学、人工智能等领域。

## Python的特点
1. **简洁易读**：Python的语法接近自然语言，容易理解和学习
2. **跨平台**：可以在Windows、macOS、Linux等多种操作系统上运行
3. **丰富的库**：拥有大量的第三方库和框架
4. **解释型语言**：无需编译，可以直接运行
5. **动态类型**：变量类型在运行时确定

## 基本语法示例

### 变量定义
```python
name = "Python"
version = 3.12
is_popular = True
```

### 函数定义
```python
def greet(name):
    return f"Hello, {name}!"

def calculate_area(length, width):
    return length * width
```

### 数据结构
- **列表（List）**：有序的可变序列
- **字典（Dictionary）**：键值对映射
- **元组（Tuple）**：有序的不可变序列
- **集合（Set）**：无序的唯一元素集合

### 控制流程
```python
# 条件语句
if version >= 3:
    print("使用Python 3")
else:
    print("使用Python 2")

# 循环语句
for i in range(5):
    print(f"数字: {i}")
```

## 面向对象编程
```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def study(self, subject):
        return f"{self.name}正在学习{subject}"
```
""", encoding='utf-8')
    
    # 机器学习文档
    ml_doc = demo_dir / "machine_learning.md"
    ml_doc.write_text("""
# 机器学习基础

## 什么是机器学习？
机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。

## 机器学习类型

### 1. 监督学习
- 使用标记的训练数据
- 包括分类和回归问题
- 常用算法：线性回归、决策树、随机森林

### 2. 无监督学习
- 从未标记数据中发现模式
- 包括聚类和降维
- 常用算法：K-means、PCA、DBSCAN

### 3. 强化学习
- 通过与环境交互学习
- 最大化累积奖励
- 应用：游戏AI、自动驾驶

## 机器学习工作流程
1. 问题定义
2. 数据收集与准备
3. 特征工程
4. 模型选择与训练
5. 模型评估
6. 模型部署

## Python机器学习库
- **Scikit-learn**：通用机器学习库
- **TensorFlow**：深度学习框架
- **PyTorch**：深度学习研究
- **Pandas**：数据处理
- **NumPy**：数值计算
""", encoding='utf-8')
    
    # 数据科学文档
    ds_doc = demo_dir / "data_science.txt"
    ds_doc.write_text("""
数据科学入门

数据科学是一个跨学科领域，结合了统计学、计算机科学和领域专业知识来从数据中提取洞察。

主要技能：
1. 编程技能（Python、R、SQL）
2. 统计学和数学基础
3. 数据可视化
4. 机器学习
5. 业务理解

数据科学流程：
1. 业务理解
2. 数据理解
3. 数据准备
4. 建模
5. 评估
6. 部署

常用工具：
- Jupyter Notebook：交互式开发环境
- Pandas：数据操作
- Matplotlib/Seaborn：数据可视化
- Scikit-learn：机器学习
- TensorFlow/PyTorch：深度学习

职业发展路径：
- 数据分析师
- 数据科学家
- 机器学习工程师
- 数据工程师
""", encoding='utf-8')
    
    return demo_dir, [python_doc, ml_doc, ds_doc]

def demonstrate_cli_commands():
    """演示CLI命令"""
    print("\n🖥️  CLI命令演示")
    print("=" * 50)
    
    demo_dir, docs = create_demo_documents()
    
    try:
        print("\n1. 创建知识库命令：")
        print(f"   knowledge new --name demo-kb --file {docs[0]} --file {docs[1]} --file {docs[2]}")
        
        print("\n2. 列出知识库命令：")
        print("   knowledge list")
        
        print("\n3. 开始问答命令：")
        print("   knowledge demo-kb review new")
        
        print("\n4. 查看历史命令：")
        print("   knowledge demo-kb review history")
        
        print("\n5. 删除知识库命令：")
        print("   knowledge delete demo-kb")
        
        print("\n6. 系统状态检查：")
        print("   knowledge status")
        
        print("\n7. 获取帮助：")
        print("   knowledge --help")
        print("   knowledge --quick-start")
        print("   knowledge --troubleshoot-all")
        
    finally:
        # 清理临时文件
        shutil.rmtree(demo_dir, ignore_errors=True)

def demonstrate_api_usage():
    """演示API使用"""
    print("\n🔧 API使用演示")
    print("=" * 50)
    
    try:
        from knowledge_base_manager import KnowledgeBaseManager
        from question_generator import QuestionGenerator
        from answer_evaluator import AnswerEvaluator
        from history_manager import HistoryManager
        
        print("\n1. 知识库管理器使用：")
        print("""
# 创建知识库管理器
manager = KnowledgeBaseManager()

# 创建知识库
kb = manager.create_knowledge_base(
    name="api-demo",
    files=["document.pdf", "notes.md"],
    description="API演示知识库"
)

# 列出知识库
all_kbs = manager.list_knowledge_bases()

# 搜索知识库
results = manager.search_knowledge_base("api-demo", "Python", k=5)
""")
        
        print("\n2. 问题生成器使用：")
        print("""
# 创建问题生成器
generator = QuestionGenerator()

# 生成问题
question = generator.generate_question("api-demo")
print(f"生成的问题: {question.content}")
""")
        
        print("\n3. 答案评估器使用：")
        print("""
# 创建答案评估器
evaluator = AnswerEvaluator()

# 评估答案
evaluation = evaluator.evaluate_answer(
    question="什么是Python？",
    user_answer="Python是一种编程语言",
    kb_name="api-demo"
)

print(f"评估结果: {evaluation.score}/100")
print(f"反馈: {evaluation.feedback}")
""")
        
        print("\n4. 历史管理器使用：")
        print("""
# 创建历史管理器
history = HistoryManager()

# 保存问答记录
record = QARecord(
    kb_name="api-demo",
    question="什么是Python？",
    user_answer="Python是编程语言",
    evaluation=evaluation
)
history.save_qa_record(record)

# 获取历史记录
history_page = history.get_history_page("api-demo", page=1, limit=10)
""")
        
    except ImportError as e:
        print(f"⚠️  无法导入模块: {e}")
        print("请确保已正确安装系统")

def demonstrate_use_cases():
    """演示使用场景"""
    print("\n📚 使用场景演示")
    print("=" * 50)
    
    scenarios = [
        {
            "title": "学生学习场景",
            "description": "学生使用系统学习编程课程",
            "steps": [
                "收集课程材料（PDF教材、课堂笔记、代码示例）",
                "创建知识库：knowledge new --name 'cs101' --file textbook.pdf --file notes.md",
                "每日学习：knowledge cs101 review new",
                "复习薄弱环节：knowledge cs101 review history --sort score",
                "导出学习记录：knowledge cs101 review history export --format csv"
            ]
        },
        {
            "title": "技术文档学习",
            "description": "开发者学习新技术的API文档",
            "steps": [
                "下载官方文档和教程",
                "创建技术知识库：knowledge new --name 'react-docs' --file api-ref.pdf --file tutorial.md",
                "针对性学习：knowledge react-docs review new",
                "团队分享：导出学习记录供团队参考"
            ]
        },
        {
            "title": "考试复习",
            "description": "准备专业考试的复习",
            "steps": [
                "整理考试材料",
                "创建复习知识库：knowledge new --name 'certification-exam' --file study-guide.pdf",
                "系统性复习：多轮问答练习",
                "查看进度：分析历史记录中的薄弱环节"
            ]
        },
        {
            "title": "企业培训",
            "description": "企业内部技能培训",
            "steps": [
                "准备培训材料",
                "创建培训知识库：knowledge new --name 'company-training' --file handbook.pdf",
                "员工自主学习：每人独立使用系统学习",
                "培训效果评估：分析学习记录和成绩"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['title']}")
        print(f"   描述: {scenario['description']}")
        print("   步骤:")
        for j, step in enumerate(scenario['steps'], 1):
            print(f"   {j}. {step}")

def demonstrate_best_practices():
    """演示最佳实践"""
    print("\n💡 最佳实践演示")
    print("=" * 50)
    
    practices = [
        {
            "category": "文档准备",
            "tips": [
                "使用清晰的文件命名：chapter1_intro.pdf, chapter2_basics.pdf",
                "保持文档格式一致，优先使用Markdown",
                "确保文档内容结构化，有清晰的标题和段落",
                "避免包含敏感信息或个人隐私"
            ]
        },
        {
            "category": "知识库管理",
            "tips": [
                "按主题创建独立知识库，避免混合不相关内容",
                "使用有意义的知识库名称：python-basics, ml-advanced",
                "定期备份重要知识库",
                "及时删除过期或不需要的知识库"
            ]
        },
        {
            "category": "学习策略",
            "tips": [
                "制定学习计划，每天固定时间使用系统",
                "从基础概念开始，循序渐进",
                "重点复习低分问题，加强薄弱环节",
                "结合实践项目，不只是理论学习"
            ]
        },
        {
            "category": "系统维护",
            "tips": [
                "定期检查系统状态：knowledge status",
                "保持模型更新：ollama pull qwen3:1.7b",
                "监控磁盘空间使用情况",
                "备份重要配置和数据"
            ]
        }
    ]
    
    for practice in practices:
        print(f"\n📋 {practice['category']}:")
        for tip in practice['tips']:
            print(f"   • {tip}")

def demonstrate_troubleshooting():
    """演示故障排除"""
    print("\n🔧 故障排除演示")
    print("=" * 50)
    
    issues = [
        {
            "problem": "Ollama服务连接失败",
            "symptoms": ["命令执行时提示连接错误", "系统状态检查显示Ollama不可用"],
            "solutions": [
                "检查Ollama服务状态：ollama list",
                "重启Ollama服务：ollama serve",
                "检查端口占用：lsof -i :11434",
                "验证模型安装：ollama pull qwen3:1.7b"
            ]
        },
        {
            "problem": "文件处理失败",
            "symptoms": ["创建知识库时文件处理错误", "不支持的文件格式提示"],
            "solutions": [
                "检查文件格式：file document.pdf",
                "验证文件权限：ls -la document.pdf",
                "转换文件格式：pandoc document.docx -o document.md",
                "检查文件大小限制"
            ]
        },
        {
            "problem": "数据库问题",
            "symptoms": ["历史记录无法保存", "知识库列表为空"],
            "solutions": [
                "检查数据目录权限：ls -la ~/.knowledge_qa/",
                "重建数据库：rm ~/.knowledge_qa/knowledge_qa.db",
                "检查磁盘空间：df -h",
                "运行系统检查：knowledge status"
            ]
        },
        {
            "problem": "性能问题",
            "symptoms": ["响应速度慢", "内存使用过高"],
            "solutions": [
                "检查系统资源：top, htop",
                "优化Ollama配置：设置内存限制",
                "清理历史数据：删除过期记录",
                "分批处理大文件"
            ]
        }
    ]
    
    for issue in issues:
        print(f"\n❌ 问题: {issue['problem']}")
        print("   症状:")
        for symptom in issue['symptoms']:
            print(f"   • {symptom}")
        print("   解决方案:")
        for solution in issue['solutions']:
            print(f"   • {solution}")

def main():
    """主演示函数"""
    print("🚀 知识库问答系统完整演示")
    print("=" * 60)
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. CLI命令演示
        demonstrate_cli_commands()
        
        # 2. API使用演示
        demonstrate_api_usage()
        
        # 3. 使用场景演示
        demonstrate_use_cases()
        
        # 4. 最佳实践演示
        demonstrate_best_practices()
        
        # 5. 故障排除演示
        demonstrate_troubleshooting()
        
        print("\n✅ 演示完成")
        print("\n📖 更多信息:")
        print("   • 使用指南: docs/usage_guide.md")
        print("   • API文档: docs/README.md")
        print("   • 示例文档: examples/sample_documents/")
        print("   • 获取帮助: knowledge --help")
        
        print("\n🎯 下一步:")
        print("   1. 安装系统: pip install -e .")
        print("   2. 启动Ollama: ollama serve")
        print("   3. 拉取模型: ollama pull qwen3:1.7b")
        print("   4. 创建知识库: knowledge new --name test --file examples/sample_documents/python_basics.md")
        print("   5. 开始学习: knowledge test review new")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查系统安装和配置")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())