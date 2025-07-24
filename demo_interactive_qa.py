#!/usr/bin/env python3
"""
Demo script for Interactive Q&A functionality
交互式问答功能演示脚本
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

def create_demo_knowledge_base():
    """创建演示用的知识库"""
    # 创建临时文件
    demo_content = """
# Python编程基础

## 什么是Python？
Python是一种高级编程语言，由Guido van Rossum在1989年发明。它具有简洁、易读的语法，广泛应用于Web开发、数据科学、人工智能等领域。

## Python的特点
1. 简洁易读：Python的语法接近自然语言，容易理解和学习
2. 跨平台：可以在Windows、macOS、Linux等多种操作系统上运行
3. 丰富的库：拥有大量的第三方库和框架
4. 解释型语言：无需编译，可以直接运行

## Python的应用领域
- Web开发：使用Django、Flask等框架
- 数据科学：使用NumPy、Pandas、Matplotlib等库
- 人工智能：使用TensorFlow、PyTorch等框架
- 自动化脚本：系统管理、文件处理等

## 基本语法示例
```python
# 变量定义
name = "Python"
version = 3.9

# 函数定义
def greet(name):
    return f"Hello, {name}!"

# 条件语句
if version >= 3:
    print("使用Python 3")
else:
    print("使用Python 2")
```
"""
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
    temp_file.write(demo_content)
    temp_file.close()
    
    return temp_file.name

def main():
    """主函数"""
    print("🚀 交互式问答功能演示")
    print("=" * 50)
    
    try:
        # 创建演示知识库文件
        demo_file = create_demo_knowledge_base()
        print(f"📝 创建演示文件: {demo_file}")
        
        print("\n📋 使用说明:")
        print("1. 首先创建知识库:")
        print(f"   python -m src.cli new --name python_demo --file {demo_file}")
        print("\n2. 然后开始问答会话:")
        print("   python -m src.cli python_demo review new")
        print("\n3. 在问答会话中:")
        print("   - 系统会自动生成问题")
        print("   - 输入您的答案")
        print("   - 系统会评估答案并给出反馈")
        print("   - 输入 'quit' 或 'exit' 退出会话")
        
        print("\n🎯 示例问题可能包括:")
        print("- Python是什么？")
        print("- Python有哪些特点？")
        print("- Python可以用于哪些领域？")
        print("- 如何定义Python函数？")
        
        print("\n💡 提示:")
        print("- 答案可以用中文或英文")
        print("- 尽量回答完整和准确")
        print("- 系统会根据知识库内容评估答案")
        
        print(f"\n🗑️  演示结束后可以删除临时文件: {demo_file}")
        
    except Exception as e:
        print(f"❌ 演示准备失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())