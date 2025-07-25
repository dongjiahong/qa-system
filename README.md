# 知识库问答系统 (Knowledge QA System)

基于 LlamaIndex、ChromaDB 和 Qwen3 的智能知识库问答系统，支持多格式文档导入、智能出题、答案评估和学习历史管理。

## 🌟 功能特性

- 📚 **多知识库管理** - 创建和管理多个独立的知识库
- 📄 **多格式支持** - 支持 PDF、TXT、MD、EPUB 文件格式
- 🤖 **智能出题** - 基于知识库内容自动生成学习问题
- ✅ **答案评估** - 智能评估用户答案并提供详细反馈
- 📊 **学习历史** - 记录和回顾问答历史，支持学习进度跟踪
- 🇨🇳 **中文优化** - 专门优化的中文文本处理和语义理解
- 🔧 **易于使用** - 直观的命令行界面和丰富的帮助系统
- 🚀 **高性能** - 优化的向量检索和模型推理
- 🧠 **智能过滤** - 自动过滤思考模型的思考过程，只保留最终答案

## 技术栈

- **Python 3.12** - 主要开发语言
- **LlamaIndex** - 文档处理和检索框架
- **ChromaDB** - 向量数据库
- **Ollama** - 本地大语言模型服务
- **Qwen3:1.7b** - 问答生成和评估模型（支持思考模型，自动过滤思考内容）
- **shaw/dmeta-embedding-zh-small-q4** - 中文向量化模型
- **Click** - 命令行界面框架
- **SQLite** - 历史记录存储

## 安装要求

### 系统依赖

1. **Python 3.12+**
2. **Ollama** - 用于运行本地大语言模型
   ```bash
   # macOS
   brew install ollama
   
   # 或从官网下载: https://ollama.ai
   ```

3. **拉取所需模型**
   ```bash
   # 拉取 Qwen3 模型
   ollama pull qwen3:1.7b
   
   # 拉取中文嵌入模型
   ollama pull shaw/dmeta-embedding-zh-small-q4
   ```

### Python 依赖

```bash
# 克隆项目
git clone <repository-url>
cd knowledge-qa-system

# 创建虚拟环境
conda create -n llamaindex python=3.12

# 安装依赖
pip install -r requirements.txt

# 按装
pip install -e .
```

## 🚀 快速开始

### 1. 启动 Ollama 服务

```bash
ollama serve
```

### 2. 创建知识库

```bash
# 使用示例文档创建知识库
knowledge new --name "python-tutorial" --file examples/sample_documents/python_basics.md

# 创建多文件知识库
knowledge new --name "programming" \
  --file examples/sample_documents/python_basics.md \
  --file examples/sample_documents/machine_learning_intro.md
```

### 3. 开始学习问答

```bash
# 开始新的问答会话
knowledge review python-tutorial new

# 示例会话：
# 🤖 正在生成问题...
# 📝 问题: Python有哪些主要特点？
# 💭 请输入您的答案 (输入 'quit' 或 'exit' 退出):
# > Python具有简洁易读的语法，跨平台支持
# 
# ✅ 评估结果:
# 分数: 85/100
# 反馈: 回答正确且较为完整...
```

### 4. 查看学习历史

```bash
# 查看历史问答记录
knowledge review python-tutorial history

# 详细查看和导出
knowledge review python-tutorial history --detailed
knowledge review python-tutorial export --format csv
```

### 5. 管理知识库

```bash
# 列出所有知识库
knowledge list

# 检查系统状态
knowledge status

# 删除知识库
knowledge delete python-tutorial
```

## 命令参考

### 创建知识库
```bash
knowledge new --name <知识库名称> --file <文件路径> [--file <文件路径>...]
```

### 问答学习
```bash
# 开始新问答
knowledge review <知识库名称> new

# 查看历史记录
knowledge review <知识库名称> history [--limit <数量>] [--page <页码>]
```

### 知识库管理
```bash
# 列出所有知识库
knowledge list

# 删除知识库
knowledge delete <知识库名称>

# 显示帮助
knowledge --help
```

## 开发指南

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=src/knowledge_qa --cov-report=html

# 运行特定测试
pytest tests/test_knowledge_base_manager.py
```

### 代码格式化

```bash
# 格式化代码
black src/ tests/

# 检查代码风格
flake8 src/ tests/

# 类型检查
mypy src/
```

## 配置

查看配置文件`cnofig.json.example`

## 📖 文档和示例

### 详细文档
- [使用指南](docs/usage_guide.md) - 完整的功能说明和使用示例
- [用户体验实现](docs/user_experience_implementation.md) - 系统特性和实现细节

### 示例文档
- [Python基础](examples/sample_documents/python_basics.md) - Python编程基础知识
- [机器学习入门](examples/sample_documents/machine_learning_intro.md) - 机器学习概念和实践

### 演示脚本
```bash
# 运行完整工作流程演示
python examples/demo_complete_workflow.py

# 运行交互式演示
python demo_interactive_qa.py
```

## 🔧 故障排除

### 快速诊断
```bash
# 系统状态检查
knowledge status

# 环境检查
knowledge --check-env

# 获取帮助
knowledge --help
knowledge --troubleshoot-all
```

### 常见问题

1. **Ollama 服务连接失败**
   ```bash
   # 检查服务状态
   ollama list
   
   # 重启服务
   ollama serve
   
   # 检查端口
   lsof -i :11434
   ```

2. **模型加载失败**
   ```bash
   # 检查已安装模型
   ollama list
   
   # 重新拉取模型
   ollama pull qwen3:1.7b
   ollama pull shaw/dmeta-embedding-zh-small-q4
   ```

3. **文件处理失败**
   ```bash
   # 检查文件格式和权限
   file document.pdf
   ls -la document.pdf
   
   # 转换文件格式
   pandoc document.docx -o document.md
   ```

4. **数据库问题**
   ```bash
   # 检查数据目录
   ls -la ~/.knowledge_qa/
   
   # 重建数据库
   rm ~/.knowledge_qa/knowledge_qa.db
   knowledge status  # 自动重建
   ```

### 获取帮助
```bash
# 命令帮助
knowledge --help-command new

# 使用示例
knowledge --examples getting_started

# 故障排除指南
knowledge --troubleshoot ollama_connection

# 快速开始指南
knowledge --quick-start
```

## 📞 支持

如果遇到问题或有功能建议，请：

1. 查看 [使用指南](docs/usage_guide.md)
2. 运行 `knowledge --troubleshoot-all`
3. 创建新的 Issue 描述问题

⭐ 如果这个项目对您有帮助，请给我们一个星标！