# 知识库问答系统 (Knowledge QA System)

基于 LlamaIndex、ChromaDB 和 Qwen3 的智能知识库问答系统，支持多格式文档导入、智能出题、答案评估和学习历史管理。

## 功能特性

- 📚 **多知识库管理** - 创建和管理多个独立的知识库
- 📄 **多格式支持** - 支持 PDF、TXT、MD、EPUB 文件格式
- 🤖 **智能出题** - 基于知识库内容自动生成学习问题
- ✅ **答案评估** - 智能评估用户答案并提供详细反馈
- 📊 **学习历史** - 记录和回顾问答历史，支持学习进度跟踪
- 🇨🇳 **中文优化** - 专门优化的中文文本处理和语义理解

## 技术栈

- **Python 3.12** - 主要开发语言
- **LlamaIndex** - 文档处理和检索框架
- **ChromaDB** - 向量数据库
- **Ollama** - 本地大语言模型服务
- **Qwen3:1.7b** - 问答生成和评估模型
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
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

## 快速开始

### 1. 启动 Ollama 服务

```bash
ollama serve
```

### 2. 创建知识库

```bash
# 创建新知识库并导入文档
knowledge new --name "python-tutorial" --file tutorial.pdf --file notes.md

# 支持多种文件格式
knowledge new --name "literature" --file book.epub --file summary.txt
```

### 3. 开始学习问答

```bash
# 开始新的问答会话
knowledge python-tutorial review new

# 系统会生成问题，你回答后会得到评估和反馈
```

### 4. 查看学习历史

```bash
# 查看历史问答记录
knowledge python-tutorial review history

# 分页查看历史记录
knowledge python-tutorial review history --limit 10 --page 1
```

### 5. 管理知识库

```bash
# 列出所有知识库
knowledge list

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
knowledge <知识库名称> review new

# 查看历史记录
knowledge <知识库名称> review history [--limit <数量>] [--page <页码>]
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

## 项目结构

```
knowledge-qa-system/
├── src/
│   ├── knowledge_qa/
│   │   ├── __init__.py
│   │   ├── cli.py                 # 命令行界面
│   │   ├── models.py              # 数据模型
│   │   ├── database.py            # 数据库操作
│   │   ├── vector_store.py        # 向量存储
│   │   ├── document_processor.py  # 文档处理
│   │   ├── knowledge_base_manager.py  # 知识库管理
│   │   ├── llm_client.py          # LLM 客户端
│   │   ├── question_generator.py  # 问题生成
│   │   ├── answer_evaluator.py    # 答案评估
│   │   └── history_manager.py     # 历史管理
├── tests/                         # 测试文件
├── docs/                          # 文档
├── examples/                      # 示例文件
├── pyproject.toml                 # 项目配置
├── README.md
└── .gitignore
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

系统支持通过环境变量进行配置：

```bash
# Ollama 服务地址
export OLLAMA_BASE_URL=http://localhost:11434

# 数据存储目录
export KNOWLEDGE_DATA_DIR=~/.knowledge_qa

# 日志级别
export LOG_LEVEL=INFO
```

## 故障排除

### 常见问题

1. **Ollama 服务连接失败**
   - 确保 Ollama 服务正在运行：`ollama serve`
   - 检查服务地址配置是否正确

2. **模型加载失败**
   - 确认已拉取所需模型：`ollama list`
   - 重新拉取模型：`ollama pull qwen3:1.7b`

3. **文件格式不支持**
   - 检查文件格式是否为 PDF、TXT、MD、EPUB
   - 确认文件未损坏且可正常打开

4. **中文处理异常**
   - 确认已安装中文嵌入模型
   - 检查文件编码是否为 UTF-8

### 日志查看

```bash
# 查看详细日志
knowledge --verbose <command>

# 日志文件位置
~/.knowledge_qa/logs/
```

## 贡献指南

1. Fork 项目
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 更新日志

### v0.1.0 (开发中)
- 基础知识库管理功能
- 多格式文档支持
- 智能问答和评估
- 学习历史记录
- 中文优化支持

## 支持

如果遇到问题或有功能建议，请：

1. 查看 [FAQ](docs/FAQ.md)
2. 搜索现有 [Issues](../../issues)
3. 创建新的 Issue 描述问题

---

**注意**: 本项目目前处于开发阶段，API 可能会发生变化。建议在生产环境使用前等待稳定版本发布。