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
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
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

## 贡献指南

1. Fork 项目
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行集成测试
pytest tests/test_integration_simple.py -v

# 运行测试并生成覆盖率报告
pytest --cov=src --cov-report=html

# 运行特定测试
pytest tests/test_knowledge_base_manager.py
```

### 测试覆盖率
当前测试覆盖率约为 72%，包括：
- 单元测试：各组件独立功能测试
- 集成测试：组件间交互测试
- 用户体验测试：CLI界面和错误处理测试

## 🚀 生产环境配置

```bash
# 设置生产环境配置
knowledge config template apply production

# 配置环境变量
export KNOWLEDGE_DATA_DIR=/var/lib/knowledge_qa
export OLLAMA_BASE_URL=http://localhost:11434
export LOG_LEVEL=INFO

# 创建系统服务（可选）
sudo systemctl enable ollama
sudo systemctl start ollama
```

## 🤝 贡献指南

### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd knowledge-qa-system

# 安装开发依赖
pip install -e ".[dev]"

# 安装pre-commit钩子
pre-commit install
```

### 代码规范
```bash
# 代码格式化
black src/ tests/
isort src/ tests/

# 代码检查
flake8 src/ tests/
mypy src/

# 运行所有检查
make lint
```

### 提交流程
1. Fork 项目
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 编写代码和测试
4. 运行测试：`pytest`
5. 提交更改：`git commit -am 'Add new feature'`
6. 推送分支：`git push origin feature/new-feature`
7. 创建 Pull Request

## 📊 性能指标

### 系统性能
- 文档处理速度：~1MB/秒
- 问题生成时间：2-5秒
- 答案评估时间：3-8秒
- 向量检索时间：<1秒

### 资源使用
- 内存使用：~500MB（基础运行）
- 磁盘空间：~100MB（每个知识库）
- CPU使用：中等（处理时）

## 🔮 路线图

### v0.2.0 (计划中)
- [ ] Web界面支持
- [ ] 批量问答模式
- [ ] 学习进度分析
- [ ] 多用户支持

### v0.3.0 (计划中)
- [ ] 插件系统
- [ ] 自定义模型支持
- [ ] 云端同步
- [ ] 移动端应用

### v1.0.0 (目标)
- [ ] 企业级功能
- [ ] 高可用部署
- [ ] 完整API文档
- [ ] 多语言支持

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [LlamaIndex](https://github.com/run-llama/llama_index) - 文档处理框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [Ollama](https://github.com/ollama/ollama) - 本地LLM服务
- [Click](https://github.com/pallets/click) - 命令行界面框架

## 📞 支持

如果遇到问题或有功能建议，请：

1. 查看 [使用指南](docs/usage_guide.md)
2. 运行 `knowledge --troubleshoot-all`
3. 搜索现有 [Issues](../../issues)
4. 创建新的 Issue 描述问题

### 社区
- 讨论：[GitHub Discussions](../../discussions)
- 问题报告：[GitHub Issues](../../issues)
- 功能请求：[GitHub Issues](../../issues)

---

**注意**: 本项目目前处于开发阶段，API 可能会发生变化。建议在生产环境使用前等待稳定版本发布。

⭐ 如果这个项目对您有帮助，请给我们一个星标！