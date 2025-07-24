# 知识库问答系统使用指南

## 目录
- [快速开始](#快速开始)
- [详细功能说明](#详细功能说明)
- [使用场景示例](#使用场景示例)
- [高级用法](#高级用法)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

## 快速开始

### 1. 环境准备

确保已安装并配置以下组件：

```bash
# 检查Python版本
python --version  # 需要 3.12+

# 检查Ollama服务
ollama --version

# 启动Ollama服务
ollama serve

# 拉取所需模型
ollama pull qwen3:1.7b
ollama pull shaw/dmeta-embedding-zh-small-q4
```

### 2. 安装系统

```bash
# 克隆项目
git clone <repository-url>
cd knowledge-qa-system

# 安装依赖
pip install -e .

# 验证安装
knowledge --help
```

### 3. 创建第一个知识库

```bash
# 使用示例文档创建知识库
knowledge new --name "python-tutorial" --file examples/sample_documents/python_basics.md

# 或使用多个文件
knowledge new --name "programming" \
  --file examples/sample_documents/python_basics.md \
  --file examples/sample_documents/machine_learning_intro.md
```

### 4. 开始学习

```bash
# 开始问答会话
knowledge review python-tutorial new

# 系统会生成问题，例如：
# 问题: Python有哪些主要特点？
# 请输入您的答案: 
```

## 详细功能说明

### 知识库管理

#### 创建知识库
```bash
# 基本创建
knowledge new --name <知识库名称> --file <文件路径>

# 多文件创建
knowledge new --name "comprehensive-guide" \
  --file doc1.pdf \
  --file doc2.md \
  --file doc3.txt

# 添加描述
knowledge new --name "ml-course" \
  --file course.pdf \
  --description "机器学习课程材料"
```

**支持的文件格式：**
- PDF文件 (`.pdf`)
- Markdown文件 (`.md`)
- 文本文件 (`.txt`)
- EPUB电子书 (`.epub`)

#### 列出知识库
```bash
# 列出所有知识库
knowledge list

# 输出示例：
# 📚 知识库列表
# ┌─────────────────┬──────────┬──────────┬─────────────────────┐
# │ 名称            │ 文件数   │ 文档数   │ 创建时间            │
# ├─────────────────┼──────────┼──────────┼─────────────────────┤
# │ python-tutorial │ 1        │ 15       │ 2024-01-15 10:30:00 │
# │ ml-course       │ 3        │ 45       │ 2024-01-16 14:20:00 │
# └─────────────────┴──────────┴──────────┴─────────────────────┘
```

#### 删除知识库
```bash
# 删除知识库（需要确认）
knowledge delete python-tutorial

# 强制删除（跳过确认）
knowledge delete python-tutorial --force
```

### 问答学习

#### 开始新的问答会话
```bash
knowledge review <知识库名称> new

# 示例会话：
# 🤖 正在生成问题...
# 
# 📝 问题: Python的主要特点有哪些？
# 
# 💭 请输入您的答案 (输入 'quit' 或 'exit' 退出):
# > Python具有简洁易读的语法，跨平台支持，丰富的库生态
# 
# ✅ 评估结果:
# 分数: 85/100
# 反馈: 回答正确且较为完整，涵盖了Python的主要特点
# 
# 💡 补充要点:
# - 解释型语言特性
# - 动态类型系统
# 
# 📚 参考答案:
# Python是一种高级编程语言，具有以下特点：
# 1. 简洁易读的语法
# 2. 跨平台支持
# 3. 丰富的第三方库
# 4. 解释型语言
# 5. 动态类型系统
# 
# 🔄 继续下一个问题？(y/n): 
```

#### 会话控制命令
在问答会话中，您可以使用以下命令：
- `quit` 或 `exit` - 退出会话
- `skip` - 跳过当前问题
- `hint` - 获取提示
- `help` - 显示帮助信息

### 历史记录管理

#### 查看历史记录
```bash
# 查看基本历史记录
knowledge review python-tutorial history

# 分页查看
knowledge review python-tutorial history --limit 5 --page 1

# 详细视图
knowledge review python-tutorial history --detailed

# 按时间排序
knowledge review python-tutorial history --sort time

# 按分数排序
knowledge review python-tutorial history --sort score
```

#### 查看特定记录详情
```bash
# 查看记录详情
knowledge review python-tutorial detail 123
```

#### 导出历史记录
```bash
# 导出为JSON格式
knowledge review python-tutorial export --format json

# 导出为CSV格式并保存到文件
knowledge review python-tutorial export --format csv --output history.csv
```

### 系统管理

#### 系统状态检查
```bash
# 检查系统状态
knowledge status

# 输出示例：
# 🔍 系统状态检查
# 
# ✅ Python环境: 3.12.0
# ✅ Ollama服务: 运行中 (http://localhost:11434)
# ✅ 数据库连接: 正常
# ✅ 向量存储: 正常
# ✅ 模型可用性:
#    - qwen3:1.7b: ✅ 可用
#    - shaw/dmeta-embedding-zh-small-q4: ✅ 可用
# 
# 📊 存储使用情况:
# - 数据目录: ~/.knowledge_qa (125 MB)
# - 知识库数量: 3
# - 历史记录数量: 47
```

#### 环境检查
```bash
# 检查环境配置
knowledge --check-env

# 输出示例：
# 🔧 环境检查
# 
# ✅ 必需依赖:
#    - Python 3.12+: ✅
#    - Ollama: ✅
#    - 所需Python包: ✅
# 
# ⚠️  配置建议:
#    - 建议设置 KNOWLEDGE_DATA_DIR 环境变量
#    - 建议增加 Ollama 内存限制
# 
# 📁 数据目录:
#    - 位置: ~/.knowledge_qa
#    - 权限: ✅ 可读写
#    - 磁盘空间: 15.2 GB 可用
```

## 使用场景示例

### 场景1：学习编程语言

```bash
# 1. 准备学习材料
# 收集Python相关的PDF教程、Markdown笔记等

# 2. 创建知识库
knowledge new --name "python-learning" \
  --file "Python核心编程.pdf" \
  --file "个人笔记.md" \
  --file "代码示例.txt"

# 3. 开始学习
knowledge review python-learning new

# 4. 定期复习
knowledge review python-learning history --sort score
# 重点复习低分问题
```

### 场景2：考试复习

```bash
# 1. 导入考试材料
knowledge new --name "final-exam" \
  --file "教材第1章.pdf" \
  --file "教材第2章.pdf" \
  --file "课堂笔记.md" \
  --file "习题集.pdf"

# 2. 系统性复习
for i in {1..10}; do
  echo "第${i}轮复习"
  knowledge review final-exam new
done

# 3. 查看学习进度
knowledge review final-exam history --detailed
```

### 场景3：技术文档学习

```bash
# 1. 创建技术文档知识库
knowledge new --name "api-docs" \
  --file "API参考手册.pdf" \
  --file "最佳实践.md" \
  --file "示例代码.txt"

# 2. 针对性学习
knowledge review api-docs new

# 3. 导出学习记录供团队分享
knowledge review api-docs export --format csv --output team-learning.csv
```

### 场景4：多语言学习

```bash
# 创建多个语言的知识库
knowledge new --name "english-grammar" --file "grammar-guide.pdf"
knowledge new --name "chinese-literature" --file "文学作品.txt"

# 分别学习
knowledge review english-grammar new
knowledge review chinese-literature new

# 查看所有知识库
knowledge list
```

## 高级用法

### 配置管理

#### 查看当前配置
```bash
knowledge config show
```

#### 使用配置模板
```bash
# 列出可用模板
knowledge config template list

# 应用开发环境模板
knowledge config template apply development

# 应用生产环境模板
knowledge config template apply production
```

#### 配置备份和恢复
```bash
# 备份当前配置
knowledge config backup --name "before-changes"

# 恢复配置
knowledge config restore --name "before-changes"
```

### 帮助系统

#### 获取命令帮助
```bash
# 查看特定命令的详细帮助
knowledge --help-command new

# 查看使用示例
knowledge --examples getting_started
knowledge --examples advanced_usage

# 查看故障排除指南
knowledge --troubleshoot-all
knowledge --troubleshoot ollama_connection
```

#### 快速开始指南
```bash
# 显示交互式快速开始指南
knowledge --quick-start
```

### 批量操作

#### 批量创建知识库
```bash
# 使用脚本批量创建
for subject in math physics chemistry; do
  knowledge new --name "${subject}-course" --file "${subject}/*.pdf"
done
```

#### 批量导出历史记录
```bash
# 导出所有知识库的历史记录
for kb in $(knowledge list --names-only); do
  knowledge review $kb export --format csv --output "${kb}-history.csv"
done
```

## 故障排除

### 常见问题

#### 1. Ollama连接失败
```bash
# 检查Ollama服务状态
ollama list

# 重启Ollama服务
pkill ollama
ollama serve

# 检查端口占用
lsof -i :11434
```

#### 2. 模型未找到
```bash
# 检查已安装的模型
ollama list

# 重新拉取模型
ollama pull qwen3:1.7b
ollama pull shaw/dmeta-embedding-zh-small-q4
```

#### 3. 文件处理失败
```bash
# 检查文件格式和权限
file document.pdf
ls -la document.pdf

# 尝试转换文件格式
pandoc document.docx -o document.md
```

#### 4. 数据库问题
```bash
# 检查数据目录权限
ls -la ~/.knowledge_qa/

# 重建数据库
rm ~/.knowledge_qa/knowledge_qa.db
knowledge status  # 会自动重建
```

### 调试模式

```bash
# 启用详细日志
knowledge --verbose <command>

# 启用调试模式
export LOG_LEVEL=DEBUG
knowledge <command>

# 查看日志文件
tail -f ~/.knowledge_qa/logs/knowledge_qa.log
```

### 性能优化

#### 大文件处理
```bash
# 对于大文件，建议分块处理
split -l 1000 large_document.txt chunk_

# 分别创建知识库
for chunk in chunk_*; do
  knowledge new --name "large-doc-$(basename $chunk)" --file "$chunk"
done
```

#### 内存优化
```bash
# 设置环境变量限制内存使用
export OLLAMA_MAX_MEMORY=4GB
export CHROMA_MAX_MEMORY=2GB

# 重启服务
ollama serve
```

## 最佳实践

### 1. 文档组织

- **按主题分类**：为不同主题创建独立的知识库
- **文件命名**：使用有意义的文件名，便于识别
- **格式统一**：尽量使用Markdown格式，便于处理

### 2. 学习策略

- **循序渐进**：从基础概念开始，逐步深入
- **定期复习**：利用历史记录功能复习薄弱环节
- **主动思考**：不要只是记忆答案，要理解原理

### 3. 系统维护

- **定期备份**：备份重要的知识库和配置
- **清理历史**：定期清理过期的历史记录
- **更新模型**：关注模型更新，及时升级

### 4. 团队协作

- **标准化配置**：团队使用统一的配置模板
- **共享知识库**：通过导出/导入功能共享知识库
- **学习记录**：导出学习记录进行团队分析

### 5. 安全考虑

- **敏感信息**：避免在知识库中包含敏感信息
- **访问控制**：在多用户环境中注意文件权限
- **数据备份**：重要数据要有备份策略

## 进阶技巧

### 1. 自定义提示词

虽然系统使用内置提示词，但可以通过配置文件自定义：

```json
{
  "question_generation": {
    "prompt_template": "基于以下内容生成一个深度思考的问题：\n{content}\n\n问题："
  },
  "answer_evaluation": {
    "prompt_template": "评估以下答案的质量：\n问题：{question}\n答案：{answer}\n参考：{reference}"
  }
}
```

### 2. 集成外部工具

```bash
# 与其他工具集成
knowledge export-api --port 8080  # 启动API服务
curl http://localhost:8080/api/generate-question -d '{"kb_name": "test"}'
```

### 3. 自动化学习

```bash
#!/bin/bash
# 自动化学习脚本
KB_NAME="daily-learning"
QUESTIONS_PER_DAY=5

for i in $(seq 1 $QUESTIONS_PER_DAY); do
  echo "问题 $i/$QUESTIONS_PER_DAY"
  timeout 300 knowledge review $KB_NAME new
done

# 生成学习报告
knowledge review $KB_NAME export --format json > daily-report.json
```

这个使用指南涵盖了系统的所有主要功能和使用场景。用户可以根据自己的需求选择相应的功能和策略。