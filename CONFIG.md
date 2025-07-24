# 配置文件说明

本目录包含了知识库问答系统的各种配置文件示例。

## 配置文件列表

### 1. config.json
完整的配置文件，包含所有可配置的选项。适合需要精细控制的用户。

### 2. config-simple.json  
简化版配置文件，只包含常用的核心配置项。适合快速开始使用。

### 3. config-dev.json
开发环境配置，启用调试模式和详细日志，便于开发和调试。

### 4. config-prod.json
生产环境配置，优化性能和稳定性，减少日志输出。

## 配置项说明

### 基础设置
- `debug`: 是否启用调试模式
- `log_level`: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Ollama 设置
- `ollama_base_url`: Ollama 服务地址
- `ollama_model`: 使用的模型名称
- `ollama_timeout`: 请求超时时间（秒）
- `ollama_max_retries`: 最大重试次数

### 嵌入模型
- `embedding_model`: 文档向量化使用的嵌入模型

### 文件处理
- `max_file_size_mb`: 单个文件最大大小（MB）
- `max_context_length`: 最大上下文长度

### 问题生成
- `question_generation_temperature`: 问题生成的温度参数（0.0-2.0）

### 答案评估
- `evaluation_temperature`: 答案评估的温度参数（0.0-2.0）

### UI 设置
- `cli_colors`: 是否启用彩色输出
- `progress_bars`: 是否显示进度条
- `verbose_output`: 是否显示详细输出

### 性能设置
- `vector_search_k`: 向量搜索返回的结果数量
- `chunk_size`: 文档分块大小
- `chunk_overlap`: 分块重叠大小

## 使用方法

### 1. 选择配置文件
根据您的使用场景选择合适的配置文件：
```bash
# 使用简化配置
cp config-simple.json ~/.knowledge_qa/config.json

# 使用开发配置
cp config-dev.json ~/.knowledge_qa/config.json

# 使用生产配置  
cp config-prod.json ~/.knowledge_qa/config.json
```

### 2. 自定义配置
复制任一配置文件并根据需要修改：
```bash
cp config-simple.json my-config.json
# 编辑 my-config.json
knowledge --config my-config.json
```

### 3. 配置文件位置
系统会按以下顺序查找配置文件：
1. 当前目录下的 `knowledge_qa.json`
2. `~/.knowledge_qa/config.json`
3. `/etc/knowledge_qa/config.json`

### 4. 环境变量
也可以通过环境变量设置配置，前缀为 `KNOWLEDGE_QA_`：
```bash
export KNOWLEDGE_QA_DEBUG=true
export KNOWLEDGE_QA_LOG_LEVEL=DEBUG
export KNOWLEDGE_QA_OLLAMA_MODEL=qwen3:1.7b
```

## 验证配置

使用以下命令验证配置是否正确：
```bash
knowledge config validate
knowledge config show
knowledge --check-env
```

## 备份和恢复

```bash
# 备份当前配置
knowledge config backup --name my-backup

# 恢复配置
knowledge config restore ~/.knowledge_qa/backups/my-backup.json

# 重置为默认配置
knowledge config reset
```
