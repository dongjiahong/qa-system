# 配置指南

## 问题生成配置

系统现在支持在配置文件中设置问题生成的默认参数，让用户可以根据需要自定义默认行为。

### 新增配置项

在 `config.json` 的 `question_generation` 部分，新增了以下配置项：

```json
{
    "question_generation": {
        "max_context_length": 4000,
        "question_generation_temperature": 0.7,
        "question_max_retries": 3,
        "default_question_difficulty": "medium",
        "default_content_selection_strategy": "diverse"
    }
}
```

### 配置选项说明

#### `default_question_difficulty`
设置默认的问题难度级别。

**可选值：**
- `"easy"` - 简单：测试基本概念和定义的理解
- `"medium"` - 中等：测试对概念的理解和应用，需要一定的分析和推理
- `"hard"` - 困难：测试深层理解和批判性思维，需要综合分析和评价

**默认值：** `"medium"`

#### `default_content_selection_strategy`
设置默认的内容选择策略，影响系统如何从知识库中选择内容来生成问题。

**可选值：**
- `"random"` - 随机选择：从知识库中随机选择内容
- `"diverse"` - 多样化选择：尝试选择不同类型的内容，提供更全面的问题覆盖
- `"recent"` - 最近内容：优先选择最近添加的内容
- `"comprehensive"` - 综合性内容：优先选择包含综合性信息的内容

**默认值：** `"diverse"`

### 使用方式

#### 1. 配置文件设置
修改 `config.json` 文件中的相应配置项，CLI会自动读取并使用这些默认值。

#### 2. CLI使用
在使用CLI进行问答会话时，系统会自动使用配置文件中的默认值：

```bash
# 开始问答会话，使用配置文件中的默认难度和策略
python -m src.cli review my_knowledge_base
```

CLI会显示当前使用的配置：
```
开始 'my_knowledge_base' 知识库问答会话
使用配置: 难度=medium, 策略=diverse
```

#### 3. 代码中使用
在代码中调用问题生成方法时，可以显式指定参数：

```python
from src.question_generator import get_question_generator
from src.models import QuestionDifficulty
from src.question_generator import ContentSelectionStrategy

generator = get_question_generator()

# 使用具体的参数值
question = generator.generate_question(
    "my_knowledge_base",
    difficulty=QuestionDifficulty.HARD,
    strategy=ContentSelectionStrategy.RANDOM
)
```

#### 4. CLI集成设计
系统采用了清晰的职责分离设计：
- **配置层**：定义和验证默认值
- **CLI层**：读取配置并传递给业务逻辑
- **业务逻辑层**：专注于核心功能实现

### 配置验证

系统使用Pydantic进行严格的配置验证：

- `default_question_difficulty` 必须是 `"easy"`, `"medium"`, `"hard"` 中的一个
- `default_content_selection_strategy` 必须是 `"random"`, `"diverse"`, `"recent"`, `"comprehensive"` 中的一个
- 如果配置文件中包含无效值，系统启动时会报错并拒绝启动
- 这确保了配置的正确性和系统的稳定性

### 最佳实践建议

1. **对于学习场景**：建议使用 `"easy"` 或 `"medium"` 难度，配合 `"diverse"` 策略
2. **对于测试场景**：建议使用 `"medium"` 或 `"hard"` 难度，配合 `"comprehensive"` 策略
3. **对于探索新内容**：建议使用 `"recent"` 策略，难度根据用户水平调整
4. **对于复习巩固**：建议使用 `"random"` 策略，确保覆盖所有内容

### 示例配置

#### 学习模式配置
```json
{
    "question_generation": {
        "default_question_difficulty": "easy",
        "default_content_selection_strategy": "diverse"
    }
}
```

#### 测试模式配置
```json
{
    "question_generation": {
        "default_question_difficulty": "hard",
        "default_content_selection_strategy": "comprehensive"
    }
}
```

#### 探索模式配置
```json
{
    "question_generation": {
        "default_question_difficulty": "medium",
        "default_content_selection_strategy": "recent"
    }
}
```