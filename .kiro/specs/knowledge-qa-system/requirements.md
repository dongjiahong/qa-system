# Requirements Document

## Introduction

本项目是一个基于知识库的智能问答系统，使用 LlamaIndex、ChromaDB、Qwen3:1.7b、shaw/dmeta-embedding-zh-small-q4、Python 3.12 和 Ollama 技术栈实现。系统通过命令行界面提供知识库管理、智能出题、答案评估和历史回顾功能，帮助用户学习和掌握知识库中的内容。

## Requirements

### Requirement 1

**User Story:** 作为用户，我希望能够创建和管理多个知识库，以便组织不同主题的学习材料

#### Acceptance Criteria

1. WHEN 用户执行 `knowledge new --name <知识库名称> --file <文件路径>` THEN 系统 SHALL 创建新的知识库并导入指定文件
2. WHEN 用户指定的文件格式为 PDF、TXT、MD 或 EPUB THEN 系统 SHALL 成功解析并存储文件内容
3. WHEN 用户创建知识库时使用已存在的名称 THEN 系统 SHALL 提示错误并拒绝创建
4. WHEN 用户指定的文件不存在或格式不支持 THEN 系统 SHALL 显示相应错误信息
5. WHEN 知识库创建成功 THEN 系统 SHALL 使用 ChromaDB 存储向量化的文档内容

### Requirement 2

**User Story:** 作为用户，我希望系统能够基于知识库内容自动生成问题，以便检验我对知识的掌握程度

#### Acceptance Criteria

1. WHEN 用户执行 `knowledge <知识库名称> review new` THEN 系统 SHALL 基于知识库内容生成一个新问题
2. WHEN 系统生成问题时 THEN 系统 SHALL 使用 LlamaIndex 和 Qwen3:1.7b 模型分析知识库内容
3. WHEN 问题生成完成 THEN 系统 SHALL 显示问题并等待用户输入答案
4. WHEN 用户指定的知识库不存在 THEN 系统 SHALL 显示错误信息
5. WHEN 知识库为空或内容不足 THEN 系统 SHALL 提示无法生成问题

### Requirement 3

**User Story:** 作为用户，我希望系统能够评估我的答案并提供反馈，以便了解我的掌握情况和需要改进的地方

#### Acceptance Criteria

1. WHEN 用户提交答案后 THEN 系统 SHALL 使用知识库内容评估答案的正确性
2. WHEN 答案评估完成 THEN 系统 SHALL 显示答案是否正确、错误点和需要补充的地方
3. WHEN 答案不完整或有错误 THEN 系统 SHALL 提供基于知识库的参考答案
4. WHEN 答案评估完成 THEN 系统 SHALL 自动保存问答记录到历史数据库
5. WHEN 系统无法评估答案时 THEN 系统 SHALL 提示用户并记录为未评估状态

### Requirement 4

**User Story:** 作为用户，我希望能够回顾历史问答记录，以便复习之前的学习内容

#### Acceptance Criteria

1. WHEN 用户执行 `knowledge <知识库名称> review history` THEN 系统 SHALL 显示该知识库的历史问答列表
2. WHEN 用户选择特定历史记录 THEN 系统 SHALL 显示完整的问题、用户答案、评估结果和参考答案
3. WHEN 历史记录为空 THEN 系统 SHALL 提示用户尚无历史记录
4. WHEN 用户查看历史记录 THEN 系统 SHALL 按时间倒序显示记录
5. WHEN 历史记录过多 THEN 系统 SHALL 支持分页显示

### Requirement 5

**User Story:** 作为用户，我希望系统能够稳定运行并提供清晰的命令行界面，以便高效使用各项功能

#### Acceptance Criteria

1. WHEN 用户执行任何命令 THEN 系统 SHALL 提供清晰的帮助信息和使用说明
2. WHEN 系统处理请求时 THEN 系统 SHALL 显示适当的进度指示
3. WHEN 发生错误时 THEN 系统 SHALL 显示有意义的错误信息和建议解决方案
4. WHEN 系统启动时 THEN 系统 SHALL 验证所有依赖组件（Ollama、ChromaDB等）的可用性
5. WHEN 用户输入无效命令 THEN 系统 SHALL 显示可用命令列表和使用示例

### Requirement 6

**User Story:** 作为用户，我希望系统能够高效处理中文内容，以便支持中文知识库的学习

#### Acceptance Criteria

1. WHEN 系统处理中文文档时 THEN 系统 SHALL 使用 shaw/dmeta-embedding-zh-small-q4 模型进行向量化
2. WHEN 系统生成中文问题时 THEN 系统 SHALL 确保问题语法正确且符合中文表达习惯
3. WHEN 系统评估中文答案时 THEN 系统 SHALL 正确理解中文语义并提供准确评估
4. WHEN 系统显示输出时 THEN 系统 SHALL 正确处理中文字符编码
5. WHEN 系统处理混合语言内容时 THEN 系统 SHALL 保持处理的一致性和准确性