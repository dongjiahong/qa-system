"""
Core data models and exception classes for Knowledge QA System
核心数据模型和异常类
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import uuid


# ============================================================================
# Exception Classes 异常类
# ============================================================================

class KnowledgeSystemError(Exception):
    """系统基础异常类"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()


class FileProcessingError(KnowledgeSystemError):
    """文件处理异常"""
    pass


class ModelServiceError(KnowledgeSystemError):
    """模型服务异常"""
    pass


class DatabaseError(KnowledgeSystemError):
    """数据库异常"""
    pass


class ValidationError(KnowledgeSystemError):
    """数据验证异常"""
    pass


class KnowledgeBaseNotFoundError(KnowledgeSystemError):
    """知识库不存在异常"""
    pass


class VectorStoreError(KnowledgeSystemError):
    """向量存储异常"""
    pass


# ============================================================================
# Enums 枚举类
# ============================================================================

class QuestionDifficulty(Enum):
    """问题难度级别"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationStatus(Enum):
    """评估状态"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    ERROR = "error"


# ============================================================================
# Core Data Models 核心数据模型
# ============================================================================

@dataclass
class KnowledgeBase:
    """知识库数据模型"""
    
    name: str
    created_at: datetime
    file_count: int = 0
    document_count: int = 0
    description: Optional[str] = None
    
    def __post_init__(self):
        """数据验证"""
        self.validate()
    
    def validate(self) -> None:
        """验证数据模型"""
        if not self.name or not self.name.strip():
            raise ValidationError("知识库名称不能为空")
        
        if len(self.name) > 100:
            raise ValidationError("知识库名称不能超过100个字符")
        
        if self.file_count < 0:
            raise ValidationError("文件数量不能为负数")
        
        if self.document_count < 0:
            raise ValidationError("文档数量不能为负数")
        
        if self.description and len(self.description) > 500:
            raise ValidationError("描述不能超过500个字符")
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "file_count": self.file_count,
            "document_count": self.document_count,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBase":
        """从字典反序列化"""
        return cls(
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            file_count=data.get("file_count", 0),
            document_count=data.get("document_count", 0),
            description=data.get("description"),
        )
    
    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeBase":
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Question:
    """问题数据模型"""
    
    content: str
    kb_name: str
    source_context: str
    difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据验证"""
        self.validate()
    
    def validate(self) -> None:
        """验证数据模型"""
        if not self.content or not self.content.strip():
            raise ValidationError("问题内容不能为空")
        
        if len(self.content) > 1000:
            raise ValidationError("问题内容不能超过1000个字符")
        
        if not self.kb_name or not self.kb_name.strip():
            raise ValidationError("知识库名称不能为空")
        
        if not self.source_context or not self.source_context.strip():
            raise ValidationError("问题来源上下文不能为空")
        
        if len(self.source_context) > 5000:
            raise ValidationError("问题来源上下文不能超过5000个字符")
        
        if not isinstance(self.difficulty, QuestionDifficulty):
            raise ValidationError("问题难度必须是有效的枚举值")
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "kb_name": self.kb_name,
            "source_context": self.source_context,
            "difficulty": self.difficulty.value,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        """从字典反序列化"""
        return cls(
            id=data["id"],
            content=data["content"],
            kb_name=data["kb_name"],
            source_context=data["source_context"],
            difficulty=QuestionDifficulty(data["difficulty"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )
    
    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Question":
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class EvaluationResult:
    """答案评估结果数据模型"""
    
    is_correct: bool
    score: float  # 0-10
    feedback: str
    reference_answer: str
    missing_points: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    status: EvaluationStatus = EvaluationStatus.SUCCESS
    
    def __post_init__(self):
        """数据验证"""
        self.validate()
    
    def validate(self) -> None:
        """验证数据模型"""
        if not isinstance(self.is_correct, bool):
            raise ValidationError("is_correct必须是布尔值")
        
        if not (0 <= self.score <= 10):
            raise ValidationError("分数必须在0-10之间")
        
        if not self.feedback or not self.feedback.strip():
            raise ValidationError("反馈内容不能为空")
        
        if len(self.feedback) > 2000:
            raise ValidationError("反馈内容不能超过2000个字符")
        
        if not self.reference_answer or not self.reference_answer.strip():
            raise ValidationError("参考答案不能为空")
        
        if len(self.reference_answer) > 2000:
            raise ValidationError("参考答案不能超过2000个字符")
        
        if not isinstance(self.status, EvaluationStatus):
            raise ValidationError("评估状态必须是有效的枚举值")
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "is_correct": self.is_correct,
            "score": self.score,
            "feedback": self.feedback,
            "reference_answer": self.reference_answer,
            "missing_points": self.missing_points,
            "strengths": self.strengths,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """从字典反序列化"""
        return cls(
            is_correct=data["is_correct"],
            score=data["score"],
            feedback=data["feedback"],
            reference_answer=data["reference_answer"],
            missing_points=data.get("missing_points", []),
            strengths=data.get("strengths", []),
            status=EvaluationStatus(data.get("status", "success")),
        )
    
    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationResult":
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class QARecord:
    """问答记录数据模型"""
    
    kb_name: str
    question: str
    user_answer: str
    evaluation: EvaluationResult
    created_at: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None  # 数据库自动生成
    
    def __post_init__(self):
        """数据验证"""
        self.validate()
    
    def validate(self) -> None:
        """验证数据模型"""
        if not self.kb_name or not self.kb_name.strip():
            raise ValidationError("知识库名称不能为空")
        
        if not self.question or not self.question.strip():
            raise ValidationError("问题内容不能为空")
        
        if len(self.question) > 1000:
            raise ValidationError("问题内容不能超过1000个字符")
        
        if not self.user_answer or not self.user_answer.strip():
            raise ValidationError("用户答案不能为空")
        
        if len(self.user_answer) > 2000:
            raise ValidationError("用户答案不能超过2000个字符")
        
        if not isinstance(self.evaluation, EvaluationResult):
            raise ValidationError("评估结果必须是EvaluationResult类型")
        
        # 验证评估结果
        self.evaluation.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "kb_name": self.kb_name,
            "question": self.question,
            "user_answer": self.user_answer,
            "evaluation": self.evaluation.to_dict(),
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QARecord":
        """从字典反序列化"""
        return cls(
            id=data.get("id"),
            kb_name=data["kb_name"],
            question=data["question"],
            user_answer=data["user_answer"],
            evaluation=EvaluationResult.from_dict(data["evaluation"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )
    
    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "QARecord":
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_summary(self) -> str:
        """获取记录摘要"""
        status_text = "正确" if self.evaluation.is_correct else "错误"
        return f"[{self.created_at.strftime('%Y-%m-%d %H:%M')}] {status_text} (分数: {self.evaluation.score:.1f})"