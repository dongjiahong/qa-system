"""
LLM Client module for Ollama and Qwen3 model integration
Ollama 和 Qwen3 模型集成客户端
"""

import json
import time
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import requests
from loguru import logger

from .config import get_config
from .models import ModelServiceError


def clean_model_response(text: str) -> str:
    """
    清理模型响应文本，移除思考标签和其他不需要的内容
    
    Args:
        text: 原始模型响应文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return text
    
    # 移除 <think> 标签及其内容（支持多行）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 移除其他可能的思考标签变体
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理多余的空白字符
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # 多个连续空行变为两个
    text = text.strip()
    
    return text


class ModelStatus(Enum):
    """模型状态枚举"""
    AVAILABLE = "available"
    LOADING = "loading"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    size: str
    digest: str
    modified_at: str
    status: ModelStatus = ModelStatus.AVAILABLE


@dataclass
class GenerationRequest:
    """生成请求数据模型"""
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = False


@dataclass
class GenerationResponse:
    """生成响应数据模型"""
    response: str
    model: str
    created_at: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url: Ollama 服务器地址
            model: 默认模型名称
        """
        self.config = get_config()
        self.base_url = base_url or self.config.ollama_base_url
        self.model = model or self.config.ollama_model
        self.timeout = self.config.ollama_timeout
        
        # 确保 base_url 不以斜杠结尾
        self.base_url = self.base_url.rstrip('/')
        
        logger.info(f"Initialized Ollama client with base_url: {self.base_url}, model: {self.model}")
    
    def _make_request(
        self, 
        endpoint: str, 
        method: str = "GET", 
        data: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        retry_delay: float = 1.0
    ) -> requests.Response:
        """
        发送 HTTP 请求，包含重试机制
        
        Args:
            endpoint: API 端点
            method: HTTP 方法
            data: 请求数据
            retries: 重试次数
            retry_delay: 重试延迟（秒）
            
        Returns:
            HTTP 响应对象
            
        Raises:
            ModelServiceError: 请求失败时抛出
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        
        last_exception = None
        
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt + 1}/{retries + 1})")
                
                if method.upper() == "GET":
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                elif method.upper() == "POST":
                    response = requests.post(
                        url, 
                        headers=headers, 
                        json=data, 
                        timeout=self.timeout
                    )
                else:
                    raise ModelServiceError(f"Unsupported HTTP method: {method}")
                
                # 检查响应状态
                if response.status_code == 200:
                    return response
                elif response.status_code == 404:
                    raise ModelServiceError(f"Endpoint not found: {url}")
                elif response.status_code == 500:
                    raise ModelServiceError(f"Server error: {response.text}")
                else:
                    raise ModelServiceError(
                        f"HTTP {response.status_code}: {response.text}"
                    )
                    
            except requests.exceptions.ConnectionError as e:
                last_exception = ModelServiceError(
                    f"Failed to connect to Ollama server at {self.base_url}. "
                    f"Please ensure Ollama is running.",
                    details={"error": str(e)}
                )
            except requests.exceptions.Timeout as e:
                last_exception = ModelServiceError(
                    f"Request timeout after {self.timeout} seconds",
                    details={"error": str(e)}
                )
            except requests.exceptions.RequestException as e:
                last_exception = ModelServiceError(
                    f"Request failed: {str(e)}",
                    details={"error": str(e)}
                )
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < retries:
                logger.warning(f"Request failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
        
        # 所有重试都失败了
        if last_exception:
            raise last_exception
        else:
            raise ModelServiceError("All retry attempts failed")
    
    def health_check(self) -> bool:
        """
        检查 Ollama 服务健康状态
        
        Returns:
            服务是否可用
        """
        try:
            response = self._make_request("/api/tags", retries=1)
            logger.info("Ollama service health check passed")
            return True
        except ModelServiceError as e:
            logger.error(f"Ollama service health check failed: {e.message}")
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """
        获取可用模型列表
        
        Returns:
            模型信息列表
            
        Raises:
            ModelServiceError: 获取模型列表失败时抛出
        """
        try:
            response = self._make_request("/api/tags")
            data = response.json()
            
            models = []
            for model_data in data.get("models", []):
                model_info = ModelInfo(
                    name=model_data["name"],
                    size=model_data.get("size", "unknown"),
                    digest=model_data.get("digest", ""),
                    modified_at=model_data.get("modified_at", ""),
                    status=ModelStatus.AVAILABLE
                )
                models.append(model_info)
            
            logger.info(f"Retrieved {len(models)} available models")
            return models
            
        except Exception as e:
            raise ModelServiceError(
                f"Failed to list models: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_model_availability(self, model_name: Optional[str] = None) -> ModelStatus:
        """
        检查指定模型是否可用
        
        Args:
            model_name: 模型名称，默认使用配置的模型
            
        Returns:
            模型状态
        """
        model_name = model_name or self.model
        
        try:
            models = self.list_models()
            for model in models:
                if model.name == model_name:
                    logger.info(f"Model {model_name} is available")
                    return ModelStatus.AVAILABLE
            
            logger.warning(f"Model {model_name} not found")
            return ModelStatus.NOT_FOUND
            
        except ModelServiceError:
            logger.error(f"Failed to check model {model_name} availability")
            return ModelStatus.ERROR
    
    def generate(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> GenerationResponse:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示词
            model: 模型名称，默认使用配置的模型
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成令牌数
            stop: 停止词列表
            
        Returns:
            生成响应
            
        Raises:
            ModelServiceError: 生成失败时抛出
        """
        model = model or self.model
        
        # 验证输入
        if not prompt or not prompt.strip():
            raise ModelServiceError("Prompt cannot be empty")
        
        if not (0.0 <= temperature <= 2.0):
            raise ModelServiceError("Temperature must be between 0.0 and 2.0")
        
        # 构建请求数据
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens is not None:
            request_data["options"]["num_predict"] = max_tokens
        
        if stop:
            request_data["options"]["stop"] = stop
        
        try:
            logger.debug(f"Generating response with model {model}")
            response = self._make_request("/api/generate", method="POST", data=request_data)
            
            response_data = response.json()
            
            # 解析响应并清理文本
            raw_response = response_data.get("response", "")
            cleaned_response = clean_model_response(raw_response)
            
            generation_response = GenerationResponse(
                response=cleaned_response,
                model=response_data.get("model", model),
                created_at=response_data.get("created_at", ""),
                done=response_data.get("done", True),
                total_duration=response_data.get("total_duration"),
                load_duration=response_data.get("load_duration"),
                prompt_eval_count=response_data.get("prompt_eval_count"),
                eval_count=response_data.get("eval_count")
            )
            
            logger.info(f"Generated response with {generation_response.eval_count} tokens")
            return generation_response
            
        except Exception as e:
            if isinstance(e, ModelServiceError):
                raise
            else:
                raise ModelServiceError(
                    f"Failed to generate response: {str(e)}",
                    details={"error": str(e), "model": model}
                )
    
    def generate_question(self, context: str, difficulty: str = "easy") -> str:
        """
        基于上下文生成问题
        
        Args:
            context: 知识库上下文
            difficulty: 问题难度 (easy/medium/hard)
            
        Returns:
            生成的问题
            
        Raises:
            ModelServiceError: 生成失败时抛出
        """
        if not context or not context.strip():
            raise ModelServiceError("Context cannot be empty for question generation")
        
        # 构建问题生成提示词
        prompt = f"""基于以下知识内容，生成一个{difficulty}难度的问题。

知识内容：
{context}

要求：
1. 问题应该测试对知识内容的理解
2. 问题应该清晰、具体
3. 问题应该有明确的答案
4. 使用中文提问

请只返回问题，不要包含其他内容："""

        try:
            response = self.generate(
                prompt=prompt,
                temperature=self.config.question_generation_temperature
            )
            
            # 响应已经在 generate 方法中被清理过了
            question = response.response.strip()
            if not question:
                raise ModelServiceError("Generated question is empty")
            
            logger.info("Successfully generated question")
            return question
            
        except Exception as e:
            if isinstance(e, ModelServiceError):
                raise
            else:
                raise ModelServiceError(
                    f"Failed to generate question: {str(e)}",
                    details={"error": str(e)}
                )
    
    def evaluate_answer(
        self, 
        question: str, 
        user_answer: str, 
        reference_context: str
    ) -> Dict[str, Any]:
        """
        评估用户答案
        
        Args:
            question: 问题
            user_answer: 用户答案
            reference_context: 参考上下文
            
        Returns:
            评估结果字典
            
        Raises:
            ModelServiceError: 评估失败时抛出
        """
        if not all([question.strip(), user_answer.strip(), reference_context.strip()]):
            raise ModelServiceError("Question, answer, and context cannot be empty")
        
        # 构建答案评估提示词
        prompt = f"""请评估以下用户答案的正确性。

问题：
{question}

用户答案：
{user_answer}

参考知识：
{reference_context}

请按照以下格式返回评估结果（必须是有效的JSON格式）：
{{
    "is_correct": true/false,
    "score": 0-10的分数,
    "feedback": "详细的反馈说明，重点关注需要改进的地方",
    "reference_answer": "基于参考知识的标准答案",
    "missing_points": ["缺失的要点1", "缺失的要点2"]
}}

评估标准：
1. 事实准确性：答案是否符合参考知识
2. 完整性：答案是否涵盖了关键要点
3. 相关性：答案是否直接回答了问题

请确保返回的是有效的JSON格式："""

        try:
            response = self.generate(
                prompt=prompt,
                temperature=self.config.evaluation_temperature
            )
            
            # 响应已经在 generate 方法中被清理过了
            # 尝试解析JSON响应
            try:
                evaluation_result = json.loads(response.response.strip())
                
                # 验证必需字段
                required_fields = ["is_correct", "score", "feedback", "reference_answer"]
                for field in required_fields:
                    if field not in evaluation_result:
                        raise ModelServiceError(f"Missing required field in evaluation: {field}")
                
                # 确保可选字段存在
                evaluation_result.setdefault("missing_points", [])
                
                logger.info("Successfully evaluated answer")
                return evaluation_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse evaluation JSON: {response.response}")
                raise ModelServiceError(
                    f"Invalid JSON response from model: {str(e)}",
                    details={"response": response.response}
                )
                
        except Exception as e:
            if isinstance(e, ModelServiceError):
                raise
            else:
                raise ModelServiceError(
                    f"Failed to evaluate answer: {str(e)}",
                    details={"error": str(e)}
                )


# 全局客户端实例
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """
    获取全局 Ollama 客户端实例
    
    Returns:
        Ollama 客户端实例
    """
    global _ollama_client
    
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    
    return _ollama_client


def reset_ollama_client() -> None:
    """重置全局客户端实例（主要用于测试）"""
    global _ollama_client
    _ollama_client = None