"""
Unit tests for LLM Client module
LLM 客户端模块单元测试
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.llm_client import (
    OllamaClient, 
    ModelStatus, 
    ModelInfo, 
    GenerationRequest,
    GenerationResponse,
    get_ollama_client,
    reset_ollama_client
)
from src.models import ModelServiceError


class TestOllamaClient:
    """Ollama 客户端测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.client = OllamaClient(
            base_url="http://localhost:11434",
            model="qwen3:1.7b"
        )
    
    def test_init_default_config(self):
        """测试默认配置初始化"""
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model == "qwen3:1.7b"
        assert client.timeout == 60
    
    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        client = OllamaClient(
            base_url="http://custom:8080",
            model="custom-model"
        )
        assert client.base_url == "http://custom:8080"
        assert client.model == "custom-model"
    
    def test_base_url_trailing_slash_removal(self):
        """测试 base_url 末尾斜杠移除"""
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"
    
    @patch('requests.get')
    def test_make_request_get_success(self, mock_get):
        """测试 GET 请求成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response
        
        response = self.client._make_request("/api/test", method="GET")
        
        assert response.status_code == 200
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/test",
            headers={"Content-Type": "application/json"},
            timeout=60
        )
    
    @patch('requests.post')
    def test_make_request_post_success(self, mock_post):
        """测试 POST 请求成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response
        
        test_data = {"key": "value"}
        response = self.client._make_request("/api/test", method="POST", data=test_data)
        
        assert response.status_code == 200
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/test",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=60
        )
    
    def test_make_request_unsupported_method(self):
        """测试不支持的 HTTP 方法"""
        with pytest.raises(ModelServiceError) as exc_info:
            self.client._make_request("/api/test", method="DELETE")
        
        assert "Unsupported HTTP method: DELETE" in str(exc_info.value)
    
    @patch('requests.get')
    def test_make_request_404_error(self, mock_get):
        """测试 404 错误处理"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client._make_request("/api/nonexistent")
        
        assert "Endpoint not found" in str(exc_info.value)
    
    @patch('requests.get')
    def test_make_request_500_error(self, mock_get):
        """测试 500 服务器错误处理"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client._make_request("/api/test")
        
        assert "Server error" in str(exc_info.value)
    
    @patch('requests.get')
    def test_make_request_connection_error(self, mock_get):
        """测试连接错误处理"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client._make_request("/api/test", retries=1)
        
        assert "Failed to connect to Ollama server" in str(exc_info.value)
    
    @patch('requests.get')
    def test_make_request_timeout_error(self, mock_get):
        """测试超时错误处理"""
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client._make_request("/api/test", retries=1)
        
        assert "Request timeout after 60 seconds" in str(exc_info.value)
    
    @patch('requests.get')
    @patch('time.sleep')
    def test_make_request_retry_mechanism(self, mock_sleep, mock_get):
        """测试重试机制"""
        # 前两次失败，第三次成功
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.ConnectionError("Connection failed"),
            Mock(status_code=200)
        ]
        
        response = self.client._make_request("/api/test", retries=2)
        
        assert response.status_code == 200
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch.object(OllamaClient, '_make_request')
    def test_health_check_success(self, mock_make_request):
        """测试健康检查成功"""
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}
        mock_make_request.return_value = mock_response
        
        result = self.client.health_check()
        
        assert result is True
        mock_make_request.assert_called_once_with("/api/tags", retries=1)
    
    @patch.object(OllamaClient, '_make_request')
    def test_health_check_failure(self, mock_make_request):
        """测试健康检查失败"""
        mock_make_request.side_effect = ModelServiceError("Connection failed")
        
        result = self.client.health_check()
        
        assert result is False
    
    @patch.object(OllamaClient, '_make_request')
    def test_list_models_success(self, mock_make_request):
        """测试获取模型列表成功"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "qwen3:1.7b",
                    "size": "1.2GB",
                    "digest": "abc123",
                    "modified_at": "2024-01-01T00:00:00Z"
                },
                {
                    "name": "llama2:7b",
                    "size": "3.8GB",
                    "digest": "def456",
                    "modified_at": "2024-01-02T00:00:00Z"
                }
            ]
        }
        mock_make_request.return_value = mock_response
        
        models = self.client.list_models()
        
        assert len(models) == 2
        assert models[0].name == "qwen3:1.7b"
        assert models[0].size == "1.2GB"
        assert models[0].status == ModelStatus.AVAILABLE
        assert models[1].name == "llama2:7b"
    
    @patch.object(OllamaClient, '_make_request')
    def test_list_models_failure(self, mock_make_request):
        """测试获取模型列表失败"""
        mock_make_request.side_effect = Exception("Network error")
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.list_models()
        
        assert "Failed to list models" in str(exc_info.value)
    
    @patch.object(OllamaClient, 'list_models')
    def test_check_model_availability_found(self, mock_list_models):
        """测试检查模型可用性 - 找到模型"""
        mock_list_models.return_value = [
            ModelInfo("qwen3:1.7b", "1.2GB", "abc123", "2024-01-01T00:00:00Z"),
            ModelInfo("llama2:7b", "3.8GB", "def456", "2024-01-02T00:00:00Z")
        ]
        
        status = self.client.check_model_availability("qwen3:1.7b")
        
        assert status == ModelStatus.AVAILABLE
    
    @patch.object(OllamaClient, 'list_models')
    def test_check_model_availability_not_found(self, mock_list_models):
        """测试检查模型可用性 - 未找到模型"""
        mock_list_models.return_value = [
            ModelInfo("llama2:7b", "3.8GB", "def456", "2024-01-02T00:00:00Z")
        ]
        
        status = self.client.check_model_availability("qwen3:1.7b")
        
        assert status == ModelStatus.NOT_FOUND
    
    @patch.object(OllamaClient, 'list_models')
    def test_check_model_availability_error(self, mock_list_models):
        """测试检查模型可用性 - 错误"""
        mock_list_models.side_effect = ModelServiceError("Service error")
        
        status = self.client.check_model_availability("qwen3:1.7b")
        
        assert status == ModelStatus.ERROR
    
    @patch.object(OllamaClient, '_make_request')
    def test_generate_success(self, mock_make_request):
        """测试文本生成成功"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Generated text response",
            "model": "qwen3:1.7b",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        mock_make_request.return_value = mock_response
        
        response = self.client.generate("Test prompt")
        
        assert isinstance(response, GenerationResponse)
        assert response.response == "Generated text response"
        assert response.model == "qwen3:1.7b"
        assert response.done is True
        assert response.eval_count == 20
        
        # 验证请求参数
        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        assert call_args[0][0] == "/api/generate"
        assert call_args[1]["method"] == "POST"
        
        request_data = call_args[1]["data"]
        assert request_data["model"] == "qwen3:1.7b"
        assert request_data["prompt"] == "Test prompt"
        assert request_data["stream"] is False
        assert request_data["options"]["temperature"] == 0.7
    
    def test_generate_empty_prompt(self):
        """测试空提示词错误"""
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.generate("")
        
        assert "Prompt cannot be empty" in str(exc_info.value)
    
    def test_generate_invalid_temperature(self):
        """测试无效温度参数"""
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.generate("Test prompt", temperature=3.0)
        
        assert "Temperature must be between 0.0 and 2.0" in str(exc_info.value)
    
    @patch.object(OllamaClient, '_make_request')
    def test_generate_with_options(self, mock_make_request):
        """测试带选项的文本生成"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Generated text",
            "model": "qwen3:1.7b",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        mock_make_request.return_value = mock_response
        
        self.client.generate(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100,
            stop=["END", "STOP"]
        )
        
        call_args = mock_make_request.call_args
        request_data = call_args[1]["data"]
        
        assert request_data["options"]["temperature"] == 0.5
        assert request_data["options"]["num_predict"] == 100
        assert request_data["options"]["stop"] == ["END", "STOP"]
    
    @patch.object(OllamaClient, 'generate')
    def test_generate_question_success(self, mock_generate):
        """测试问题生成成功"""
        mock_response = GenerationResponse(
            response="这是一个测试问题？",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        mock_generate.return_value = mock_response
        
        question = self.client.generate_question("测试知识内容", "medium")
        
        assert question == "这是一个测试问题？"
        mock_generate.assert_called_once()
        
        # 验证提示词包含必要内容
        call_args = mock_generate.call_args
        prompt = call_args[1]["prompt"]
        assert "测试知识内容" in prompt
        assert "medium" in prompt
        assert "中文提问" in prompt
    
    def test_generate_question_empty_context(self):
        """测试空上下文问题生成"""
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.generate_question("")
        
        assert "Context cannot be empty" in str(exc_info.value)
    
    @patch.object(OllamaClient, 'generate')
    def test_generate_question_empty_response(self, mock_generate):
        """测试问题生成空响应"""
        mock_response = GenerationResponse(
            response="   ",  # 空白响应
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        mock_generate.return_value = mock_response
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.generate_question("测试内容")
        
        assert "Generated question is empty" in str(exc_info.value)
    
    @patch.object(OllamaClient, 'generate')
    def test_evaluate_answer_success(self, mock_generate):
        """测试答案评估成功"""
        evaluation_json = {
            "is_correct": True,
            "score": 85,
            "feedback": "答案基本正确",
            "reference_answer": "标准答案",
            "missing_points": ["缺失点1"],
            "strengths": ["优点1", "优点2"]
        }
        
        mock_response = GenerationResponse(
            response=json.dumps(evaluation_json, ensure_ascii=False),
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        mock_generate.return_value = mock_response
        
        result = self.client.evaluate_answer(
            question="测试问题？",
            user_answer="用户答案",
            reference_context="参考上下文"
        )
        
        assert result["is_correct"] is True
        assert result["score"] == 85
        assert result["feedback"] == "答案基本正确"
        assert result["reference_answer"] == "标准答案"
        assert result["missing_points"] == ["缺失点1"]
        assert result["strengths"] == ["优点1", "优点2"]
    
    def test_evaluate_answer_empty_inputs(self):
        """测试答案评估空输入"""
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.evaluate_answer("", "答案", "上下文")
        
        assert "cannot be empty" in str(exc_info.value)
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.evaluate_answer("问题", "", "上下文")
        
        assert "cannot be empty" in str(exc_info.value)
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.evaluate_answer("问题", "答案", "")
        
        assert "cannot be empty" in str(exc_info.value)
    
    @patch.object(OllamaClient, 'generate')
    def test_evaluate_answer_invalid_json(self, mock_generate):
        """测试答案评估无效JSON响应"""
        mock_response = GenerationResponse(
            response="这不是有效的JSON",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        mock_generate.return_value = mock_response
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.evaluate_answer("问题", "答案", "上下文")
        
        assert "Invalid JSON response" in str(exc_info.value)
    
    @patch.object(OllamaClient, 'generate')
    def test_evaluate_answer_missing_fields(self, mock_generate):
        """测试答案评估缺失必需字段"""
        incomplete_json = {
            "is_correct": True,
            "score": 85
            # 缺失 feedback 和 reference_answer
        }
        
        mock_response = GenerationResponse(
            response=json.dumps(incomplete_json),
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        mock_generate.return_value = mock_response
        
        with pytest.raises(ModelServiceError) as exc_info:
            self.client.evaluate_answer("问题", "答案", "上下文")
        
        assert "Missing required field" in str(exc_info.value)


class TestGlobalClient:
    """全局客户端测试类"""
    
    def setup_method(self):
        """测试前重置全局客户端"""
        reset_ollama_client()
    
    def test_get_ollama_client_singleton(self):
        """测试全局客户端单例模式"""
        client1 = get_ollama_client()
        client2 = get_ollama_client()
        
        assert client1 is client2
        assert isinstance(client1, OllamaClient)
    
    def test_reset_ollama_client(self):
        """测试重置全局客户端"""
        client1 = get_ollama_client()
        reset_ollama_client()
        client2 = get_ollama_client()
        
        assert client1 is not client2


class TestDataModels:
    """数据模型测试类"""
    
    def test_model_info_creation(self):
        """测试模型信息创建"""
        model_info = ModelInfo(
            name="qwen3:1.7b",
            size="1.2GB",
            digest="abc123",
            modified_at="2024-01-01T00:00:00Z"
        )
        
        assert model_info.name == "qwen3:1.7b"
        assert model_info.size == "1.2GB"
        assert model_info.status == ModelStatus.AVAILABLE
    
    def test_generation_request_creation(self):
        """测试生成请求创建"""
        request = GenerationRequest(
            model="qwen3:1.7b",
            prompt="Test prompt",
            temperature=0.8,
            max_tokens=100
        )
        
        assert request.model == "qwen3:1.7b"
        assert request.prompt == "Test prompt"
        assert request.temperature == 0.8
        assert request.max_tokens == 100
        assert request.stream is False
    
    def test_generation_response_creation(self):
        """测试生成响应创建"""
        response = GenerationResponse(
            response="Generated text",
            model="qwen3:1.7b",
            created_at="2024-01-01T00:00:00Z",
            done=True,
            eval_count=20
        )
        
        assert response.response == "Generated text"
        assert response.model == "qwen3:1.7b"
        assert response.done is True
        assert response.eval_count == 20


@pytest.mark.integration
class TestOllamaIntegration:
    """Ollama 集成测试类（需要运行的 Ollama 服务）"""
    
    def setup_method(self):
        """测试前设置"""
        self.client = OllamaClient()
    
    def test_real_health_check(self):
        """测试真实健康检查（需要 Ollama 服务运行）"""
        # 这个测试需要真实的 Ollama 服务
        # 在 CI 环境中可能会跳过
        try:
            result = self.client.health_check()
            # 如果服务可用，应该返回 True
            # 如果服务不可用，应该返回 False
            assert isinstance(result, bool)
        except Exception:
            pytest.skip("Ollama service not available for integration test")
    
    def test_real_model_list(self):
        """测试真实模型列表获取（需要 Ollama 服务运行）"""
        try:
            models = self.client.list_models()
            assert isinstance(models, list)
            # 如果有模型，验证模型信息结构
            if models:
                assert isinstance(models[0], ModelInfo)
                assert hasattr(models[0], 'name')
                assert hasattr(models[0], 'size')
        except ModelServiceError:
            pytest.skip("Ollama service not available for integration test")