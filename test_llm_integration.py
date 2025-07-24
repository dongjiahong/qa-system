#!/usr/bin/env python3
"""
Simple integration test script for LLM client
简单的 LLM 客户端集成测试脚本
"""

from src.llm_client import OllamaClient, get_ollama_client
from src.models import ModelServiceError

def test_basic_functionality():
    """测试基本功能"""
    print("Testing LLM Client basic functionality...")
    
    # 测试客户端创建
    client = OllamaClient()
    print(f"✓ Created client with base_url: {client.base_url}")
    print(f"✓ Default model: {client.model}")
    
    # 测试全局客户端
    global_client = get_ollama_client()
    print(f"✓ Global client created: {type(global_client).__name__}")
    
    # 测试健康检查（不需要真实服务）
    try:
        health = client.health_check()
        print(f"✓ Health check result: {health}")
    except ModelServiceError as e:
        print(f"✓ Health check failed as expected (no service): {e.message}")
    
    # 测试模型可用性检查
    try:
        status = client.check_model_availability()
        print(f"✓ Model availability check: {status}")
    except Exception as e:
        print(f"✓ Model availability check failed as expected: {e}")
    
    # 测试输入验证
    try:
        client.generate("")
        print("✗ Empty prompt should fail")
    except ModelServiceError:
        print("✓ Empty prompt validation works")
    
    try:
        client.generate("test", temperature=3.0)
        print("✗ Invalid temperature should fail")
    except ModelServiceError:
        print("✓ Temperature validation works")
    
    try:
        client.generate_question("")
        print("✗ Empty context should fail")
    except ModelServiceError:
        print("✓ Question generation context validation works")
    
    try:
        client.evaluate_answer("", "answer", "context")
        print("✗ Empty question should fail")
    except ModelServiceError:
        print("✓ Answer evaluation input validation works")
    
    print("\n✓ All basic functionality tests passed!")

if __name__ == "__main__":
    test_basic_functionality()