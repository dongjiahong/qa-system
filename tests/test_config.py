"""
Tests for configuration module
"""

import pytest
from pathlib import Path

from src.config import Settings, setup_logging, setup_directories


def test_settings_default_values():
    """Test that settings have correct default values"""
    settings = Settings()
    
    assert settings.app_name == "Knowledge QA System"
    assert settings.debug is False
    assert settings.ollama_model == "qwen3:1.7b"
    assert settings.embedding_model == "shaw/dmeta-embedding-zh-small-q4"
    assert settings.log_level == "INFO"


def test_settings_from_env(monkeypatch):
    """Test that settings can be loaded from environment variables"""
    monkeypatch.setenv("KNOWLEDGE_QA_DEBUG", "true")
    monkeypatch.setenv("KNOWLEDGE_QA_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("KNOWLEDGE_QA_OLLAMA_MODEL", "custom-model")
    
    settings = Settings()
    
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    assert settings.ollama_model == "custom-model"


def test_setup_directories(test_settings, temp_dir):
    """Test that setup_directories creates necessary directories"""
    setup_directories(test_settings)
    
    # Check that directories were created
    assert (temp_dir / "chroma").exists()
    assert (temp_dir / "chroma").is_dir()


def test_supported_file_extensions():
    """Test that supported file extensions are correctly defined"""
    settings = Settings()
    
    expected_extensions = [".pdf", ".txt", ".md", ".epub"]
    assert settings.supported_file_extensions == expected_extensions