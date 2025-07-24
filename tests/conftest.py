"""
Pytest configuration and fixtures for Knowledge QA System tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

from src.config import Settings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings with temporary directories"""
    return Settings(
        db_path=str(temp_dir / "test.db"),
        chroma_persist_directory=str(temp_dir / "chroma"),
        log_file=str(temp_dir / "test.log"),
        debug=True,
        log_level="DEBUG"
    )


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    mock_client = Mock()
    mock_client.generate.return_value = {"response": "Test response"}
    return mock_client