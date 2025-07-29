"""
Configuration module for Knowledge QA System
配置模块
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from loguru import logger
from pydantic import ConfigDict, Field, validator
from pydantic_settings import BaseSettings

from .models import ValidationError


class Settings(BaseSettings):
    """Application settings with enhanced validation and configuration support"""
    
    # Application settings
    app_name: str = "Knowledge QA System"
    version: str = "0.1.0"
    debug: bool = False
    
    # Database settings
    db_path: str = "data/qa_history.db"
    
    # ChromaDB settings
    chroma_persist_directory: str = "data/chroma_db"
    
    @property
    def chroma_db_path(self) -> str:
        """Get ChromaDB persistence path"""
        return self.chroma_persist_directory
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_timeout: int = Field(default=60, ge=10, le=300)
    ollama_max_retries: int = Field(default=3, ge=1, le=10)
    ollama_retry_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Embedding model
    embedding_model: str = "shaw/dmeta-embedding-zh-small-q4"
    
    # Logging settings
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: str = "logs/knowledge_qa.log"
    log_max_size: str = "10 MB"
    log_retention: str = "7 days"
    
    # File processing settings
    supported_file_extensions: List[str] = [".pdf", ".txt", ".md", ".epub"]
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    max_files_per_kb: int = Field(default=100, ge=1, le=1000)
    
    # Question generation settings
    max_context_length: int = Field(default=4000, ge=1000, le=8000)
    question_generation_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    question_max_retries: int = Field(default=3, ge=1, le=10)
    default_question_difficulty: str = Field(default="easy", pattern="^(easy|medium|hard)$")
    default_content_selection_strategy: str = Field(default="diverse", pattern="^(random|diverse|recent|comprehensive)$")
    
    # Answer evaluation settings
    evaluation_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    evaluation_max_retries: int = Field(default=3, ge=1, le=10)
    
    # UI settings
    cli_colors: bool = True
    progress_bars: bool = True
    verbose_output: bool = False
    
    # Performance settings
    vector_search_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    
    # Health check settings
    health_check_timeout: int = Field(default=10, ge=1, le=60)
    
    model_config = ConfigDict(
        env_file=".env",
        env_prefix="KNOWLEDGE_QA_",
        validate_assignment=True
    )
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """Validate chunk overlap is less than chunk size"""
        chunk_size = values.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate environment and dependencies"""
        validation_results = {
            "status": "healthy",
            "issues": [],
            "warnings": [],
            "components": {}
        }
        
        # Check required directories
        required_dirs = [
            Path(self.db_path).parent,
            Path(self.chroma_persist_directory),
            Path(self.log_file).parent
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    validation_results["warnings"].append(f"Created missing directory: {dir_path}")
                except Exception as e:
                    validation_results["issues"].append(f"Cannot create directory {dir_path}: {e}")
                    validation_results["status"] = "unhealthy"
        
        # Check file permissions
        try:
            test_file = Path(self.db_path).parent / ".test_write"
            test_file.touch()
            test_file.unlink()
            validation_results["components"]["filesystem"] = {"status": "healthy"}
        except Exception as e:
            validation_results["issues"].append(f"Filesystem write permission error: {e}")
            validation_results["status"] = "unhealthy"
            validation_results["components"]["filesystem"] = {"status": "unhealthy", "error": str(e)}
        
        # Check Ollama connectivity (basic check)
        validation_results["components"]["ollama"] = {
            "status": "unknown",
            "url": self.ollama_base_url,
            "model": self.ollama_model
        }
        
        return validation_results


def load_config_file(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if config_path is None:
        # Try default locations
        possible_paths = [
            Path.cwd() / "config.json",
            Path.cwd() / "knowledge_qa.json",
            Path("/etc/knowledge_qa/config.json")
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            logger.info(f"Loaded configuration from: {config_path}")
            
            # 将嵌套的配置结构转换为扁平结构
            flattened_config = _flatten_config(config_data)
            return flattened_config
        except Exception as e:
            logger.warning(f"Failed to load config file {config_path}: {e}")
    
    return {}


def _flatten_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将嵌套的配置结构转换为扁平结构
    
    Args:
        config_data: 嵌套的配置数据
        
    Returns:
        Dict[str, Any]: 扁平化的配置数据
    """
    flattened = {}
    
    # 处理顶级配置
    for key, value in config_data.items():
        if not isinstance(value, dict):
            flattened[key] = value
    
    # 处理数据库配置
    if "database" in config_data:
        db_config = config_data["database"]
        flattened.update({
            "db_path": db_config.get("db_path", "data/qa_history.db"),
            "chroma_persist_directory": db_config.get("chroma_persist_directory", "data/chroma_db")
        })
    
    # 处理Ollama配置
    if "ollama" in config_data:
        ollama_config = config_data["ollama"]
        flattened.update({
            "ollama_base_url": ollama_config.get("ollama_base_url", "http://localhost:11434"),
            "ollama_model": ollama_config.get("ollama_model", "qwen3:1.7b"),
            "ollama_timeout": ollama_config.get("ollama_timeout", 60),
            "ollama_max_retries": ollama_config.get("ollama_max_retries", 3),
            "ollama_retry_delay": ollama_config.get("ollama_retry_delay", 1.0)
        })
    
    # 处理嵌入模型配置
    if "embedding" in config_data:
        embedding_config = config_data["embedding"]
        flattened.update({
            "embedding_model": embedding_config.get("embedding_model", "shaw/dmeta-embedding-zh-small-q4")
        })
    
    # 处理日志配置
    if "logging" in config_data:
        logging_config = config_data["logging"]
        flattened.update({
            "log_level": logging_config.get("log_level", "INFO"),
            "log_file": logging_config.get("log_file", "logs/knowledge_qa.log"),
            "log_max_size": logging_config.get("log_max_size", "10 MB"),
            "log_retention": logging_config.get("log_retention", "7 days")
        })
    
    # 处理文件处理配置
    if "file_processing" in config_data:
        file_config = config_data["file_processing"]
        flattened.update({
            "supported_file_extensions": file_config.get("supported_file_extensions", [".pdf", ".txt", ".md", ".epub"]),
            "max_file_size_mb": file_config.get("max_file_size_mb", 100),
            "max_files_per_kb": file_config.get("max_files_per_kb", 100)
        })
    
    # 处理问题生成配置
    if "question_generation" in config_data:
        qg_config = config_data["question_generation"]
        flattened.update({
            "max_context_length": qg_config.get("max_context_length", 4000),
            "question_generation_temperature": qg_config.get("question_generation_temperature", 0.7),
            "question_max_retries": qg_config.get("question_max_retries", 3)
        })
    
    # 处理答案评估配置
    if "answer_evaluation" in config_data:
        ae_config = config_data["answer_evaluation"]
        flattened.update({
            "evaluation_temperature": ae_config.get("evaluation_temperature", 0.3),
            "evaluation_max_retries": ae_config.get("evaluation_max_retries", 3)
        })
    
    # 处理UI配置
    if "ui" in config_data:
        ui_config = config_data["ui"]
        flattened.update({
            "cli_colors": ui_config.get("cli_colors", True),
            "progress_bars": ui_config.get("progress_bars", True),
            "verbose_output": ui_config.get("verbose_output", False)
        })
    
    # 处理性能配置
    if "performance" in config_data:
        perf_config = config_data["performance"]
        flattened.update({
            "vector_search_k": perf_config.get("vector_search_k", 5),
            "chunk_size": perf_config.get("chunk_size", 1000),
            "chunk_overlap": perf_config.get("chunk_overlap", 200)
        })
    
    # 处理健康检查配置
    if "health_check" in config_data:
        health_config = config_data["health_check"]
        flattened.update({
            "health_check_timeout": health_config.get("health_check_timeout", 10)
        })
    
    return flattened


def save_config_file(settings: Settings, config_path: Optional[Path] = None) -> None:
    """Save current configuration to JSON file"""
    if config_path is None:
        config_dir = Path.home() / ".knowledge_qa"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
    
    # Convert settings to nested dict structure
    config_data = {
        "app_name": settings.app_name,
        "version": settings.version,
        "debug": settings.debug,
        "database": {
            "db_path": settings.db_path,
            "chroma_persist_directory": settings.chroma_persist_directory
        },
        "ollama": {
            "ollama_base_url": settings.ollama_base_url,
            "ollama_model": settings.ollama_model,
            "ollama_timeout": settings.ollama_timeout,
            "ollama_max_retries": settings.ollama_max_retries,
            "ollama_retry_delay": settings.ollama_retry_delay
        },
        "embedding": {
            "embedding_model": settings.embedding_model
        },
        "logging": {
            "log_level": settings.log_level,
            "log_file": settings.log_file,
            "log_max_size": settings.log_max_size,
            "log_retention": settings.log_retention
        },
        "file_processing": {
            "supported_file_extensions": settings.supported_file_extensions,
            "max_file_size_mb": settings.max_file_size_mb,
            "max_files_per_kb": settings.max_files_per_kb
        },
        "question_generation": {
            "max_context_length": settings.max_context_length,
            "question_generation_temperature": settings.question_generation_temperature,
            "question_max_retries": settings.question_max_retries
        },
        "answer_evaluation": {
            "evaluation_temperature": settings.evaluation_temperature,
            "evaluation_max_retries": settings.evaluation_max_retries
        },
        "ui": {
            "cli_colors": settings.cli_colors,
            "progress_bars": settings.progress_bars,
            "verbose_output": settings.verbose_output
        },
        "performance": {
            "vector_search_k": settings.vector_search_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap
        },
        "health_check": {
            "health_check_timeout": settings.health_check_timeout
        }
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Configuration saved to: {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config file {config_path}: {e}")
        raise ValidationError(f"无法保存配置文件: {e}")


def setup_logging(settings: Settings) -> None:
    """Setup logging configuration with enhanced error handling"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_file).parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Cannot create log directory {log_dir}: {e}")
        # Fallback to current directory
        settings.log_file = "knowledge_qa.log"
    
    # Remove default logger
    logger.remove()
    
    # Add console logger with color support
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    if not settings.cli_colors:
        console_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.log_level,
        format=console_format,
        colorize=settings.cli_colors,
    )
    
    # Add file logger with rotation and compression
    try:
        logger.add(
            sink=settings.log_file,
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=settings.log_max_size,
            retention=settings.log_retention,
            compression="zip",
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Warning: Cannot setup file logging: {e}")


def validate_system_requirements() -> Dict[str, Any]:
    """Validate system requirements and dependencies"""
    validation_results = {
        "status": "healthy",
        "issues": [],
        "warnings": [],
        "components": {}
    }
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version < (3, 12):
        validation_results["issues"].append(
            f"Python 3.12+ required, found {python_version.major}.{python_version.minor}"
        )
        validation_results["status"] = "unhealthy"
    
    validation_results["components"]["python"] = {
        "status": "healthy" if python_version >= (3, 12) else "unhealthy",
        "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    }
    
    # Check required packages
    required_packages = [
        "click", "rich", "pydantic", "loguru", "chromadb", 
        "llama-index", "requests", "sqlite3"
    ]
    
    for package in required_packages:
        try:
            if package == "sqlite3":
                import sqlite3
            else:
                __import__(package.replace("-", "_"))
            validation_results["components"][package] = {"status": "healthy"}
        except ImportError:
            validation_results["issues"].append(f"Required package missing: {package}")
            validation_results["status"] = "unhealthy"
            validation_results["components"][package] = {"status": "missing"}
    
    return validation_results


def setup_directories(settings: Settings) -> None:
    """Create necessary directories"""
    
    directories = [
        Path(settings.db_path).parent,
        Path(settings.chroma_persist_directory),
        Path(settings.log_file).parent,
        Path("data"),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


# Global settings instance
_settings = None


def get_config(config_path: Optional[Path] = None, force_reload: bool = False) -> Settings:
    """Get global settings instance with configuration file support"""
    global _settings
    
    if _settings is None or force_reload:
        # Load configuration from file if available
        config_data = load_config_file(config_path)
        
        # Create settings with file configuration
        _settings = Settings(**config_data)
        
        # Setup logging and directories
        setup_logging(_settings)
        setup_directories(_settings)
        
        logger.info(f"Initialized {_settings.app_name} v{_settings.version}")
        
        # Validate environment
        env_validation = _settings.validate_environment()
        if env_validation["warnings"]:
            for warning in env_validation["warnings"]:
                logger.warning(warning)
        
        if env_validation["issues"]:
            for issue in env_validation["issues"]:
                logger.error(issue)
    
    return _settings


def reload_config(config_path: Optional[Path] = None) -> Settings:
    """Reload configuration from file"""
    return get_config(config_path, force_reload=True)


# Initialize default settings
settings = get_config()