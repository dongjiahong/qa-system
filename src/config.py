"""
Configuration module for Knowledge QA System
配置模块
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    app_name: str = "Knowledge QA System"
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
    ollama_timeout: int = 60
    
    # Embedding model
    embedding_model: str = "shaw/dmeta-embedding-zh-small-q4"
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/knowledge_qa.log"
    
    # File processing settings
    supported_file_extensions: list = [".pdf", ".txt", ".md", ".epub"]
    max_file_size_mb: int = 100
    
    # Question generation settings
    max_context_length: int = 4000
    question_generation_temperature: float = 0.7
    
    # Answer evaluation settings
    evaluation_temperature: float = 0.3
    
    model_config = ConfigDict(
        env_file=".env",
        env_prefix="KNOWLEDGE_QA_"
    )


def setup_logging(settings: Settings) -> None:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )
    
    # Add file logger
    logger.add(
        sink=settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
    )


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
settings = Settings()

# Setup logging and directories
setup_logging(settings)
setup_directories(settings)

logger.info(f"Initialized {settings.app_name} configuration")


def get_config() -> Settings:
    """Get global settings instance"""
    return settings