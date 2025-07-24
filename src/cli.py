"""
Command Line Interface for Knowledge QA System
命令行界面
"""

import click
from loguru import logger

from .config import settings


@click.group()
@click.version_option(version="0.1.0")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(debug: bool) -> None:
    """Knowledge QA System - 知识库问答系统
    
    A command-line tool for creating knowledge bases and interactive Q&A sessions.
    """
    if debug:
        settings.debug = True
        settings.log_level = "DEBUG"
        logger.info("Debug mode enabled")
    
    logger.info("Knowledge QA System initialized")


@main.command()
def version() -> None:
    """Show version information"""
    click.echo(f"{settings.app_name} v0.1.0")


if __name__ == "__main__":
    main()