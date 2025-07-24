#!/usr/bin/env python3
"""
Environment check tool for Knowledge QA System.
This script verifies that all required dependencies and services are available.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import requests


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color


class EnvironmentChecker:
    """Environment checker for Knowledge QA System."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        
    def log_info(self, message: str):
        """Log info message."""
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
    
    def log_success(self, message: str):
        """Log success message."""
        print(f"{Colors.GREEN}[✓]{Colors.NC} {message}")
        self.checks_passed += 1
    
    def log_error(self, message: str):
        """Log error message."""
        print(f"{Colors.RED}[✗]{Colors.NC} {message}")
        self.checks_failed += 1
    
    def log_warning(self, message: str):
        """Log warning message."""
        print(f"{Colors.YELLOW}[!]{Colors.NC} {message}")
        self.warnings += 1
    
    def run_command(self, cmd: str, capture_output: bool = True) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=capture_output, text=True, timeout=30
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    def check_python_version(self):
        """Check Python version."""
        self.log_info("Checking Python version...")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor >= 12:
            self.log_success(f"Python {version_str} (✓ >= 3.12)")
        else:
            self.log_error(f"Python {version_str} (✗ requires >= 3.12)")
    
    def check_system_info(self):
        """Check system information."""
        self.log_info("System Information:")
        print(f"  OS: {platform.system()} {platform.release()}")
        print(f"  Architecture: {platform.machine()}")
        print(f"  Python: {sys.version}")
    
    def check_required_packages(self):
        """Check if required Python packages are installed."""
        self.log_info("Checking required Python packages...")
        
        required_packages = [
            "click",
            "pydantic",
            "llama_index",
            "chromadb",
            "ollama",
            "requests",
            "rich",
            "loguru",
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.log_success(f"Package '{package}' is installed")
            except ImportError:
                self.log_error(f"Package '{package}' is not installed")
    
    def check_docker(self):
        """Check Docker installation and status."""
        self.log_info("Checking Docker...")
        
        # Check if Docker is installed
        success, output = self.run_command("docker --version")
        if success:
            self.log_success(f"Docker is installed: {output}")
        else:
            self.log_error("Docker is not installed")
            return
        
        # Check if Docker daemon is running
        success, _ = self.run_command("docker info")
        if success:
            self.log_success("Docker daemon is running")
        else:
            self.log_error("Docker daemon is not running")
    
    def check_docker_compose(self):
        """Check Docker Compose installation."""
        self.log_info("Checking Docker Compose...")
        
        # Try docker-compose command
        success, output = self.run_command("docker-compose --version")
        if success:
            self.log_success(f"Docker Compose is installed: {output}")
            return
        
        # Try docker compose command (newer syntax)
        success, output = self.run_command("docker compose version")
        if success:
            self.log_success(f"Docker Compose is installed: {output}")
        else:
            self.log_error("Docker Compose is not installed")
    
    def check_ollama_service(self):
        """Check if Ollama service is running."""
        self.log_info("Checking Ollama service...")
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.log_success("Ollama service is running")
                
                # Check if Qwen model is available
                tags = response.json()
                models = [model["name"] for model in tags.get("models", [])]
                if any("qwen" in model.lower() for model in models):
                    self.log_success("Qwen model is available")
                else:
                    self.log_warning("Qwen model not found. Run: ollama pull qwen:1.7b")
            else:
                self.log_error(f"Ollama service returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.log_warning("Ollama service is not running (this is OK if using Docker)")
        except Exception as e:
            self.log_error(f"Error checking Ollama service: {e}")
    
    def check_chromadb_service(self):
        """Check if ChromaDB service is running."""
        self.log_info("Checking ChromaDB service...")
        
        try:
            response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=5)
            if response.status_code == 200:
                self.log_success("ChromaDB service is running")
            else:
                self.log_error(f"ChromaDB service returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.log_warning("ChromaDB service is not running (this is OK if using Docker)")
        except Exception as e:
            self.log_error(f"Error checking ChromaDB service: {e}")
    
    def check_project_structure(self):
        """Check project structure and required files."""
        self.log_info("Checking project structure...")
        
        required_files = [
            "pyproject.toml",
            "requirements.txt",
            "src/cli.py",
            "src/models.py",
            "src/database.py",
            "src/vector_store.py",
            "src/llm_client.py",
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                self.log_success(f"Found {file_path}")
            else:
                self.log_error(f"Missing {file_path}")
        
        # Check data directory
        data_dir = Path("data")
        if data_dir.exists():
            self.log_success("Data directory exists")
        else:
            self.log_warning("Data directory doesn't exist (will be created automatically)")
    
    def check_environment_variables(self):
        """Check environment variables and configuration."""
        self.log_info("Checking environment configuration...")
        
        # Check for .env file
        env_file = Path(".env")
        if env_file.exists():
            self.log_success(".env file exists")
        else:
            env_example = Path(".env.example")
            if env_example.exists():
                self.log_warning(".env file not found, but .env.example exists")
            else:
                self.log_error("Neither .env nor .env.example found")
        
        # Check important environment variables
        important_vars = [
            "KNOWLEDGE_DATA_DIR",
            "KNOWLEDGE_LOG_DIR",
        ]
        
        for var in important_vars:
            value = os.getenv(var)
            if value:
                self.log_success(f"{var} = {value}")
            else:
                self.log_warning(f"{var} not set (will use default)")
    
    def check_disk_space(self):
        """Check available disk space."""
        self.log_info("Checking disk space...")
        
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            
            free_gb = free // (1024**3)
            total_gb = total // (1024**3)
            
            if free_gb >= 5:
                self.log_success(f"Disk space: {free_gb}GB free / {total_gb}GB total")
            elif free_gb >= 1:
                self.log_warning(f"Low disk space: {free_gb}GB free / {total_gb}GB total")
            else:
                self.log_error(f"Very low disk space: {free_gb}GB free / {total_gb}GB total")
        except Exception as e:
            self.log_error(f"Error checking disk space: {e}")
    
    def run_all_checks(self):
        """Run all environment checks."""
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
        print(f"{Colors.WHITE}Knowledge QA System - Environment Check{Colors.NC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
        print()
        
        self.check_system_info()
        print()
        
        self.check_python_version()
        self.check_required_packages()
        print()
        
        self.check_docker()
        self.check_docker_compose()
        print()
        
        self.check_ollama_service()
        self.check_chromadb_service()
        print()
        
        self.check_project_structure()
        self.check_environment_variables()
        self.check_disk_space()
        
        print()
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
        print(f"{Colors.WHITE}Summary{Colors.NC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
        
        print(f"{Colors.GREEN}Checks passed: {self.checks_passed}{Colors.NC}")
        if self.warnings > 0:
            print(f"{Colors.YELLOW}Warnings: {self.warnings}{Colors.NC}")
        if self.checks_failed > 0:
            print(f"{Colors.RED}Checks failed: {self.checks_failed}{Colors.NC}")
        
        if self.checks_failed == 0:
            print(f"\n{Colors.GREEN}✓ Environment is ready for Knowledge QA System!{Colors.NC}")
            return True
        else:
            print(f"\n{Colors.RED}✗ Environment has issues that need to be resolved.{Colors.NC}")
            return False


def main():
    """Main function."""
    checker = EnvironmentChecker()
    success = checker.run_all_checks()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()