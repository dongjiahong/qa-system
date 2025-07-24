# Makefile for Knowledge QA System

.PHONY: help install install-dev test lint format clean run build package deploy docker-build docker-deploy check-env

help:
	@echo "Available commands:"
	@echo "  install     - Install production dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  clean       - Clean build artifacts"
	@echo "  run         - Run the application"
	@echo "  build       - Build package distribution"
	@echo "  package     - Build and package for distribution"
	@echo "  deploy      - Deploy using Docker Compose"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-deploy - Deploy with Docker"
	@echo "  check-env   - Check environment and dependencies"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.cli --help

build:
	python -m build

package:
	python build_package.py

deploy:
	./deploy.sh deploy

docker-build:
	./deploy.sh build

docker-deploy:
	./deploy.sh deploy

check-env:
	python check_environment.py