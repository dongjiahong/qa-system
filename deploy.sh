#!/bin/bash

# Knowledge QA System Deployment Script
# This script helps deploy the Knowledge QA System in various environments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE_NAME="knowledge-qa-system"
DOCKER_TAG="latest"
COMPOSE_FILE="docker-compose.yml"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log_success "All dependencies are available"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p data logs
    
    # Copy environment file if it doesn't exist
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            log_info "Created .env file from .env.example"
        else
            log_warning ".env.example not found, creating minimal .env file"
            cat > .env << EOF
# Knowledge QA System Configuration
KNOWLEDGE_DATA_DIR=./data
KNOWLEDGE_LOG_DIR=./logs
OLLAMA_HOST=http://ollama:11434
CHROMADB_HOST=http://chromadb:8000
EOF
        fi
    fi
    
    log_success "Environment setup completed"
}

build_image() {
    log_info "Building Docker image..."
    
    docker build -t ${DOCKER_IMAGE_NAME}:${DOCKER_TAG} .
    
    log_success "Docker image built successfully"
}

deploy_with_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Pull latest images
    docker-compose -f ${COMPOSE_FILE} pull
    
    # Build and start services
    docker-compose -f ${COMPOSE_FILE} up -d --build
    
    log_success "Services deployed successfully"
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_services_health
}

check_services_health() {
    log_info "Checking service health..."
    
    # Check Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        log_success "Ollama service is healthy"
    else
        log_warning "Ollama service may not be ready yet"
    fi
    
    # Check ChromaDB
    if curl -s http://localhost:8000/api/v1/heartbeat > /dev/null; then
        log_success "ChromaDB service is healthy"
    else
        log_warning "ChromaDB service may not be ready yet"
    fi
    
    # Check main application
    if docker-compose -f ${COMPOSE_FILE} exec -T knowledge-qa knowledge --help > /dev/null; then
        log_success "Knowledge QA application is healthy"
    else
        log_warning "Knowledge QA application may not be ready yet"
    fi
}

show_status() {
    log_info "Service status:"
    docker-compose -f ${COMPOSE_FILE} ps
    
    echo ""
    log_info "Service logs (last 20 lines):"
    docker-compose -f ${COMPOSE_FILE} logs --tail=20
}

stop_services() {
    log_info "Stopping services..."
    docker-compose -f ${COMPOSE_FILE} down
    log_success "Services stopped"
}

cleanup() {
    log_info "Cleaning up..."
    docker-compose -f ${COMPOSE_FILE} down -v --remove-orphans
    docker image prune -f
    log_success "Cleanup completed"
}

install_local() {
    log_info "Installing locally..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    if [[ $(echo "$python_version >= 3.12" | bc -l) -eq 0 ]]; then
        log_error "Python 3.12 or higher is required. Current version: $python_version"
        exit 1
    fi
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_info "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Install in development mode
    pip install -e .
    
    log_success "Local installation completed"
    log_info "To use the application, activate the virtual environment:"
    log_info "  source venv/bin/activate"
    log_info "  knowledge --help"
}

show_usage() {
    echo "Knowledge QA System Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy      Deploy using Docker Compose (default)"
    echo "  build       Build Docker image only"
    echo "  start       Start existing services"
    echo "  stop        Stop running services"
    echo "  restart     Restart services"
    echo "  status      Show service status and logs"
    echo "  cleanup     Stop services and clean up"
    echo "  local       Install locally without Docker"
    echo "  check       Check environment and dependencies"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy           # Deploy all services"
    echo "  $0 local            # Install locally"
    echo "  $0 status           # Check service status"
    echo "  $0 cleanup          # Clean up everything"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        check_dependencies
        setup_environment
        deploy_with_docker_compose
        ;;
    "build")
        check_dependencies
        build_image
        ;;
    "start")
        check_dependencies
        docker-compose -f ${COMPOSE_FILE} start
        log_success "Services started"
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        check_dependencies
        docker-compose -f ${COMPOSE_FILE} restart
        log_success "Services restarted"
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "local")
        install_local
        ;;
    "check")
        check_dependencies
        python3 check_environment.py
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        log_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac