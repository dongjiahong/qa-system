# Knowledge QA System - Deployment Guide

This guide covers various deployment options for the Knowledge QA System.

## Quick Start

### Using Docker (Recommended)

1. **Check environment**:
   ```bash
   python check_environment.py
   # or
   make check-env
   ```

2. **Deploy with Docker Compose**:
   ```bash
   ./deploy.sh deploy
   # or
   make deploy
   ```

3. **Check service status**:
   ```bash
   ./deploy.sh status
   ```

### Local Installation

1. **Install locally**:
   ```bash
   ./deploy.sh local
   ```

2. **Activate environment and use**:
   ```bash
   source venv/bin/activate
   knowledge --help
   ```

## Deployment Options

### 1. Docker Compose Deployment (Production Ready)

This is the recommended deployment method that includes all required services.

**Services included:**
- Knowledge QA System (main application)
- Ollama (LLM service)
- ChromaDB (vector database)

**Commands:**
```bash
# Deploy all services
./deploy.sh deploy

# Check status
./deploy.sh status

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart

# Clean up everything
./deploy.sh cleanup
```

**Configuration:**
- Edit `docker-compose.yml` to customize service settings
- Modify `.env` file for environment variables
- Data persists in `./data` and `./logs` directories

### 2. Docker Image Only

Build and run just the main application container:

```bash
# Build image
./deploy.sh build

# Run container manually
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  knowledge-qa-system:latest \
  knowledge --help
```

### 3. Local Development Installation

For development or when Docker is not available:

```bash
# Install locally
./deploy.sh local

# Activate virtual environment
source venv/bin/activate

# Use the application
knowledge --help
```

## Environment Check

Before deployment, run the environment check tool:

```bash
python check_environment.py
```

This will verify:
- Python version (>= 3.12)
- Required packages
- Docker installation
- Service availability
- Project structure
- Disk space

## Package Building and Distribution

### Build Package

```bash
# Build wheel and source distribution
python build_package.py

# Or using make
make package
```

### Upload to PyPI

```bash
# Upload to TestPyPI (for testing)
python build_package.py --upload test

# Upload to PyPI (production)
python build_package.py --upload prod
```

## Configuration

### Environment Variables

Key environment variables (set in `.env` file):

```bash
# Data and logging
KNOWLEDGE_DATA_DIR=./data
KNOWLEDGE_LOG_DIR=./logs

# Service endpoints (for Docker deployment)
OLLAMA_HOST=http://ollama:11434
CHROMADB_HOST=http://chromadb:8000

# For local deployment
OLLAMA_HOST=http://localhost:11434
CHROMADB_HOST=http://localhost:8000
```

### Docker Configuration

**Dockerfile features:**
- Multi-stage build for optimization
- Non-root user for security
- Health checks
- Proper caching layers

**docker-compose.yml features:**
- Service orchestration
- Volume persistence
- Network isolation
- Automatic restarts
- Service dependencies

## Troubleshooting

### Common Issues

1. **Docker daemon not running**:
   ```bash
   sudo systemctl start docker
   ```

2. **Port conflicts**:
   - Ollama: Change port 11434 in docker-compose.yml
   - ChromaDB: Change port 8000 in docker-compose.yml

3. **Permission issues**:
   ```bash
   sudo chown -R $USER:$USER data logs
   ```

4. **Service not ready**:
   ```bash
   # Wait for services to start
   ./deploy.sh status
   
   # Check logs
   docker-compose logs -f
   ```

### Service Health Checks

**Check Ollama**:
```bash
curl http://localhost:11434/api/tags
```

**Check ChromaDB**:
```bash
curl http://localhost:8000/api/v1/heartbeat
```

**Check main application**:
```bash
docker-compose exec knowledge-qa knowledge --help
```

### Logs

**View all logs**:
```bash
docker-compose logs -f
```

**View specific service logs**:
```bash
docker-compose logs -f knowledge-qa
docker-compose logs -f ollama
docker-compose logs -f chromadb
```

## Performance Tuning

### Resource Limits

Edit `docker-compose.yml` to add resource limits:

```yaml
services:
  knowledge-qa:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Volume Optimization

For better performance, use named volumes instead of bind mounts:

```yaml
volumes:
  knowledge_data:
    driver: local
  knowledge_logs:
    driver: local
```

## Security Considerations

1. **Non-root containers**: All containers run as non-root users
2. **Network isolation**: Services communicate through internal network
3. **Volume permissions**: Proper file permissions for mounted volumes
4. **Environment variables**: Sensitive data should be in `.env` file
5. **Firewall**: Only expose necessary ports

## Monitoring

### Health Checks

All services include health checks:
- Application: CLI command execution
- Ollama: API endpoint check
- ChromaDB: Heartbeat endpoint

### Metrics

For production monitoring, consider adding:
- Prometheus metrics
- Grafana dashboards
- Log aggregation (ELK stack)
- Alerting (AlertManager)

## Backup and Recovery

### Data Backup

```bash
# Backup data directory
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Backup Docker volumes
docker run --rm -v knowledge-qa_chromadb_data:/data -v $(pwd):/backup alpine tar czf /backup/chromadb-backup.tar.gz /data
```

### Recovery

```bash
# Restore data directory
tar -xzf backup-20240101.tar.gz

# Restore Docker volumes
docker run --rm -v knowledge-qa_chromadb_data:/data -v $(pwd):/backup alpine tar xzf /backup/chromadb-backup.tar.gz -C /
```

## Scaling

For high-load scenarios:

1. **Horizontal scaling**: Run multiple application instances
2. **Load balancing**: Use nginx or HAProxy
3. **Database scaling**: Use ChromaDB clustering
4. **Caching**: Add Redis for query caching

## Support

For deployment issues:
1. Run environment check: `python check_environment.py`
2. Check service logs: `./deploy.sh status`
3. Verify configuration files
4. Check system resources (CPU, memory, disk)