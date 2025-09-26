# FinScope Deployment Guide

This guide covers different deployment strategies for the FinScope application, from development to production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Development Deployment](#development-deployment)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring and Logging](#monitoring-and-logging)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 20GB minimum
- **Network**: Stable internet connection for API integrations

### Software Requirements

- **Node.js**: 18.0.0 or higher
- **Python**: 3.8 or higher
- **PostgreSQL**: 12.0 or higher (optional, SQLite fallback available)
- **Redis**: 6.0 or higher (optional, in-memory fallback available)
- **Docker**: 20.10.0 or higher (for containerized deployment)
- **Docker Compose**: 2.0.0 or higher

## Environment Variables

Create a `.env` file based on `.env.example` and configure the following variables:

### Required Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/finscope

# JWT Configuration
SECRET_KEY=your-super-secret-jwt-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Settings
ENVIRONMENT=production
DEBUG=false
CORS_ORIGINS=https://yourdomain.com
```

### API Keys (Optional but Recommended)

```bash
# Market Data APIs
COINGECKO_API_KEY=your-coingecko-api-key
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-api-key
FINNHUB_API_KEY=your-finnhub-api-key

# News APIs
NEWS_API_KEY=your-newsapi-key

# AI/ML APIs
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

## Development Deployment

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FinScope
   ```

2. **Run setup script**:
   ```bash
   # Linux/macOS
   chmod +x setup-dev.sh
   ./setup-dev.sh
   
   # Windows
   .\setup-dev.ps1
   ```

3. **Start development servers**:
   ```bash
   # Terminal 1 - Frontend
   npm run dev
   
   # Terminal 2 - Backend
   cd backend
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\Activate.ps1  # Windows
   uvicorn main:app --reload
   ```

### Manual Setup

If the setup script doesn't work, follow these manual steps:

1. **Install frontend dependencies**:
   ```bash
   npm install
   ```

2. **Setup backend**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Docker Deployment

### Development with Docker

1. **Start all services**:
   ```bash
   docker-compose up -d
   ```

2. **View logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Stop services**:
   ```bash
   docker-compose down
   ```

### Production with Docker

1. **Create production environment file**:
   ```bash
   cp .env.example .env.production
   # Edit .env.production with production values
   ```

2. **Build and start production containers**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

3. **Generate SSL certificates**:
   ```bash
   cd nginx
   ./generate-ssl.sh
   ```

## Production Deployment

### Server Setup

1. **Update system packages**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Docker and Docker Compose**:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

3. **Install Nginx (if not using Docker)**:
   ```bash
   sudo apt install nginx -y
   ```

### Application Deployment

1. **Clone repository**:
   ```bash
   git clone <repository-url> /opt/finscope
   cd /opt/finscope
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with production values
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

4. **Setup SSL certificates**:
   ```bash
   # For Let's Encrypt (recommended)
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d yourdomain.com
   
   # Or generate self-signed certificates
   cd nginx && ./generate-ssl.sh
   ```

### Reverse Proxy Configuration

If using external Nginx (not Docker), create `/etc/nginx/sites-available/finscope`:

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/finscope /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS

1. **Create ECS cluster**:
   ```bash
   aws ecs create-cluster --cluster-name finscope-cluster
   ```

2. **Build and push Docker images**:
   ```bash
   # Build images
   docker build -t finscope-frontend .
   docker build -t finscope-backend ./backend
   
   # Tag and push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   docker tag finscope-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/finscope-frontend:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/finscope-frontend:latest
   ```

3. **Create task definitions and services** using AWS Console or CLI

#### Using AWS Elastic Beanstalk

1. **Install EB CLI**:
   ```bash
   pip install awsebcli
   ```

2. **Initialize and deploy**:
   ```bash
   eb init
   eb create finscope-prod
   eb deploy
   ```

### Google Cloud Platform

#### Using Google Cloud Run

1. **Build and push images**:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/finscope-frontend
   gcloud builds submit --tag gcr.io/PROJECT-ID/finscope-backend ./backend
   ```

2. **Deploy services**:
   ```bash
   gcloud run deploy finscope-frontend --image gcr.io/PROJECT-ID/finscope-frontend --platform managed
   gcloud run deploy finscope-backend --image gcr.io/PROJECT-ID/finscope-backend --platform managed
   ```

### Digital Ocean

#### Using App Platform

1. **Create app specification** (`app.yaml`):
   ```yaml
   name: finscope
   services:
   - name: frontend
     source_dir: /
     github:
       repo: your-username/finscope
       branch: main
     run_command: npm start
     environment_slug: node-js
     instance_count: 1
     instance_size_slug: basic-xxs
   - name: backend
     source_dir: /backend
     github:
       repo: your-username/finscope
       branch: main
     run_command: uvicorn main:app --host 0.0.0.0 --port 8080
     environment_slug: python
     instance_count: 1
     instance_size_slug: basic-xxs
   databases:
   - name: finscope-db
     engine: PG
     version: "12"
   ```

2. **Deploy**:
   ```bash
   doctl apps create --spec app.yaml
   ```

## Monitoring and Logging

### Application Monitoring

1. **Health Checks**:
   - Frontend: `https://yourdomain.com/api/health`
   - Backend: `https://yourdomain.com/health`

2. **Log Aggregation**:
   ```bash
   # View Docker logs
   docker-compose logs -f
   
   # View specific service logs
   docker-compose logs -f backend
   docker-compose logs -f frontend
   ```

3. **Performance Monitoring**:
   - Use tools like New Relic, DataDog, or Prometheus
   - Monitor CPU, memory, and disk usage
   - Track API response times and error rates

### Database Monitoring

1. **PostgreSQL monitoring**:
   ```sql
   -- Check active connections
   SELECT count(*) FROM pg_stat_activity;
   
   -- Check database size
   SELECT pg_size_pretty(pg_database_size('finscope'));
   ```

2. **Backup strategy**:
   ```bash
   # Create backup
   docker-compose exec postgres pg_dump -U finscope_user finscope > backup.sql
   
   # Restore backup
   docker-compose exec -T postgres psql -U finscope_user finscope < backup.sql
   ```

## Security Considerations

### SSL/TLS Configuration

1. **Use strong SSL configuration**:
   - TLS 1.2 or higher
   - Strong cipher suites
   - HSTS headers

2. **Certificate management**:
   - Use Let's Encrypt for free SSL certificates
   - Set up automatic renewal
   - Monitor certificate expiration

### Application Security

1. **Environment variables**:
   - Never commit secrets to version control
   - Use strong, unique passwords
   - Rotate API keys regularly

2. **Network security**:
   - Use firewalls to restrict access
   - Implement rate limiting
   - Use VPN for administrative access

3. **Database security**:
   - Use strong database passwords
   - Limit database access to application only
   - Enable database encryption at rest

### Regular Updates

1. **Keep dependencies updated**:
   ```bash
   # Update Node.js dependencies
   npm audit fix
   
   # Update Python dependencies
   pip list --outdated
   ```

2. **System updates**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

## Troubleshooting

### Common Issues

1. **Database connection errors**:
   - Check database credentials in `.env`
   - Verify database is running
   - Check network connectivity

2. **API key errors**:
   - Verify API keys are correct
   - Check API rate limits
   - Ensure APIs are accessible from your server

3. **Docker issues**:
   ```bash
   # Rebuild containers
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   
   # Check container logs
   docker-compose logs -f [service-name]
   
   # Check container status
   docker-compose ps
   ```

4. **SSL certificate issues**:
   ```bash
   # Check certificate validity
   openssl x509 -in nginx/ssl/cert.pem -text -noout
   
   # Test SSL configuration
   openssl s_client -connect yourdomain.com:443
   ```

### Performance Issues

1. **High CPU usage**:
   - Check for infinite loops in code
   - Monitor API call frequency
   - Consider caching strategies

2. **High memory usage**:
   - Check for memory leaks
   - Monitor database query performance
   - Consider increasing server resources

3. **Slow API responses**:
   - Check database query performance
   - Monitor external API response times
   - Implement caching where appropriate

### Getting Help

1. **Check logs first**:
   ```bash
   # Application logs
   docker-compose logs -f
   
   # System logs
   sudo journalctl -u docker
   sudo journalctl -u nginx
   ```

2. **Debug mode**:
   - Set `DEBUG=true` in `.env` for detailed error messages
   - Use browser developer tools for frontend issues
   - Check API documentation at `/docs` endpoint

3. **Community support**:
   - Check GitHub issues
   - Join community forums
   - Contact support team

---

## Conclusion

This deployment guide covers various scenarios from development to production. Choose the deployment method that best fits your needs and infrastructure. Always test deployments in a staging environment before deploying to production.

For additional help or questions, please refer to the main README.md or contact the development team.