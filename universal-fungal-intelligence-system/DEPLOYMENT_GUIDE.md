# Universal Fungal Intelligence System - Deployment Guide

## 🚀 Deployment Options Overview

The Universal Fungal Intelligence System with the enhanced Molecular 3D Visualization Component can be deployed in several ways:

1. **Local Development** - Quick setup for development and testing
2. **Docker Container** - Containerized deployment
3. **Google Cloud Platform (GCP)** - Cloud deployment with BigQuery integration
4. **Streamlit Cloud** - Web deployment for the UI
5. **Heroku** - Alternative cloud deployment

## 📋 Prerequisites

### System Requirements
- Python 3.7+ (3.9+ recommended)
- 4GB+ RAM (8GB+ for large molecule processing)
- 2GB+ disk space
- Internet connection for API calls

### Required Tools
- Git
- pip (Python package manager)
- Docker (for containerized deployment)
- Google Cloud SDK (for GCP deployment)

## 🏠 Option 1: Local Development Deployment

### Quick Start (Recommended for Testing)

```bash
# 1. Clone the repository
git clone <repository-url>
cd universal-fungal-intelligence-system

# 2. Install dependencies
./install_molecular_3d_deps.sh

# 3. Test the installation
python3 test_quick.py

# 4. Run the web UI
python3 run_web_ui.py
```

### Manual Installation

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install additional dependencies for 3D visualization
pip install py3Dmol plotly

# 3. Test the molecular 3D component
python3 test_molecular_3d.py

# 4. Start the application
python3 run_web_ui.py
```

### Access the Application
- **Web UI**: http://localhost:8501
- **API**: http://localhost:8000 (if running main.py)

## 🐳 Option 2: Docker Deployment

### Using Docker Compose (Recommended)

```bash
# 1. Navigate to the docker directory
cd docker

# 2. Build and run with Docker Compose
docker-compose up --build

# 3. Access the application
# Web UI: http://localhost:8501
# API: http://localhost:5000
```

### Manual Docker Build

```bash
# 1. Build the Docker image
docker build -f docker/Dockerfile -t fungal-intelligence .

# 2. Run the container
docker run -p 8501:8501 -p 5000:5000 fungal-intelligence

# 3. Access the application
# Web UI: http://localhost:8501
# API: http://localhost:5000
```

### Docker with Volume Mounting (Development)

```bash
# Run with source code mounted for development
docker run -p 8501:8501 -p 5000:5000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
  fungal-intelligence
```

## ☁️ Option 3: Google Cloud Platform (GCP) Deployment

### Prerequisites
- Google Cloud account
- Google Cloud SDK installed
- Billing enabled on GCP project

### Automated Deployment

```bash
# 1. Authenticate with GCP
gcloud auth login

# 2. Set your project ID
gcloud config set project YOUR_PROJECT_ID

# 3. Run the automated deployment script
python3 scripts/deploy_to_gcp.py

# 4. Follow the prompts to complete setup
```

### Manual GCP Setup

```bash
# 1. Enable required APIs
gcloud services enable \
  bigquery.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com

# 2. Create service account
gcloud iam service-accounts create fungal-intelligence-sa \
  --display-name="Fungal Intelligence System"

# 3. Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:fungal-intelligence-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

# 4. Deploy to Cloud Run
gcloud run deploy fungal-intelligence \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### GCP with BigQuery Integration

```bash
# 1. Set up BigQuery dataset
bq mk --dataset YOUR_PROJECT_ID:crowe_ml_pipeline

# 2. Run analysis with BigQuery export
python3 scripts/deploy_to_gcp.py --no-export

# 3. View results in BigQuery console
# https://console.cloud.google.com/bigquery
```

## 🌐 Option 4: Streamlit Cloud Deployment

### Prerequisites
- Streamlit Cloud account (free tier available)
- GitHub repository with the code

### Deployment Steps

1. **Prepare the Repository**
   ```bash
   # Ensure requirements.txt includes all dependencies
   pip freeze > requirements.txt
   
   # Create .streamlit/config.toml for configuration
   mkdir -p .streamlit
   cat > .streamlit/config.toml << EOF
   [server]
   port = 8501
   address = "0.0.0.0"
   
   [browser]
   gatherUsageStats = false
   EOF
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Connect your GitHub repository
   - Set the main file path: `src/web_ui/app.py`
   - Deploy

3. **Access the Application**
   - Your app will be available at: `https://your-app-name.streamlit.app`

## 🚀 Option 5: Heroku Deployment

### Prerequisites
- Heroku account
- Heroku CLI installed

### Deployment Steps

```bash
# 1. Install Heroku CLI
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# 2. Login to Heroku
heroku login

# 3. Create Heroku app
heroku create your-fungal-intelligence-app

# 4. Set buildpacks
heroku buildpacks:set heroku/python

# 5. Deploy
git push heroku main

# 6. Open the app
heroku open
```

### Heroku Configuration

Create `Procfile`:
```
web: streamlit run src/web_ui/app.py --server.port=$PORT --server.address=0.0.0.0
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Database configuration
export DATABASE_URL="sqlite:///app/database.db"
export DATABASE_URL="postgresql://user:pass@host:port/db"

# Google Cloud configuration
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-key.json"
export GCP_PROJECT_ID="your-project-id"

# Application configuration
export ENV="production"
export DEBUG="false"
export LOG_LEVEL="INFO"

# Molecular 3D visualization
export CACHE_DIR=".molecular_cache"
export VIEWER_HEIGHT="500"
```

### Configuration Files

#### `.streamlit/config.toml`
```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

#### `config/production.py`
```python
import os

# Production configuration
DEBUG = False
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app/database.db')
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
CACHE_DIR = os.getenv('CACHE_DIR', '.molecular_cache')
```

## 🧪 Testing Deployment

### Pre-deployment Testing

```bash
# 1. Run unit tests
python3 -m pytest tests/unit/

# 2. Run integration tests
python3 -m pytest tests/integration/

# 3. Test molecular 3D component
python3 test_molecular_3d.py

# 4. Test web UI
python3 run_web_ui.py
# Then visit http://localhost:8501 and test features
```

### Post-deployment Testing

```bash
# 1. Test basic functionality
curl http://your-app-url/health

# 2. Test molecular 3D visualization
# Visit the web UI and test with example compounds

# 3. Test API endpoints (if applicable)
curl http://your-app-url/api/compounds

# 4. Test BigQuery integration (GCP only)
python3 scripts/test_bigquery_integration.py
```

## 📊 Monitoring and Logging

### Application Logs

```bash
# Local development
tail -f logs/app.log

# Docker
docker logs fungal-intelligence-container

# GCP Cloud Run
gcloud logging read "resource.type=cloud_run_revision"

# Heroku
heroku logs --tail
```

### Performance Monitoring

```bash
# Monitor memory usage
htop

# Monitor disk usage
df -h

# Monitor network connections
netstat -tulpn

# Monitor application performance
python3 scripts/monitor_performance.py
```

## 🔒 Security Considerations

### Production Security Checklist

- [ ] Use HTTPS in production
- [ ] Set secure environment variables
- [ ] Enable authentication (if required)
- [ ] Configure CORS properly
- [ ] Set up rate limiting
- [ ] Enable input validation
- [ ] Use secure database connections
- [ ] Regular security updates

### Security Configuration

```python
# Security headers
SECURE_HEADERS = {
    'X-Frame-Options': 'DENY',
    'X-Content-Type-Options': 'nosniff',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
}

# Rate limiting
RATE_LIMIT = "100 per minute"
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run src/web_ui/app.py --server.port 8502
```

#### 2. Dependencies Not Found
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Clear cache
pip cache purge
```

#### 3. RDKit Installation Issues
```bash
# Install RDKit via conda (recommended)
conda install -c conda-forge rdkit

# Or use Docker with RDKit pre-installed
docker run -it --rm -p 8501:8501 rdkit/rdkit-python3
```

#### 4. GCP Authentication Issues
```bash
# Re-authenticate
gcloud auth login

# Set application credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
```

### Debug Mode

```bash
# Enable debug mode
export DEBUG="true"
export LOG_LEVEL="DEBUG"

# Run with verbose logging
python3 run_web_ui.py --verbose
```

## 📈 Scaling Considerations

### Horizontal Scaling

```bash
# Docker Swarm
docker swarm init
docker stack deploy -c docker-compose.yml fungal-intelligence

# Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# GCP Cloud Run (auto-scaling)
gcloud run deploy fungal-intelligence \
  --source . \
  --platform managed \
  --region us-central1 \
  --min-instances 0 \
  --max-instances 10
```

### Vertical Scaling

```bash
# Increase memory limit
docker run -m 4g fungal-intelligence

# Increase CPU limit
docker run --cpus=2 fungal-intelligence
```

## 🎯 Next Steps

### Immediate Actions
1. **Choose deployment option** based on your needs
2. **Install dependencies** using the provided scripts
3. **Test the application** locally first
4. **Deploy to your chosen platform**
5. **Monitor and optimize** performance

### Future Enhancements
1. **Set up CI/CD pipeline** for automated deployments
2. **Implement monitoring** and alerting
3. **Add load balancing** for high traffic
4. **Set up backup and recovery** procedures
5. **Implement blue-green deployments**

## 📞 Support

For deployment issues:
1. Check the troubleshooting section
2. Review logs for error messages
3. Test with minimal configuration
4. Consult the documentation
5. Open an issue in the repository

---

**Happy Deploying! 🚀**

The enhanced Molecular 3D Visualization Component is now ready for production deployment across all major platforms. 