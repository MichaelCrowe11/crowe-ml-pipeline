# 🚀 Fly.io Deployment Guide for Crowe ML Pipeline

## Overview

This guide covers deploying the complete Crowe ML Pipeline platform to Fly.io, including:
- Main ML Pipeline API
- Vision Platform (Next.js 3D UI)
- CriOS Research OS
- Persistent storage for models and data

## Prerequisites

1. **Fly.io Account**: Sign up at [fly.io](https://fly.io)
2. **Fly CLI**: Install the Fly CLI
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```
3. **Authentication**: Log in to Fly
   ```bash
   flyctl auth login
   ```

## Quick Deploy

### Automated Deployment

Run the deployment script for a complete setup:

```bash
chmod +x deploy.sh
./deploy.sh
```

This will:
- Deploy the main ML Pipeline
- Deploy the Vision Platform
- Set up persistent volumes
- Configure scaling
- Set up custom domains (optional)

## Manual Deployment Steps

### 1. Deploy Main ML Pipeline

```bash
# Create the app
flyctl apps create crowe-ml-pipeline

# Set secrets
flyctl secrets set CLAUDE_API_KEY="your-key-here" --app crowe-ml-pipeline
flyctl secrets set GCP_PROJECT_ID="your-project-id" --app crowe-ml-pipeline

# Deploy
flyctl deploy --app crowe-ml-pipeline

# Create persistent volume for data
flyctl volumes create crowe_data --size 10 --app crowe-ml-pipeline
```

### 2. Deploy Vision Platform

```bash
cd crowe-vision-platform

# Create the app
flyctl apps create crowe-vision

# Build and deploy
npm install
npm run build
flyctl deploy --app crowe-vision

# Create cache volume
flyctl volumes create vision_cache --size 1 --app crowe-vision

cd ..
```

### 3. Configure Scaling

```bash
# Scale the main API (2 instances)
flyctl scale count 2 --app crowe-ml-pipeline

# Set memory limits
flyctl scale memory 2048 --app crowe-ml-pipeline
flyctl scale memory 1024 --app crowe-vision
```

### 4. Set Up Custom Domains (Optional)

```bash
# Add domains
flyctl domains add yourdomain.com --app crowe-ml-pipeline
flyctl domains add app.yourdomain.com --app crowe-vision

# Get DNS records to add
flyctl domains list --app crowe-ml-pipeline
flyctl domains list --app crowe-vision
```

## Environment Variables

### Required Secrets

Set these using `flyctl secrets set`:

```bash
# AI Configuration
CLAUDE_API_KEY=sk-ant-...
QWEN_ENDPOINT=https://your-qwen-endpoint

# Google Cloud
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=base64-encoded-service-account

# Database (if using external DB)
DATABASE_URL=postgresql://user:pass@host:5432/db
```

### Application Configuration

These are set in `fly.toml`:

```toml
[env]
  PORT = "8080"
  NODE_ENV = "production"
  PYTHON_VERSION = "3.10"
  NEXT_PUBLIC_API_URL = "https://crowe-ml-pipeline.fly.dev"
```

## Deployed URLs

After deployment, your services will be available at:

- **Main API**: https://crowe-ml-pipeline.fly.dev
- **Vision Platform**: https://crowe-vision.fly.dev
- **API Documentation**: https://crowe-ml-pipeline.fly.dev/docs
- **Health Check**: https://crowe-ml-pipeline.fly.dev/health

## Monitoring & Management

### View Logs

```bash
# Main API logs
flyctl logs --app crowe-ml-pipeline

# Vision Platform logs
flyctl logs --app crowe-vision

# Stream logs
flyctl logs --app crowe-ml-pipeline -f
```

### SSH into Container

```bash
flyctl ssh console --app crowe-ml-pipeline
```

### Check Status

```bash
flyctl status --app crowe-ml-pipeline
flyctl status --app crowe-vision
```

### Monitor Resources

```bash
# View metrics dashboard
flyctl dashboard --app crowe-ml-pipeline

# Check resource usage
flyctl scale show --app crowe-ml-pipeline
```

## CI/CD with GitHub Actions

The repository includes a GitHub Actions workflow for automated deployment:

1. **Set up GitHub Secrets**:
   ```
   FLY_API_TOKEN: Your Fly.io API token
   SLACK_WEBHOOK: (Optional) For deployment notifications
   ```

2. **Get your Fly API token**:
   ```bash
   flyctl auth token
   ```

3. **Add to GitHub**:
   - Go to Settings → Secrets → Actions
   - Add `FLY_API_TOKEN` with your token

4. **Deploy on Push**:
   - Push to `main` or `production` branch
   - GitHub Actions will automatically deploy

## Troubleshooting

### Common Issues

1. **Port binding errors**:
   ```bash
   # Ensure PORT environment variable is used
   PORT = process.env.PORT || 8080
   ```

2. **Memory issues**:
   ```bash
   # Increase memory allocation
   flyctl scale memory 4096 --app crowe-ml-pipeline
   ```

3. **Volume mounting errors**:
   ```bash
   # List volumes
   flyctl volumes list --app crowe-ml-pipeline
   
   # Attach volume to app
   flyctl volumes attach <volume-id> --app crowe-ml-pipeline
   ```

4. **Build failures**:
   ```bash
   # Build locally first
   docker build -f Dockerfile.fly -t crowe-ml-pipeline .
   
   # Test locally
   docker run -p 8080:8080 crowe-ml-pipeline
   ```

### Health Checks

The platform includes health check endpoints:

```bash
# Check main API
curl https://crowe-ml-pipeline.fly.dev/health

# Check Vision Platform
curl https://crowe-vision.fly.dev/

# Expected response
{
  "status": "healthy",
  "services": {
    "ml_pipeline": "online",
    "fungal_intelligence": "online",
    "molecular_analyzer": "online",
    "bioactivity_predictor": "online"
  }
}
```

## Backup & Recovery

### Backup Data Volumes

```bash
# Create snapshot
flyctl volumes snapshots create vol_<id> --app crowe-ml-pipeline

# List snapshots
flyctl volumes snapshots list --app crowe-ml-pipeline
```

### Restore from Snapshot

```bash
# Create new volume from snapshot
flyctl volumes create crowe_data_restore \
  --snapshot-id <snapshot-id> \
  --app crowe-ml-pipeline
```

## Cost Optimization

### Fly.io Pricing

- **Hobby Plan**: Free tier includes:
  - 3 shared-cpu-1x VMs (256MB RAM)
  - 3GB persistent storage
  - 160GB outbound data transfer

- **Scale Plan**: For production:
  - ~$0.0067/hour per shared-cpu-1x (256MB)
  - ~$0.15/GB for persistent volumes
  - ~$0.02/GB for bandwidth

### Optimization Tips

1. **Use shared CPUs** for non-intensive workloads
2. **Scale horizontally** instead of vertically when possible
3. **Enable auto-scaling** for variable loads
4. **Use caching** to reduce compute needs
5. **Optimize Docker images** to reduce deployment time

## Security Best Practices

1. **Use secrets** for sensitive data
2. **Enable HTTPS** (automatic with Fly.io)
3. **Restrict CORS** in production
4. **Implement rate limiting**
5. **Regular security updates**
6. **Monitor access logs**

## Support & Resources

- **Fly.io Documentation**: https://fly.io/docs
- **Fly.io Status**: https://status.fly.io
- **Community Forum**: https://community.fly.io
- **Platform Issues**: Open issue on GitHub

## Next Steps

After deployment:

1. ✅ Verify all endpoints are accessible
2. ✅ Test the Vision Platform UI
3. ✅ Run a test compound analysis
4. ✅ Check logs for any errors
5. ✅ Set up monitoring alerts
6. ✅ Configure backups
7. ✅ Document API endpoints for users

---

**Deployment Version**: 2.0.0  
**Platform**: Fly.io  
**Last Updated**: January 2025

