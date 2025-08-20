#!/bin/bash
# Comprehensive deployment script for Crowe ML Pipeline on Fly.io

set -e

echo "🚀 Deploying Crowe ML Pipeline to Fly.io"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if fly CLI is installed
if ! command -v flyctl &> /dev/null; then
    print_error "Fly CLI not found. Installing..."
    curl -L https://fly.io/install.sh | sh
    export FLYCTL_INSTALL="/home/$USER/.fly"
    export PATH="$FLYCTL_INSTALL/bin:$PATH"
fi

# Function to deploy a service
deploy_service() {
    local service_name=$1
    local service_dir=$2
    local fly_config=$3
    
    echo ""
    echo "Deploying $service_name..."
    echo "------------------------"
    
    cd "$service_dir"
    
    # Check if app exists
    if flyctl apps list | grep -q "$service_name"; then
        print_warning "App $service_name already exists, updating..."
    else
        print_status "Creating app $service_name..."
        flyctl apps create "$service_name" --org personal || true
    fi
    
    # Deploy the service
    if [ -f "$fly_config" ]; then
        flyctl deploy --config "$fly_config" --strategy immediate
        print_status "$service_name deployed successfully!"
    else
        print_error "Configuration file $fly_config not found!"
        return 1
    fi
    
    cd - > /dev/null
}

# Main deployment process
main() {
    echo ""
    print_status "Starting deployment process..."
    
    # 1. Set up secrets and environment variables
    print_status "Setting up secrets..."
    
    # Check if we need to set secrets
    read -p "Do you want to set up API keys? (y/n): " setup_keys
    if [ "$setup_keys" = "y" ]; then
        read -p "Enter Claude API Key (or press Enter to skip): " claude_key
        if [ ! -z "$claude_key" ]; then
            flyctl secrets set CLAUDE_API_KEY="$claude_key" --app crowe-ml-pipeline
        fi
        
        read -p "Enter Google Cloud Project ID: " gcp_project
        if [ ! -z "$gcp_project" ]; then
            flyctl secrets set GCP_PROJECT_ID="$gcp_project" --app crowe-ml-pipeline
        fi
    fi
    
    # 2. Build and deploy main ML Pipeline
    print_status "Building main ML Pipeline..."
    
    # Create production requirements file if using poetry
    if [ -f "pyproject.toml" ]; then
        poetry export -f requirements.txt --output requirements.txt --without-hashes
    fi
    
    # Deploy main application
    deploy_service "crowe-ml-pipeline" "." "fly.toml"
    
    # 3. Deploy Vision Platform
    print_status "Deploying Vision Platform..."
    
    # Build Vision Platform
    cd crowe-vision-platform
    npm install
    npm run build
    cd ..
    
    deploy_service "crowe-vision" "crowe-vision-platform" "crowe-vision-platform/fly.toml"
    
    # 4. Create volumes for persistent storage
    print_status "Setting up persistent storage..."
    
    flyctl volumes create crowe_data --size 10 --app crowe-ml-pipeline --region sjc || true
    flyctl volumes create vision_cache --size 1 --app crowe-vision --region sjc || true
    
    # 5. Scale applications
    print_status "Scaling applications..."
    
    flyctl scale count 2 --app crowe-ml-pipeline
    flyctl scale count 1 --app crowe-vision
    
    # 6. Set up custom domains (optional)
    read -p "Do you have a custom domain to set up? (y/n): " setup_domain
    if [ "$setup_domain" = "y" ]; then
        read -p "Enter your domain (e.g., croweml.com): " domain
        
        flyctl domains add "$domain" --app crowe-ml-pipeline
        flyctl domains add "api.$domain" --app crowe-ml-pipeline
        flyctl domains add "app.$domain" --app crowe-vision
        
        print_status "Domain setup complete. Add these DNS records:"
        flyctl domains list --app crowe-ml-pipeline
        flyctl domains list --app crowe-vision
    fi
    
    # 7. Display deployment information
    echo ""
    echo "========================================"
    print_status "Deployment Complete!"
    echo "========================================"
    echo ""
    echo "🌐 Your applications are available at:"
    echo "   Main API: https://crowe-ml-pipeline.fly.dev"
    echo "   Vision Platform: https://crowe-vision.fly.dev"
    echo ""
    echo "📊 Monitoring:"
    echo "   Main API: https://fly.io/apps/crowe-ml-pipeline"
    echo "   Vision Platform: https://fly.io/apps/crowe-vision"
    echo ""
    echo "🔧 Useful commands:"
    echo "   View logs: flyctl logs --app crowe-ml-pipeline"
    echo "   SSH into app: flyctl ssh console --app crowe-ml-pipeline"
    echo "   Check status: flyctl status --app crowe-ml-pipeline"
    echo "   Scale app: flyctl scale count 3 --app crowe-ml-pipeline"
    echo ""
}

# Run main deployment
main

# Optional: Open the deployed app
read -p "Would you like to open the deployed application? (y/n): " open_app
if [ "$open_app" = "y" ]; then
    flyctl open --app crowe-vision
fi
