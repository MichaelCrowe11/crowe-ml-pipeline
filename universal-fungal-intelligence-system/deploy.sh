#!/bin/bash

# Universal Fungal Intelligence System - Enhanced Deployment Script
# This script provides easy deployment options for the enhanced Molecular 3D Visualization Component

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        required_version="3.7"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"; then
            print_success "Python $python_version found"
            return 0
        else
            print_error "Python 3.7+ required, found $python_version"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if [ -f "install_molecular_3d_deps.sh" ]; then
        chmod +x install_molecular_3d_deps.sh
        ./install_molecular_3d_deps.sh
    else
        print_warning "install_molecular_3d_deps.sh not found, installing manually..."
        pip3 install -r requirements.txt
        pip3 install py3Dmol plotly matplotlib seaborn
    fi
    
    print_success "Dependencies installed"
}

# Function to test the installation
test_installation() {
    print_status "Testing installation..."
    
    if python3 test_quick.py; then
        print_success "Quick test passed"
    else
        print_warning "Quick test failed, but continuing..."
    fi
}

# Function to deploy locally
deploy_local() {
    print_status "Deploying locally..."
    
    # Check if port 8501 is available
    if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
        print_warning "Port 8501 is already in use"
        read -p "Do you want to kill the process using port 8501? (y/n): " kill_process
        if [[ $kill_process =~ ^[Yy]$ ]]; then
            sudo kill -9 $(lsof -t -i:8501)
            sleep 2
        else
            print_error "Cannot deploy: port 8501 is in use"
            exit 1
        fi
    fi
    
    # Start the application
    print_status "Starting Streamlit application..."
    python3 run_web_ui.py &
    STREAMLIT_PID=$!
    
    # Wait for the application to start
    sleep 5
    
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        print_success "Application started successfully!"
        echo ""
        echo "🌐 Access your application at:"
        echo "   http://localhost:8501"
        echo ""
        echo "📊 Molecular 3D Visualization features:"
        echo "   - Interactive 3D molecular visualization"
        echo "   - Complete bond visualization (single, double, triple, aromatic)"
        echo "   - Export functionality (JSON, CSV, SDF)"
        echo "   - Batch processing capabilities"
        echo "   - Enhanced molecular properties"
        echo ""
        echo "Press Ctrl+C to stop the application"
        
        # Wait for user to stop
        wait $STREAMLIT_PID
    else
        print_error "Failed to start application"
        kill $STREAMLIT_PID 2>/dev/null || true
        exit 1
    fi
}

# Function to deploy with Docker
deploy_docker() {
    print_status "Deploying with Docker..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Build and run with Docker Compose
    cd docker
    print_status "Building Docker containers..."
    docker-compose build
    
    print_status "Starting Docker containers..."
    docker-compose up -d
    
    # Wait for the application to start
    sleep 10
    
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        print_success "Docker deployment successful!"
        echo ""
        echo "🌐 Access your application at:"
        echo "   http://localhost:8501"
        echo ""
        echo "📊 Docker services running:"
        docker-compose ps
        echo ""
        echo "To stop: cd docker && docker-compose down"
    else
        print_error "Docker deployment failed"
        docker-compose logs
        exit 1
    fi
}

# Function to deploy to GCP
deploy_gcp() {
    print_status "Deploying to Google Cloud Platform..."
    
    if ! command_exists gcloud; then
        print_error "Google Cloud SDK is not installed"
        echo "Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "Not authenticated with GCP"
        gcloud auth login
    fi
    
    # Run the GCP deployment script
    if [ -f "scripts/deploy_to_gcp.py" ]; then
        python3 scripts/deploy_to_gcp.py
    else
        print_error "GCP deployment script not found"
        exit 1
    fi
}

# Function to deploy to Streamlit Cloud
deploy_streamlit_cloud() {
    print_status "Preparing for Streamlit Cloud deployment..."
    
    # Create .streamlit directory and config
    mkdir -p .streamlit
    
    cat > .streamlit/config.toml << EOF
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
EOF
    
    # Update requirements.txt for Streamlit Cloud
    pip3 freeze > requirements.txt
    
    print_success "Streamlit Cloud configuration created!"
    echo ""
    echo "📋 Next steps for Streamlit Cloud deployment:"
    echo "1. Push your code to GitHub"
    echo "2. Go to https://share.streamlit.io/"
    echo "3. Connect your GitHub repository"
    echo "4. Set the main file path: src/web_ui/app.py"
    echo "5. Deploy"
    echo ""
    echo "Your app will be available at: https://your-app-name.streamlit.app"
}

# Function to show help
show_help() {
    echo "Universal Fungal Intelligence System - Enhanced Deployment Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  local     Deploy locally with Streamlit"
    echo "  docker    Deploy with Docker Compose"
    echo "  gcp       Deploy to Google Cloud Platform"
    echo "  streamlit Deploy to Streamlit Cloud (prepares configuration)"
    echo "  test      Run tests only"
    echo "  install   Install dependencies only"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local      # Deploy locally"
    echo "  $0 docker     # Deploy with Docker"
    echo "  $0 gcp        # Deploy to GCP"
    echo ""
    echo "For more information, see DEPLOYMENT_GUIDE.md"
}

# Main script
main() {
    echo "🍄 Universal Fungal Intelligence System - Enhanced Deployment"
    echo "=========================================================="
    echo ""
    
    # Check Python version
    if ! check_python_version; then
        exit 1
    fi
    
    # Parse command line arguments
    case "${1:-local}" in
        "local")
            install_dependencies
            test_installation
            deploy_local
            ;;
        "docker")
            deploy_docker
            ;;
        "gcp")
            install_dependencies
            deploy_gcp
            ;;
        "streamlit")
            deploy_streamlit_cloud
            ;;
        "test")
            install_dependencies
            test_installation
            print_status "Running full test suite..."
            python3 test_molecular_3d.py
            ;;
        "install")
            install_dependencies
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 