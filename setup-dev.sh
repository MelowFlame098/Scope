#!/bin/bash

# FinScope Development Setup Script
# This script sets up the development environment for FinScope

set -e  # Exit on any error

echo "🚀 Setting up FinScope Development Environment..."

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

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ from https://python.org/"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip3."
        exit 1
    fi
    
    # Check Docker (optional)
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Docker setup will be skipped."
        DOCKER_AVAILABLE=false
    else
        DOCKER_AVAILABLE=true
    fi
    
    print_success "All requirements checked!"
}

# Setup environment variables
setup_env() {
    print_status "Setting up environment variables..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_success "Created .env file from .env.example"
        print_warning "Please update the .env file with your actual API keys and configuration"
    else
        print_warning ".env file already exists, skipping..."
    fi
}

# Install frontend dependencies
setup_frontend() {
    print_status "Installing frontend dependencies..."
    
    npm install
    
    print_success "Frontend dependencies installed!"
}

# Setup Python virtual environment and install backend dependencies
setup_backend() {
    print_status "Setting up backend environment..."
    
    cd backend
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created Python virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    pip install -r requirements.txt
    
    print_success "Backend dependencies installed!"
    
    cd ..
}

# Setup database (if Docker is available)
setup_database() {
    if [ "$DOCKER_AVAILABLE" = true ]; then
        print_status "Setting up database with Docker..."
        
        # Start only PostgreSQL and Redis
        docker-compose up -d postgres redis
        
        # Wait for database to be ready
        print_status "Waiting for database to be ready..."
        sleep 10
        
        print_success "Database setup completed!"
    else
        print_warning "Docker not available. Please set up PostgreSQL and Redis manually."
        print_warning "Alternatively, the application will use SQLite as fallback."
    fi
}

# Generate SSL certificates for development
setup_ssl() {
    print_status "Generating SSL certificates for development..."
    
    cd nginx
    chmod +x generate-ssl.sh
    ./generate-ssl.sh
    cd ..
    
    print_success "SSL certificates generated!"
}

# Create necessary directories
setup_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p backend/logs
    mkdir -p nginx/ssl
    
    print_success "Directories created!"
}

# Main setup function
main() {
    echo "=========================================="
    echo "       FinScope Development Setup        "
    echo "=========================================="
    echo ""
    
    check_requirements
    setup_directories
    setup_env
    setup_frontend
    setup_backend
    setup_database
    setup_ssl
    
    echo ""
    echo "=========================================="
    print_success "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Update the .env file with your API keys"
    echo "2. Start the development servers:"
    echo "   Frontend: npm run dev"
    echo "   Backend: cd backend && source venv/bin/activate && uvicorn main:app --reload"
    echo "3. Or use Docker: docker-compose up"
    echo ""
    echo "Access the application:"
    echo "- Frontend: http://localhost:3000"
    echo "- Backend API: http://localhost:8000"
    echo "- API Documentation: http://localhost:8000/docs"
    echo ""
    print_warning "Don't forget to update your .env file with actual API keys!"
}

# Run main function
main "$@"