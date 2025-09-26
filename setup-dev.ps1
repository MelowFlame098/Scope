# FinScope Development Setup Script for Windows
# This script sets up the development environment for FinScope on Windows

param(
    [switch]$SkipDocker,
    [switch]$Help
)

if ($Help) {
    Write-Host "FinScope Development Setup Script" -ForegroundColor Cyan
    Write-Host "Usage: .\setup-dev.ps1 [-SkipDocker] [-Help]" -ForegroundColor White
    Write-Host "  -SkipDocker: Skip Docker-related setup" -ForegroundColor Gray
    Write-Host "  -Help: Show this help message" -ForegroundColor Gray
    exit 0
}

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up FinScope Development Environment..." -ForegroundColor Cyan

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if required tools are installed
function Test-Requirements {
    Write-Status "Checking requirements..."
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Success "Node.js found: $nodeVersion"
    } catch {
        Write-Error "Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    }
    
    # Check npm
    try {
        $npmVersion = npm --version
        Write-Success "npm found: $npmVersion"
    } catch {
        Write-Error "npm is not installed. Please install npm."
        exit 1
    }
    
    # Check Python
    try {
        $pythonVersion = python --version
        Write-Success "Python found: $pythonVersion"
    } catch {
        Write-Error "Python is not installed. Please install Python 3.8+ from https://python.org/"
        exit 1
    }
    
    # Check pip
    try {
        $pipVersion = pip --version
        Write-Success "pip found: $pipVersion"
    } catch {
        Write-Error "pip is not installed. Please install pip."
        exit 1
    }
    
    # Check Docker (optional)
    if (-not $SkipDocker) {
        try {
            $dockerVersion = docker --version
            Write-Success "Docker found: $dockerVersion"
            $script:DockerAvailable = $true
        } catch {
            Write-Warning "Docker is not installed. Docker setup will be skipped."
            $script:DockerAvailable = $false
        }
    } else {
        Write-Status "Skipping Docker check as requested."
        $script:DockerAvailable = $false
    }
    
    Write-Success "Requirements check completed!"
}

# Setup environment variables
function Set-Environment {
    Write-Status "Setting up environment variables..."
    
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Success "Created .env file from .env.example"
        Write-Warning "Please update the .env file with your actual API keys and configuration"
    } else {
        Write-Warning ".env file already exists, skipping..."
    }
}

# Install frontend dependencies
function Install-Frontend {
    Write-Status "Installing frontend dependencies..."
    
    try {
        npm install
        Write-Success "Frontend dependencies installed!"
    } catch {
        Write-Error "Failed to install frontend dependencies: $_"
        exit 1
    }
}

# Setup Python virtual environment and install backend dependencies
function Install-Backend {
    Write-Status "Setting up backend environment..."
    
    Set-Location "backend"
    
    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        python -m venv venv
        Write-Success "Created Python virtual environment"
    }
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install dependencies
    pip install -r requirements.txt
    
    Write-Success "Backend dependencies installed!"
    
    Set-Location ".."
}

# Setup database (if Docker is available)
function Set-Database {
    if ($script:DockerAvailable) {
        Write-Status "Setting up database with Docker..."
        
        try {
            # Start only PostgreSQL and Redis
            docker-compose up -d postgres redis
            
            # Wait for database to be ready
            Write-Status "Waiting for database to be ready..."
            Start-Sleep -Seconds 10
            
            Write-Success "Database setup completed!"
        } catch {
            Write-Warning "Failed to start Docker containers: $_"
            Write-Warning "Please start the containers manually or set up PostgreSQL and Redis locally."
        }
    } else {
        Write-Warning "Docker not available. Please set up PostgreSQL and Redis manually."
        Write-Warning "Alternatively, the application will use SQLite as fallback."
    }
}

# Generate SSL certificates for development
function Set-SSL {
    Write-Status "Setting up SSL certificates for development..."
    
    if (-not (Test-Path "nginx\ssl")) {
        New-Item -ItemType Directory -Path "nginx\ssl" -Force | Out-Null
    }
    
    # Check if OpenSSL is available
    try {
        $opensslVersion = openssl version
        Write-Status "OpenSSL found, generating certificates..."
        
        Set-Location "nginx"
        
        # Generate private key
        openssl genrsa -out ssl\key.pem 2048
        
        # Generate certificate
        openssl req -new -x509 -key ssl\key.pem -out ssl\cert.pem -days 365 -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"
        
        Set-Location ".."
        
        Write-Success "SSL certificates generated!"
    } catch {
        Write-Warning "OpenSSL not found. SSL certificates not generated."
        Write-Warning "You can install OpenSSL or use the application without HTTPS in development."
    }
}

# Create necessary directories
function New-Directories {
    Write-Status "Creating necessary directories..."
    
    $directories = @(
        "backend\logs",
        "nginx\ssl"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directories created!"
}

# Main setup function
function Start-Setup {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "       FinScope Development Setup        " -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    try {
        Test-Requirements
        New-Directories
        Set-Environment
        Install-Frontend
        Install-Backend
        Set-Database
        Set-SSL
        
        Write-Host ""
        Write-Host "==========================================" -ForegroundColor Green
        Write-Success "Setup completed successfully!"
        Write-Host "==========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Update the .env file with your API keys" -ForegroundColor White
        Write-Host "2. Start the development servers:" -ForegroundColor White
        Write-Host "   Frontend: npm run dev" -ForegroundColor Gray
        Write-Host "   Backend: cd backend && .\venv\Scripts\Activate.ps1 && uvicorn main:app --reload" -ForegroundColor Gray
        Write-Host "3. Or use Docker: docker-compose up" -ForegroundColor White
        Write-Host ""
        Write-Host "Access the application:" -ForegroundColor Cyan
        Write-Host "- Frontend: http://localhost:3000" -ForegroundColor White
        Write-Host "- Backend API: http://localhost:8000" -ForegroundColor White
        Write-Host "- API Documentation: http://localhost:8000/docs" -ForegroundColor White
        Write-Host ""
        Write-Warning "Don't forget to update your .env file with actual API keys!"
        
    } catch {
        Write-Error "Setup failed: $_"
        exit 1
    }
}

# Run main function
Start-Setup