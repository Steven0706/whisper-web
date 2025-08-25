#!/bin/bash

# Whisper Web Installation Script
# This script sets up a complete Whisper transcription service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

print_info "Starting Whisper Web installation..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.10"

if (( $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) )); then
    print_error "Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
print_success "Python $PYTHON_VERSION detected"

# Check for NVIDIA GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    GPU_AVAILABLE=true
else
    print_info "No NVIDIA GPU detected, will use CPU mode"
    GPU_AVAILABLE=false
fi

# Create virtual environment
print_info "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
print_success "Virtual environment created"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools > /dev/null 2>&1
print_success "Pip upgraded"

# Install PyTorch
print_info "Installing PyTorch (this may take a while)..."
if [ "$GPU_AVAILABLE" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 > /dev/null 2>&1
    print_success "PyTorch installed with CUDA support"
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
    print_success "PyTorch installed (CPU only)"
fi

# Install requirements
print_info "Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
print_success "Dependencies installed"

# Create directories
print_info "Creating necessary directories..."
mkdir -p uploads transcriptions recorded_audio templates
print_success "Directories created"

# Copy configuration
if [ ! -f .env ]; then
    print_info "Creating configuration file..."
    cp .env.example .env
    print_success "Configuration file created (.env)"
    print_info "Please edit .env to customize your settings"
else
    print_info "Configuration file already exists, skipping..."
fi

# Download a small model for testing
print_info "Downloading Whisper base model for initial setup..."
python3 -c "import whisper; whisper.load_model('base')" > /dev/null 2>&1
print_success "Whisper base model downloaded"

# Create systemd service
print_info "Would you like to install as a system service? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/whisper-web.service"
    
    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Whisper Web Speech-to-Text Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$(pwd)/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    print_success "Systemd service installed"
    print_info "To start the service: sudo systemctl start whisper-web"
    print_info "To enable auto-start: sudo systemctl enable whisper-web"
fi

# Test the installation
print_info "Testing installation..."
python3 -c "import whisper, torch, fastapi; print('All modules loaded successfully')" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

# Final instructions
echo ""
echo "========================================="
echo -e "${GREEN}Installation completed successfully!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit configuration: nano .env"
echo "2. Start the service:"
echo "   - Development: python app.py"
echo "   - Production: sudo systemctl start whisper-web"
echo "3. Access the web interface: http://localhost:5000"
echo ""
echo "For GPU users:"
echo "   Edit .env and set WHISPER_MODEL=large-v3 for best quality"
echo ""
print_info "Happy transcribing! 🎙️"