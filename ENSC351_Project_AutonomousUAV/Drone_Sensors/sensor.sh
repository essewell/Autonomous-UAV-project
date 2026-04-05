# Drone Controller Setup Script for BeagleBone
# This script installs all necessary dependencies for the person-following drone system

set -e  # Exit on any error

echo "=================================================="
echo "Drone Controller Setup Script"
echo "=================================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run this script as root (use sudo)"
    exit 1
fi

# Update system packages
echo "Updating system packages..."
apt update
apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
apt install -y \
    python3 \
    python3-pip \
    python3-opencv \
    git \
    build-essential \
    cmake \
    libatlas-base-dev \
    gfortran

# Install Python packages for computer vision
echo "Installing Python packages for computer vision..."
pip3 install --break-system-packages \
    ultralytics \
    opencv-python \
    pillow \
    numpy \
    onnxruntime \
    onnx \
    onnxslim

# Download and setup YOLO model
echo "Setting up YOLO model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
echo "Exporting YOLO model to ONNX..."
python3 export_onnx.py
