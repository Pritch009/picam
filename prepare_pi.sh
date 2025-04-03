#!/bin/bash

# This script prepares the Raspberry Pi for running the code.
# It installs the necessary dependencies and sets up a virtual environment.
# Update and upgrade the system
# Ensure the script is run with superuser privileges
if [ "$EUID" -ne 0 ]; then
		echo "Please run as root"
		exit
fi

sudo apt update
sudo apt upgrade -y
sudo apt install -y libgl1-mesa-glx libcap-dev python3-dev python3-picamera2

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv --system-site-packages venv
source venv/bin/activate
echo "Virtual environment created and activated."

echo "Installing python dependencies..."
# Install the required package
pip install -r requirements.txt
pip install -r pi_requirements.txt

echo "All dependencies installed!"