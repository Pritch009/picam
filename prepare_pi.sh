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
sudo apt install -y libgl1-mesa-glx libcap-dev python3-dev

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required package
pip install -r requirements.txt
pip install -r pi_requirements.txt