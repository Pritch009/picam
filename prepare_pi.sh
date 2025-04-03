#!/bin/bash

sudo apt update
sudo apt upgrade -y
sudo apt install libgl1-mesa-glx

pip install -r requirements.txt
pip install -r pi_requirements.txt